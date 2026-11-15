import { useCallback, useEffect, useRef, useState } from "react";
import { FlaskConical, Play, RefreshCw, Square } from "lucide-react";
import { api } from "./api";
import EvidenceDrawer from "./EvidenceDrawer";
import AutomatedComparison from "./AutomatedComparison";

const statusNames = { ready: "Sẵn sàng", running: "Đang chạy", completed: "Hoàn tất", cancelled: "Đã hủy", paused: "Tạm dừng", pending: "Chờ chạy", failed: "Thất bại" };
const defaults = { experiment_id: "", smoke: false, question_ids: "", timeout_seconds: "180", max_retries: "0", requests_per_minute: "6", prompt_overrides: "{}", tool_descriptions: "{}", mcp_enabled: false, allowed_mcp_servers: "" };

function variantObject(text, allowed, label) {
  let value;
  try { value = JSON.parse(text); } catch { throw new Error(`${label}: JSON không hợp lệ.`); }
  if (!value || Array.isArray(value) || typeof value !== "object") throw new Error(`${label}: cần một JSON object.`);
  for (const [key, content] of Object.entries(value)) {
    if (!allowed.includes(key)) throw new Error(`${label}: tài nguyên không được hỗ trợ: ${key}.`);
    if (typeof content !== "string" || !content.trim()) throw new Error(`${label}: ${key} cần nội dung văn bản.`);
  }
  return value;
}
function creationOptions(form, capabilities) {
  if (!form.experiment_id) throw new Error("Hãy chọn experiment.");
  const numeric = {};
  for (const [key, label, min, max, integer] of [["timeout_seconds", "Timeout", 0.05, 1800, false], ["max_retries", "Retry", 0, 3, true], ["requests_per_minute", "Tần suất", 1, 600, true]]) {
    const value = Number(form[key]);
    if (!Number.isFinite(value) || value < min || value > max || (integer && !Number.isInteger(value))) throw new Error(`${label}: nhập số ${integer ? "nguyên " : ""}từ ${min} đến ${max}.`);
    numeric[key] = value;
  }
  return { experiment_id: form.experiment_id, smoke: form.smoke, question_ids: [...new Set(form.question_ids.split(/[\s,]+/).filter(Boolean))], ...numeric, prompt_overrides: variantObject(form.prompt_overrides, capabilities.prompt_resources || [], "Prompt overrides"), tool_descriptions: variantObject(form.tool_descriptions, capabilities.tools || [], "Tool descriptions"), mcp_enabled: form.mcp_enabled, allowed_mcp_servers: form.mcp_enabled ? [...new Set(form.allowed_mcp_servers.split(/[\s,]+/).filter(Boolean))] : [] };
}
function NumberField({ label, value, onChange, min, max, step = 1 }) { return <label className="eval-field"><span>{label}</span><input type="number" min={min} max={max} step={step} required value={value} onChange={e => onChange(e.target.value)}/></label>; }

export default function BenchmarkWorkspace() {
  const [experiments, setExperiments] = useState([]), [jobs, setJobs] = useState([]), [capabilities, setCapabilities] = useState(null), [error, setError] = useState(""), [loading, setLoading] = useState(true), [selected, setSelected] = useState(null), [comparison, setComparison] = useState(false);
  useEffect(() => {
    const controller = new AbortController(); let live = true;
    Promise.all([api("/experiments", { signal: controller.signal }), api("/benchmarks", { signal: controller.signal }), api("/benchmark/capabilities", { signal: controller.signal })]).then(([exps, saved, caps]) => { if (live) { setExperiments(exps); setJobs(saved); setCapabilities(caps); } }).catch(e => { if (live && e.name !== "AbortError") setError(e.message); }).finally(() => { if (live) setLoading(false); });
    return () => { live = false; controller.abort(); };
  }, []);
  const record = useCallback(job => { setJobs(previous => [job, ...previous.filter(value => value.id !== job.id)]); }, []);
  return <div className="eval-page v11-page benchmark-page"><div className="page-heading"><div><div className="eyebrow">REPRODUCIBLE EVALUATION · V1.2</div><h1>Benchmarks</h1><p>Đóng băng cấu hình, chạy trên snapshot và xem bằng chứng cho từng câu.</p></div><button aria-pressed={comparison} onClick={() => setComparison(!comparison)}>{comparison ? "Quay lại benchmark" : "So sánh tự động"}</button></div>
    {error && <div className="inline-error" role="alert">{error}</div>}
    {loading && <p role="status"><RefreshCw size={16} className="spin"/> Đang tải benchmark…</p>}
    {capabilities && !capabilities.live_available && <p className="review-warning" role="status">{capabilities.reason || "Chạy model chưa khả dụng trong môi trường này."}</p>}
    {comparison ? <AutomatedComparison experiments={experiments} benchmarks={jobs}/> : !loading && capabilities && <>
      <BenchmarkCreator experiments={experiments} capabilities={capabilities} onCreated={job => { record(job); setSelected({ id: job.id, initial: job }); }}/>
      <div className="benchmark-layout"><section className="panel benchmark-list" aria-label="Danh sách benchmark"><div className="panel-heading"><div><h2>Benchmark đã lưu</h2><p>Tạo job chỉ lưu cấu hình; Start/Resume cần xác nhận riêng.</p></div></div>{jobs.length ? jobs.map(job => <button className={`experiment-item ${selected?.id === job.id ? "selected" : ""}`} key={job.id} aria-pressed={selected?.id === job.id} onClick={() => setSelected({ id: job.id })}><strong>{job.id}</strong><small>{experiments.find(e => e.id === job.experiment_id)?.name || job.experiment_id} · {statusNames[job.status] || job.status}</small></button>) : <p className="helper">Chưa có benchmark.</p>}</section>
        {selected ? <BenchmarkDetail key={selected.id} id={selected.id} initial={selected.initial} capabilities={capabilities} onChanged={record}/> : <section className="panel empty-notice">Chọn benchmark để xem cấu hình và tiến độ.</section>}
      </div>
    </>}
  </div>;
}

function BenchmarkCreator({ experiments, capabilities, onCreated }) {
  const [form, setForm] = useState(defaults), [preview, setPreview] = useState(null), [error, setError] = useState(""), [busy, setBusy] = useState(false);
  const mutation = useRef(false), mounted = useRef(true), errorNode = useRef(null);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  function update(key, value) { setForm(previous => ({ ...previous, [key]: value })); setPreview(null); setError(""); }
  function review(event) { event.preventDefault(); try { setPreview(creationOptions(form, capabilities)); setError(""); } catch(e) { setPreview(null); setError(e.message); queueMicrotask(() => errorNode.current?.focus()); } }
  async function create() {
    if (!preview || mutation.current) return;
    mutation.current = true; setBusy(true); setError("");
    try { const job = await api("/benchmarks", { method: "POST", body: JSON.stringify(preview) }); if (mounted.current) { setPreview(null); onCreated(job); } }
    catch(e) { if (mounted.current) setError(e.message); }
    finally { mutation.current = false; if (mounted.current) setBusy(false); }
  }
  const current = experiments.find(e => e.id === form.experiment_id);
  return <details className="panel benchmark-create" open><summary><FlaskConical size={17}/> Tạo benchmark</summary><form className="stack" onSubmit={review}><fieldset className="eval-form-fields" disabled={busy}>
    <p className="helper">Provider, model, chế độ chat và retrieval lấy từ experiment. Mỗi biến thể prompt/tool cần một experiment riêng. Prompt và mô tả tool bên dưới chỉ áp dụng cho benchmark này.</p>
    {error && <div className="inline-error" role="alert" ref={errorNode} tabIndex={-1}>{error}</div>}
    <div className="form-grid"><label className="eval-field"><span>Experiment benchmark</span><select required value={form.experiment_id} onChange={e => update("experiment_id", e.target.value)}><option value="">Chọn experiment…</option>{experiments.map(e => <option key={e.id} value={e.id}>{e.name} · {e.snapshot_id}</option>)}</select></label><label className="eval-field"><span>Question IDs (để trống: tất cả)</span><input value={form.question_ids} onChange={e => update("question_ids", e.target.value)} placeholder="HIST-EVAL-001, HIST-EVAL-002"/></label></div>
    {current && <div className="metadata-strip"><span>Snapshot<strong>{current.snapshot_id}</strong></span><span>Model<strong>{current.provider || "Chưa ghi"} / {current.model_id || "Chưa ghi"}</strong></span><span>Chat / Backend<strong>{current.chat_mode || "Chưa ghi"} / {current.dense_backend || "Chưa ghi"}</strong></span><span>Final K<strong>{current.final_k ?? "Chưa ghi"}</strong></span></div>}
    <label className="checkbox-line"><input type="checkbox" checked={form.smoke} onChange={e => update("smoke", e.target.checked)}/> Chỉ chạy smoke set</label>
    <div className="benchmark-options"><NumberField label="Timeout mỗi câu (giây)" value={form.timeout_seconds} onChange={value => update("timeout_seconds", value)} min={0.05} max={1800} step={0.05}/><NumberField label="Số lần retry tối đa" value={form.max_retries} onChange={value => update("max_retries", value)} min={0} max={3}/><NumberField label="Yêu cầu mỗi phút" value={form.requests_per_minute} onChange={value => update("requests_per_minute", value)} min={1} max={600}/></div><p className="helper">Mặc định không retry. Mỗi retry đã chọn có thể phát sinh thêm chi phí.</p>
    <details><summary>Biến thể prompt & mô tả tool</summary><div className="form-grid benchmark-variants"><label className="eval-field"><span>Prompt overrides (JSON)</span><small>Tài nguyên hỗ trợ: {(capabilities.prompt_resources || []).join(", ") || "Không có"}</small><textarea aria-label="Prompt overrides (JSON)" rows={5} value={form.prompt_overrides} onChange={e => update("prompt_overrides", e.target.value)} spellCheck={false}/></label><label className="eval-field"><span>Tool descriptions (JSON)</span><small>Chỉ thay mô tả: {(capabilities.tools || []).join(", ") || "Không có"}</small><textarea aria-label="Tool descriptions (JSON)" rows={5} value={form.tool_descriptions} onChange={e => update("tool_descriptions", e.target.value)} spellCheck={false}/></label></div></details>
    <details><summary>MCP</summary><label className="checkbox-line"><input type="checkbox" checked={form.mcp_enabled} onChange={e => update("mcp_enabled", e.target.checked)}/> Cho phép MCP trong benchmark</label>{form.mcp_enabled && <label className="eval-field"><span>MCP servers được phép</span><input value={form.allowed_mcp_servers} onChange={e => update("allowed_mcp_servers", e.target.value)} placeholder="Tên server, phân tách bằng dấu phẩy"/></label>}</details>
    <button className="primary" type="submit" disabled={!form.experiment_id || busy}>Xem cấu hình trước khi tạo</button>
    {preview && <section className="benchmark-preview stack" aria-label="Xem trước benchmark"><h3>Xem trước cấu hình</h3><p>Chưa gọi model. Hash của cấu hình đã đóng băng sẽ xuất hiện sau khi tạo.</p><pre>{JSON.stringify({ experiment: current, options: preview }, null, 2)}</pre><button className="primary" type="button" disabled={busy} onClick={create}>{busy ? "Đang tạo…" : "Tạo benchmark chưa chạy"}</button></section>}
    {!experiments.length && <p className="helper">Tạo experiment gắn snapshot trước tại workspace Experiments.</p>}
  </fieldset></form></details>;
}

function BenchmarkDetail({ id, initial, capabilities, onChanged }) {
  const [data, setData] = useState(initial || null), [error, setError] = useState(""), [busy, setBusy] = useState(false), [confirmed, setConfirmed] = useState(false), [inspected, setInspected] = useState("");
  const mounted = useRef(true), mutation = useRef(false), revision = useRef(0);
  const [observedAt, setObservedAt] = useState(Date.now);
  const closeInspector = useCallback(() => setInspected(""), []);
  useEffect(() => { const controller = new AbortController(); mounted.current = true; if (!initial) api(`/benchmarks/${encodeURIComponent(id)}`, { signal: controller.signal }).then(value => { if (mounted.current) { setData(value); setObservedAt(Date.now()); } }).catch(e => { if (mounted.current && e.name !== "AbortError") setError(e.message); }); return () => { mounted.current = false; controller.abort(); }; }, [id, initial]);
  useEffect(() => {
    if (data?.status !== "running" || busy) return;
    const controller = new AbortController(); let live = true, timer;
    async function poll() {
      const request = revision.current;
      try { const value = await api(`/benchmarks/${encodeURIComponent(id)}`, { signal: controller.signal }); if (!live || request !== revision.current) return; setData(value); setObservedAt(Date.now()); onChanged(value); setError(""); if (value.status === "running") timer = setTimeout(poll, 2500); }
      catch(e) { if (live && request === revision.current && e.name !== "AbortError") { setError(`Không cập nhật được tiến độ: ${e.message}`); timer = setTimeout(poll, 5000); } }
    }
    timer = setTimeout(poll, 2500);
    return () => { live = false; clearTimeout(timer); controller.abort(); };
  }, [id, data?.status, busy, onChanged]);
  async function action(kind) {
    if (mutation.current || (kind === "start" && (!confirmed || !capabilities.live_available))) return;
    mutation.current = true; revision.current++; setBusy(true); setError(""); setConfirmed(false);
    try { const result = await api(`/benchmarks/${encodeURIComponent(id)}/${kind}`, { method: "POST", body: JSON.stringify(kind === "start" ? { confirm_live: true } : {}) }); if (mounted.current) { setData(result); setObservedAt(Date.now()); onChanged(result); } }
    catch(e) { if (mounted.current) setError(e.message); }
    finally { mutation.current = false; if (mounted.current) setBusy(false); }
  }
  const expired = data?.status === "running" && data.lease_expires_at != null && data.lease_expires_at * 1000 <= observedAt;
  const counts = data?.counts || {}, terminal = (counts.completed || 0) + (counts.failed || 0) + (counts.cancelled || 0), canStart = (["ready", "paused", "cancelled"].includes(data?.status) || expired) && (counts.total || 0) > (counts.completed || 0);
  return <section className="panel stack benchmark-detail" aria-label="Chi tiết benchmark">{error && <p className="inline-error" role="alert">{error}</p>}{!data ? <p role="status">Đang tải cấu hình…</p> : <>
    <div className="panel-heading"><div><h2>Chi tiết benchmark</h2><p><code>{data.id}</code> · snapshot {data.snapshot_id}</p></div><span className={`eval-badge benchmark-state ${data.status}`}>{statusNames[data.status] || data.status}</span></div>
    <dl className="benchmark-hashes">{[["Configuration hash", data.configuration_hash], ["Prompt hash", data.prompt_hash], ["Tool hash", data.tool_hash]].map(([label, value]) => <div key={label}><dt>{label}</dt><dd><code>{value || "Chưa ghi"}</code></dd></div>)}</dl>
    <details><summary>Cấu hình đã đóng băng</summary><pre className="benchmark-json">{JSON.stringify({ configuration: data.configuration, options: data.options }, null, 2)}</pre></details>
    {expired && <p className="review-warning">Lease của worker đã hết hạn. Request trước có thể đã phát sinh chi phí; Resume chỉ chạy sau xác nhận của bạn.</p>}
    {data.error_code && <p className="inline-error">Mã lỗi job: {data.error_code}</p>}
    {data.status === "paused" && <p className="review-warning">Lần chạy trước đã tạm dừng. Request bị gián đoạn có thể đã phát sinh chi phí; kiểm tra attempts và lỗi trước khi Resume.</p>}
    {canStart && <div className="benchmark-confirm"><label className="checkbox-line"><input type="checkbox" checked={confirmed} disabled={busy} onChange={e => setConfirmed(e.target.checked)}/> Tôi xác nhận gọi model có tính phí cho lần Start/Resume này.</label><button className="primary" disabled={busy || !confirmed || !capabilities.live_available} onClick={() => action("start")}><Play size={15}/>{data.status === "ready" ? "Start" : "Resume"} · gọi model có tính phí</button></div>}
    {["running", "ready", "paused"].includes(data.status) && <button disabled={busy || data.cancel_requested} onClick={() => action("cancel")}><Square size={14}/>{data.cancel_requested ? "Đang yêu cầu hủy…" : "Hủy benchmark"}</button>}
    <div className="benchmark-progress" role="status" aria-live="polite" aria-atomic="true"><strong>{terminal}/{counts.total || 0} câu đã kết thúc</strong><div className="eval-progress" role="progressbar" aria-label="Tiến độ benchmark" aria-valuemin={0} aria-valuemax={counts.total || 0} aria-valuenow={terminal}><span style={{ width: `${counts.total ? Math.min(100, terminal / counts.total * 100) : 0}%` }}/></div><p>Chờ {counts.pending ?? 0} · Đang chạy {counts.running ?? 0} · Hoàn tất {counts.completed ?? 0} · Lỗi {counts.failed ?? 0} · Hủy {counts.cancelled ?? 0}</p></div>
    <div className="table-scroll"><table className="eval-table benchmark-table"><caption>Kết quả từng câu</caption><thead><tr><th>Câu hỏi</th><th>Trạng thái</th><th>Attempts</th><th>Lỗi</th><th>Run & bằng chứng</th></tr></thead><tbody>{(data.items || []).map(item => <tr key={item.question_id}><th><code>{item.question_id}</code></th><td>{statusNames[item.status] || item.status}</td><td>{item.attempts ?? 0}</td><td>{item.error_code || "—"}</td><td>{item.run_id ? <button onClick={() => setInspected(item.run_id)}>Xem bằng chứng {item.question_id}</button> : "Chưa có run"}</td></tr>)}</tbody></table></div>
    {inspected && <EvidenceDrawer runId={inspected} onClose={closeInspector}/>}
  </>}</section>;
}
