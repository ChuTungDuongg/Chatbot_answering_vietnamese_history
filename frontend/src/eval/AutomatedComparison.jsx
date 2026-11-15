import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import EvidenceDrawer from "./EvidenceDrawer";

const display = value => value == null ? "Chưa ghi" : typeof value === "boolean" ? (value ? "Đạt" : "Không đạt") : typeof value === "number" ? value.toFixed(3) : "Chưa ghi";
function pairedChange(pair, directions) {
  const known = pair.paired > 0 && typeof pair.mean_delta === "number" && Number.isFinite(pair.mean_delta);
  if (!known) return { delta: "Chưa ghi", label: "Chưa đủ cặp dữ liệu", state: "neutral" };
  const delta = `${pair.mean_delta > 0 ? "+" : ""}${pair.mean_delta.toFixed(3)}`;
  const direction = pair.direction || directions?.[pair.metric];
  if (!["higher", "lower"].includes(direction)) return { delta, label: "Chưa rõ chiều đánh giá", state: "neutral" };
  if (pair.mean_delta === 0) return { delta, label: "Không đổi", state: "neutral" };
  const regression = direction === "higher" ? pair.mean_delta < 0 : pair.mean_delta > 0;
  return { delta, label: regression ? "B hồi quy so với A" : "B cải thiện so với A", state: regression ? "regression" : "improvement" };
}
export default function AutomatedComparison({ experiments, benchmarks = [] }) {
  const [selected, setSelected] = useState([]), [jobs, setJobs] = useState({}), [data, setData] = useState(null), [error, setError] = useState(""), [busy, setBusy] = useState(false), [detail, setDetail] = useState("");
  const generation = useRef(0);
  useEffect(() => () => { generation.current++; }, []);
  function invalidate() { generation.current++; setData(null); setError(""); setBusy(false); setDetail(""); }
  const sameSnapshot = new Set(experiments.filter(e => selected.includes(e.id)).map(e => e.snapshot_id)).size <= 1;
  async function compare() {
    if (selected.length < 2 || !sameSnapshot) return;
    const request = ++generation.current; setBusy(true); setError(""); setData(null);
    const benchmarkIds = selected.map(id => jobs[id]).filter(Boolean);
    try { const result = await api("/benchmark/compare", { method: "POST", body: JSON.stringify({ experiment_ids: selected, ...(benchmarkIds.length ? { benchmark_ids: benchmarkIds } : {}) }) }); if (request === generation.current) setData(result); }
    catch(e) { if (request === generation.current) setError(e.message); }
    finally { if (request === generation.current) setBusy(false); }
  }
  const metricKeys = data ? [...new Set(data.experiments.flatMap(e => Object.keys(e.metrics || {})))] : [];
  return <section className="panel stack automated-comparison" role="region" aria-label="So sánh đo lường tự động"><div className="panel-heading"><div><h2>So sánh đo lường tự động</h2><p>Cùng snapshot. Các chỉ số từ vựng, hành vi và trích dẫn không xác minh tính đúng đắn lịch sử.</p></div></div>{error && <p className="inline-error" role="alert">{error}</p>}
    <div className="experiment-options">{experiments.map(exp => <div className="automated-option" key={exp.id}><label><input type="checkbox" checked={selected.includes(exp.id)} onChange={e => { setSelected(previous => e.target.checked ? [...previous, exp.id] : previous.filter(id => id !== exp.id)); invalidate(); }}/><span><strong>{exp.name}</strong><small>{exp.model_id || "Chưa ghi"} · {exp.snapshot_id}</small></span></label>{selected.includes(exp.id) && <label className="eval-field"><span>Benchmark của {exp.name}</span><select value={jobs[exp.id] || ""} onChange={e => { setJobs(previous => ({ ...previous, [exp.id]: e.target.value })); invalidate(); }}><option value="">Benchmark mới nhất</option>{benchmarks.filter(job => job.experiment_id === exp.id).map(job => <option key={job.id} value={job.id}>{job.id}</option>)}</select></label>}</div>)}</div>
    {!sameSnapshot && <p role="alert" className="inline-error">Các experiment cần dùng cùng snapshot bất biến.</p>}
    <button className="primary" disabled={selected.length < 2 || !sameSnapshot || busy || (selected.some(id => jobs[id]) && !selected.every(id => jobs[id]))} onClick={compare}>{busy ? "Đang tính…" : "So sánh đo lường"}</button>
    {selected.some(id => jobs[id]) && !selected.every(id => jobs[id]) && <p className="helper">Chọn một benchmark cho mỗi experiment hoặc để tất cả ở benchmark mới nhất.</p>}
    {data && <><p className="comparison-note">Snapshot {data.snapshot_id}. {data.configuration_differences?.length ? `Khác cấu hình: ${data.configuration_differences.join(", ")}.` : "Cấu hình không có khác biệt được ghi nhận."}</p>
      <div className="table-scroll"><table className="eval-table"><caption>Các phép đo riêng biệt</caption><thead><tr><th>Chỉ số</th>{data.experiments.map(exp => <th key={exp.id}>{exp.name}</th>)}</tr></thead><tbody>
        <tr><th>Câu đã đo / tổng</th>{data.experiments.map(exp => <td key={exp.id}>{exp.evaluated_questions}/{exp.total_questions}</td>)}</tr><tr><th>Thiếu đo lường / thất bại</th>{data.experiments.map(exp => <td key={exp.id}>{exp.missing_measurements} / {exp.failed_questions}</td>)}</tr>
        {metricKeys.map(key => <tr key={key}><th><code>{key}</code>{data.metric_directions?.[key] && <small className="metric-direction">{data.metric_directions[key] === "higher" ? "Cao hơn tốt hơn" : "Thấp hơn tốt hơn"}</small>}</th>{data.experiments.map(exp => <td key={exp.id}>{display(exp.metrics?.[key])}</td>)}</tr>)}
        {[ ["latency_ms", "Latency (ms)"], ["input_tokens", "Input tokens"], ["output_tokens", "Output tokens"], ["tool_calls", "Tool calls"], ["cost", "Chi phí đã ghi"] ].map(([key, label]) => <tr key={key}><th>{label}</th>{data.experiments.map(exp => <td key={exp.id}>{display(exp[key])}</td>)}</tr>)}
        <tr><th>Điểm human (thủ công)</th>{data.experiments.map(exp => <td key={exp.id}>{display(exp.quality?.human_mean)}</td>)}</tr><tr><th>Điểm judge (đã nhập)</th>{data.experiments.map(exp => <td key={exp.id}>{display(exp.quality?.judge_mean)}</td>)}</tr>
      </tbody></table></div>
      {!!data.pairwise?.length && <details className="paired-metrics" open><summary>Chênh lệch & hồi quy theo cặp</summary><p className="helper">Δ = B − A, chỉ tính các câu có cả hai giá trị. Hồi quy dựa trên chiều tốt hơn của từng chỉ số; không phải kết luận về độ chính xác lịch sử.</p><div className="table-scroll"><table className="eval-table"><caption>Chênh lệch A/B theo chỉ số</caption><thead><tr><th>Chỉ số & cặp so sánh</th><th>Δ (B − A)</th><th>Diễn giải</th><th>A thắng / Hòa / B thắng</th></tr></thead><tbody>{data.pairwise.map((pair, index) => {
        const change = pairedChange(pair, data.metric_directions);
        return <tr key={index}><th><code>{pair.metric}</code><small className="metric-direction">A: {experiments.find(e => e.id === pair.a)?.name || pair.a}<br/>B: {experiments.find(e => e.id === pair.b)?.name || pair.b}</small></th><td>{change.delta}</td><td><span className={`paired-assessment ${change.state}`}>{change.label}</span><small className="metric-direction">{pair.paired ?? 0} cặp</small></td><td>{pair.paired > 0 ? `${pair.wins} / ${pair.ties} / ${pair.losses}` : "Chưa ghi"}</td></tr>;
      })}</tbody></table></div></details>}
      {!!data.rows?.length && <div className="table-scroll"><table className="eval-table"><caption>Run theo từng câu</caption><thead><tr><th>Câu hỏi</th>{data.experiments.map(exp => <th key={exp.id}>{exp.name}</th>)}</tr></thead><tbody>{data.rows.map(row => <tr key={row.question.id}><th><code>{row.question.id}</code><p>{row.question.question}</p></th>{data.experiments.map(exp => <td key={exp.id}>{row.cells?.[exp.id]?.run_id ? <button onClick={() => setDetail(row.cells[exp.id].run_id)}>Xem bằng chứng</button> : "Chưa ghi"}</td>)}</tr>)}</tbody></table></div>}
    </>}
    {detail && <EvidenceDrawer runId={detail} onClose={() => setDetail("")}/>}
  </section>;
}
