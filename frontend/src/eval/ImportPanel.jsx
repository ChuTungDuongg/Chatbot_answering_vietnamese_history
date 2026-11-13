import { useEffect, useRef, useState } from "react";
import { Upload } from "lucide-react";
import { api } from "./api";

export default function ImportPanel({ kind, onApplied, fallbackQuestion }) {
  const [file, setFile] = useState(null), [content, setContent] = useState(""), [format, setFormat] = useState("json"), [preview, setPreview] = useState(null), [policy, setPolicy] = useState("reject"), [message, setMessage] = useState(""), [busy, setBusy] = useState(false), [applying, setApplying] = useState(false);
  const generation = useRef(0);
  useEffect(() => () => { generation.current += 1; }, []);
  async function load(e) {
    const selected = e.target.files?.[0]; e.target.value = "";
    const request = ++generation.current;
    setPreview(null); setMessage(""); setFile(null); setContent(""); setPolicy("reject"); setBusy(false);
    if (!selected) return;
    if (selected.size > 2_000_000) { setMessage("Tệp vượt quá 2 MB."); return; }
    const nextFormat = selected.name.toLowerCase().endsWith(".jsonl") ? "jsonl" : selected.name.toLowerCase().endsWith(".csv") ? "csv" : "json";
    setBusy(true);
    try {
    let text = await selected.text();
    if (request !== generation.current) return;
    if (kind === "responses" && fallbackQuestion && selected.name.toLowerCase().endsWith(".json")) {
      try { const data = JSON.parse(text); const rows = Array.isArray(data) ? data : data.responses || [data]; text = JSON.stringify(rows.map(row => ({...row, question_id: row.question_id || fallbackQuestion}))); } catch { /* Server reports parse error. */ }
    }
    if (kind === "responses" && fallbackQuestion && selected.name.toLowerCase().endsWith(".jsonl")) {
      text = text.split(/\r?\n/).map(line => { if (!line.trim()) return line; try { const row = JSON.parse(line); return JSON.stringify({...row, question_id: row.question_id || fallbackQuestion}); } catch { return line; } }).join("\n");
    }
    setFile(selected.name); setFormat(nextFormat); setContent(text);
    } catch (e) { if (request === generation.current) setMessage(`Không đọc được tệp: ${e.message}`); }
    finally { if (request === generation.current) setBusy(false); }
  }
  async function requestPreview() {
    const request = ++generation.current, source = {kind, format, content};
    setBusy(true); setMessage(""); setPreview(null);
    try { const result = await api("/batch/preview", {method:"POST", body:JSON.stringify(source)}); if (request === generation.current) setPreview({...result, source}); }
    catch (e) { if (request === generation.current) setMessage(e.message); }
    finally { if (request === generation.current) setBusy(false); }
  }
  async function commit() {
    if (!preview || busy || !window.confirm(`Xác nhận nhập ${preview.count} bản ghi với chính sách ${policy}?`)) return;
    const request = ++generation.current;
    setBusy(true); setApplying(true); setMessage("");
    try { const result = await api("/batch/apply", {method:"POST", body:JSON.stringify({...preview.source, token:preview.token, policy})}); if (request === generation.current) { setMessage(`Đã xử lý ${result.imported} bản ghi.`); setPreview(null); onApplied?.(); } }
    catch (e) { if (request === generation.current) setMessage(e.message); }
    finally { if (request === generation.current) { setBusy(false); setApplying(false); } }
  }
  return <div className="import-panel"><label className="eval-field"><span>Chọn JSON, JSONL hoặc CSV</span><input type="file" accept=".json,.jsonl,.csv,application/json,text/csv" onChange={load} disabled={applying}/></label>{file && <p className="helper">{file} · tối đa 1.000 bản ghi / 2 MB</p>}<button disabled={!file || busy} onClick={requestPreview}><Upload size={15}/> Kiểm tra trước khi nhập</button>{preview && <div className="import-preview"><strong>{preview.valid} / {preview.count} bản ghi hợp lệ</strong><p>{preview.conflicts.length} ID đã tồn tại · {preview.errors.length} lỗi</p>{preview.errors.map((e,i) => <p className="import-error" key={i}>Dòng {e.line}: {e.message}</p>)}{preview.conflicts.slice(0,10).map(c => <p key={c.id}>Dòng {c.line}: {c.id} · revision {c.current_version}</p>)}{preview.sample.map((row,i) => <p key={i}>Dòng {row.line} · {row.id} {row.text && `— ${row.text}`}</p>)}{kind === "questions" && preview.conflicts.length > 0 && <label className="eval-field"><span>Xử lý ID đã tồn tại</span><select value={policy} disabled={applying} onChange={e => setPolicy(e.target.value)}><option value="reject">Không ghi đè</option><option value="skip">Bỏ qua</option><option value="overwrite">Ghi đè nếu revision khớp</option></select></label>}<button className="primary" disabled={busy || !!preview.errors.length || (kind === "questions" && preview.conflicts.length > 0 && policy === "reject")} onClick={commit}>Xác nhận nhập toàn bộ</button></div>}{message && <p role="status" className="helper">{message}</p>}</div>;
}
