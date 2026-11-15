const groups = ["checks", "tools", "evidence", "answer_metrics", "citation_metrics", "grounding_metrics", "semantic_checks"];
const display = value => value == null ? "Chưa ghi" : typeof value === "boolean" ? value ? "Đạt" : "Không đạt" : typeof value === "number" ? value.toFixed(3) : "Chưa ghi";

export default function RunMeasurements({ data, error, loading }) {
  const metrics = data?.metrics;
  const diagnostics = data?.diagnostics;
  return <section className="run-measurements" role="region" aria-label="Đo lường tự động"><h3>Đo lường tự động</h3><p className="helper">Các phép đo tất định về hành vi, độ trùng từ vựng và trích dẫn; không xác minh tính đúng đắn lịch sử. Dữ liệu thiếu được giữ ở trạng thái chưa ghi.</p>
    {loading ? <p>Đang tải đo lường…</p> : error ? <p className="helper">Không tải được đo lường: {error}</p> : !data ? <p>Chưa ghi đo lường cho run này.</p> : <>
      {metrics && <dl className="measurement-values">{groups.flatMap(group => Object.entries(metrics[group] || {}).filter(([, value]) => value == null || ["number", "boolean"].includes(typeof value)).map(([key, value]) => <div key={`${group}.${key}`}><dt><code>{group}.{key}</code></dt><dd>{display(value)}</dd></div>))}</dl>}
      <h3>Chẩn đoán tool an toàn</h3><p>Planning exit reason: {diagnostics?.planning?.exit_reason || "Chưa ghi"}</p>
      {diagnostics?.tool_calls == null ? <p>Chưa ghi tool trace.</p> : diagnostics.tool_calls.length ? <div className="table-scroll"><table className="eval-table tool-diagnostics"><caption>Metadata của tool; không hiển thị arguments hoặc reasoning thô</caption><thead><tr><th>Tool</th><th>Trạng thái / lỗi</th><th>Arguments hợp lệ</th><th>Fingerprint</th><th>Thời gian</th></tr></thead><tbody>{diagnostics.tool_calls.map((call, index) => <tr key={index}><th>{call.name || "Chưa ghi"}</th><td>{call.status || "Chưa ghi"}{call.error_code && <small>{call.error_code}</small>}</td><td>{display(call.arguments_valid)}</td><td><code>{call.argument_fingerprint || "Chưa ghi"}</code></td><td>{call.duration_ms == null ? "Chưa ghi" : `${call.duration_ms} ms`}</td></tr>)}</tbody></table></div> : <p>Không có tool call trong trace đã ghi.</p>}
      {diagnostics?.capture && <p className="helper">Capture: {[["tool_trace", "tool trace"], ["retrieved_evidence", "retrieved evidence"], ["final_context", "final context"]].map(([key, label]) => `${label}: ${diagnostics.capture[key] === true ? "Có" : "Chưa ghi"}`).join(" · ")}</p>}
    </>}
  </section>;
}
