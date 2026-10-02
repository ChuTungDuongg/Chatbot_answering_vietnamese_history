import { Check, Circle, LoaderCircle, Square, X } from "lucide-react";

export default function RetrievalProgress({ pipeline = [], backend, content, status }) {
  if (!pipeline.length) return null;
  const active = pipeline.findLast((item) => item.state === "started");
  const retrieved = pipeline.some((item) => item.stage === "dense_search" && item.state === "completed");
  const mcpLabels = [...new Set(pipeline.filter((item) => item.provider === "mcp" && item.state === "completed").map((item) => item.label.replace(/^Tra cứu /, "")))];
  const label = status === "cancelled" ? "Đã dừng" : status === "error" ? "Không thể hoàn tất"
    : content ? `${retrieved ? `Đã truy xuất bằng ${backend === "qdrant" ? "Qdrant" : "FAISS"}` : "Đã chuẩn bị ngữ cảnh"}${mcpLabels.length ? ` · ${mcpLabels.join(", ")}` : ""}${active ? " · Đang tạo câu trả lời" : ""}`
      : active?.label ?? "Đã chuẩn bị nguồn";
  return <details className="retrieval-progress" open={!content || undefined} role="status" aria-live="polite">
    <summary>{active && !["error", "cancelled", "done"].includes(status) && <LoaderCircle className="pipeline-spinner" aria-hidden="true" />}{label}</summary>
    <ol>{pipeline.map((item) => {
      const Icon = { completed: Check, started: LoaderCircle, failed: X, cancelled: Square }[item.state] ?? Circle;
      return <li key={item.stage} data-state={item.state}><Icon aria-hidden="true" className={item.state === "started" ? "pipeline-spinner" : ""} /><span>{item.label}</span></li>;
    })}</ol>
  </details>;
}
