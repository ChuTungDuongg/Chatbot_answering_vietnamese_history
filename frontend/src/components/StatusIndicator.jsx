import { LoaderCircle } from "lucide-react";

const STATUS_LABELS = {
  processing: "Đang tìm bằng chứng phù hợp",
  retrieval_started: "Đang truy xuất kho sử liệu",
  reranking: "Đang xếp hạng bằng chứng",
  generating: "Đang soạn câu trả lời",
  validating: "Đang kiểm tra độ chính xác",
  agentic_analyzing: "Đang phân tích câu hỏi...",
  agentic_local_search: "Đang tìm trong kho sử liệu...",
  agentic_external_check: "Đang kiểm tra thêm nguồn ngoài...",
  agentic_evidence_check: "Đang đối chiếu bằng chứng...",
  agentic_answering: "Đang soạn câu trả lời...",
  hybrid_retrieval: "Đang truy xuất kho sử liệu...",
  hybrid_answering: "Đang soạn câu trả lời...",
  fast_retrieval: "Đang tìm nhanh trong kho sử liệu...",
  fast_answering: "Đang chuẩn bị câu trả lời...",
  validated: "Đã kiểm tra câu trả lời",
  streaming: "Đang trả lời",
  cancelled: "Đã dừng tạo câu trả lời",
  error: "Không thể hoàn tất câu trả lời",
};

function StatusIndicator({ status }) {
  if (!status || ["idle", "done"].includes(status)) return null;

  const isActive = !["error", "cancelled"].includes(status);

  return (
    <div className={`status-indicator status-${status}`} role="status">
      {isActive && <LoaderCircle className="spin" />}
      <span>{STATUS_LABELS[status] ?? "Đang xử lý"}</span>
    </div>
  );
}

export default StatusIndicator;
