const STATUS_LABELS = {
  idle: "",
  processing: "Đang xử lý...",
  retrieval_started: "Đang truy xuất tài liệu...",
  reranking: "Đang xếp hạng bằng chứng...",
  generating: "Đang tạo câu trả lời...",
  validating: "Đang kiểm tra câu trả lời...",
  validated: "Đã kiểm tra câu trả lời.",
  streaming: "Đang trả lời...",
  done: "",
  cancelled: "Đã dừng.",
  error: "Đã xảy ra lỗi.",
};

function StatusIndicator({ status }) {
  if (!status || status === "idle" || status === "done") {
    return null;
  }

  const label = STATUS_LABELS[status] ?? status;

  return (
    <div className={`status-indicator status-${status}`}>
      {status !== "error" && status !== "cancelled" && (
        <span className="status-dot" />
      )}
      <span>{label}</span>
    </div>
  );
}

export default StatusIndicator;
