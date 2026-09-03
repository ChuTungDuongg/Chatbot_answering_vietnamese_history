import { progressLabel } from "../services/progressLabels";

function StatusIndicator({ status }) {
  if (!status || ["idle", "done"].includes(status)) return null;
  const inactive = ["error", "cancelled"].includes(status);
  const label = status === "error" ? "Không thể hoàn tất câu trả lời"
    : status === "cancelled" ? "Đã dừng tạo câu trả lời"
      : progressLabel(status);
  return <div className={`status-indicator ${inactive ? `status-${status}` : ""}`} role="status">
    {!inactive && <span className="thinking-dots" aria-hidden="true"><i /><i /><i /></span>}
    <span>{label}</span>
  </div>;
}

export default StatusIndicator;
