export function progressLabel(status) {
  return {
    central_loading: "Đang khởi động mô hình...",
    central_tools: "Đang tìm tư liệu...",
    central_answering: "Đang tổng hợp câu trả lời...",
  }[status] ?? "Đang tìm và tổng hợp tư liệu...";
}
