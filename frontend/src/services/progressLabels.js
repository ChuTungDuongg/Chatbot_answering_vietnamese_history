export function progressLabel(status) {
  return {
    request_preparation: "Phân tích câu hỏi...",
    query_analysis: "Phân tích câu hỏi...",
    embedding: "Tạo embedding truy vấn...",
    dense_search: "Truy vấn nguồn...",
    bm25_search: "Truy vấn BM25...",
    fusion: "Hợp nhất kết quả...",
    rerank: "Rerank tư liệu...",
    context_selection: "Chuẩn bị nguồn...",
    generation: "Tạo câu trả lời...",
    central_loading: "Đang khởi động mô hình...",
    central_tools: "Đang tìm tư liệu...",
    central_answering: "Đang tổng hợp câu trả lời...",
  }[status] ?? "Đang tìm và tổng hợp tư liệu...";
}
