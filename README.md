# Sử Việt

**Trợ lý tra cứu lịch sử có dẫn nguồn.**

Sử Việt kết hợp truy xuất tư liệu và mô hình ngôn ngữ để hỗ trợ tìm hiểu lịch sử Việt Nam. Câu trả lời được truyền trực tiếp khi model sinh, kèm nguồn tham khảo để người dùng kiểm tra và đối chiếu.

Ứng dụng hỗ trợ hội thoại, tra cứu tài liệu PDF/ảnh tải lên và lựa chọn FAISS hoặc Qdrant cho từng câu hỏi. Trên Modal, một URL HTTPS phục vụ giao diện và API sau màn hình đăng nhập bằng mật khẩu.

| Chế độ | Cách hoạt động | Model |
| --- | --- | --- |
| **Hybrid RAG** | Truy xuất tư liệu rồi sinh câu trả lời từ context. | Qwen3-4B, vanilla hoặc SFT adapter |
| **Central Agent** | Lập kế hoạch, dùng công cụ được cấp quyền và tổng hợp câu trả lời. | Qwen3-8B, vanilla |

Backend dùng FastAPI; frontend dùng React/Vite. Retrieval kết hợp E5, FAISS/Qdrant, BM25S và cross-encoder reranking trên Corpus V1.

Inference có thể chọn Qwen local hoặc OpenAI Responses API cho từng request, độc lập với chat mode và dense backend. OpenAI mặc định tắt và được bảo vệ bằng phiên đăng nhập và hạn mức phía máy chủ; xem [cấu hình bảo mật](docs/CONFIGURATION.md#openai-responses-api). Không đặt OpenAI API key trong trình duyệt.

Modal mặc định chạy CPU với OpenAI API; chỉ đặt `LOCAL_INFERENCE_ENABLED=true` khi muốn cấp GPU và chạy Qwen. Central ưu tiên Qdrant cho tư liệu Việt Nam và Wikipedia cho lịch sử thế giới, đồng thời tôn trọng lựa chọn nguồn/công cụ của người dùng.

## Bắt đầu

- [Cài đặt và chạy local](docs/SETUP.md)
- [Triển khai frontend và API trên Modal](docs/DEPLOYMENT.md)
- [Khôi phục corpus, index và model artifacts](docs/ARTIFACTS.md)

Clone repository chưa bao gồm corpus/index lớn, adapter weights hoặc model cache. Chế độ `api-only` cho phép khởi động API mà chưa cần các artifacts này.

## Tài liệu

| Nội dung | Hướng dẫn |
| --- | --- |
| Thành phần và luồng xử lý | [Kiến trúc hệ thống](docs/ARCHITECTURE.md) |
| Biến môi trường, model và giới hạn | [Cấu hình](docs/CONFIGURATION.md) |
| Corpus và tìm kiếm | [Xây corpus](docs/CORPUS_BUILD.md) · [Retrieval](docs/RETRIEVAL.md) · [Qdrant migration](docs/QDRANT_MIGRATION.md) |
| Công cụ ngoài | [Central tools và MCP](docs/TOOLS_AND_MCP.md) |
| Bảo trì prompt và built-in tools | [Markdown prompts và YAML tool contracts](docs/PROMPTS_AND_TOOLS.md) |
| Training và kiểm chứng chất lượng | [Training](docs/TRAINING_AND_EVALUATION.md) · [Evaluation](docs/EVALUATION.md) |
| Hiệu năng và vận hành | [Benchmark](docs/BENCHMARKS.md) · [Telemetry](docs/RUNTIME_TELEMETRY.md) |
| Làm việc với mã nguồn | [Phát triển và kiểm tra](docs/DEVELOPMENT.md) |

Xem [mục lục docs/](docs/README.md) để tra cứu toàn bộ tài liệu. Kết quả tự động và trích dẫn cần được kiểm chứng trên dữ liệu thực tế; [hướng dẫn evaluation](docs/EVALUATION.md) ghi rõ phạm vi metric và các giới hạn đã biết.
