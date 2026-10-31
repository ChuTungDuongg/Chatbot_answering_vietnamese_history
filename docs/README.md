# Sử Việt — tài liệu hệ thống

Tài liệu mô tả mã nguồn và cấu hình đang có trong repository. Giá trị mặc định của ứng dụng lấy từ [`app/config.py`](../app/config.py); cấu hình triển khai Modal lấy từ [`modal_app.py`](../modal_app.py). Thông số corpus/index của một bản dữ liệu lấy từ manifest đi kèm bản đó.

## Đọc theo nhu cầu

| Cần tìm | Tài liệu |
| --- | --- |
| Thành phần, luồng xử lý, API, lưu trữ | [Kiến trúc](ARCHITECTURE.md) |
| Cài đặt và chạy backend/frontend | [Chạy local](SETUP.md) |
| Biến môi trường, model, token budget, giới hạn | [Cấu hình](CONFIGURATION.md) |
| Frontend CPU, API GPU, deploy và kiểm tra Modal | [Triển khai](DEPLOYMENT.md) |
| Bộ dữ liệu, đường dẫn, khôi phục, bảo toàn dữ liệu | [Artifacts](ARTIFACTS.md) |
| Xây corpus từ UVW-2026 trên Colab/Drive | [Xây corpus](CORPUS_BUILD.md) |
| Chuyển Qdrant Cloud về Docker local, resume/verify | [Qdrant migration](QDRANT_MIGRATION.md) |
| E5, FAISS/Qdrant, BM25S, tạo và kiểm tra index | [Retrieval](RETRIEVAL.md) |
| Công cụ Central, MCP và quyền từng request | [Tools và MCP](TOOLS_AND_MCP.md) |
| Nhãn câu hỏi, retrieval và các metric chất lượng | [Đánh giá](EVALUATION.md) |
| Quy trình gán nhãn có người kiểm duyệt | [GOLD annotation](GOLD_ANNOTATION_GUIDE.md) |
| Gán nhãn tự động và các batch SILVER | [SILVER benchmark](SILVER_BENCHMARK.md) |
| SILVER V1 cố định, QLoRA, adapter và six-way benchmark | [Training và đánh giá model](TRAINING_AND_EVALUATION.md) |
| Đo HTTP/SSE, định nghĩa latency và mẫu thí nghiệm | [Benchmark](BENCHMARKS.md) |
| Backend vLLM và thí nghiệm Transformers/vLLM | [vLLM](VLLM_LATENCY_EXPERIMENTS.md) |
| Serving monitoring, planner, token, finish reason và timing | [Telemetry](RUNTIME_TELEMETRY.md) |
| Bố cục mã nguồn, kiểm tra Python/frontend và CI | [Phát triển](DEVELOPMENT.md) |

## Phân biệt các lựa chọn

| Lựa chọn | Ý nghĩa |
| --- | --- |
| `APP_MODE=api-only/retrieval-only/full` | Thành phần được khởi tạo khi backend chạy. |
| Chat `hybrid/central` | Luồng trả lời và model dùng cho câu hỏi. |
| `MODEL_VARIANT=vanilla/sft` | Hybrid dùng model gốc hay gắn PEFT adapter. |
| `INFERENCE_BACKEND=transformers/vllm` | Engine chạy model. vLLM chỉ hỗ trợ vanilla và triển khai riêng từng chat mode. |
| `retrieval_backend=faiss/qdrant` | Dense search của từng request, trong số backend đã sẵn sàng. |

Corpus V1 là mặc định. V0 là dữ liệu cũ dùng khi cấu hình rõ ràng. Các nhãn SILVER là nhãn tự động; các câu hỏi fixture dùng kiểm tra pipeline. Kết quả test hoặc một lần smoke không chứng minh độ chính xác lịch sử hay hiệu năng triển khai.
