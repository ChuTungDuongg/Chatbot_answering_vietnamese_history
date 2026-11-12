# Cấu hình hệ thống

[`app/config.py`](../app/config.py) khai báo và kiểm tra cấu hình bằng Pydantic Settings. Backend đọc `.env` tại thư mục chạy; lệnh trong tài liệu được chạy từ root repository. Biến môi trường của process được ưu tiên hơn `.env`. Các đường dẫn tương đối trong Settings được chuẩn hóa theo root repository. Khởi động lại backend sau khi đổi cấu hình.

[`.env.example`](../.env.example) là mẫu cấu hình local. Modal có các giá trị được đặt riêng trong [`modal_app.py`](../modal_app.py); xem [triển khai](DEPLOYMENT.md).

## Khởi tạo và đường dẫn

| Biến | Mặc định local | Ý nghĩa |
| --- | --- | --- |
| `APP_MODE` | `api-only` | `api-only`, `retrieval-only`, `full`. |
| `DEVICE` / `DTYPE` | `cpu` / `bfloat16` | Thiết bị và dtype; cần phù hợp phần cứng chạy model. |
| `ARTIFACT_ROOT` | `./artifacts/corpus_v1` | Root dữ liệu V1. |
| `CORPUS_PATH` | `<root>/chunks.jsonl` với V1 | Override file corpus. |
| `RETRIEVAL_ROOT` | `<root>/retrieval` | Thư mục index. |
| `INFERENCE_CONFIG_PATH` | `<root>/runtime/inference_config.json` với V1 | Thông số thuật toán retrieval. |
| `RUNTIME_MANIFEST_PATH` | `<root>/runtime/manifest.json` với V1 | Danh tính và số lượng corpus/index. |
| `CHAT_DATABASE_PATH` | `./data/chat.sqlite3` | Lưu hội thoại, messages và attachments. |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Danh sách origin, phân cách dấu phẩy. |

Khi đặt cả `CORPUS_PATH` và `RETRIEVAL_ROOT`, đường dẫn runtime mặc định nằm tại `runtime/` cạnh corpus. Hai override runtime riêng vẫn có ưu tiên cao hơn. Muốn dùng V0, đặt `ARTIFACT_ROOT=./artifacts/vn_history_deployment` và bỏ các override V1; Settings dùng layout `corpus/`, `config/` của V0. Xem [Artifacts](ARTIFACTS.md).

## Model và sinh câu trả lời

| Biến | Mặc định | Ràng buộc |
| --- | --- | --- |
| `HYBRID_MODEL_ID` | `Qwen/Qwen3-4B-Instruct-2507` | Settings yêu cầu đúng ID này. |
| `CENTRAL_MODEL_ID` | `Qwen/Qwen3-8B` | Settings yêu cầu đúng ID này; Central dùng vanilla. |
| `MODEL_VARIANT` | `vanilla` | `sft` yêu cầu `MODEL_ADAPTER_PATH`. |
| `MODEL_ADAPTER_PATH` | Không đặt | Thư mục PEFT có config và weights phù hợp model gốc. |
| `HYBRID_MODEL_REVISION`, `CENTRAL_MODEL_REVISION` | Không đặt | Có thể pin revision để tái lập. |
| `MODEL_CACHE_DIR` / `MODEL_LOCAL_FILES_ONLY` | Không đặt / `false` | Cache model và chế độ chỉ đọc file local. |
| `RUNTIME_LOADING_STRATEGY` | `lazy` | Central tải khi được gọi; `eager` tải từ startup. Hybrid luôn tải từ startup trong `full`. |
| `ENABLE_HYBRID_MODE`, `ENABLE_CENTRAL_MODE` | `true`, `true` | Bật runtime tương ứng. |
| `DEFAULT_INFERENCE_MODE` | `hybrid` | Chat mode khi request không chỉ định. |
| `DO_SAMPLE` / `ENABLE_THINKING` | `false` / `false` | Greedy decoding, tắt thinking. |
| `MODEL_TEMPERATURE` / `MODEL_TOP_P` | `0.7` / `1.0` | Temperature không bật sampling khi `DO_SAMPLE=false`. |
| `HYBRID_MAX_NEW_TOKENS` | `1536` | Chung cho vanilla và SFT. |
| `CENTRAL_ACTION_MAX_NEW_TOKENS` | `256` | Budget mỗi lần planner sinh action. |
| `CENTRAL_FINAL_MAX_NEW_TOKENS` | `1536` | Budget câu trả lời cuối. |
| `CENTRAL_MAX_ACTION_ROUNDS` | `4` | Cho phép `0..8`; completion hợp lệ có thể dừng sau round đầu. |
| `CENTRAL_OPENAI_PLANNING_REASONING_EFFORT` | `low` | Reasoning riêng cho planner OpenAI: low/medium/high/xhigh; `inherit` dùng effort request như trước. Final answer luôn giữ effort đã chọn. Không áp dụng cho local Qwen. |
| `CENTRAL_MAX_TOOL_CALLS_PER_ROUND` | `4` | Calls được thực thi tối đa mỗi round (1..16). |
| `CENTRAL_MAX_TOTAL_TOOL_CALLS` | `12` | Tổng calls được thực thi mỗi request (1..64). |
| `CENTRAL_MAX_IDENTICAL_TOOL_CALLS` | `1` | Repeat limit với arguments đã normalize (1..3). |
| `CENTRAL_MAX_PARALLEL_TOOL_CALLS` | `3` | Chỉ tool có policy an toàn và explicit independence (1..8). |
| `CENTRAL_EXECUTION_DEADLINE_SECONDS` | `180` | Planning + tools; final generation có timeout riêng. |
| `CENTRAL_TOOL_TIMEOUT_SECONDS` | `30` | Mỗi tool, tính cả chờ resource gate. |
| `CENTRAL_MAX_OBSERVATION_CHARS` | `8000` | JSON evidence observation mỗi call. |
| `CENTRAL_MAX_TOTAL_OBSERVATION_CHARS` | `48000` | Evidence observations/request; fixed native error closures đếm riêng. |
| `CENTRAL_MAX_PLANNING_OUTPUT_TOKENS` | `40000` | Actual usage cộng dồn, cap output round tiếp theo. |
| `CENTRAL_MAX_PLANNING_INPUT_TOKENS` | `120000` | Dừng sau round chạm budget; không ước lượng token chưa đo. |

Token budget là trần; model có thể dừng sớm bằng EOS. Live API/UI chỉ có hai chat modes, không có `response_mode`. Các giá trị `concise/standard/detailed` chỉ còn phục vụ dữ liệu và đánh giá offline đã cố định.

## OpenAI Responses API

OpenAI mặc định tắt. Modal bắt buộc xác thực bằng mật khẩu; cookie phiên đã xác thực cho phép sử dụng OpenAI mà không cần nhập token dịch vụ. `X-Client-ID` chỉ phân vùng hội thoại, không xác thực. API key và token dịch vụ chỉ được giữ phía server. Quota SQLite là lớp giới hạn bổ sung; cấu hình Modal giới hạn một container và không bảo đảm quota đồng bộ giữa nhiều replicas. Không mở rộng replicas khi chưa có cơ chế quota chung.

Đặt `OPENAI_ENABLED=true` và cung cấp `OPENAI_API_KEY` qua Modal Secret `MODAL_OPENAI_SECRET_NAME`. Trường hợp tích hợp dịch vụ cũ không dùng session vẫn cần `OPENAI_TRUSTED_INTERNAL_ONLY=true` và `OPENAI_ACCESS_TOKEN` riêng trên server. Không đưa secret vào frontend, Git hoặc log. `/ready` không trả secrets.

| Biến | Mặc định |
| --- | --- |
| OPENAI_DEFAULT_MODEL | gpt-6-luna; cho phép thêm gpt-6.1-terra |
| OPENAI_DEFAULT_REASONING_EFFORT | medium; cho phép low, medium, high, xhigh |
| OPENAI_TIMEOUT_SECONDS | 180 |
| OPENAI_MAX_OUTPUT_TOKENS | 25000 mỗi model call, bao gồm reasoning |
| OPENAI_MAX_REQUESTS_PER_HOUR / OPENAI_MAX_REQUESTS_PER_DAY | 20 / 100 trên database |
| OPENAI_MAX_CONCURRENT_REQUESTS | 2 trên database |
| LOCAL_INFERENCE_ENABLED | `true` trong Settings local; `.env.example` và Modal mặc định `false` để tránh tải Qwen |
| MODAL_RETRIEVAL_GPU_CLASS | Rỗng: CPU; `T4` hoặc `L4` cấp GPU cho E5/reranker khi Qwen tắt |
| MODAL_GPU_CLASS | `L4`; chỉ dùng khi `LOCAL_INFERENCE_ENABLED=true` |

Central có thể gọi model tới số action rounds đã cấu hình cộng một lần sinh cuối; có thêm aggregate planning budgets bên trên. Usage không có từ provider để null; input budget có thể vượt bởi một round đang chạy, không phải hard USD cap. High và Very High có thể tăng thời gian và chi phí. API trả incomplete nếu hết budget và ứng dụng không lưu answer đó. Requests dùng store=false; điều này không tự tạo bảo đảm Zero Data Retention cho tài khoản. Model access, quota và chính sách dữ liệu phải được xác minh với tài khoản OpenAI. Chi tiết dừng và protocol overhead: [Central v2](AGENTIC_RAG_V2.md).

Payload cũ mặc định local. Với OpenAI, gửi provider là openai, tùy chọn model và reasoning_effort, và cookie phiên đăng nhập. Model/effort ngoài allowlist bị từ chối. Status stream báo đang xử lý, không hiển thị reasoning nội bộ. Khi model không khả dụng hoặc bị rate limit, thử lại hoặc chọn provider khác; không tự chuyển provider.

## Dense retrieval

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `RETRIEVAL_DENSE_BACKEND` | `faiss` | Backend mặc định, bắt buộc tải được. |
| `RETRIEVAL_AVAILABLE_BACKENDS` | Chỉ backend mặc định | Đặt `faiss,qdrant` để bật cả hai; phải chứa backend mặc định. |
| `QDRANT_URL` / `QDRANT_API_KEY` | Không đặt | Kết nối Qdrant; key cần khi server yêu cầu. |
| `QDRANT_COLLECTION` | `vn_history_v1_e5` | Collection phải khớp manifest. |
| `QDRANT_HNSW_EF` | Không đặt | Dùng mặc định server nếu không override. |
| `QDRANT_TIMEOUT_SECONDS` | `30` | Khoảng `1..300` giây. |

`retrieval_backend` trong request chọn một backend đã khả dụng. Backend tùy chọn lỗi khi startup không được advertise; backend mặc định lỗi làm startup thất bại. Request chọn backend không khả dụng hoặc lỗi giữa chừng không tự đổi sang backend khác. Thông số E5/BM25S/RRF/reranker nằm trong artifact config, xem [Retrieval](RETRIEVAL.md).

## Engine và công cụ ngoài

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `INFERENCE_BACKEND` | `transformers` | `vllm` cần Linux CUDA, môi trường riêng và chỉ bật một chat runtime. |
| `VLLM_ENABLE_PREFIX_CACHING` | `false` | Prefix caching. |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.75` | Lớn hơn 0 và nhỏ hơn 1. |
| `VLLM_MAX_NUM_SEQS` | `8` | Số sequence tối đa trong engine. |
| `VLLM_MAX_MODEL_LEN` | Không đặt | Dùng context native nếu không giới hạn rõ. |
| `VLLM_ENFORCE_EAGER` | `false` | Tùy chọn engine cho thí nghiệm/debug. |
| `CENTRAL_ENABLE_DOCUMENTS`, `CENTRAL_ENABLE_WIKIPEDIA` | `true`, `true` | Built-in tools của Central. |
| `CENTRAL_ENABLE_WEB` | `false` | Cần cấu hình provider/key khi bật web search. |
| `WEB_SEARCH_PROVIDER` / `WEB_SEARCH_API_KEY` | `local-only` / không đặt | Provider và credentials riêng. |
| `MCP_ENABLED` | `false` | Bật kết nối MCP lúc startup Central/full. |
| `MCP_CONFIG_PATH` | `./config/mcp_servers.local.json` | Cấu hình server riêng, Git-ignored. |
| `MCP_MAX_TOOLS_PER_REQUEST` | `8` | Khoảng `1..32`. |
| `MCP_SCHEMA_BUDGET_BYTES` | `16384` | Khoảng `1024..65536`. |

vLLM hiện từ chối SFT/LoRA. Xem [vLLM](VLLM_LATENCY_EXPERIMENTS.md) và [Tools/MCP](TOOLS_AND_MCP.md) cho quy trình sử dụng. `SILVER_LLM_BASE_URL` và `SILVER_LLM_API_KEY` chỉ phục vụ [annotation offline](SILVER_BENCHMARK.md), không cấu hình model của live app.

## Xác thực và observability

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `AUTH_ENABLED` | `false` local | Modal luôn bật; production không thể chạy khi tắt. |
| `AUTH_PASSWORD_HASH` | Không đặt | Hash Argon2id trong Modal Secret. |
| `AUTH_SESSION_SECRET` | Không đặt | Secret ký cookie ngẫu nhiên, ít nhất 32 byte. |
| `MODAL_AUTH_SECRET_NAME` | `vn-history-auth` | Tên Modal Secret bắt buộc. |
| `VITE_API_BASE_URL` | Rỗng | Frontend image dùng cùng origin. |
| `VITE_SHOW_DEBUG_TRACE` | `false` | Hiện giao diện trace nếu bật. |
| `MODAL_OBSERVABILITY_SECRET_NAME` | Không đặt | Secret chứa `OBSERVABILITY_ADMIN_TOKEN`. |
| `OBSERVABILITY_ADMIN_TOKEN` | Không đặt | Token riêng cho `/api/v1/ops/summary`; vẫn bắt buộc khi đã đăng nhập. |

Xem [Deployment](DEPLOYMENT.md) để tạo và xoay vòng secret. Không đưa admin token vào frontend bundle.

## Prompt và built-in declarations

Markdown trong `app/prompts/` và YAML `app/tools/definitions/tools.yaml` là tài nguyên đi kèm mã nguồn, không cấu hình bằng đường dẫn từ người dùng hay biến môi trường. Chỉnh file cần restart/redeploy để làm mới cache. Startup fail nếu thiếu prompt hoặc YAML/Pydantic contract lệch nhau; không cần secret mới. Giữ nguyên whitespace của frozen offline/SFT prompt. Xem [hướng dẫn bảo trì](PROMPTS_AND_TOOLS.md).
