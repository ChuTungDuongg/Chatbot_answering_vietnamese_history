# Chạy và triển khai Modal

[`modal_app.py`](../modal_app.py) đọc `.env` tại root trước khi tạo cấu hình; biến môi trường terminal có ưu tiên cao hơn. Chạy lệnh từ root repository với Modal CLI trong Python environment đã cài dependencies.

## Tài nguyên và cấu hình

| Thuộc tính | Mặc định entrypoint |
| --- | --- |
| App / functions | `vn-history-rag-api` / GPU `fastapi_app` và CPU `frontend_app` |
| Engine / mode | Transformers, `APP_MODE=full`, Hybrid và Central đều bật |
| GPU / CPU / RAM | GPU qua `MODAL_GPU_CLASS`; ví dụ dưới dùng `A100-40GB` / 4 CPU / 32 GiB |
| Container / HTTP inputs | Tối thiểu 0, tối đa 1 container; `MODAL_MAX_INPUTS=1` |
| Timeouts | Invocation 600 s, startup 900 s |
| Scale down | 120 s; override `CENTRAL_SCALEDOWN_WINDOW_SECONDS` |
| Artifact Volume | `vn-history-artifacts` → `/artifacts` |
| Model cache Volume | `vn-history-hf-cache` → `/hf-cache` |
| Conversation Volume | `vn-history-chat-data` → `/data`, SQLite `/data/chat.sqlite3` |
| Corpus | `/artifacts/corpus_v1/chunks.jsonl` |
| Retrieval / runtime metadata | `/artifacts/corpus_v1/retrieval`, `/artifacts/corpus_v1/runtime` |
| SFT adapter | `/artifacts/models/qwen3_4b_sft_v1/run_best_b4_ga4_e2/adapter` |

Artifact/cache Volumes phải có sẵn; conversation Volume có thể được tạo khi chưa có. Base weights thiếu có thể được tải về cache. Các đường dẫn, `DEVICE=cuda`, `DTYPE=bfloat16`, greedy decoding, Central budgets và một số built-in flags được entrypoint đặt trực tiếp; không phải mọi biến trong `.env` đều được chuyển vào container.

Đặt `MODAL_GPU_CLASS=A100-40GB` trong `.env` cho cấu hình chạy đồng thời hai model ở hướng dẫn này; fallback trong bản main là `L4`, phù hợp các benchmark một model hơn là cấu hình combined trong ví dụ này. Benchmark helper mặc định L4 và chỉ chạy một model.

CPU frontend dùng 0,5 CPU, 512 MiB, timeout 600 s, 0..2 containers và tối đa 30 inputs/container. [`Dockerfile.frontend`](../Dockerfile.frontend) build React bằng Node 24 rồi phục vụ static assets/proxy bằng Python, không chứa corpus hoặc model.

Các override được hỗ trợ gồm `MODAL_APP_NAME`, `MODAL_GPU_CLASS`, `MODAL_MAX_INPUTS`, `INFERENCE_BACKEND`, `MODAL_RUNTIME_IMAGE`, `MODEL_VARIANT`, `HYBRID_MAX_NEW_TOKENS`, loading strategy, model revisions, dense backend/available backends, MCP, `MODAL_GPU_API_URL` và `MODAL_OBSERVABILITY_SECRET_NAME`. Xem entrypoint trước khi thêm override khác. `MODAL_SERVER_URL` là cấu hình Modal client tùy chọn, không phải URL API chatbot.

## Chuẩn bị một lần

```powershell
if (!(Test-Path .env)) { Copy-Item .env.example .env }
modal token new
modal volume list --json
modal secret list --json
```

Chỉ chạy `modal token new` khi chưa đăng nhập. Khôi phục/upload artifacts theo [Artifacts](ARTIFACTS.md) nếu Volume chưa có dữ liệu; không cần upload lại mỗi lần serve.

FAISS-only không cần Qdrant Secret. Muốn bật cả hai dense backend, đặt `.env`:

```dotenv
MODAL_GPU_CLASS=A100-40GB
RETRIEVAL_DENSE_BACKEND=faiss
RETRIEVAL_AVAILABLE_BACKENDS=faiss,qdrant
MODAL_QDRANT_SECRET_NAME=vn-history-qdrant
```

Secret phải tồn tại và chứa `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`. Tạo/cập nhật Secret qua Modal với credentials riêng. Entrypoint tham chiếu Secret ngay khi chọn Qdrant trong available backends; thiếu Secret/required keys có thể chặn deploy trước khi runtime kiểm tra optional lane. Collection và manifest phải khớp V1.

Đổi Hybrid vanilla/SFT bằng `MODEL_VARIANT=vanilla` hoặc `sft` rồi restart/redeploy. Entrypoint tự đặt adapter path trên Volume. Central vẫn vanilla. `HYBRID_MAX_NEW_TOKENS` được chuyển nếu override, còn mặc định lấy từ Settings. MCP mặc định tắt; xem [Tools/MCP](TOOLS_AND_MCP.md) cho config và `MODAL_MCP_SECRET_NAME`.

## Development và deployment

```powershell
modal serve modal_app.py
```

Dùng URL CLI trả về; `Ctrl+C` kết thúc phiên development. Đặt URL đó vào `VITE_API_BASE_URL` của `frontend/.env` và restart Vite.

Để triển khai endpoint lâu dài:

```powershell
modal deploy modal_app.py
modal app logs vn-history-rag-api --since 2h
```

Lần deploy đầu in hai endpoint. Đặt URL **GPU `fastapi_app`** vào root `.env` hoặc shell, rồi deploy lại để cấu hình gateway:

```powershell
$env:MODAL_GPU_API_URL = 'https://YOUR-GPU-FUNCTION.modal.run'
modal deploy modal_app.py
```

Mở URL **CPU `frontend_app`** để dùng UI trên cloud. `VITE_API_BASE_URL` rỗng trong image dùng same-origin `/api/v1`; không cần frontend process local. Đổi GPU URL cần redeploy frontend function. Gateway chuyển uploads, owner/auth headers, lỗi HTTP và SSE; timeout upstream trả 504, lỗi kết nối trả 502, chưa cấu hình URL trả 503.

`/health` trên CPU kiểm tra gateway và `backend_configured`; nó không phải health của GPU. `/ready` trên CPU proxy tới GPU. Kiểm tra JSON readiness và một request thực tế; Central lazy có thể thêm thời gian tải model vào câu hỏi đầu.

Để chạy image CPU riêng:

```powershell
docker build -f Dockerfile.frontend -t vn-history-ui .
docker run --rm -p 8080:8000 -e MODAL_GPU_API_URL='https://YOUR-GPU-FUNCTION.modal.run' vn-history-ui
```

Serving summary tùy chọn cần Secret chứa `OBSERVABILITY_ADMIN_TOKEN`; đặt `MODAL_OBSERVABILITY_SECRET_NAME` rồi redeploy. Lấy summary bằng Bearer token từ môi trường admin, không đưa vào frontend bundle. Xem [Telemetry](RUNTIME_TELEMETRY.md).

## Smoke test và kiểm tra Volume

```powershell
New-Item -ItemType Directory -Force reports/restore | Out-Null
modal volume ls vn-history-artifacts /corpus_v1/runtime --json
modal run --detach --write-result reports/restore/runtime_smoke.json modal_app.py::runtime_smoke --generate --central
modal run --write-result reports/restore/qdrant_smoke.json scripts/modal_qdrant_smoke.py::qdrant_smoke
```

GPU smoke có thể khởi tạo tài nguyên tính phí. `runtime_smoke` dùng database tạm, đọc V1 và kiểm tra retrieval/generation; Qdrant smoke kiểm tra count/schema và mẫu payload/vector. Đây là kiểm tra vận hành, không phải benchmark chất lượng hay xác minh toàn bộ remote vectors.

Với endpoint đã chạy:

```powershell
$base = 'https://YOUR-ENDPOINT.modal.run'
Invoke-RestMethod "$base/ready"
$headers = @{'X-Client-ID'='terminal-smoke'}
$c = Invoke-RestMethod -Method Post -Uri "$base/api/v1/conversations" -Headers $headers -ContentType 'application/json' -Body '{}'
$body = @{conversation_id=$c.id; question='Chiến thắng Bạch Đằng năm 938 có ý nghĩa gì?'; mode='hybrid'; retrieval_backend='faiss'; final_k=3} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$base/api/v1/chat" -Headers $headers -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

Timeout client cần đủ cho cold startup và request. Đo latency dùng [HTTP/SSE benchmark](BENCHMARKS.md), không lấy thời gian tải model hoặc CLI smoke làm answer TTFT.

vLLM cần image riêng và triển khai một chat mode; helper benchmark mặc định L4, tách biệt cấu hình combined đã chọn A100 ở đây. Xem [vLLM](VLLM_LATENCY_EXPERIMENTS.md).
