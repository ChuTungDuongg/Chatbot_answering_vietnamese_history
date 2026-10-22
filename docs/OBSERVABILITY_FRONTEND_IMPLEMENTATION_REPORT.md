# Báo cáo hoàn thiện observability, câu trả lời live và frontend Modal

Ngày kiểm tra: 23/10/2026, múi giờ Asia/Saigon. Repository: ChuTungDuongg/Chatbot_answering_vietnamese_history. PR: #3, nhánh feat/observability-natural-answers-modal-ui-20261023. Base main: 2edd78f0f82db0d3fb75d0fec347e504d9584e4a. Bản PR được audit ban đầu: f62a856dcff4ace8226d89335ecd1be46ab42b3e.

## 1. Phạm vi và tiêu chí kết thúc

Hoàn thiện ba phần đã tồn tại trong PR; không viết lại retrieval, planner hay frontend. Giữ nguyên model IDs, adapters, cấu hình decoding, token budgets, các lane FAISS/Qdrant, schema SSE và prompt offline đã đóng băng.

Checkout main có thay đổi chưa commit từ trước. Toàn bộ sửa chữa nằm ở worktree riêng gắn với chính nhánh PR; không đưa các thay đổi chưa commit trên main vào PR.

Theo yêu cầu Phase 6 bổ sung: chỉ ready và Squash and Merge sau khi toàn bộ kiểm tra cần thiết, GitHub CI và deployment validation đạt. Có blocker hoặc kiểm tra còn pending thì giữ PR chưa merge.

## 2. Kiến trúc thực tế

~~~mermaid
flowchart LR
    B[Browser] --> C[Modal frontend_app: CPU]
    C --> S[Vite dist: static files]
    C -->|REST / multipart / raw SSE| G[Modal fastapi_app: GPU]
    G --> R[Retrieval FAISS / Qdrant / BM25 / reranker]
    G --> H[Hybrid Qwen3 4B]
    G --> A[Central Qwen3 8B]
    G --> M[ServingMetrics + numeric JSON logs]
    G --> V[Existing artifacts / hf-cache / chat-data volumes]
~~~

frontend_app không mount model/corpus volumes, không nhận secrets của GPU và không import app.main. GET /, file assets và /health được xử lý trên CPU. /api/v1/* và /ready mới được chuyển tới GPU. Lưu ý frontend React hiện có bootstrap API để đọc hội thoại và capabilities: mở UI trong trình duyệt có thể kéo theo các API này và khởi động GPU. Việc tải riêng static bytes không khởi động GPU.

Giữ nguyên các volume vn-history-artifacts, vn-history-hf-cache, vn-history-chat-data và cách cấp secret Qdrant. Cấu hình triển khai dùng lại .env hiện có: A100-40GB, vanilla, bật Hybrid/Central, FAISS mặc định và cả FAISS/Qdrant khả dụng; không đổi default model configuration trong source.

## 3. Audit đủ 18 file ban đầu

| File trong PR ban đầu | Kết luận và xử lý |
|---|---|
| .env.example | URL GPU và tên secret chỉ là cấu hình; không chứa token thật. |
| .github/workflows/ci.yml | Giữ Python compile/tests và frontend tests/lint/build; bổ sung image build/check và CPU smoke. |
| Dockerfile.frontend | Multi-stage Node build → Python CPU; ép API origin rỗng và debug false khi build production. |
| README.md | Giữ hướng dẫn hai container; liên kết tài liệu vận hành. |
| app/api/routes.py | Sửa thời gian cancellation và quyền sở hữu generator tại tầng ASGI; giữ tên, thứ tự và payload các event. |
| app/main.py | Bearer compare theo bytes; no-store; đọc cache stats mà không load model; thêm counters ASGI. |
| app/observability.py | Sửa cắt mất request khi vượt 500; numeric allowlist, fixed labels, phân tách outcome, throughput và memory scope. |
| app/rag/prompting.py | Chỉ sửa live; nhấn mạnh không đổi quân số thành thương vong; giữ bytes offline. |
| docs/SERVING_OBSERVABILITY_MODAL_FRONTEND.md | Cập nhật metric semantics và loại lệnh CLI chưa có trong SDK hiện tại. |
| frontend/.env.example | Rỗng nghĩa là same origin trong cloud; local có thể chọn origin rõ ràng. |
| frontend/README.md | Giữ hướng dẫn phát triển local và production gateway. |
| frontend/src/services/api.js | Giữ same origin; phát hiện EOF thiếu done; luôn đóng reader khi lỗi/abort/callback lỗi. |
| frontend_server/__init__.py | Package CPU riêng, không kéo dependency inference. |
| frontend_server/server.py | Sửa cancellation trước header, cleanup, upload encoding, repeated query và HTTP headers/status. |
| modal_app.py | CPU function độc lập; optional secret kiểm tra required key; giữ resource/model/volume cấu hình GPU. |
| tests/test_baseline_runtime.py | Các assertion live dùng LIVE_SYSTEM_PROMPT, contract và retrieval assertions còn nguyên. |
| tests/test_frontend_gateway.py | Giữ test ban đầu; thêm suite HTTP thật và chạy lại trong image Linux. |
| tests/test_serving_observability.py | Giữ kiểm tra percentile arithmetic; đồng bộ kiểu response an toàn. |

Các file bổ sung tập trung vào regression và vận hành: cache_metrics.py, các test ASGI/cache/Waterloo, smoke script, prompt hashes, .gitattributes và báo cáo này. Không thay đổi nội dung benchmark/evaluation/training/config model. .gitattributes giữ JSONL latency_v1 ở LF để hash không đổi khi checkout Windows; blob dữ liệu và manifest vẫn khớp main.

Baseline ban đầu: 364 passed, 7 failed. Sáu failure là assertion live còn dùng prompt cũ, ở test_central_planning, test_dynamic_retrieval và test_mcp_integration; một failure là test_latency_experiments đọc SILVER split generated/ignored không có trong checkout. Đã thay kiểm tra đó bằng xác minh frozen asset + manifest đã commit và test sampler với dataset tạm có thể tái tạo. Không tạo lại workload hoặc đổi manifest.

## 4. Observability: định nghĩa và tính đúng

### Ba lớp metric khác nhau

| Lớp | Nguồn và ý nghĩa | Không nên suy diễn |
|---|---|---|
| Application | Accepted chat executions, outcome success/error/cancelled, TTFT, retrieval/planning/generation/e2e, sample percentiles; HTTP admission/status counters riêng. | Không phải số liệu toàn cluster hoặc thời gian đầy đủ từ browser qua cold start. |
| Modal infrastructure | Dashboard/host metrics: containers, queue/calls, CPU/RAM, cold starts, GPU device telemetry. | Không thay được semantic error trong SSE HTTP 200, hoặc model TTFT. |
| Engine/GPU/cache | vLLM scheduler KV-cache usage và token queries/hits; Torch allocator của process nếu CUDA đã khởi tạo. | HF disk cache không phải prefix cache; allocated/reserved không phải GPU utilization hay tổng VRAM worker. |

### Application metrics

- e2e_ms: timestamp request trace ở backend tới request_finished hiện hữu. Không bao gồm browser/proxy/cold-start bên ngoài application; không đổi baseline timing contract.
- final_answer_ttft_ms: tới answer_delta đầu tiên của final answer, có retrieval/planner trước đó; không lấy status event làm TTFT.
- model_ttft_ms: từ ModelDone của final generation; cancellation thiếu ModelDone giữ null.
- generation_ms, tokens_per_second: thời gian và token count do runtime cung cấp. Không suy token từ ký tự hay số SSE chunk.
- p50/p95 dùng nội suy tuyến tính; tối đa 500 sample cho từng mode/backend/outcome. Success, error và cancelled tách riêng để error ngắn không làm sai latency thành công.
- requests_started, inflight và totals là execution counters. http là nhóm riêng cho chat, chat_stream, retrieve, bao gồm cả 422/503 trước khi generation chạy.
- Request window dùng tối đa 301 bucket giây, đếm 300 bucket gần nhất. Không bị cắt request khi traffic vượt 500; độ phân giải 1 giây được xuất rõ.
- Throughput request/giây khác với per-request token/giây. known_output_tokens_per_second chỉ cộng request có token count đã biết, kèm coverage; không giả định request hủy có 0 token.
- Chỉ nhận số finite không âm, loại bool, NaN, infinity, chuỗi và key tùy ý. Labels chỉ có tập hữu hạn.

HTTP 200 có thể chứa SSE error: HTTP counters vẫn phản ánh status 2xx, còn execution outcome là error. Client rời đi giữa stream được tính cancelled vì chưa gửi terminal body. Success ở execution không bảo đảm browser đã nhận mọi byte.

### Cancellation và error

Đã tái hiện red rồi green cho các lỗi:

1. Cancelled execution không có finished mark.
2. Generator bị bỏ treo sau ASGI send OSError.
3. ASGI 2.3 trả về bình thường sau disconnect làm HTTP counter ghi complete sai.
4. Proxy bị cancel lúc đang chờ upstream headers không đóng client.
5. Frontend coi EOF chưa có done là kết thúc bình thường.

ChatStreamingResponse đóng body iterator ngay trong task đang consume, tránh reset ContextVar ở task khác. Cleanup được shield trong cancel scope. Proxy sở hữu response/client tới hết vòng đời ASGI; không đợi garbage collection. HTTP lỗi trước header thành 502/504 an toàn; status/body của backend được giữ. Nếu upstream lỗi sau khi đã gửi header, kết nối bị ngắt, không thể đổi lại HTTP status; frontend nhận EOF thiếu done và báo lỗi.

### Cache và bộ nhớ

Đã đối chiếu API chính thức của vLLM 0.23.0: AsyncLLM.from_engine_args nhận stat_loggers; logger record nhận SchedulerStats. Adapter chỉ giữ số, không giữ prompt, token IDs hay nội dung response.

- Prefix enabled đọc effective configuration, hoặc cấu hình yêu cầu khi model chưa load.
- queries/hits là token counters từ scheduler. Hit rate = tổng hits / tổng queries từ lần reset cache gần nhất; mẫu số 0 trả null.
- KV usage lấy scheduler fraction [0,1]; nếu có nhiều engine ranks, xuất giá trị lớn nhất. Runtime hiện tại dùng một engine rank.
- Chưa load/không có stats/unsupported engine → null với status rõ ràng.
- KV sample quá 15 giây → stale và utilization null. Counter prefix vẫn là lịch sử có tuổi quan sát, không được gọi là hit rate 5 phút.
- Torch memory chỉ đọc nếu CUDA đã initialized; không khởi tạo CUDA để đọc monitoring. vLLM workers là process riêng nên metric này không đại diện tổng VRAM.
- Process peak RSS dùng resource trên Linux; Windows không có API đó thì status unavailable, không giả 0.

Metric trong RAM là process_local, không durable, reset khi scale to zero/restart. Không cộng/average percentile của các replica để tạo global p95.

## 5. Bảo mật

/api/v1/ops/summary tắt bằng 404 nếu chưa có OBSERVABILITY_ADMIN_TOKEN. Token sai trả 403; constant-time compare theo UTF-8 bytes không lỗi 500 với non-ASCII header. Response thành công có Cache-Control: no-store, proxy cũng giữ no-store cho ops.

Khi được cấp, token chỉ ở secret GPU; không có trong Vite env/bundle hoặc frontend image. Code chỉ log fixed dimensions và numeric measurements; không log owner IDs, câu hỏi, câu trả lời, source text hoặc private cache fields. Test cố ý đưa chuỗi PRIVATE qua metric input xác nhận không xuất ra snapshot/log.

X-Client-ID hiện hữu là định danh anonymous conversation, không phải authentication. Public API vẫn cần authentication/rate limiting trước sử dụng rộng rãi. Không tự đổi cơ chế ownership hoặc SSE error schema trong task này. Không thêm monitoring scrape giữ GPU sống khi idle.

## 6. Câu trả lời tiếng Việt và Waterloo

LIVE_SYSTEM_PROMPT áp dụng cho final answer của Hybrid/Central. SYSTEM_PROMPT và cả ba explicit response modes concise/standard/detailed giữ nguyên bytes với main. Sáu hash regression bao phủ cả RAG và no-RAG prompts, lấy golden từ main trước khi sửa.

Live instruction yêu cầu trả lời trực tiếp, dùng đoạn văn tự nhiên, trích dẫn sau luận điểm; nguồn thiếu hoặc mâu thuẫn thì nêu giới hạn cụ thể. Số quân không tới chiến trường không đồng nghĩa thiệt mạng. Không cài sẵn số liệu Waterloo vào prompt production.

Đối chiếu lịch sử: [National Army Museum](https://www.nam.ac.uk/explore/battle-waterloo) mô tả khoảng 33.000 quân được tách đi truy kích sau Ligny: đây là quân số. Không có cơ sở đổi số đó thành số tử trận tại Wavre. [Fondation Napoléon](https://www.napoleon.org/en/history-of-the-two-empires/articles/bullet-point-35-how-did-napoleon-manage-to-lose-the-battle-of-waterloo/) cho thấy vai trò phối hợp Phổ–Wellington và các yếu tố chỉ huy/thời điểm; không thể coi một kịch bản Grouchy đến là bảo đảm đảo ngược kết quả.

scripts/check_live_answer_regressions.py dùng lại HTTP/SSE client benchmark hiện có, tạo hội thoại smoke độc lập và lưu output synthetic. Heuristic đánh dấu mở đầu máy móc, 33.000 thành thương vong, kết quả phản thực tế chắc chắn, invalid/missing citation. Đây là kiểm tra từ khóa hỗ trợ reviewer, không phải chứng minh entailment; không có flag vẫn cần xem nguồn và đọc câu trả lời. File capture không phải telemetry production.

## 7. Frontend/proxy và container

- Build Vite trong Node 24; runtime Python CPU chỉ chứa dist và frontend_server.
- Build ép VITE_API_BASE_URL rỗng, VITE_SHOW_DEBUG_TRACE=false; không phụ thuộc localhost.
- Local .env, node_modules, dist cũ, model/cache/corpus không vào build context.
- Chuyển raw bytes, không parse/re-encode SSE, không buffer toàn bộ upload.
- Giữ multipart Content-Type/Content-Encoding/Content-Length, request identity, repeated query params.
- Giữ error status, Retry-After, WWW-Authenticate, no-content 204 và headers cần thiết; không follow redirects server-side.
- Redirect tới cùng upstream được đổi thành relative location để browser giữ same origin.
- /index.html luôn no-cache; SPA fallback không nuốt /api/* hoặc /assets/*.
- Timeout connect 30s, write 90s, upstream read gap 600s. Modal request/container timeouts vẫn là giới hạn riêng.
- Đã test first SSE bytes trước khi upstream được phép kết thúc, browser abort và abort trước headers bằng hai HTTP server thật.

## 8. Kết quả kiểm tra thực tế

| Kiểm tra | Kết quả đã xác nhận |
|---|---|
| Python baseline ban đầu | 364 passed, 7 failed, 116.65s |
| Python full trước hai test ASGI cuối và logging | 394 passed, 109.79s |
| Python final | 399 passed, 67.93s; gồm remote import, ASGI và live-quality regressions |
| Frontend node:test | 67 passed |
| Frontend Vitest | 78 passed, 11 files |
| ESLint | Exit 0 |
| Vite production build | Exit 0; 2079 modules, JS khoảng 412 kB / gzip 127 kB |
| Python compileall | Exit 0 cho app, frontend_server, scripts, evaluation, benchmarks |
| docker build --check | Passed, no warnings |
| docker build Dockerfile.frontend | Passed; image sha256:5d6bd638c538f8a2db61e9e657c480f5f3dc9fa68390db1cfff972d8af719a2a |
| Gateway tests trong image Linux production | 8 passed, 1 Starlette TestClient deprecation warning |
| Docker static/health/index-cache/proxy 503 smoke | Passed; image cuối không có Torch hoặc node executable |
| Playwright | 45 passed, 30.6s, desktop/tablet/mobile |
| Review độc lập | Đã sửa hai lỗi ASGI được reviewer tái hiện; reviewer xác nhận không còn important/critical finding |
| GitHub Actions | Chưa có kết quả cho commit sửa cuối; không coi local pass là CI pass |

Node test suite có cảnh báo cổng Vite HMR 24678 dùng đồng thời từ các test SSR hiện hữu; các assertion vẫn pass. Không phải lỗi của production build.

## 9. Modal: trạng thái và lệnh triển khai

Đã xác nhận authenticated profile, các volume và secret vn-history-qdrant. Deployment cũ vn-history-rag-api đang stopped trước lượt này. Không đọc/in token hoặc giá trị secret trong báo cáo. Secret observability riêng chưa được cấp; endpoint giữ disabled cho tới khi cấu hình.

Đã deploy thành công hai function từ worktree, nạp cấu hình .env của checkout chính vào subprocess mà không copy secret vào source/image. Cả frontend multi-stage và backend giữ nguyên requirements đã build trên Modal.

- CPU: https://tungduong156gli--vn-history-rag-api-frontend-app.us-east.modal.run
- GPU: https://tungduong156gli--vn-history-rag-api-fastapi-app.us-east.modal.run
- /health CPU trả 200, backend_configured=true; HTML trả 200.
- /ready qua CPU trả 200, ready=true, corpus_chunks=624288, backend Transformers.
- Upload PDF synthetic trả 201, ready, 1 chunk; GET hội thoại 200; delete attachment và conversation 204.
- Validation lỗi trả 422; ops chưa cấp secret trả 404 như thiết kế.
- Log Modal thực có serving_request numeric cho Central success, không có prompt/answer.

Deployment thật phát hiện CPU import modal_app.py bị lỗi thiếu dotenv; đã sửa chỉ load dotenv trong modal.is_local(), thêm test remote import và redeploy thành công. Docker local không tự tái hiện được lỗi này vì chạy gateway trực tiếp.

Capture Waterloo ban đầu qua Central/FAISS thành công transport: client TTFT 49008 ms, E2E 75173 ms; 339 output tokens. Qua Hybrid/Qdrant: TTFT 6336 ms, E2E 66558 ms. Đây chỉ là hai smoke sample, không phải p50/p95 benchmark, không dùng để kết luận hiệu năng. Central phân biệt 33.000 là quân bị giữ ngoài Waterloo, không gọi là tử trận, nhưng có kết luận thiếu hỗ trợ về phối hợp của liên minh. Hybrid còn suy diễn từ Wavre sang toàn bộ quân Phổ. Heuristic đầu tiên không bắt hết lỗi; human review đã phát hiện, bổ sung counterexample regression và siết prompt về causal relation, đúng chủ thể/thời gian và phạm vi trận/chiến dịch. Đang chạy lại live capture sau chỉnh sửa này; chưa chấp nhận chất lượng chỉ dựa vào flags rỗng.

Quy trình deployment đã hỗ trợ:

~~~powershell
# .env đã có cấu hình runtime phù hợp và Modal CLI đã authenticate
python -m modal deploy modal_app.py

# Copy URL fastapi_app từ kết quả deployment; không dùng localhost
$env:MODAL_GPU_API_URL = "https://YOUR-GPU-FUNCTION.modal.run"
python -m modal deploy modal_app.py

# Optional: secret đã tồn tại và chứa OBSERVABILITY_ADMIN_TOKEN
$env:MODAL_OBSERVABILITY_SECRET_NAME = "vn-history-observability"
python -m modal deploy modal_app.py

# Capture synthetic real-model output qua CPU frontend
python -m scripts.check_live_answer_regressions --base-url "https://YOUR-CPU-FUNCTION.modal.run" --mode central --retrieval-backend faiss --output reports/pr3/waterloo-central.json

# Lệnh logs đã kiểm tra bằng CLI help hiện tại
python -m modal app logs vn-history-rag-api --since 2h
~~~

Modal CLI hiện cài không có subcommand function; vì vậy đã bỏ các lệnh modal function metrics chưa được xác thực khỏi guide. Dùng Modal Dashboard cho infrastructure/GPU metrics.

Static/proxy/upload readiness đã được kiểm tra trên Modal thật; acceptance về factual consistency của câu trả lời đang chờ lần kiểm tra lại. Cache counters vLLM thật trên GPU chưa được đo, vì production đang dùng Transformers. GPU thật cho vLLM optional backend cần deployment Linux CUDA riêng theo requirements-vllm.txt; không đổi production Transformers chỉ để thử cache.

## 10. Giới hạn và cải tiến tiếp theo

1. Đưa histogram/counters sang OpenTelemetry/collector bền vững để có multi-replica metrics và cảnh báo; giữ label allowlist, tránh content logs.
2. Dashboard riêng cho queue/cold-start/infrastructure, engine/cache và application latencies; đo client E2E bằng HTTP/SSE harness hiện có.
3. Khóa dependency CPU gateway và image digest theo bản đã kiểm chứng để tránh drift trong future builds.
4. Thêm authentication/rate limiting cho demo public; quản lý và rotate monitoring secret.
5. Đánh giá câu trả lời thật theo bộ câu hỏi, blind A/B naturalness, factuality, citation entailment và contradiction; không dùng prompt/unit test làm bằng chứng mô hình luôn tuân thủ.
6. Theo dõi retrieval/thread cancellation: Python không thể dừng cưỡng bức một hàm native đã chạy trong to_thread; cancel flag/ngắt generation và không tiếp tục bước mới, còn hàm retrieval native đang chạy có thể phải hoàn tất.
7. Theo dõi error sau khi response headers đã gửi bằng terminated-stream metrics/client EOF; HTTP status không thể đổi ngược.

## 11. Cổng merge Phase 6

Chưa merge. Còn cổng factual consistency live và CI của commit cuối. Chỉ chuyển ready và squash sau khi final Python/frontend/lint/build, CI, review và deployment validation đều đạt. Nếu bất kỳ required check chưa xong hoặc deployment chưa đạt, giữ draft và ghi blocker cụ thể vào đây.
