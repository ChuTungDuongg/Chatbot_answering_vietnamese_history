# Dynamic FAISS / Qdrant retrieval và pipeline progress

## Kết quả và phạm vi

App giữ đúng hai chat modes `hybrid` / `central`. Composer thêm lựa chọn dense backend độc lập
FAISS / Qdrant, áp dụng theo request, không restart backend và không thay model.
Live API/frontend không có response detail modes. Hybrid/Central tiếp tục đưa chunk evidence
và lịch sử vào một grounded `SYSTEM_PROMPT`; citation `[chunk_id]` giữ nguyên.

Đã inspect repository và fetch `origin/main` trước khi sửa. Điểm xuất phát: branch `main`,
HEAD và remote main cùng SHA `b3fcd5b4b950e289822092cfd448f3e03558d20c`.
Máy kiểm tra: Windows, Python 3.11 trong `.conda`, Modal CLI 1.5.3; integration trên Modal A100.
Delivery theo yêu cầu cuối của user: commit + normal push lên `origin/main` sau final checks.
Không deploy production. MCP extension được mô tả trong [MCP_INTEGRATION_REPORT.md](MCP_INTEGRATION_REPORT.md).

**DEFAULT MODAL CORPUS: V1** — `/artifacts/corpus_v1/chunks.jsonl`, Volume `vn-history-artifacts`.
Corpus local: `artifacts/corpus_v1/chunks.jsonl`, 624.288 chunks. Qdrant: `vn_history_v1_e5`,
named vector `dense_e5`, dimension 768, Cosine, full precision, embedding `intfloat/multilingual-e5-base`.

## Kiến trúc trước / sau

Trước: `RAGService` chỉ load dense lane do `RETRIEVAL_DENSE_BACKEND` chọn khi startup.
Sau: `dense_retrievers` giữ các lane đã validate; `get_dense_retriever(name)` trả instance được reuse.
Corpus, BM25S, E5 và cross-encoder vẫn là một bộ tài nguyên chung. FAISS đọc index một lần,
Qdrant tạo client một lần với timeout mặc định 30 giây và đóng khi shutdown.
Anchor embeddings có khóa khởi tạo một lần để tránh duplicate initialization khi requests đồng thời.

`RETRIEVAL_DENSE_BACKEND` vẫn chọn default (mặc định FAISS). Không khai báo
`RETRIEVAL_AVAILABLE_BACKENDS` thì chỉ load default. `faiss,qdrant` opt-in cả hai lane.
Default không load được: startup fail. Optional Qdrant thiếu credentials/manifest hoặc validate thất bại:
FAISS vẫn hoạt động và Qdrant không được advertise. Dynamic hai lane yêu cầu layout V1 đã validate.

Validation tái sử dụng corpus count/SHA-256, ordered chunk-ID SHA-256, manifest fingerprints,
embedding model/dimension/resolved revision, FAISS count/dimension, Qdrant collection count/status,
named vector Cosine/full precision và HNSW config. Mỗi Qdrant hit còn đối chiếu point ID ↔ chunk ID.
Không rebuild hoặc ingest index. Validation remote dùng manifest + collection config/count + hit identity,
không download lại toàn bộ vectors để tính checksum remote.

Request giữ backend trong biến local rồi truyền qua Hybrid runtime → `retrieve(dense_backend=...)`.
Central giữ lựa chọn trong `ToolExecutionContext`; `SearchHistoryTool.run_with_context()` truyền nó
vào cùng retriever, kể cả fallback `search_history`. Các tool Wikipedia/web/documents giữ behavior cũ.
Không đổi pointer dense toàn cục, env hoặc service theo request. Các test chạy hai lane đồng thời,
gồm Central/Qdrant và Hybrid/FAISS, xác nhận không có backend bleed và tài nguyên không bị thay thế.

Query analysis/expansion, dense fetch K, BM25, RRF weights, comparison balancing, metadata bonuses,
reranker và context selection giữ nguyên. Qdrant HNSW và FAISS exact search có thể trả dense neighbors
khác nhau; task này không thay search policy để ép hai kết quả giống nhau.

## API / capabilities / lỗi

`ChatRequest` và `RetrieveRequest` thêm `retrieval_backend: Literal["faiss", "qdrant"] | None`.
Omitted/null dùng default server; giá trị khác trả 422. Áp dụng `/api/v1/chat`, `/chat/stream`, `/retrieve`.
`ChatResponse`, `RetrieveResponse`, SSE `done`/error và debug trace ghi backend đã chọn/thực thi.
`RetrieveResponse` thêm `timings_ms`.

`GET /ready` giữ các field cũ và bổ sung:

```json
{
  "default_dense_backend": "faiss",
  "qdrant_loaded": true,
  "dense_backends": {"faiss": {"available": true}, "qdrant": {"available": true}},
  "retrieval": {"default_backend": "faiss", "available_backends": ["faiss", "qdrant"]}
}
```

Frontend dùng capability làm source of truth, lưu preference tại `vn-history-retrieval-backend-v1`.
Preference Qdrant cũ được normalize về default nếu lane không available. Khi chưa xác nhận capability,
chưa cho gửi chat. Runtime không silently fallback: unavailable trả 503 trước khi ghi conversation;
network failure giữa stream phát `error` và `done.status=error`, giữ conversation cũ và ngừng generation.
User nhận thông báo Qdrant an toàn, không URL/key/header. Logs lỗi của phần mới chỉ ghi error type.
Nếu domain gate bỏ retrieval, progress không tuyên bố đã chạy dense search.

## SSE và UI progress

Status đầu tiên được yield trước history/retrieval. Callback từ worker sử dụng
`loop.call_soon_threadsafe` → `asyncio.Queue`; coroutine đợi queue hoặc `prepare()` hoàn tất bằng
`asyncio.wait(FIRST_COMPLETED)`. Không polling retrieval, busy-loop hay timer giả progress.
Answer delta đi qua ngay khi model phát, không đợi debug trace.

```json
{
  "stage": "dense_search",
  "state": "completed",
  "message": "Truy vấn Qdrant",
  "retrieval_backend": "qdrant",
  "request_id": "request-uuid",
  "mode": "hybrid",
  "latency_ms": 18.4
}
```

States: `started`, `completed`, `failed`. Stable IDs: `request_preparation`, `query_analysis`,
`embedding`, `dense_search`, `bm25_search`, `fusion`, `rerank`, `context_selection`,
`attachment_search`, `prompt_preparation`, `tool_selection`, `tool:<registered-name>`, `generation`.
Chỉ emit stage đã chạy; query variants có thể lặp embedding/dense/BM25 events trong cùng message.
Central hiển thị chi tiết lần history search đầu, các lần sau giữ tool events và cộng đầy đủ metrics;
việc coalesce này hạn chế số event khi nhiều action rounds. Planning/tool names phản ánh thực thi thật.

`RetrievalProgress` là component riêng, compact, `role=status`, spinner/check/error, dark/light/mobile.
Có answer text thì collapse thành summary; done/abort/error dừng spinner. Pipeline nằm trên từng
assistant message và được giữ qua sync bằng server message ID. Request/scope protections cũ giữ
nguyên để SSE cũ không cập nhật conversation mới. Attachments, drag/drop/paste, citations, stop/send
và hai chat modes đều có test regression.

## Timing / latency sanity

Done metrics giữ `retrieval_ms`, model/final TTFT, e2e, tokens/s, ITL; thêm
`query_analysis_ms`, `embedding_ms`, `dense_search_ms`, `bm25_search_ms`, `fusion_ms`,
`rerank_ms`, `context_selection_ms`. Debug: `retrieval.backend`, `retrieval.timings_ms`;
DeveloperTrace hiện Dense backend và breakdown.
Stage timings đo active work; embedding/dense/BM25 cộng các query variants, Central cộng các history
searches. Query analysis bao gồm domain classifier/anchor embedding như pipeline cũ. `retrieval_ms`
còn bao gồm orchestration và attachments ở Hybrid; Central giữ semantics thời gian history tool cũ.
Không cộng stage timings vào retrieval_ms lần thứ hai; không xem tổng stage là e2e.

Modal A100, V1 đầy đủ, cùng câu hỏi: “Ngô Quyền giành chiến thắng trên sông Bạch Đằng vào năm nào?”.
Mỗi lane warm-up một lần; 3 mẫu Hybrid/lane, đổi thứ tự lane giữa rounds, conversation mới mỗi lần.
Vòng paired đầu chạy trước điều chỉnh event theo từng query variant. Đơn vị ms, p95 nội suy từ mẫu nhỏ:

| Hybrid metric | FAISS p50 / p95 | Qdrant p50 / p95 |
| --- | ---: | ---: |
| Dense | 181,61 / 181,83 | 1.044,07 / 1.646,86 |
| Retrieval | 553,31 / 555,81 | 1.417,08 / 2.051,67 |
| Final answer TTFT | 697,93 / 702,85 | 1.561,90 / 2.198,19 |
| E2E | 2.175,65 / 2.180,37 | 3.051,68 / 3.710,97 |

Smoke cuối trên source hiện tại có 1 mẫu Hybrid/lane sau warm-up; p50=p95:

| Final smoke metric | FAISS | Qdrant |
| --- | ---: | ---: |
| Dense | 152,36 | 991,96 |
| Retrieval | 565,57 | 1.393,68 |
| Final answer TTFT | 768,80 | 1.592,64 |
| E2E | 3.892,92 | 4.666,37 |

Hybrid final smoke phát 24 status events/request, Central 28; first status server 0,33–0,56 ms.
Đây là sanity check trên môi trường này, không kết luận tốc độ chung hay sửa benchmark frozen.
Central warm-up một round/lane và 1 mẫu/lane cũng PASS; p50=p95 với n=1 nên chỉ có ý nghĩa smoke.
Số liệu dùng server timestamps; HTTP TestClient integration không đo browser network TTFT.
Raw samples/corpus identity lưu ở `reports/dynamic_retrieval/2026-10-03_paired_modal.json` và
`2026-10-03_final_modal.json` (gitignored); final smoke ghi model ID, resolved revision,
generation/retrieval settings. Hai app smoke đã hoàn tất và dừng, không để GPU chạy nền.

## Chạy local với cả hai lane

PowerShell trong repo; credentials nhập vào process, không ghi key thật vào command history:

```powershell
Set-Location (git rev-parse --show-toplevel)
$env:PYTHONUTF8 = '1'
$env:RETRIEVAL_DENSE_BACKEND = 'faiss'
$env:RETRIEVAL_AVAILABLE_BACKENDS = 'faiss,qdrant'
$env:ARTIFACT_ROOT = (Join-Path (Get-Location) 'artifacts/corpus_v1')
$env:QDRANT_COLLECTION = 'vn_history_v1_e5'
$env:QDRANT_TIMEOUT_SECONDS = '30'
$env:QDRANT_URL = Read-Host 'QDRANT_URL của cluster V1'
$taskQdrantKey = Read-Host 'QDRANT_API_KEY' -AsSecureString
$env:QDRANT_API_KEY = [Net.NetworkCredential]::new('', $taskQdrantKey).Password

# API retrieval local, không load LLM:
$env:APP_MODE = 'retrieval-only'
$env:DEVICE = 'cpu'
.\.conda\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Máy hiện tại có khoảng 16 GiB RAM: chưa kiểm chứng full Qwen4B+8B local và không phù hợp chạy cả
hai LLM float32 cùng corpus ở RAM này. Để chạy **chat local** trên máy CUDA đủ VRAM/RAM, dùng cùng
env phía trên, thay `APP_MODE=full`, `DEVICE=cuda`, `DTYPE=bfloat16` trước lệnh uvicorn.
Đây là cấu hình startup một lần; đổi FAISS/Qdrant trong UI không restart. Không đổi model IDs,
quantization hoặc generation settings để né giới hạn máy.

Terminal thứ hai cho frontend:

```powershell
Set-Location (git rev-parse --show-toplevel)
$env:VITE_API_BASE_URL = 'http://127.0.0.1:8000'
npm --prefix frontend run dev
```

Local `retrieval-only` dùng `/retrieve`, chưa có chat generation. Để dùng App chat trên máy này,
chạy frontend local và API Modal theo block sau.

## Modal với hai lane / terminal

Secret hiện có `vn-history-qdrant` cung cấp `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`.
Mount `/artifacts` và corpus V1 giữ nguyên, cache `/hf-cache`; không cần upload/re-index lại.

```powershell
Set-Location (git rev-parse --show-toplevel)
$repo = (Get-Location).Path
$env:Path = "$repo\.conda;$repo\.conda\Scripts;$repo\.conda\Library\bin;$env:Path"
$env:PYTHONUTF8 = '1'
$env:MODAL_SERVER_URL = 'https://api.modal2.com'
$env:RETRIEVAL_DENSE_BACKEND = 'faiss'
$env:RETRIEVAL_AVAILABLE_BACKENDS = 'faiss,qdrant'
$env:MODAL_QDRANT_SECRET_NAME = 'vn-history-qdrant'
modal --version
modal secret list --json
modal serve modal_app.py
```

Frontend terminal: đặt `VITE_API_BASE_URL` bằng URL development CLI in ra rồi `npm --prefix frontend run dev`.
Ctrl+C dừng development. Chỉ khi muốn deploy bản mới: `modal deploy modal_app.py`;
logs: `modal app logs vn-history-rag-api --tail 100`. Chưa chạy deploy trong task này.

Read-only integration/paired benchmark (có dùng A100):

```powershell
New-Item -ItemType Directory -Force reports/dynamic_retrieval | Out-Null
modal run --detach --write-result reports/dynamic_retrieval/new_modal_smoke.json scripts/modal_dynamic_retrieval_smoke.py::dynamic_smoke --repeats 3

# HTTP benchmark lên API local/full hoặc API Modal đang serve; output phải là file mới:
python scripts/benchmark_dynamic_retrieval.py --base-url http://127.0.0.1:8000 --repeats 3 --warmup 1 --mode hybrid --output reports/dynamic_retrieval/new_hybrid.json
```

Script yêu cầu `/ready` advertise cả hai lane, tạo/xóa conversation riêng của smoke,
không đụng conversation người dùng hoặc build artifact. Nếu thiếu credentials, optional integration
không chạy; toàn bộ unit tests dùng mocks/tiny fixtures và không cần key thật.

## Ví dụ request

Tạo conversation bằng `POST /api/v1/conversations`, giữ cùng `X-Client-ID` cho các calls.
Thay UUID ví dụ bằng ID trả về; body cho `/api/v1/chat` hoặc `/api/v1/chat/stream`:

```json
{"conversation_id":"00000000-0000-0000-0000-000000000001","question":"Bạch Đằng năm 938 có ý nghĩa gì?","mode":"hybrid","retrieval_backend":"faiss","final_k":3}
```

```json
{"conversation_id":"00000000-0000-0000-0000-000000000001","question":"Bạch Đằng năm 938 có ý nghĩa gì?","mode":"hybrid","retrieval_backend":"qdrant","final_k":3}
```

```json
{"conversation_id":"00000000-0000-0000-0000-000000000001","question":"Bạch Đằng năm 938 có ý nghĩa gì?","mode":"central","retrieval_backend":"qdrant","final_k":3}
```

Retrieval riêng: `{"question":"Bạch Đằng năm 938?","retrieval_backend":"qdrant","final_k":3}`.

## Validation / bảo toàn research

| Check | Kết quả |
| --- | --- |
| `$env:PYTHONCASEOK='1'; .\.conda\python.exe -m pytest -q` | **289 passed**, bao gồm MCP extension |
| `npm --prefix frontend test -- --run` | **66 Node + 74 Vitest = 140 passed** |
| `npm --prefix frontend run test:ui` | **45 passed**, desktop/tablet/mobile, 390px mobile |
| `npm --prefix frontend run build` | PASS, production bundle |
| `npm --prefix frontend run lint` | PASS |
| Real Modal V1 Hybrid/Central × FAISS/Qdrant | PASS; resources reused |
| Protected-file audit | **192 files unchanged**; SHA-256 small files, size/mtime large files |
| `git diff --check` | PASS |

Windows `PYTHONCASEOK=1` phục vụ import lowercase `training` với historical thư mục `Training`;
không rename hoặc sửa historical code.
Browser tests bắt được race focus dropdown mới; đã sửa bằng layout effect và kiểm chứng trên ba viewport.
Không còn blocker từ lần test đầu.

`rg` sau thay đổi: không có live frontend/API/runtime response mode; các references hợp lệ còn
trong `app/rag/prompting.py`, `response_modes.py`, `evaluation/six_way.py`, `silver_v1.py`,
`silver_citation_v2.py`, `Training/train_qwen3.py` và các test compatibility. Các file này/datasets
không bị sửa. Search dùng boundary để phân biệt `response_mode` với FastAPI `response_model`.
Không mutate corpus/index/model/PEFT/ZIP, không regenerate nghiên cứu hoặc đổi citation/model IDs.

## Files thay đổi

Danh sách chính xác được ghi ở phần cuối report. Không xóa file trong task này.

## Giới hạn còn lại

- Startup/cold model loading vẫn có chi phí một lần; Central lazy load Qwen8B lần đầu như trước.
- Availability phản ánh validation tại startup; outage sau đó báo lỗi request. Không tự reconnect/reload
  resources hoặc chuyển backend. Muốn enable lane chưa load cần cấu hình lại startup; lane đã load
  thì UI switching hoàn toàn theo request.
- Số mẫu latency nhỏ, không đại diện benchmark toàn suite. Dense timing cộng query variants,
  không phải latency một network query riêng lẻ. P95 Central n=1 không có ý nghĩa thống kê.
- Source được chuẩn bị commit/push theo yêu cầu cuối của user. Production deploy là bước riêng;
  dùng `modal serve` để review trước khi tự deploy.

## Danh sách working-tree files

Danh sách delivery gồm dynamic retrieval + MCP extension:

| Status | File |
| --- | --- |
| Modified | `.dockerignore` |
| Modified | `.env.example` |
| Modified | `.gitignore` |
| Modified | `MODAL_QUICKSTART.md` |
| Modified | `README.md` |
| Modified | `app/api/routes.py` |
| Modified | `app/central/runtime.py` |
| Modified | `app/config.py` |
| Modified | `app/main.py` |
| Modified | `app/rag/dense_backend.py` |
| Modified | `app/rag/hybrid_runtime.py` |
| Modified | `app/rag/retrieval.py` |
| Modified | `app/rag/retriever.py` |
| Modified | `app/schemas.py` |
| Modified | `app/services/metadata.py` |
| Modified | `app/services/rag_service.py` |
| Modified | `app/tools/local_search.py` |
| Modified | `app/tools/registry.py` |
| Modified | `frontend/e2e/brandMark.spec.js` |
| Modified | `frontend/e2e/chatLayout.spec.js` |
| Modified | `frontend/e2e/clipboard.spec.js` |
| Modified | `frontend/e2e/conversation.spec.js` |
| Modified | `frontend/src/App.jsx` |
| Modified | `frontend/src/components/ChatInput.jsx` |
| Modified | `frontend/src/components/ChatMessage.jsx` |
| Modified | `frontend/src/components/DeveloperTrace.jsx` |
| Modified | `frontend/src/components/RetrievedChunks.jsx` |
| Modified | `frontend/src/hooks/useChatStream.js` |
| Modified | `frontend/src/services/api.js` |
| Modified | `frontend/src/services/progressLabels.js` |
| Modified | `frontend/src/state/chatSessionReducer.js` |
| Modified | `frontend/src/styles/composer.css` |
| Modified | `frontend/src/styles/conversation.css` |
| Modified | `frontend/tests/ui/App.test.jsx` |
| Modified | `frontend/tests/ui/api.test.jsx` |
| Modified | `frontend/tests/ui/useChatStream.test.jsx` |
| Modified | `modal_app.py` |
| Modified | `requirements.txt` |
| Modified | `tests/test_baseline_runtime.py` |
| Added | `DYNAMIC_RETRIEVAL_REPORT.md` |
| Added | `MCP_INTEGRATION_REPORT.md` |
| Added | `app/mcp/__init__.py` |
| Added | `app/mcp/adapters.py` |
| Added | `app/mcp/client.py` |
| Added | `app/mcp/manager.py` |
| Added | `app/mcp/normalization.py` |
| Added | `app/mcp/schemas.py` |
| Added | `app/rag/backends.py` |
| Added | `app/rag/progress.py` |
| Added | `app/tools/policy.py` |
| Added | `config/mcp_servers.example.json` |
| Added | `frontend/src/components/RetrievalBackendSelector.jsx` |
| Added | `frontend/src/components/RetrievalProgress.jsx` |
| Added | `frontend/src/components/ToolSteering.jsx` |
| Added | `frontend/src/config/retrievalBackends.js` |
| Added | `frontend/src/hooks/useRetrievalBackend.js` |
| Added | `frontend/src/hooks/useToolSteering.js` |
| Added | `frontend/tests/ui/retrievalBackend.test.jsx` |
| Added | `frontend/tests/ui/toolSteering.test.jsx` |
| Added | `scripts/benchmark_dynamic_retrieval.py` |
| Added | `scripts/mcp_demo_server.py` |
| Added | `scripts/modal_dynamic_retrieval_smoke.py` |
| Added | `scripts/modal_mcp_smoke.py` |
| Added | `tests/test_dynamic_retrieval.py` |
| Added | `tests/test_mcp_integration.py` |

**Deleted:** không có. Secrets/data/model/ZIP không được stage. Delivery theo yêu cầu cuối: commit + normal push `origin/main`; production không deploy.
