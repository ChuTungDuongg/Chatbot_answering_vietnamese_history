# MCP integration + request steering

Ngày kiểm tra: 2026-10-03. Workspace `Chatbot_answering_vietnamese_history`, Windows,
Python 3.11.15 trong `.conda`, Modal 1.5.3, Node 24.21.0. Branch `main`,
HEAD/remote main lúc inspect: `b3fcd5b4b950e289822092cfd448f3e03558d20c`.
Delivery theo yêu cầu cuối của user: commit + normal push lên `origin/main` sau checks.
Không deploy production. Commit SHA cuối xem `git log -1` và final response.

**DEFAULT MODAL CORPUS: V1** — `/artifacts/corpus_v1/chunks.jsonl`,
Volume `vn-history-artifacts`. App vẫn chỉ có Hybrid RAG và Central Agent;
response detail modes không được đưa trở lại.

## Architecture và SDK

Trước đây Central có registry built-in, planning bằng Qwen3-8B và final grounded generation.
Sau thay đổi, built-ins và MCP adapters dùng cùng registry. Một immutable `ToolView`
lưu allowlist riêng cho mỗi request, lọc schemas trước khi planner nhận chúng;
`ToolExecutionContext` mang backend, permissions, cancellation và telemetry riêng.
Không mutate registry hoặc dense selector theo request.

```text
Central request → tool policy/view → planner → registry
                                            ├─ search_history → shared HybridRetriever → FAISS/Qdrant + BM25
                                            ├─ built-in documents/Wikipedia/web
                                            └─ MCP adapter → cached MCPManager → SDK client/server
                                               bounded untrusted evidence → final grounded prompt
Hybrid request → shared HybridRetriever → final grounded prompt
```

Dependency pin: **`mcp==2.2.0`**, SDK Python chính thức, và **`jsonschema==4.26.0`**
để validate discovered input schemas/arguments. SDK hỗ trợ stdio và Streamable HTTP,
đàm phán protocol với server; app không tự implement MCP wire protocol.
Config `transport="http"` gọi `streamable_http_client`, không phải REST tool tự chế.
Tham khảo [SDK v2.2.0](https://github.com/modelcontextprotocol/python-sdk/blob/v2.2.0/README.md)
và [release v2.2.0](https://github.com/modelcontextprotocol/python-sdk/releases/tag/v2.2.0).

## Lifecycle, discovery và latency

- MCP chỉ initialize khi `APP_MODE=full`, Central được bật và `MCP_ENABLED=true`.
  Hybrid-only/API-only/retrieval-only không mở MCP connection.
- FastAPI lifespan đọc config và mở một client/transport cho mỗi enabled server,
  discover tools một lần, validate/cache schemas và register adapters trước khi load model.
- Owner task giữ async context của SDK suốt lifespan; enter/exit cùng task để đóng
  AnyIO scopes, HTTP sessions và stdio processes đúng cách. Shutdown close sessions.
- Client và cached schemas được reuse; không reconnect, ping, discover hoặc spawn
  process theo từng question/tool call. Concurrent calls qua cùng client đã được test.
- Server optional lỗi được đánh dấu unavailable, app tiếp tục. `required=true` lỗi
  làm startup fail rõ ràng và cleanup các kết nối đã mở.
- Discovery tối đa 256 tools/server; mỗi schema tối đa 8 KiB. Request mặc định tối đa
  8 MCP tools và 16 KiB external schema budget. Vượt budget trả HTTP 400 để chọn ít tool hơn.
- MCP disabled không có transport/network/tool discovery trên hot path. Microcheck
  10.000 lần tạo request tool view trên máy này: khoảng **13,94 µs/lần**, một built-in.
  Đây là overhead policy đo cục bộ, không phải benchmark TTFT của LLM hoặc mạng.
- Chi phí `mcp_connection_ms` là startup/discovery, được ghi riêng và không cộng
  lại như chi phí reconnect mỗi request. `mcp_tool_ms` đo tool execution thật.

## Configuration và secure defaults

`config/mcp_servers.example.json` có tất cả server **disabled**. Không có demo hardcode
trong application logic. `scripts/mcp_demo_server.py` chỉ phục vụ dev/tests.

```dotenv
MCP_ENABLED=false
MCP_CONFIG_PATH=./config/mcp_servers.local.json
MCP_MAX_TOOLS_PER_REQUEST=8
MCP_SCHEMA_BUDGET_BYTES=16384
```

Private config bị ignore cả Git và Docker build context. Credentials dùng env refs;
không ghi key/token trong JSON. `allowed_tools=[]` mặc định expose **zero** MCP tools.
`read_only_only=true` chỉ expose tool có annotation read-only. Operator cần chọn
allowlist và xác minh server; annotation không thay thế sandbox của chính server đó.

Ví dụ HTTP server thật, sau này chỉ thêm config/env, không sửa core:

```json
{
  "servers": {
    "research": {
      "label": "Research MCP",
      "enabled": true,
      "required": false,
      "transport": "http",
      "url_env": "RESEARCH_MCP_URL",
      "headers_env": {"Authorization": "RESEARCH_MCP_AUTHORIZATION"},
      "allowed_tools": ["lookup"],
      "read_only_only": true,
      "connect_timeout_seconds": 15,
      "tool_timeout_seconds": 30,
      "max_result_chars": 6000
    }
  }
}
```

`RESEARCH_MCP_AUTHORIZATION` chứa header value đầy đủ, ví dụ Bearer token được đặt
ngoài Git trong environment/Modal Secret. Không nhập secret vào chat. Nếu server
không cần auth thì bỏ mapping `headers_env`; không dùng reference rỗng.

stdio server thật khai báo `command`, `args`, optional `cwd` relative repo và `env_refs`
(destination env name → host env reference). Literal `command="python"` resolve tới
Python của runtime. Executable/packages của server phải tồn tại trong host/container;
app không tự tải/chạy executable do frontend gửi. Với filesystem server, operator
cấu hình allowed roots/sandbox trong server definition.

## Request contract và permissions

```json
{
  "conversation_id": "00000000-0000-0000-0000-000000000001",
  "question": "Bạch Đằng năm 938 có ý nghĩa gì?",
  "mode": "central",
  "retrieval_backend": "qdrant",
  "steering": {
    "mcp_enabled": true,
    "allowed_mcp_servers": ["research"],
    "allowed_tools": ["search_history", "mcp__research__lookup__d71b3589"],
    "mcp_failure_policy": "continue"
  },
  "debug": true
}
```

Thay conversation ID bằng ID được tạo qua `/api/v1/conversations`. Tool IDs lấy từ
`/ready`; example lookup ID ở trên là mapping deterministic cho server `research`,
remote tool `lookup`. Namespace: `mcp__<server>__<sanitized_tool>__<sha256-prefix>`;
hash phân biệt tên như `a.b` và `a_b`, không overwrite built-in.

- Omit steering: configured built-ins hoạt động như trước, **không expose MCP**.
- `mcp_enabled=true` và server allowlist chỉ mở tools đã được backend config/discovery
  cho phép. Unknown/unavailable server hoặc tool vượt quyền trả 400 trước khi ghi message.
- `allowed_tools=null`/omitted: tất cả built-ins hiện có cộng MCP subset của servers đã chọn.
  `allowed_tools=[]`: không tool; automatic history fallback cũng bị tắt.
- `mcp_failure_policy="continue"`: tool failure được ghi rõ, planner có thể tiếp tục với
  evidence khác được phép. `"fail"`: dừng request, error an toàn, non-stream HTTP 503.
- Hybrid gửi steering trực tiếp tới API nhận 422. Frontend Hybrid không serialize steering.
- Natural-language steering hỗ trợ hạn chế đơn giản: “chỉ dùng kho sử liệu local”,
  “không dùng Wikipedia”, “không dùng MCP”, tương đương English. Prompt text không mở quyền;
  “dùng MCP research” chỉ hữu ích nếu checkbox/request đã cho phép server đó.
- `search_history` tiếp tục được planner ưu tiên cho lịch sử và fallback khi được phép.
  Backend FAISS/Qdrant đi theo cùng request context. MCP bổ sung tối đa 2 evidence items,
  không chiếm các local-evidence slots hiện có. Khi history bị tắt, debug đánh dấu
  `history_retrieval_executed=false`; backend field vẫn biểu thị lựa chọn cho các lần
  history search của request, không giả vờ dense search đã chạy.

## Capabilities và frontend

`GET /ready` giữ các readiness fields cũ và retrieval capabilities; bổ sung:

```json
{
  "tools": [{"id":"search_history","label":"Kho sử liệu","available":true}],
  "mcp": {
    "enabled": true,
    "servers": [{"id":"research","label":"Research MCP","available":true,
      "tools":[{"id":"mcp__research__lookup__d71b3589","label":"lookup"}]}]
  },
  "tool_policy": {"max_mcp_tools":8,"schema_budget_bytes":16384}
}
```

Không có URL, command, headers, env values hoặc input schema trong ordinary UI capabilities.
Composer có Hybrid/Central selector và FAISS/Qdrant selector độc lập. Tools icon chỉ hiện
trong Central; popover compact có built-in, enabled-server và per-tool checkboxes.
Unavailable server disabled; capability thay đổi normalize selection/payload. Không có
localStorage MCP permissions; mặc định mỗi lần mở trang chỉ built-ins. FAISS/Qdrant
preference vẫn lưu `vn-history-retrieval-backend-v1` và normalize theo server.
Selection bị khóa khi streaming. Over-budget chặn Send và vẫn cho deselect tool.

## SSE, telemetry, cancellation và error boundary

Status đầu tiên vẫn được gửi ngay khi request được chấp nhận, trước retrieval/planning.
MCP dùng cùng SSE `status` pipeline; không polling UI hoặc timer progress.

```json
{"stage":"tool:mcp__research__lookup__d71b3589","state":"started",
 "message":"Tra cứu Research MCP","provider":"mcp","server":"research","tool":"lookup",
 "request_id":"...","retrieval_backend":"qdrant","mode":"central"}
```

Completed có `state="completed"`, `latency_ms`; failed có `state="failed"`.
Stable stage ID cho từng tool, repeated calls cập nhật cùng row; toàn bộ calls còn trong debug trace.
Progress panel collapse sau token đầu, summary giữ MCP label; Stop/error/done dừng spinner.
Reducer giữ request/conversation scope guards, không đưa stale SSE vào conversation mới.
DeveloperTrace có requested/effective steering, provider/server/tool, success/result count,
`mcp_connection_ms`, `mcp_tool_ms`. UI thường không hiển thị raw milliseconds.

Connect timeout mặc định 15s; tool timeout 30s, config bounded. AbortController/disconnect
→ request cancellation/event → cancel SDK call task → cleanup per-call waiters.
Event flag kiểm tra mỗi 50ms khi đang chờ tool; ASGI disconnect watcher hiện có kiểm tra
250ms. Không đóng connection dùng chung chỉ vì một request Stop. Native SDK transports
và fake timeout/cancel/reuse đã được kiểm thử.

Không trả raw exception, tool arguments, transport envelope hoặc auth values trong trace.
Logs của integration chỉ server ID/error type; stdio stderr bị chặn và transport diagnostics
được suppress để tránh URL/envelope leak. Known secret refs/header token/endpoint được redact
khỏi tool text. JSON secret keys bị loại, unsafe URL schemes/userinfo bị từ chối, secret
query fields bị bỏ. External `$ref` trong input schema bị từ chối.

## Grounding và provenance

Tool output là untrusted evidence, không system/developer instructions. Planning prompt có
boundary khi MCP được expose; final prompt vẫn **nguyên `SYSTEM_PROMPT`** của live app.
`build_messages` vẫn đưa title/text và IDs của history chunks vào context cho LLM observe.
Output giới hạn tổng chars, depth 4, 128 structured nodes, tối đa 16 content blocks.
Text/resource-link/text-resource được normalize; image/audio/binary không đổ vào prompt.

MCP sources dùng `source_kind="mcp"`, `chunk_id="mcp:<server>:<digest>"`, remote source ID
được namespace `mcp:<server>:...`; không giả mạo Corpus V1 chunk IDs. Public URL/source provenance
được giữ, source drawer ghi “Nguồn MCP”. Corpus V1 citations `[chunk_id]` không đổi;
existing display-index renderer cũng resolve external MCP IDs riêng.

## Chạy local/dev từ PowerShell

Terminal trong repo:

```powershell
Set-Location (git rev-parse --show-toplevel)
$env:Path = "$PWD\.conda;$PWD\.conda\Scripts;$PWD\.conda\Library\bin;$env:Path"
$env:PYTHONUTF8 = '1'
python -m pip install -r requirements.txt
```

Enable **demo stdio** trong private config (không overwrite file có sẵn):

```powershell
if (Test-Path config/mcp_servers.local.json) { throw 'Config exists; edit it deliberately.' }
$cfg = Get-Content config/mcp_servers.example.json -Raw | ConvertFrom-Json -AsHashtable
$cfg.servers.demo.enabled = $true
$cfg | ConvertTo-Json -Depth 12 | Set-Content -Encoding utf8 config/mcp_servers.local.json
$env:MCP_ENABLED = 'true'
$env:MCP_CONFIG_PATH = 'config/mcp_servers.local.json'
```

Để thử **Streamable HTTP demo**, terminal riêng:

```powershell
.\.conda\python.exe -m scripts.mcp_demo_server --transport streamable-http --port 8383
```

Định nghĩa server dev HTTP (private config, chỉ thay khi chủ động chọn HTTP demo):

```json
{"servers":{"research":{"label":"Research demo","enabled":true,"transport":"http",
 "url":"http://127.0.0.1:8383/mcp","allowed_tools":["lookup"],"read_only_only":true}}}
```

Chạy full local với cả hai dense lanes trên host có CUDA/RAM đủ cho V1 + models:

```powershell
$env:APP_MODE = 'full'
$env:DEVICE = 'cuda'
$env:DTYPE = 'bfloat16'
$env:ARTIFACT_ROOT = './artifacts/corpus_v1'
$env:RETRIEVAL_DENSE_BACKEND = 'faiss'
$env:RETRIEVAL_AVAILABLE_BACKENDS = 'faiss,qdrant'
$env:QDRANT_URL = '<your-cluster-URL>'
$env:QDRANT_API_KEY = '<your-private-key>'
$env:QDRANT_COLLECTION = 'vn_history_v1_e5'
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Credentials ở đây là placeholders, cần local env/.env riêng; Modal Secret không tự cung cấp
chúng cho local Python. Máy hiện tại 16 GiB/CPU không được kiểm chứng chạy full models;
đã kiểm thử full V1/models trên Modal A100. Với FAISS-only, đặt available `faiss`, không cần Qdrant env.
Terminal frontend:

```powershell
$env:VITE_API_BASE_URL = 'http://127.0.0.1:8000'
npm --prefix frontend run dev
```

Capabilities: `Invoke-RestMethod http://127.0.0.1:8000/ready`. Chọn Central → Tools → server/tool.
Để tắt MCP: `$env:MCP_ENABLED='false'` trước startup. Đổi quyền/backend đã available không cần restart.

## Modal

`modal_app.py` đọc `MCP_ENABLED`, `MCP_CONFIG_PATH`; khi bật, copy private config vào
`/etc/vn-history/mcp_servers.json`. Optional `MODAL_MCP_SECRET_NAME` đưa env references
vào container. Server HTTP cần endpoint reachable từ Modal; `127.0.0.1` của máy local
không reachable từ remote container. stdio executable phải có trong image.

```powershell
$env:MODAL_SERVER_URL = 'https://api.modal2.com'
$env:RETRIEVAL_DENSE_BACKEND = 'faiss'
$env:RETRIEVAL_AVAILABLE_BACKENDS = 'faiss,qdrant'
$env:MODAL_QDRANT_SECRET_NAME = 'vn-history-qdrant'
# Only after configuring a real server reachable from Modal:
$env:MCP_ENABLED = 'true'
$env:MCP_CONFIG_PATH = 'config/mcp_servers.local.json'
$env:MODAL_MCP_SECRET_NAME = '<your-MCP-Secret-name>'
.\.conda\Scripts\modal.exe serve modal_app.py
```

Mặc định MCP false; demo không được deploy tự động. Các lệnh deploy/logs ở
[MODAL_QUICKSTART.md](MODAL_QUICKSTART.md); production deployment là bước riêng sau code delivery.

## Validation

| Check | Result |
| --- | --- |
| Full pytest | 289 passed |
| MCP tests | 24 passed; fake clients + actual local stdio/HTTP SDK transports |
| Frontend tests | 66 Node + 74 Vitest = 140 passed |
| Browser tests | 45 passed; desktop/tablet/390px mobile |
| Frontend build/lint | PASS; 2079 modules production build |
| Real Modal V1 Hybrid/Central × FAISS/Qdrant after MCP dependency | PASS |
| Real Qwen3-8B → configured SDK MCP stdio → grounded answer | PASS; client/resources reused |
| pip dependency check | PASS |
| Protected artifacts/research/ZIP audit | 192 unchanged |
| Live response-mode search | None in live frontend/API/runtimes; compatibility modules retained |

Coverage: disabled, discovery/cache, allowlists/read-only, namespace collision, schema budget,
permissions concurrency, Central backend propagation, optional/required startup failure, real lifecycle,
timeout, cancellation/reuse/non-cooperative cleanup, malformed/huge results, secrets, source provenance, planner subset,
fixed grounded prompt/chunks, SSE/DeveloperTrace, frontend payload, stale conversation events,
abort, responsive popover/progress and capability normalization. No external server dependency.

Windows full suite uses `$env:PYTHONCASEOK='1'` for existing historical `Training` imports:

```powershell
$env:PYTHONCASEOK = '1'
.\.conda\python.exe -m pytest -q
npm --prefix frontend test -- --run
npm --prefix frontend run test:ui
npm --prefix frontend run lint
npm --prefix frontend run build
```

Frozen SILVER V1/Citation V2, SFT/training artifacts, corpus, PEFT and retrieval indexes remain unchanged.
Dynamic FAISS/Qdrant architecture/API/progress and paired latency results:
[DYNAMIC_RETRIEVAL_REPORT.md](DYNAMIC_RETRIEVAL_REPORT.md).

### Actual model / Modal MCP smoke

`scripts/modal_mcp_smoke.py` tạo private temporary demo config và SQLite trong container dev.
MCP-only request có allowlist rõ ràng để kiểm chứng model gọi tool; request tiếp theo bỏ steering
để kiểm chứng default history retrieval. Demo không đi vào production app defaults.

```powershell
$env:MCP_ENABLED = 'false' # Dev function initializes its own temporary demo config.
$env:RETRIEVAL_AVAILABLE_BACKENDS = 'faiss,qdrant'
.\.conda\Scripts\modal.exe run --write-result reports/dynamic_retrieval/mcp-smoke-new.json scripts/modal_mcp_smoke.py::mcp_smoke
```

Kết quả lưu local, gitignored: `reports/dynamic_retrieval/2026-10-03_real_mcp_modal.json`.
624.288 V1 chunks loaded; actual `Qwen/Qwen3-8B` gọi `mcp__demo__lookup__316d035f`, success=true,
1 normalized source, `mcp_tool_ms=7,93`, startup/discovery `mcp_connection_ms=1.463,06`.
First status 0,078ms. Final TTFT 28.180,60ms / e2e 31.969,32ms ở request đầu gồm cold lazy
Qwen8B loading/planning; không phải MCP network latency. Request history tiếp theo không có
MCP call, 3 history sources từ Qdrant; final TTFT 6.289,20ms / e2e 8.063,77ms.
Client và corpus/BM25/embedder/reranker/FAISS instances không đổi. Smoke app đã completed/stopped.

Lần demo Modal đầu thiếu module search path trong child process của stdio. Dev script đã sửa
`cwd` bằng path resolve từ module được mount; rerun pass. Application core không hardcode demo
hoặc đường dẫn máy cũ. Server stdio thật cần executable/cwd/package path hợp lệ như config docs.

## Known limitations

- Chưa có MCP server production/credentials thật theo lựa chọn của user. Actual SDK transports
  đã test với demo, gồm real Qwen3-8B trên Modal; unit SSE/permissions dùng fake model để deterministic.
- Discovery chỉ lúc startup; thay config/tool schemas cần startup lại. Không rediscover trên hot path.
- Cancellation dừng chờ và gửi cancel qua SDK khi supported; server ngoài không hợp tác có thể
  tiếp tục tác vụ của nó. App không bảo đảm undo remote side effects. Default chỉ read-only allowlist.
- OAuth browser flow không được thêm vào UI; HTTP auth hiện dùng configured env headers.
- Size/schema/permissions boundaries hạn chế rủi ro; prompt grounding không bảo đảm LLM tuyệt đối
  miễn nhiễm mọi prompt injection. App không xem MCP output như trusted instructions.
- Demo content là synthetic, không xác minh lịch sử; không dùng làm benchmark khoa học.
- MCP tool timeout/failure có trạng thái failed rõ ràng; continuation chỉ theo policy đã chọn.
- Thêm server stdio thật có executable/dependencies mới cần chuẩn bị môi trường đó;
  HTTP server đã reachable thì chỉ config/env, không sửa core code.

## Files

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
