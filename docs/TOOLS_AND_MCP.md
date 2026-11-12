# Central tools và MCP

[`ToolRegistry`](../app/tools/registry.py) validate arguments và chạy tools theo request context. Context mang owner, conversation, dense backend, permissions, cancellation và telemetry; model không tự chọn scope dữ liệu.

Built-in declarations lấy từ [`tools.yaml`](../app/tools/definitions/tools.yaml), đối chiếu với Pydantic ở `app/tools/builtin/*/tool.py`. Mỗi tool có `TOOL.md`; import module cũ tiếp tục hoạt động. MCP giữ schema discovery động và không đưa vào YAML. Xem [hướng dẫn thêm prompt/tool](PROMPTS_AND_TOOLS.md) và contract tests.

| Built-in tool | Điều kiện | Chức năng |
| --- | --- | --- |
| `search_history` | Central full, khi request cho phép | Retrieval corpus dùng chung với Hybrid. |
| `search_uploaded_documents` | `CENTRAL_ENABLE_DOCUMENTS=true` | Tìm trong attachments của owner/conversation hiện tại. |
| `search_wikipedia`, `fetch_wikipedia_page` | `CENTRAL_ENABLE_WIKIPEDIA=true` | Tìm và đọc Wikipedia. |
| `search_web`, `fetch_web_page` | `CENTRAL_ENABLE_WEB=true` | Search provider đã cấu hình và đọc trang. |

Web mặc định tắt. History fallback chỉ chạy khi tool policy cho phép và phù hợp phạm vi local; xem [Telemetry](RUNTIME_TELEMETRY.md) cho external routing và early exit.

## Architecture và SDK

Central dùng provider được chọn theo request: OpenAI Responses API hoặc Qwen3-8B local cho planning và final grounded generation.

Central v2 có [scheduler và execution budgets](AGENTIC_RAG_V2.md): planner chọn calls dựa trên observations, backend kiểm tra quyền/dependencies trước khi chạy. `__central_independent` là optional planner-only boolean, không sửa public YAML signatures; flag không đủ để cấp parallel execution. Built-in phải có policy Python tường minh; history/uploaded-document retrieval dùng chung resource gate. MCP luôn chạy tuần tự trong executor và shared session được serialize theo server qua các request. Reader chỉ nhận ID/title/URL đã quan sát hoặc người dùng cung cấp rõ; speculative same-round fetch trả lỗi an toàn để planner lập lại call. Timeout vẫn tôn trọng `mcp_failure_policy=fail`. Trace chỉ chứa metadata/outcomes, không raw arguments hoặc reasoning.
Built-ins và MCP adapters dùng cùng registry. Một immutable `ToolView`
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
- MCP disabled không có transport/network/tool discovery trên hot path.
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
boundary khi MCP được expose; final prompt dùng chung **`LIVE_SYSTEM_PROMPT`** của live app (grounding cộng live-only style), không nhận instructions từ tool output.
`build_messages` vẫn đưa title/text và IDs của history chunks vào context cho LLM observe.
Output giới hạn tổng chars, depth 4, 128 structured nodes, tối đa 16 content blocks.
Text/resource-link/text-resource được normalize; image/audio/binary không đổ vào prompt.

MCP sources dùng `source_kind="mcp"`, `chunk_id="mcp:<server>:<digest>"`, remote source ID
được namespace `mcp:<server>:...`; không giả mạo Corpus V1 chunk IDs. Public URL/source provenance
được giữ, source drawer ghi “Nguồn MCP”. Corpus V1 citations `[chunk_id]` không đổi;
existing display-index renderer cũng resolve external MCP IDs riêng.


## Thiết lập local và Modal

Copy [`config/mcp_servers.example.json`](../config/mcp_servers.example.json) thành `config/mcp_servers.local.json` nếu file chưa có, rồi chỉnh server đã chọn. Tất cả servers trong mẫu đều disabled. Bật demo stdio bằng `servers.demo.enabled=true`; `allowed_tools` và read-only policy đã có trong mẫu. Demo phục vụ kiểm thử, không dùng làm nguồn lịch sử.

Đặt trong `.env` trước khi khởi động full backend:

```dotenv
MCP_ENABLED=true
MCP_CONFIG_PATH=./config/mcp_servers.local.json
```

Streamable HTTP demo có thể chạy ở terminal riêng:

```powershell
python -m scripts.mcp_demo_server --transport streamable-http --port 8383
```

Config server HTTP dùng `transport=http`, URL tương ứng như `http://127.0.0.1:8383/mcp`, allowlist tool cụ thể. Lấy tool IDs thực tế qua `/ready`, chọn Central và cấp quyền trong Tools popover. Thay config/schema cần restart để discover lại; chọn quyền từng request không cần restart.

Trên Modal, entrypoint copy private config tới `/etc/vn-history/mcp_servers.json` khi bật MCP. `MODAL_MCP_SECRET_NAME` tùy chọn đưa credentials theo env refs vào container. HTTP endpoint phải reachable từ Modal; loopback của máy local không phải server remote. Stdio executable và dependencies phải có trong image. Xem [Deployment](DEPLOYMENT.md).

Cancellation dừng chờ và gửi cancel qua SDK khi supported; server ngoài có thể vẫn tiếp tục tác vụ. HTTP auth hiện dùng configured environment headers, chưa có OAuth browser flow trong UI. Tool annotations không thay sandbox/server permissions; output được giới hạn và coi là evidence không đáng tin cậy về mặt chỉ dẫn.
