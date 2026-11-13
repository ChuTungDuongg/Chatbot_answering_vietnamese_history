# Sử Việt

**Trợ lý tra cứu lịch sử có dẫn nguồn.**

Sử Việt kết hợp truy xuất tư liệu và mô hình ngôn ngữ để hỗ trợ tìm hiểu lịch sử Việt Nam. Câu trả lời được truyền trực tiếp khi model sinh, kèm nguồn tham khảo để người dùng kiểm tra và đối chiếu.

Ứng dụng hỗ trợ hội thoại, tra cứu tài liệu PDF/ảnh tải lên và lựa chọn FAISS hoặc Qdrant cho từng câu hỏi. Trên Modal, một URL HTTPS phục vụ giao diện và API sau màn hình đăng nhập bằng mật khẩu.

| Chế độ | Cách hoạt động | Model |
| --- | --- | --- |
| **Hybrid RAG** | Truy xuất tư liệu rồi sinh câu trả lời từ context. | Local Qwen3-4B vanilla/SFT hoặc OpenAI Luna/Terra |
| **Central Agent** | Lập kế hoạch, dùng công cụ được cấp quyền và tổng hợp câu trả lời. | Local Qwen3-8B vanilla hoặc OpenAI Luna/Terra |

Backend dùng FastAPI; frontend dùng React/Vite. Retrieval kết hợp E5, FAISS/Qdrant, BM25S và cross-encoder reranking trên Corpus V1.

Inference có thể chọn Qwen local hoặc OpenAI Responses API cho từng request, độc lập với chat mode và dense backend. OpenAI mặc định tắt và được bảo vệ bằng phiên đăng nhập và hạn mức phía máy chủ; xem [cấu hình bảo mật](docs/CONFIGURATION.md#openai-responses-api). Không đặt OpenAI API key trong trình duyệt.

Modal hỗ trợ OpenAI API với retrieval trên CPU hoặc GPU nhỏ: giữ `LOCAL_INFERENCE_ENABLED=false` và đặt `MODAL_RETRIEVAL_GPU_CLASS=T4` để tăng tốc E5/reranker; để trống biến GPU retrieval để chạy CPU-only. Chỉ đặt `LOCAL_INFERENCE_ENABLED=true` khi muốn tải Qwen. Central ưu tiên Qdrant cho tư liệu Việt Nam và Wikipedia cho lịch sử thế giới, đồng thời tôn trọng lựa chọn nguồn/công cụ của người dùng.

[Central Agent v2](docs/AGENTIC_RAG_V2.md) bổ sung dependency checks, chạy song song tool an toàn, replan theo evidence và budgets riêng từng request. Câu hỏi đơn giản vẫn có thể kết thúc sau một retrieval. Evaluation offline và giới hạn của kết quả được ghi kèm trong tài liệu.

## Inference provider: Local và API LLM

| Provider | Model / API ID | Chat mode | Cấu hình reasoning |
| --- | --- | --- | --- |
| Local Qwen | `Qwen/Qwen3-4B-Instruct-2507` vanilla/SFT | Hybrid RAG | Local generation settings; Transformers hoặc vLLM theo cấu hình |
| Local Qwen | `Qwen/Qwen3-8B` vanilla | Central Agent | Action/final generation settings của Qwen |
| **OpenAI API LLM** | **GPT-6 Luna — `gpt-6-luna`** (default OpenAI) | **Hybrid RAG và Central Agent** | Final: Low `low`, Medium `medium`, High `high`, Very High `xhigh` |
| **OpenAI API LLM** | **GPT-6.1 Terra — `gpt-6.1-terra`** | **Hybrid RAG và Central Agent** | Cùng bốn reasoning levels, chọn theo từng request |

Provider/model/effort không ghi đè cấu hình của user khác. OpenAI dùng Responses API chính thức, native streaming/function calling và evidence của ứng dụng. Central OpenAI có planning effort riêng, default `low`, còn final giữ effort UI đã chọn; `inherit` dùng cùng effort cho cả hai. Key chỉ ở server/Modal Secret. `LOCAL_INFERENCE_ENABLED=false` cho phép dùng API LLM mà không tải Qwen; T4/L4 có thể chỉ phục vụ E5/reranker.

## Kiến trúc hệ thống và agent

Ứng dụng dùng **một Central Agent với planner lặp**, cùng đường Hybrid RAG trực tiếp. Planner và final generation là hai giai đoạn của cùng request; scheduler, policy, retriever và guardrails là mã backend xác định, không phải các LLM agents riêng. Provider, chat mode và dense backend được chọn độc lập theo request.

### 1. Toàn hệ thống: một URL, hai đường trả lời

```mermaid
flowchart TB
    Owner["Chủ ứng dụng / trình duyệt"] --> URL["Một URL HTTPS trên Modal"]
    URL --> Auth["Password login + signed session cookie<br/>HttpOnly / Secure / SameSite + CSRF + login limits"]
    Auth --> UI["React / Vite static assets<br/>Hybrid hoặc Central; provider; model; reasoning; FAISS/Qdrant"]
    UI --> API["FastAPI /api/v1<br/>Auth + owner scope + validation + OpenAI admission limits"]
    API --> History["Conversation history + attachment scope"]
    API --> Router{"Chat mode của request"}

    subgraph Hybrid["Hybrid RAG: đường trực tiếp"]
        Router -->|hybrid| Analyze["Question analysis + query variants"]
        Analyze --> Retrieve["Shared retrieval service"]
        Retrieve --> Grounded["Context selection + build_messages<br/>Vietnamese grounded instructions + history"]
    end

    subgraph Central["Central Agent: một vòng plan / observe / replan"]
        Router -->|central| View["Request-scoped ToolView<br/>Built-in + enabled MCP allowlists / schema budget"]
        View --> Planner["LLM planner<br/>Chọn calls, arguments, independence và completion"]
        Planner --> Executor["AgentExecutor + ToolRegistry<br/>Permissions / Pydantic / dependencies / budgets / cancellation"]
        Executor --> Builtins["Built-in tools<br/>History, uploads, Wikipedia, web"]
        Executor --> MCP["MCP manager<br/>Dynamic schemas + allowed servers/tools<br/>Serialized shared server sessions"]
        Builtins -->|search_history / uploaded docs| Retrieve
        Builtins --> External["Wikipedia API / configured web provider<br/>Safe page fetching"]
        Builtins --> Observe["Sanitized bounded observations + provenance"]
        MCP --> Observe
        Observe --> Decide{"Đủ selected evidence + explicit completion?"}
        Decide -->|Chưa đủ và còn budget| Planner
        Decide -->|Đủ hoặc phải dừng| Select["Diverse context selection<br/>Reserve required domains / attachments; qualify gaps"]
        Select --> Grounded
    end

    Grounded --> Provider{"Inference provider của request"}
    Planner -. "Cùng provider/model; options riêng cho planning" .-> Provider
    Provider --> Local["Local Qwen<br/>Transformers hoặc vLLM<br/>Qwen3-4B Hybrid / Qwen3-8B Central"]
    Provider --> OpenAI["OpenAI async Responses API<br/>gpt-6-luna / gpt-6.1-terra<br/>store=False; native function calls; shared async client"]
    Local --> Stream["ModelDelta / ModelDone → SSE<br/>status / answer_delta / sources / debug_trace / done / error"]
    OpenAI --> Stream
    Stream --> Validate["Grounding + citation IDs + completion validation<br/>Không lưu failed/incomplete answer như thành công"]
    Validate --> DB[("SQLite trên named Modal Volume<br/>Conversations / attachments / sessions / usage limits")]
    Stream --> UI
    API -. "Request IDs, measured latency/usage, no hidden reasoning" .-> Metrics["Telemetry + protected admin monitoring"]
```

Mũi tên provider của planner biểu thị việc dùng adapter chung; **chỉ final generation** phát `answer_delta` cho người dùng. Tool-call arguments và private reasoning không được stream thành câu trả lời. OpenAI API key và internal credentials ở server/Modal Secrets.

### 2. Retrieval dùng chung, không nhân đôi RAG theo provider

```mermaid
flowchart LR
    Query["Question / refined tool query"] --> E5["Multilingual E5 embedding"]
    E5 --> Dense{"Dense backend đã sẵn sàng<br/>FAISS hoặc Qdrant"}
    Dense --> FAISS["FAISS index"]
    Dense --> Qdrant["Qdrant Cloud / configured collection"]
    Query --> BM25["BM25S lexical retrieval"]
    FAISS --> RRF["RRF fusion"]
    Qdrant --> RRF
    BM25 --> RRF
    RRF --> Rank["Cross-encoder reranker"]
    Rank --> Context["Selected chunks + stable chunk_id + provenance"]
    Context --> H["Hybrid grounded prompt"]
    Context --> C["Central tool observations / final context"]
    Upload["PDF / images → extraction / OCR<br/>Owner/conversation-scoped documents"] --> Temp["Temporary document retrieval"]
    Temp --> H
    Temp --> C
```

Chọn OpenAI không bỏ qua retrieval. Trong cấu hình API + T4, **E5/reranker chạy trên T4**, còn LLM chạy qua OpenAI; Qwen không được tải. FAISS/Qdrant theo lựa chọn request, không tự đổi backend khi lỗi. Central ưu tiên corpus cho Việt Nam, external evidence cho lịch sử thế giới, trong giới hạn nguồn/công cụ người dùng cho phép.

### 3. Thuật toán vòng lặp Central Agent

**Mã thực thi vòng lặp:** [`CentralRuntime.prepare()`](app/central/runtime.py); **admission/scheduling/tool execution:** [`AgentExecutor.execute()`](app/central/execution.py). Hai provider cùng dùng harness này: OpenAI nhận native function calls, Qwen dùng Hermes decoder.

```mermaid
flowchart TD
    Start["Nhận question + history + steering + provider/model"] --> Init["Tạo request-local state<br/>ToolView, immutable limits, evidence store, counters, deadline"]
    Init --> Cancel{"Đã hủy / browser disconnect?"}
    Cancel -->|Có| Abort["Cancel active async tasks / cooperative workers<br/>Close upstream stream; không lưu successful answer"]
    Cancel -->|Chưa| Budget{"Còn round, deadline<br/>và planning token budget?"}
    Budget -->|Không| ExitBudget["Ghi exit_reason<br/>Round / deadline / token budget"]
    Budget -->|Có| Plan["Gọi LLM planner<br/>Question + available schemas + prior sanitized observations"]
    Plan --> Decode{"Native OpenAI hay local Qwen?"}
    Decode -->|OpenAI| Native["Giữ reasoning/output items và native call IDs<br/>Malformed arguments nhận error output"]
    Decode -->|Qwen| Hermes["Decode Hermes tool calls<br/>Structured completion control"]
    Native --> Envelope{"Envelope hợp lệ<br/>và không quá 64 calls?"}
    Hermes --> Envelope
    Envelope -->|Không| Error["SSE error; đóng resources<br/>Không lưu completed answer"]
    Envelope -->|Có| Calls{"Planner có yêu cầu tool?"}
    Calls -->|Không| NoTools["exit_reason = planner_no_tools"]
    Calls -->|Có| Controls["Strip planner-only controls<br/>completion / explicit independence"]
    Controls --> Admit["Admission từng call<br/>Allowlist + schema + domain + round/total limits<br/>Observed ID/URL + normalized repeat key"]
    Admit --> Allowed{"Call được chấp nhận?"}
    Allowed -->|Không| Rejected["Sanitized rejected outcome<br/>Giữ original call ID; không execute"]
    Allowed -->|Có| Parallel{"Explicitly independent + read-only<br/>+ parallel_safe + resource không xung đột?"}
    Parallel -->|Có| Batch["Bounded concurrent batch<br/>Tối đa 3 safe calls"]
    Parallel -->|Không| Serial["Sequential execution<br/>Retrieval gate / shared MCP session gate"]
    Batch --> Tool["ToolRegistry validate + execute<br/>Isolated context/metrics; per-tool timeout"]
    Serial --> Tool
    Tool --> Failure{"Lỗi tool?"}
    Failure -->|MCP failure_policy = fail| Error
    Failure -->|Recoverable / timeout / empty| Outcome["Success hoặc fixed error outcome"]
    Failure -->|Không| Outcome
    Rejected --> Observe["Restore model call order<br/>Merge completed metrics; ingest evidence/provenance"]
    Outcome --> Observe
    Observe --> Feedback["Bounded valid JSON observations<br/>Native output cho mọi accepted call ID<br/>Tool output là untrusted data"]
    Feedback --> Enough{"Explicit completion + mọi call usable<br/>+ selected evidence cover scope<br/>+ không truncated/pending dependency?"}
    Enough -->|Đủ| Sufficient["exit_reason = evidence_sufficient"]
    Enough -->|Chưa đủ| Remaining{"Còn tool/observation budget<br/>và chưa lặp vô ích / hết deadline?"}
    Remaining -->|Còn| Replan["Replan theo observation<br/>Sửa query / đổi approved tool / fetch observed ID"]
    Replan --> Cancel
    Remaining -->|Hết| Stop["Ghi safe stopping reason<br/>Không tự bịa thêm nguồn"]
    NoTools --> Fallback{"Fallback cũ còn hợp lệ<br/>và được phép, còn budget?"}
    Stop --> Fallback
    ExitBudget --> Fallback
    Fallback -->|Có| FallbackTool["History hoặc Wikipedia fallback<br/>Qua cùng executor, permissions và budgets"]
    FallbackTool --> SelectFinal["Chọn context cuối; reserve required source kinds<br/>Gaps → qualified execution instruction"]
    Fallback -->|Không| SelectFinal
    Sufficient --> SelectFinal
    SelectFinal --> Final["Grounded final generation<br/>Final reasoning người dùng chọn + exact native replay"]
    Final --> SSE["Stream answer_delta<br/>Filter unknown citation IDs"]
    SSE --> Valid{"Completed + nonempty<br/>+ Central citations hợp lệ?"}
    Valid -->|Có| Save["Persist answer + sources + trace<br/>SSE sources + done"]
    Valid -->|Không| Error
```

Agent quyết định query/tool/completion bằng LLM; backend quyết định quyền, admission, concurrency và budgets. Chẳng hạn search Wikipedia trả page IDs trong round 1; planner chỉ tạo fetch dựa trên IDs đó ở round 2. Kết quả không rỗng chưa đủ để dừng. Mọi đường lỗi/hủy giữ nguyên isolation và không tạo bản ghi assistant hoàn tất giả.

### 4. Trình tự gọi tools song song và phụ thuộc

```mermaid
sequenceDiagram
    participant U as User / SSE
    participant C as CentralRuntime
    participant P as LLM Planner
    participant E as Executor + Registry
    participant T as Approved Tools
    participant F as Final LLM
    U->>C: Question + history + provider/model/effort + steering
    C->>C: Build immutable limits, ToolView, request-local state
    loop Tối đa 4 rounds, dừng sớm khi đủ
        C->>P: Trusted instructions + available schemas + sanitized observations
        Note over P: OpenAI planner default low, configurable or inherit
        P-->>C: Native calls / Qwen calls + independent + completion controls
        C->>E: Validate allowlists, arguments, observed IDs/URLs, repeat/call/deadline budgets
        alt Calls độc lập và backend policy xác nhận an toàn
            par Safe read-only tool A
                E->>T: Search topic A
                T-->>E: Evidence A
            and Safe read-only tool B
                E->>T: Search topic B
                T-->>E: Evidence B
            end
        else Dependency chưa quan sát hoặc shared resource
            E->>E: Reject speculative reader / serialize shared retrieval or MCP
        end
        E-->>C: Outcomes theo call order + original IDs + isolated metrics
        C->>C: Sanitize observations, select evidence, check full scope and completion
        Note over C,P: Search → observe real page IDs → fetch ở round sau, không đoán ID
        C-->>U: status + operational debug metadata khi bật
    end
    C->>F: Selected evidence + grounded instructions + native replay khi cần
    Note over F: Giữ reasoning người dùng chọn, kể cả Very High / xhigh
    F-->>U: answer_delta liên tục
    C->>C: Validate citations/completion, save successful answer
    C-->>U: sources + done, hoặc error khi không hợp lệ
    Note over U,T: Stop/disconnect truyền cancellation tới planner, tool tasks và stream
```

Default: 4 rounds, 4 calls/round, 12 tool executions/request, một lần cho cùng normalized arguments, tối đa 3 calls song song đã được xác nhận an toàn. Planning + tools có deadline 180 giây; mỗi tool 30 giây. History/upload retrieval dùng chung resource gate; MCP không tự chạy song song chỉ vì remote tool ghi read-only. Budget hết vẫn phải đóng mọi native call ID và chỉ trả lời phần evidence hỗ trợ.

**Latency:** OpenAI planner mặc định dùng `CENTRAL_OPENAI_PLANNING_REASONING_EFFORT=low`; UI reasoning tiếp tục áp dụng cho final answer. `inherit` khôi phục việc dùng cùng effort cho mọi bước. Câu hỏi nhiều ý có thể gom independent searches trong một round, rồi gom readers đã biết IDs ở round tiếp theo. Backend vẫn kiểm tra sufficiency và quyền; số round thực tế phụ thuộc evidence. Xem [telemetry và giới hạn đánh giá](docs/AGENTIC_RAG_V2.md).

### 5. Chạy Modal gắn với terminal và Ctrl+C

```mermaid
flowchart LR
    Command["python scripts/modal_session.py"] --> Mode{"Chế độ"}
    Mode -->|Mặc định| Deploy["modal deploy modal_app.py<br/>Stable public URL + existing Secrets/Volumes"]
    Mode -->|--attach| Existing["Lookup deployment hiện có"]
    Deploy --> ID["Resolve exact App ID + selected Modal environment"]
    Existing --> ID
    ID --> Follow["Follow logs + lấy URL fastapi_app từ Modal"]
    Follow --> Warm["Đánh thức container<br/>Poll /health trong thời gian giới hạn"]
    Warm -->|HTTP 200 + status ok| Ready["Sử Việt is ready: URL<br/>Tự mở trình duyệt; chờ Ctrl+C"]
    Warm -->|Timeout hoặc Ctrl+C| Cleanup["Stop/reap local log process<br/>modal app stop APP_ID --yes"]
    Ready -->|Ctrl+C| Cleanup
    Cleanup --> Offline["App và containers dừng<br/>Named Volumes / Secrets được giữ"]
    Offline -->|Chạy lại script| Deploy
```

PowerShell tại root repository, với environment/Secrets đã cấu hình:

```powershell
# Deploy, chờ web sẵn sàng rồi tự mở trình duyệt; Ctrl+C dừng app Modal.
& '.\.venv\Scripts\python.exe' scripts/modal_session.py

# Hoặc điều khiển deployment đang có, không deploy lại.
& '.\.venv\Scripts\python.exe' scripts/modal_session.py --attach
```

Chờ dòng **`Sử Việt is ready: URL`**; thông báo deploy thành công của Modal chỉ xác nhận đã đăng ký deployment, chưa xác nhận container/web đã khởi động. Script tự warm container và chờ tối đa 960 giây, có tiến độ trong terminal, rồi mở URL. Dùng `--no-browser` nếu chỉ muốn in URL, hoặc `--ready-timeout 1200` để đổi giới hạn chờ. Khởi động lỗi/timeout trả exit code khác 0 và dừng app; Ctrl+C trong lúc chờ cũng dừng app.

Chờ thông báo đã dừng trước khi tắt máy. Mất điện, force-kill terminal hoặc mất mạng không bảo đảm gửi được stop tới cloud; script báo lệnh stop thủ công nếu không xác nhận được. `modal deploy modal_app.py` thông thường vẫn là deployment độc lập với terminal. Xem [Deployment](docs/DEPLOYMENT.md) cho cấu hình T4/API-only và lifecycle đầy đủ.

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
| Dataset và đánh giá thủ công nội bộ | [Sử Việt Eval Studio v1.1](docs/EVAL_STUDIO.md) |
| Hiệu năng và vận hành | [Benchmark](docs/BENCHMARKS.md) · [Telemetry](docs/RUNTIME_TELEMETRY.md) |
| Làm việc với mã nguồn | [Phát triển và kiểm tra](docs/DEVELOPMENT.md) |

Xem [mục lục docs/](docs/README.md) để tra cứu toàn bộ tài liệu. Kết quả tự động và trích dẫn cần được kiểm chứng trên dữ liệu thực tế; [hướng dẫn evaluation](docs/EVALUATION.md) ghi rõ phạm vi metric và các giới hạn đã biết.
