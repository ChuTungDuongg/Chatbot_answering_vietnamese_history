# Prompt resources và built-in tools

## Hợp đồng trước và sau refactor

Baseline được chụp từ commit `aba83fd71ac1b51a7f7b340b5c62dff8fd8c2e18` trước khi thay mã nguồn. `tests/fixtures/prompt_contracts.json` lưu nguyên văn constant công khai, message Hybrid có history, ba response mode offline, SFT grounding và sáu instruction annotation SILVER. `tests/fixtures/builtin_tool_contracts.json` lưu JSON Schema Pydantic của sáu built-in. Fixture SHA-256 evaluation có sẵn vẫn giữ nguyên.

Trước đây prompt là literal trong `rag/prompting.py`, `rag/response_modes.py`, `central/planning.py`, `central/runtime.py` và `evaluation/annotation/silver_logic.py`. Tool đặt cạnh client/helper trong năm module; registry lấy schema từ Pydantic và description từ Python. Hiện prompt nằm trong Markdown, declaration built-in trong YAML và mỗi implementation có module riêng. Python vẫn lắp message, validate input, inject dependency, kiểm tra quyền và thực thi.

Tham khảo cấu trúc tài nguyên của [ResearchAgentToolEval](https://github.com/lit1numyt/K5-L3B-DAY03-4x6.1Sol-ResearchAgentToolEval/tree/7c7864a4a27a43c8093b0b76c9dca5b59c07f4a9/starter_v0). Registry production và chính sách bảo mật hiện tại của Sử Việt được giữ lại.

## Cấu trúc tài nguyên

```text
app/
  prompts/
    __init__.py
    loader.py
    shared/grounded_system.md
    hybrid/live_style.md
    central/
      planner.md
      agentic_loop.md
      independent_parameter.md
      execution_limit.md
      completion_parameter.md
      external_domain.md
      grounded_final.md
    offline/
      no_rag.md
      sft_grounding.md
      modes/{concise,standard,detailed}.md
      semantic/
        generate.md
        relevance.md
        verify.md
        contradiction.md
        deep_answer.md
        deep_verify.md
        json_suffix.md
  tools/
    declarations.py
    definitions/tools.yaml
    registry.py
    policy.py
    wikipedia_client.py
    web_providers.py
    builtin/
      __init__.py
      search_history/{__init__.py,tool.py,TOOL.md}
      search_uploaded_documents/{__init__.py,tool.py,TOOL.md}
      search_wikipedia/{__init__.py,tool.py,TOOL.md}
      fetch_wikipedia_page/{__init__.py,tool.py,TOOL.md}
      search_web/{__init__.py,tool.py,TOOL.md}
      fetch_web_page/{__init__.py,tool.py,TOOL.md}
    local_search.py, attachment_search.py, wikipedia.py,
    web_search.py, page_fetcher.py   # compatibility imports
```

## Sửa hoặc thêm prompt

`load_prompt(name)` chỉ nhận tên trong `PROMPT_NAMES`, resolve tương đối với `loader.py` và cache theo process. Không có đường dẫn từ HTTP request, network fetch, template engine hoặc LLM call. Các constant import prompt một lần; FastAPI lifespan validate/warm toàn bộ 21 file trước khi tạo service, đọc corpus hay load model. File thiếu, UTF-8 lỗi, rỗng, BOM, NUL hoặc CRLF làm startup fail rõ ràng. Git attributes giữ Markdown ở LF trên Windows.

Sửa file Markdown rồi restart process/redeploy. Không trim hoặc tự thêm newline: một số instruction có dấu cách đầu/cuối để nối chuỗi. `shared/grounded_system.md` dùng chung cho Hybrid và final answer Central. `LIVE_SYSTEM_PROMPT` vẫn là `SYSTEM_PROMPT + " " + LIVE_STYLE_INSTRUCTION`. Các import `SYSTEM_PROMPT`, `LIVE_SYSTEM_PROMPT`, `LIVE_STYLE_INSTRUCTION`, `PLANNER_INSTRUCTION`, `MODE_INSTRUCTIONS`, `SFT_GROUNDING_INSTRUCTION` và builder cũ tiếp tục dùng được.

Khi thêm prompt, tạo file UTF-8/LF, thêm logical name vào `PROMPT_NAMES`, load thành constant tại module sử dụng và thêm kiểm tra composition. Context formatting, truncation, history, question và runtime values vẫn ở Python. Không sửa prompt offline/SFT/evaluation đã freeze hoặc cập nhật fixture chỉ để test pass; thay đổi nghiên cứu có chủ đích cần version và review riêng.

## Tool declarations và runtime validation

`definitions/tools.yaml` version 1 là nguồn model-visible declaration cho sáu built-in. Description giải thích lúc nên dùng và không nên dùng; parameter description giải thích ý nghĩa input. `yaml.safe_load` đọc file một lần, reject cấu trúc sai, tên/tool trùng, mapping key trùng, alias, schema không hợp lệ và required field không tồn tại. Không import class hay cấp quyền từ YAML.

| Tool | Signature giữ nguyên | Required |
| --- | --- | --- |
| `search_history` | `query: string(minLength=1)`, `top_k: integer=8 [1,20]` | `query` |
| `search_uploaded_documents` | `query: string(minLength=1)`, `top_k: integer=8 [1,20]` | `query` |
| `search_wikipedia` | `query: string(minLength=1)`, `language: vi\|en=vi`, `top_k: integer=5 [1,10]` | `query` |
| `fetch_wikipedia_page` | `page_id_or_title: string(minLength=1)`, `language: vi\|en=vi`, `max_chars: integer=8000 [200,20000]` | `page_id_or_title` |
| `search_web` | `query: string(minLength=1)`, `top_k: integer=5 [1,10]` | `query` |
| `fetch_web_page` | `url: string(minLength=8)`, `max_chars: integer=8000 [200,20000]` | `url` |

Pydantic input models trong `tool.py` tiếp tục validate runtime, kể cả defaults và giới hạn cũ. `validate_builtin_contracts()` so tập tên YAML với `BUILTIN_TOOL_TYPES`, rồi so schema với Pydantic; chỉ bỏ annotation tài liệu như title/description, không bỏ types, required, enums, bounds hoặc defaults. Kiểm tra chạy tại startup và khi registry lấy descriptor built-in, ngoài hot path. Thêm hoặc đổi argument chỉ ở một phía làm fail sớm.

`BuiltinTool` cung cấp descriptor từ YAML. Registry vẫn generic: kiểm tra tên trùng, validate arguments, gọi sync/async, truyền `ToolExecutionContext`, enforce allowlist, xử lý lỗi và telemetry. YAML không tự register tool. `app/main.py` tiếp tục quyết định tool nào được bật và inject retriever/client. `ToolView` lọc schema theo request trước khi planner nhận; quyền vẫn kiểm tra lúc thực thi. Adapter OpenAI Responses và envelope Qwen/Hermes giữ nguyên. Planner completion flag vẫn thêm vào bản copy và strip trước thực thi.

Central v2 nối `agentic_loop.md` vào live planner bằng `AGENTIC_PLANNER_INSTRUCTION`; constant `PLANNER_INSTRUCTION` và frozen fixtures vẫn nguyên byte. Optional independence flag cũng chỉ thêm vào planner schema copy. Khi thêm tool, mặc định chạy thận trọng; chỉ cấp `ToolExecutionPolicy(read_only=True, parallel_safe=True)` sau khi xác minh shared clients/state an toàn. Tool dùng shared resource có `exclusive_resource` gate. YAML/TOOL.md không tự cấp quyền parallel hay quyền sử dụng MCP. Xem [Central v2](AGENTIC_RAG_V2.md).

`TOOL.md` là tài liệu người bảo trì, không load vào prompt và không cấp quyền. Frontmatter ghi provider/kind, environment, inputs/outputs, side effects và confirmation. HTTP helper Wikipedia và search providers chỉ có một implementation dùng chung. Module cũ re-export để import và test hiện tại vẫn hoạt động.

## Ví dụ thêm built-in

Ví dụ tool đọc mốc thời gian từ dependency nội bộ:

```python
# app/tools/builtin/lookup_timeline/tool.py
from pydantic import BaseModel, Field
from app.tools.declarations import BuiltinTool, builtin_description

class LookupTimelineInput(BaseModel):
    year: int = Field(ge=1, le=2100)

class LookupTimelineTool(BuiltinTool):
    name = "lookup_timeline"
    description = builtin_description(name)
    input_schema = LookupTimelineInput

    def __init__(self, timeline):
        self.timeline = timeline

    def run(self, arguments):
        return self.timeline.lookup(arguments.year)
```

Thêm declaration vào `tools:` (ví dụ này chưa được đăng ký trong app):

```yaml
- name: lookup_timeline
  description: >-
    Read known events for an exact year from the configured timeline.
    Use for year-specific evidence; do not use for open-ended research.
  parameters:
    type: object
    properties:
      year:
        type: integer
        minimum: 1
        maximum: 2100
        description: Exact calendar year to look up.
    required: [year]
```

Thêm `TOOL.md` có frontmatter như tool hiện có, export class trong `BUILTIN_TOOL_TYPES`, register instance có dependency trong startup khi cấu hình cho phép, rồi viết contract/execution/permission tests. Không cần sửa planner, retrieval hay model adapter. Nếu tool cần owner/conversation, dùng `run_with_context` và enforce scope như uploaded-document tool. Tool có side effect cần thiết kế quyền/confirmation tương ứng.

## MCP tiếp tục discovery động

Không đưa MCP tools vào YAML. `MCPManager` vẫn discover và validate schema từ server, enforce server/tool allowlists, collision protection, schema budget và failure policy. MCP adapter cung cấp descriptor trực tiếp cho cùng registry, không đọc declaration built-in. Tool output vẫn là untrusted evidence. YAML không mở rộng quyền MCP hoặc thay call IDs/reasoning replay của OpenAI.

## Kiểm tra và đóng gói

```powershell
python -m pytest -q tests/test_prompt_contracts.py tests/test_prompt_loader.py tests/test_tool_contracts.py tests/test_resource_packaging.py
python -m pytest -q tests/test_central_planning.py tests/test_mcp_integration.py tests/test_openai_responses.py
python -m pytest -q
```

Dùng environment test không chứa cấu hình deployment cá nhân. Nếu `.env` local đang tắt local inference để dùng API, đặt `$env:LOCAL_INFERENCE_ENABLED='true'` cho regression suite có fake Qwen; không tải weights hoặc gọi API trả phí. Suite remote-only vẫn tự kiểm tra local inference tắt.

Hai Docker runtime đã `COPY app /workspace/app`, bao gồm Markdown/YAML. Frontend-only image không cần tài nguyên backend. Modal source digest bao gồm `app/**/*.md` và `app/**/*.yaml`, nên thay prompt/schema được nhận diện khi build. `.dockerignore` loại `.env`, private MCP config và reports. Test packaging copy cây `app` sang runtime tạm, validate từ working directory khác và không cần corpus/model artifacts. Không cần secret hay biến môi trường mới.

Không thay đổi retrieval, số lượt agent, citation mapping, conversation persistence, SSE, auth, provider selection hoặc frontend. Automated tests dùng fake model/HTTP/MCP; chúng không chứng minh chất lượng model thật hoặc thay thế live smoke trên deployment tương ứng.
