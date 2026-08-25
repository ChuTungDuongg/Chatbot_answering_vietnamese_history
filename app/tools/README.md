# 🧰 Agent Tools

[⬅️ Backend](../README.md) · [🧠 Agents](../agents/README.md)

`ToolRegistry` validate arguments bằng Pydantic trước khi gọi tool, hỗ trợ sync/async implementation và trả `ToolCallRecord` không chứa raw secret.

| Tool name | Input chính | Output |
|---|---|---|
| `search_history` | query, top_k | Chunks từ HybridRetriever hiện có. |
| `search_web` | query, top_k | Search results; rỗng trong local-only mode. |
| `fetch_web_page` | URL, max_chars | Clean text, title, final URL, content type. |
| `retrieve_evidence` | query, top_k, session_id | Evidence đã thu thập trong session. |
| `inspect_evidence` | IDs, session_id | Full evidence rows theo ID. |

## 🌐 Web provider

```dotenv
WEB_SEARCH_PROVIDER=local-only
WEB_SEARCH_API_KEY=
```

Hoặc `WEB_SEARCH_PROVIDER=tavily` và cung cấp key qua environment/Modal Secret. Fetcher timeout 10 giây, đọc tối đa 1 MB, chỉ nhận HTML/plain text/XHTML, extract title, clean whitespace và không crawl recursive.

## 🧺 Session evidence

`SessionEvidenceStore` hỗ trợ `add_documents`, `search`, `deduplicate`, `get/all` và `remove_session`. Dữ liệu phân vùng bằng session ID; orchestrator cleanup khi request kết thúc. Web chunks dùng ID hash ổn định trong session và không được đưa vào corpus/index vĩnh viễn.

## ➕ Thêm tool

1. Tạo Pydantic input schema.
2. Khai báo `name`, `description`, `input_schema` và `run`.
3. Register một lần trong lifespan ở `app/main.py`.
4. Thêm unit test registry/schema/error path.

Tool không được log API key, response page đầy đủ hoặc hidden model reasoning.
