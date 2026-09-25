# Central tools

The Central Agent uses `ToolRegistry` to validate model-supplied arguments with Pydantic, run tools, and record bounded results or errors. `ToolExecutionContext` carries request, owner, and conversation IDs separately from the tool schema shown to the model. Tool implementations do not load Qwen or create retrieval indexes.

| Tool | Availability | Purpose |
|---|---|---|
| `search_history` | Always in full Central mode | Query the same historical retriever used by Hybrid RAG. |
| `search_uploaded_documents` | When `CENTRAL_ENABLE_DOCUMENTS=true` | Search documents attached to the current owner and conversation. |
| `search_wikipedia`, `fetch_wikipedia_page` | When `CENTRAL_ENABLE_WIKIPEDIA=true` | Find and read public encyclopedia pages. |
| `search_web`, `fetch_web_page` | When `CENTRAL_ENABLE_WEB=true` | Optional web search and bounded page extraction. |

The Central runtime guarantees a local `search_history` call if its tool rounds did not already make one. Web search is disabled by default. Uploaded-document search uses host-supplied conversation scope; model text cannot select another user's attachment store.

To add a tool, define a Pydantic input model and a class with `name`, `description`, `input_schema`, and `run` or `run_with_context`. Register it in `app/main.py` and test argument validation, expected output, and error handling. Never log secrets, full private page content, or hidden model reasoning.
