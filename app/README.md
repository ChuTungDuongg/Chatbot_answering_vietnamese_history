# ⚡ FastAPI Runtime

[🏠 Project README](../README.md) · [🧠 Agents](agents/README.md) · [🧰 Tools](tools/README.md) · [📦 Artifacts](../artifacts/README.md)

`app/` giữ API tương thích frontend hiện tại và nối Hybrid RAG vào orchestrator ba vai trò LLM.

## 🗂️ Module map

```text
app/
├── main.py                 # lifespan, service/model/tool/orchestrator cache
├── config.py               # Pydantic environment settings + artifact paths
├── agents/                 # Research, Evidence, History, shared Qwen3 runtime
├── tools/                  # typed registry và 5 tools
├── rag/
│   ├── retrieval.py        # E5 + FAISS + BM25S + RRF + BGE reranker
│   ├── prompting.py        # dynamic grounded prompts + token budget
│   ├── generation.py       # History generation, repair, global/temp merge
│   └── guards.py           # source/year/format/completeness checks
├── api/
│   ├── routes.py           # retrieve, chat, validated SSE
│   └── conversations.py    # conversation CRUD + attachments
├── chat/
│   ├── store.py            # SQLite persistence
│   └── attachments.py      # PDF/OCR/chunk/embed temporary corpus
└── services/rag_service.py # artifact validation và cached model/index loading
```

## 🔁 Request flow

```text
/api/v1/chat
  → load recent conversation history
  → AgentOrchestrator
  → ResearchAgent tool loop
  → session-scoped evidence
  → EvidenceCriticAgent
  → optional one research retry
  → HistoryAnswererAgent
  → generation guards / one repair
  → persist answer + cited sources
```

`/api/v1/chat/stream` là validated SSE: backend hoàn tất generation/guards trước, sau đó stream answer đã được chấp nhận. Đây không phải raw token stream.

## 🎚️ Runtime modes

| Mode | Load |
|---|---|
| `api-only` | FastAPI + SQLite; không corpus/model. |
| `retrieval-only` | Corpus, E5, FAISS, BM25S, reranker. |
| `full` | Retrieval + History model + agents/orchestrator. |

Trong `full`, `AGENT_CONTROLLER=model` nạp shared Qwen3 base 4-bit và hai PEFT adapters. `deterministic` giữ API chạy khi chưa có adapter, nhưng không phải 3-LLM production mode.

## 🔐 Environment

```dotenv
APP_MODE=full
DEVICE=cuda
DTYPE=bfloat16
ARTIFACT_ROOT=./artifacts/vn_history_deployment
HISTORY_MODEL_PATH=./artifacts/vn_history_deployment/history_answerer/model
AGENT_CONTROLLER=model
RESEARCH_AGENT_MODEL=Qwen/Qwen3-4B-Instruct-2507
RESEARCH_AGENT_ADAPTER_PATH=./artifacts/vn_history_deployment/research_agent/adapter
EVIDENCE_AGENT_MODEL=Qwen/Qwen3-4B-Instruct-2507
EVIDENCE_AGENT_ADAPTER_PATH=./artifacts/vn_history_deployment/evidence_agent/adapter
MAX_AGENT_STEPS=6
MAX_WEB_SEARCHES=3
MAX_PAGE_FETCHES=5
WEB_SEARCH_PROVIDER=local-only
CHAT_DATABASE_PATH=./data/chat.sqlite3
CORS_ORIGINS=http://localhost:5173
```

Không log API key, raw page lớn hoặc model reasoning. Runtime logs tool name, request/conversation ID, steps, latency và evidence count.

## 🚀 Start

```bash
python -m pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Kiểm tra:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

`ready=true` mới có nghĩa artifact/model đã load xong.

## 🧱 Invariants

- History Answerer chỉ nhận text contexts đã chọn.
- Web evidence không tự ghi vào permanent corpus.
- Evidence store phân vùng theo session và cleanup sau request.
- Evidence model không được chọn ID ngoài candidate set.
- Tool loop và retry đều có giới hạn hữu hạn.
- Model/index được tạo trong lifespan, không load lại theo request.
- Conversation history là context, không được coi là historical evidence.
- Frontend endpoints và SSE event contract được giữ tương thích.
