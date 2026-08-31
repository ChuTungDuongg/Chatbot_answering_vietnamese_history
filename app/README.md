# ⚡ FastAPI Runtime

[🏠 Project README](../README.md) · [🧠 Agents](agents/README.md) · [🧰 Tools](tools/README.md) · [📦 Artifacts](../artifacts/README.md)

`app/` giữ API tương thích frontend hiện tại và nối Hybrid RAG vào orchestrator ba vai trò LLM.

## 🗂️ Module map

```text
app/
├── main.py                 # lifespan, service/model/tool/orchestrator cache
├── config.py               # Pydantic environment settings + artifact paths
├── chat_modes.py           # Fast/Hybrid/Agent enum + legacy API aliases
├── agents/                 # Research, Evidence, History, shared Qwen3 runtime
├── tools/                  # typed registry, request context và 6 tools
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
└── services/
    ├── chat_mode_router.py # one app-level dispatch point
    ├── fast_service.py     # bounded low-latency direct-RAG facade
    └── rag_service.py      # artifact validation và cached model/index loading
```

## 🔁 Request flow

```text
/api/v1/chat
  → load recent conversation history
  → AgentOrchestrator
  → ResearchAgent tool loop
  → search uploaded PDF/image chunks trong đúng conversation
  → session-scoped evidence
  → EvidenceCriticAgent
  → optional one research retry
  → HistoryAnswererAgent
  → generation guards / one repair
  → persist answer + cited sources
```

`/api/v1/chat/stream` là validated SSE: backend hoàn tất generation/guards trước, sau đó stream answer đã được chấp nhận. Đây không phải raw token stream.

## 💬 User-facing chat modes

| Mode | App path |
|---|---|
| `fast` | `FastChatService` → direct Hybrid retrieval/History answer path với tối đa 3 contexts. |
| `hybrid` | Existing three-role pipeline: Hybrid retrieval → Research → Evidence → History. |
| `agent` | `CentralAgent` chọn Fast cho câu hỏi đơn giản hoặc bounded three-role/tool path khi cần. |

API cũ gửi `hybrid_rag` hoặc `agentic_rag` vẫn giữ đúng execution behavior cũ và được map lần lượt về `fast` và `hybrid`. `CentralAgent` là orchestration facade ở app layer, không phải model thứ tư.

## 🎚️ Runtime modes

| Mode | Load |
|---|---|
| `api-only` | FastAPI + SQLite; không corpus/model. |
| `retrieval-only` | Corpus, E5, FAISS, BM25S, reranker. |
| `full` | Retrieval + shared Qwen3 backend + three role adapters/orchestrator. |

Trong `full`, `LLM_BACKEND=transformers` nạp một shared Qwen3 base 4-bit và ba PEFT adapters. `LLM_BACKEND=vllm` gọi endpoint OpenAI-compatible đã phục vụ đúng ba tên role. `deterministic` giữ API chạy khi chưa có adapter, nhưng không đại diện chất lượng 3-LLM.

## 🔐 Environment

```dotenv
APP_MODE=full
DEVICE=cuda
DTYPE=bfloat16
ARTIFACT_ROOT=./artifacts/vn_history_deployment
LLM_BACKEND=transformers
SHARED_BASE_MODEL_ID=Qwen/Qwen3-4B-Instruct-2507
AGENT_CONTROLLER=model
RESEARCH_AGENT_MODEL=Qwen/Qwen3-4B-Instruct-2507
RESEARCH_AGENT_ADAPTER_PATH=./artifacts/vn_history_deployment/adapters/research
EVIDENCE_AGENT_MODEL=Qwen/Qwen3-4B-Instruct-2507
EVIDENCE_AGENT_ADAPTER_PATH=./artifacts/vn_history_deployment/adapters/evidence
HISTORY_AGENT_ADAPTER_PATH=./artifacts/vn_history_deployment/adapters/history
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
