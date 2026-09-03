# 🧠 Agent Runtime

[⬅️ Backend](../README.md) · [🧰 Tool contracts](../tools/README.md)

## Package ownership

```text
app/agents/
  common/            shared runtime/lazy loading, registry, cache, tool codec,
                     EvidenceChunk, comparison helpers and domain gate
  research/          ResearchAgent, research config/policy/result
  evidence/          EvidenceCriticAgent, evidence schemas/prompts/validation
  history_answerer/  HistoryAnswererAgent and answer contract
  central/           independent CentralAgent/model/config, semantic parsing,
                     evidence selection, synthesis, citations and repairs
  three_llm/         AgentOrchestrator for the three legacy roles
  hybrid.py          existing HybridRAGOrchestrator
  config.py, schemas.py, prompts.py, orchestrator.py  compatibility re-exports
```

Public role imports are `from app.agents.research import ResearchAgent`,
`from app.agents.evidence import EvidenceCriticAgent`,
`from app.agents.history_answerer import HistoryAnswererAgent`, and
`from app.agents.central import CentralAgent, CentralAgentConfig, CentralModelRuntime`.
`from app.agents.three_llm import AgentOrchestrator` owns legacy orchestration.
Public package exports resolve lazily; imports do not construct models.

`common` contains only shared runtime/data helpers and does not import concrete
agents. Central imports its own modules, shared helpers and tool/RAG infrastructure;
it does not import/delegate to legacy agents or `three_llm`. Production imports
neither `training` nor `evaluation`. The four flat compatibility files have no
classes, functions or business logic; callers are migrated to canonical packages.

Within Central, `semantics.py` owns value types and lexical normalization;
`question.py` owns parsing/query planning; `depth.py` owns causal breadth policy;
`evidence.py` owns evidence packet planning. `citation_support.py` classifies
supported paragraphs below `citations.py`; `citation_recovery.py` performs bounded
recovery above validation. These boundaries remove deferred import cycles without
changing the existing generalization algorithms. Package dependency and duplicate
implementation gates live in `tests/test_agent_package_architecture.py`.

## 🎭 Ba vai trò

| Thành phần | Model/adapter | Trách nhiệm |
|---|---|---|
| `ResearchAgent` | Qwen3 + Research adapter | PLAN/ACTION/OBSERVATION/FINISH, chọn typed tool. |
| `EvidenceCriticAgent` | Qwen3 + Evidence adapter | Lọc, dedup, conflict check, compress, validate IDs. |
| `HistoryAnswererAgent` | Qwen3 + History adapter | Viết grounded Vietnamese answer và citations. |

`SharedAgentModelRuntime` nạp Qwen3 base một lần ở NF4 4-bit, load ba adapter tên `research`, `evidence`, `history`, rồi chuyển đúng adapter dưới lock trước generation. Metadata base mismatch hoặc role chưa load bị từ chối. `VLLMOpenAIBackend` giữ cùng interface và ba model name nhưng không tự quản lý server.

`CentralModelRuntime` là runtime riêng cho `Qwen/Qwen3-8B`. Mặc định Central V2 dùng base model, không gắn PEFT; chỉ `CENTRAL_AGENT_ADAPTER_PATH` khác rỗng mới validate và gắn adapter tương lai. `CentralAgent` dùng state machine `PREPARE → INITIAL_GROUNDING → ACTION/TOOL_EXECUTION → SYNTHESIS → QUALITY_REPAIR → FINAL`. Mọi câu hỏi lịch sử được host gọi `search_history` trước khi có thể tổng hợp câu trả lời. Tools được truyền bằng Qwen chat template và lời gọi Hermes/Qwen có cấu trúc, không dùng ReAct `Action:` hoặc parser rải rác. Nó không import/delegate sang `AgentOrchestrator`, `ResearchAgent`, `EvidenceCriticAgent` hoặc `HistoryAnswererAgent`. Hai runtime dùng `LazyRuntime` để không cùng khởi tạo khi chưa cần.

Central base model dùng Hugging Face cache riêng ở `/hf-cache/hub` trên Modal. Runtime log `central_cache_hit`, resolved snapshot và thời gian resolve/load/adapter-load; `CENTRAL_AGENT_LOCAL_FILES_ONLY=true` chỉ nên bật sau khi `scripts/modal_seed_hf_cache.py --validate-only` đã pass.
Lazy model initialization có timeout riêng (`CENTRAL_MODEL_LOAD_TIMEOUT_SECONDS`) và không ăn vào agent reasoning budget (`CENTRAL_AGENT_TIMEOUT_SECONDS`). Quality repair cũng có quota riêng một generation, nên normal budget ba generation không còn làm mất lượt sửa khi final phân tích vẫn quá nông.

## 🔁 Orchestration

```text
Research run (max 6 steps)
  → prefetch evidence từ PDF/ảnh của đúng conversation nếu có
  → Evidence critique
  → nếu insufficient và model controller đang bật: thêm tối đa 1 research round
  → Evidence critique lại
  → History answer + guards
  → cleanup SessionEvidenceStore
```

Research policy chỉ trả JSON action hoặc finish. Evidence policy chỉ trả structured JSON. Parser có tối đa một repair cho JSON không hợp lệ; Pydantic và candidate-ID validation chạy trước khi evidence tới History model. Attachment được prefetch bằng request scope nội bộ và cũng xuất hiện dưới dạng tool `search_uploaded_documents`; model không được nhìn thấy hoặc tự chọn owner/conversation ID.

## 🧪 Unit testing

Tests dùng fake generator/tool và deterministic path. Không khởi tạo `SharedAgentModelRuntime`, do đó không tải Qwen3.
