# LangGraph workflow architecture

LangGraph 1.2.11 is used only for request-scoped workflow orchestration. The
graphs wrap the existing retrieval, model, tool, validation, and response
functions. They do not replace the SQLite conversation/message/attachment
store, training pipelines, or evaluation logs. Production graphs compile
without a checkpointer; an optional injected checkpointer exists only for
isolated tests/debugging.

```text
ChatModeRouter
  ├─ hybrid   → HybridGraph
  ├─ three_llm → ThreeLLMGraph
  └─ central  → CentralGraph
```

## Graphs

```text
HybridGraph
START → prepare → hybrid_retrieval → prepare_evidence
      → answer_generation → finalize → END
```

The `prepare` node retains the existing deterministic domain gate. Hybrid has
one History Answerer call when evidence is available and no agent loop.

```text
ThreeLLMGraph
START → prepare → research_agent → evidence_critic
      → [optional bounded research_retry → evidence_retry]
      → history_answerer → finalize → END
```

The same role models, prompts, evidence contracts, one retry policy, and
session-evidence cleanup remain in force.

```text
CentralGraph
START → prepare → initial_grounding
      → [bounded action ↔ structured tool execution]
      → synthesis → validation
      → [citation_repair → revalidate]
      → [quality_repair → revalidate_after_repair]
      → final → END
             ↘ insufficient_evidence → final
```

Central nodes call the existing capability-based question analysis, target and
facet retrieval, native Qwen/Hermes structured tool execution, evidence
sufficiency, citation checks, and repair functions. Central imports no legacy
role agent and has no route to `ThreeLLMGraph`.

## State and dependencies

`CommonGraphState` contains only one-execution fields such as request and
conversation identifiers, mode, question, result, route, errors, and normalized
trace. Each graph extends it with mode-specific state. Central keeps its domain
`CentralAgentState` as a request-local nested value so existing deterministic
policy remains authoritative.

Graph construction receives dependencies. In particular, `CentralGraphDependencies`
injects the model provider and records `model_variant` separately from topology.
Base and adapter V2 therefore use the same nodes, edges, host policy, prompts,
retrieval, tools, validation, and repair code. The topology fingerprint hashes
only stable graph/host-policy metadata and intentionally excludes adapter state.

## Streaming, telemetry, persistence, and evaluation

The API still calls each runtime through its existing `chat()` facade. Results
flow through the unchanged SSE adapter and preserve the existing status/event
schema. LangGraph internal event objects are not sent to the frontend.

Every result includes a framework-independent `graph_trace` with graph and node
names, timestamps, latency, call/token deltas, error, and route. Evaluation raw
records retain that trace plus topology metadata, while `evaluation/logs/`
remains the durable rescoring artifact. No production module imports evaluation
or training code, and training has no LangGraph dependency.
