"""Explicit lightweight LangGraph workflow for canonical Hybrid chat."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.common.domain_gate import _domain_gate, _scoped_response
from app.agents.common.graph import GRAPH_VERSION, CommonGraphState, GraphTopology, WorkflowGraph, extend_route, traced_node
from app.telemetry import current_request_telemetry


class HybridState(CommonGraphState, total=False):
    final_k: int
    history: list[dict[str, str]]
    owner_id: str | None
    started: float
    gate: dict[str, Any]
    normalized_history: list[dict[str, str]]
    retrieval_question: str
    history_used: list[dict[str, str]]
    retrieval: dict[str, Any]
    retrieval_elapsed_ms: float
    contexts: list[dict[str, Any]]
    analysis: dict[str, Any]
    tool_trace: list[str]


@dataclass(frozen=True)
class HybridGraphDependencies:
    retriever: Any
    retrieval_runtime: Any
    answerer: Any
    checkpointer: Any | None = None


TOPOLOGY = GraphTopology(
    name="HybridGraph",
    version=GRAPH_VERSION,
    nodes=("prepare", "hybrid_retrieval", "prepare_evidence", "answer_generation", "finalize"),
    edges=(("__start__", "prepare"), ("hybrid_retrieval", "prepare_evidence"),
           ("prepare_evidence", "answer_generation"), ("answer_generation", "finalize"),
           ("finalize", "__end__")),
    conditional_routes=(("prepare", (("continue", "hybrid_retrieval"), ("scoped", "finalize"))),),
)


def build_hybrid_graph(dependencies: HybridGraphDependencies) -> WorkflowGraph[HybridState]:
    deps = dependencies

    def prepare(state: HybridState) -> dict[str, Any]:
        started = float(state.get("started") or time.perf_counter())
        question = state["question"]
        gate = _domain_gate(deps.retriever, question)
        scoped = _scoped_response(question=question, gate=gate, mode="hybrid", started=started,
                                  answer_depth="standard")
        route = "scoped" if scoped is not None else "continue"
        return {"started": started, "gate": gate, "result": scoped, "next_route": route,
                "graph_route": extend_route(state, route)}

    def retrieve(state: HybridState) -> dict[str, Any]:
        question = state["question"]
        selected_final_k = max(1, int(state.get("final_k") or getattr(deps.retriever, "final_context_k", 6)))
        normalized_history = deps.retrieval_runtime.normalize_history(state.get("history"), current_question=question)
        retrieval_question, history_used = deps.retrieval_runtime.build_retrieval_question(question, normalized_history)
        retrieval_started = time.perf_counter()
        retrieval = deps.retriever.retrieve(retrieval_question, final_k=selected_final_k)
        elapsed_ms = (time.perf_counter() - retrieval_started) * 1000
        telemetry = current_request_telemetry()
        if telemetry is not None:
            telemetry.retrieval_ms += elapsed_ms
        return {"normalized_history": normalized_history, "retrieval_question": retrieval_question,
                "history_used": history_used, "retrieval": retrieval, "retrieval_elapsed_ms": elapsed_ms}

    def prepare_evidence(state: HybridState) -> dict[str, Any]:
        question = state["question"]
        retrieval = dict(state["retrieval"])
        contexts = list(retrieval.get("final_context", []))
        analysis = deps.retriever.analyze_question(question)
        gate = state["gate"]
        retrieval.update({
            "question": question, "analysis": analysis, "retrieval_question": state["retrieval_question"],
            "history_used_for_retrieval": state["history_used"], "global_final_context": contexts,
            "global_context_count": sum(item.get("source_kind") != "attachment" for item in contexts),
            "temporary_context_count": sum(item.get("source_kind") == "attachment" for item in contexts),
            "temporary_context_relevant": False,
            "context_title_diversity": deps.retriever.context_title_diversity(contexts),
            "domain_gate_result": retrieval.get("domain_gate_result") or gate.get("domain_gate_result"),
            "domain_gate_reason": retrieval.get("domain_gate_reason") or gate.get("domain_gate_reason"),
        })
        return {"retrieval": retrieval, "contexts": contexts, "analysis": analysis,
                "tool_trace": [*retrieval.get("tool_trace", []), "mode:hybrid", "hybrid:retriever"]}

    def answer(state: HybridState) -> dict[str, Any]:
        retrieval = state["retrieval"]
        result = deps.answerer.answer(
            question=state["question"], contexts=state["contexts"] if not retrieval.get("is_ood") else [],
            analysis=state["analysis"], tool_trace=state["tool_trace"], is_ood=bool(retrieval.get("is_ood")),
            ood_reason=str(retrieval.get("ood_reason") or ""), history=None, request_id=state.get("request_id"),
            answer_depth="standard", avoid_generic_source_prefix=True, inference_mode="hybrid",
        )
        return {"result": result}

    def finalize(state: HybridState) -> dict[str, Any]:
        result = dict(state.get("result") or {})
        if not result:
            raise RuntimeError("Hybrid graph completed without a result")
        if "retrieval" in state:
            elapsed_ms = float(state["retrieval_elapsed_ms"])
            result.update({"inference_mode": "hybrid", "agentic": False, "retrieval": state["retrieval"],
                           "retrieval_latency_sec": elapsed_ms / 1000,
                           "total_latency_sec": time.perf_counter() - state["started"]})
            provenance = result.setdefault("answer_provenance", {})
            provenance.update({
                "mode": "hybrid", "research_generation_calls": 0, "evidence_generation_calls": 0,
                "history_input_evidence_count": len(result.get("history_debug", {}).get("input_evidence_ids", [])),
                "history_input_claim_count": result.get("history_debug", {}).get("input_claim_count", 0),
                "history_input_source_kind_counts": result.get("history_debug", {}).get("input_source_kind_counts", {}),
                "total_llm_calls": int(provenance.get("history_generation_calls") or 0),
            })
            result["performance_debug"] = {
                "retrieval_latency_ms": elapsed_ms,
                "history_first_latency_ms": result.get("history_debug", {}).get("first_latency_ms"),
                "history_retry_latency_ms": result.get("history_debug", {}).get("retry_latency_ms"),
                "history_total_latency_ms": result.get("history_debug", {}).get("total_latency_ms"),
                "total_latency_ms": result["total_latency_sec"] * 1000,
                "research_generation_calls": 0, "evidence_generation_calls": 0,
                "history_generation_calls": provenance.get("history_generation_calls", 0),
                "total_llm_calls": provenance.get("total_llm_calls", 0),
            }
        result.setdefault("inference_mode", "hybrid")
        result.setdefault("answer_provenance", {}).setdefault("mode", "hybrid")
        return {"result": result, "status": str(result.get("status") or "ok")}

    builder = StateGraph(HybridState)
    for name, function in (("prepare", prepare), ("hybrid_retrieval", retrieve),
                           ("prepare_evidence", prepare_evidence), ("answer_generation", answer),
                           ("finalize", finalize)):
        builder.add_node(name, traced_node(graph_name=TOPOLOGY.name, graph_version=TOPOLOGY.version,
                                          node_name=name, function=function))
    builder.add_edge(START, "prepare")
    builder.add_conditional_edges("prepare", lambda state: state["next_route"],
                                  {"continue": "hybrid_retrieval", "scoped": "finalize"})
    builder.add_edge("hybrid_retrieval", "prepare_evidence")
    builder.add_edge("prepare_evidence", "answer_generation")
    builder.add_edge("answer_generation", "finalize")
    builder.add_edge("finalize", END)
    return WorkflowGraph(compiled=builder.compile(checkpointer=deps.checkpointer), topology=TOPOLOGY,
                         checkpointer_enabled=deps.checkpointer is not None)
