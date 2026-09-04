"""Explicit LangGraph topology for the isolated Central workflow."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.common.graph import GRAPH_VERSION, CommonGraphState, GraphTopology, WorkflowGraph, traced_node


class CentralGraphState(CommonGraphState, total=False):
    history: list[dict[str, str]]
    owner_id: str | None
    started: float
    progress: dict[str, Any]
    model_ready_task: Any
    question_analysis: Any
    central_state: Any
    tool_context: Any
    schemas: list[dict[str, Any]]
    packet: list[Any]
    synthesis_generation: Any
    quality_issues: list[str]
    citations: Any
    action_round: int


@dataclass(frozen=True)
class CentralGraphDependencies:
    controller: Any
    model_provider: Any
    model_variant: str = "base"
    checkpointer: Any | None = None


TOPOLOGY = GraphTopology(
    name="CentralGraph",
    version=GRAPH_VERSION,
    nodes=("prepare", "initial_grounding", "action", "synthesis", "validation",
           "citation_repair", "revalidate", "quality_repair", "revalidate_after_repair",
           "insufficient_evidence", "final"),
    edges=(("__start__", "prepare"), ("prepare", "initial_grounding"),
           ("citation_repair", "revalidate"), ("quality_repair", "revalidate_after_repair"),
           ("revalidate_after_repair", "final"), ("insufficient_evidence", "final"),
           ("final", "__end__")),
    conditional_routes=(
        ("initial_grounding", (("action", "action"), ("synthesis", "synthesis"),
                               ("insufficient", "insufficient_evidence"))),
        ("action", (("action", "action"), ("synthesis", "synthesis"),
                    ("insufficient", "insufficient_evidence"))),
        ("synthesis", (("validation", "validation"), ("insufficient", "insufficient_evidence"))),
        ("validation", (("citation_repair", "citation_repair"),
                        ("quality_repair", "quality_repair"), ("final", "final"))),
        ("revalidate", (("quality_repair", "quality_repair"), ("final", "final"))),
    ),
    host_config_version="central-host-policy-v2",
)


def build_central_graph(
    dependencies: CentralGraphDependencies | None = None,
    *,
    controller: Any | None = None,
    model_provider: Any | None = None,
    model_variant: str = "base",
    checkpointer: Any | None = None,
) -> WorkflowGraph[CentralGraphState]:
    """Build one topology for both base and adapter variants.

    ``model_provider`` is metadata/injection, not a topology switch. Production
    nodes call the controller's existing deterministic/model/tool functions.
    """

    deps = dependencies or CentralGraphDependencies(
        controller=controller,
        model_provider=model_provider,
        model_variant=model_variant,
        checkpointer=checkpointer,
    )
    if deps.controller is None:
        raise ValueError("Central graph requires an injected workflow controller")
    if deps.model_provider is None:
        raise ValueError("Central graph requires an injected model provider")

    builder = StateGraph(CentralGraphState)
    methods = {
        "prepare": deps.controller._graph_prepare,
        "initial_grounding": deps.controller._graph_initial_grounding,
        "action": deps.controller._graph_action,
        "synthesis": deps.controller._graph_synthesis,
        "validation": deps.controller._graph_validation,
        "citation_repair": deps.controller._graph_citation_repair,
        "revalidate": deps.controller._graph_revalidate,
        "quality_repair": deps.controller._graph_quality_repair,
        "revalidate_after_repair": deps.controller._graph_revalidate_after_repair,
        "insufficient_evidence": deps.controller._graph_insufficient,
        "final": deps.controller._graph_final,
    }
    for name, function in methods.items():
        builder.add_node(name, traced_node(graph_name=TOPOLOGY.name, graph_version=TOPOLOGY.version,
                                          node_name=name, function=function))
    route = lambda state: state["next_route"]
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "initial_grounding")
    builder.add_conditional_edges("initial_grounding", route, {
        "action": "action", "synthesis": "synthesis", "insufficient": "insufficient_evidence",
    })
    builder.add_conditional_edges("action", route, {
        "action": "action", "synthesis": "synthesis", "insufficient": "insufficient_evidence",
    })
    builder.add_conditional_edges("synthesis", route, {
        "validation": "validation", "insufficient": "insufficient_evidence",
    })
    builder.add_conditional_edges("validation", route, {
        "citation_repair": "citation_repair", "quality_repair": "quality_repair", "final": "final",
    })
    builder.add_edge("citation_repair", "revalidate")
    builder.add_conditional_edges("revalidate", route, {"quality_repair": "quality_repair", "final": "final"})
    builder.add_edge("quality_repair", "revalidate_after_repair")
    builder.add_edge("revalidate_after_repair", "final")
    builder.add_edge("insufficient_evidence", "final")
    builder.add_edge("final", END)
    return WorkflowGraph(
        compiled=builder.compile(checkpointer=deps.checkpointer),
        topology=TOPOLOGY,
        model_variant=deps.model_variant,
        checkpointer_enabled=deps.checkpointer is not None,
    )


__all__ = ["CentralGraphDependencies", "CentralGraphState", "build_central_graph"]
