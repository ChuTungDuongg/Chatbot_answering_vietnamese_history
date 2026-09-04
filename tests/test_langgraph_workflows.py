from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from app.agents.central.graph import CentralGraphDependencies, build_central_graph
from app.agents.hybrid_graph import TOPOLOGY as HYBRID_TOPOLOGY
from app.agents.three_llm.graph import TOPOLOGY as THREE_LLM_TOPOLOGY


ROOT = Path(__file__).resolve().parents[1]


class FakeCentralController:
    def __init__(self, path: str):
        self.path = path

    async def _graph_prepare(self, state):
        domain = SimpleNamespace(model_calls=0, tool_calls=0, input_tokens=0, output_tokens=0)
        return {"central_state": domain}

    async def _graph_initial_grounding(self, state):
        route = "insufficient" if self.path == "insufficient" else "synthesis"
        return {"next_route": route, "graph_route": [route]}

    async def _graph_action(self, state):
        return {"next_route": "synthesis", "graph_route": [*state.get("graph_route", []), "synthesis"]}

    async def _graph_synthesis(self, state):
        return {"next_route": "validation", "graph_route": [*state.get("graph_route", []), "validation"]}

    async def _graph_validation(self, state):
        route = "quality_repair" if self.path == "repair" else "final"
        return {"next_route": route, "graph_route": [*state.get("graph_route", []), route]}

    async def _graph_citation_repair(self, state):
        return {}

    async def _graph_revalidate(self, state):
        return {"next_route": "final", "graph_route": [*state.get("graph_route", []), "final"]}

    async def _graph_quality_repair(self, state):
        state["central_state"].model_calls += 1
        return {"central_state": state["central_state"]}

    async def _graph_revalidate_after_repair(self, state):
        return {}

    async def _graph_insufficient(self, state):
        return {}

    async def _graph_final(self, state):
        return {"result": {"answer": self.path, "status": "ok", "inference_mode": "central"}}


def _central(path="normal", variant="base"):
    provider = object()
    return build_central_graph(CentralGraphDependencies(
        controller=FakeCentralController(path), model_provider=provider, model_variant=variant,
    ))


def test_graph_topologies_are_mode_isolated():
    assert not set(HYBRID_TOPOLOGY.nodes) & {"research_agent", "evidence_critic", "history_answerer", "central"}
    assert {"research_agent", "evidence_critic", "history_answerer"} <= set(THREE_LLM_TOPOLOGY.nodes)
    assert not any("central" in node for node in THREE_LLM_TOPOLOGY.nodes)
    central_nodes = set(_central().topology.nodes)
    assert {"initial_grounding", "action", "synthesis", "validation", "quality_repair", "final"} <= central_nodes
    assert not central_nodes & {"research_agent", "evidence_critic", "history_answerer"}


def test_central_base_and_adapter_share_the_exact_topology():
    base = _central(variant="base")
    adapted = _central(variant="adapted")
    assert base.topology == adapted.topology
    assert base.topology_fingerprint == adapted.topology_fingerprint
    assert base.model_variant == "base"
    assert adapted.model_variant == "adapted"
    assert set(base.get_graph().nodes) == set(adapted.get_graph().nodes)
    assert {(edge.source, edge.target, edge.conditional) for edge in base.get_graph().edges} == {
        (edge.source, edge.target, edge.conditional) for edge in adapted.get_graph().edges
    }


def test_central_fake_routes_emit_normalized_traces():
    async def execute(path):
        graph = _central(path)
        return await graph.ainvoke({"question": "q", "graph_trace": [], "graph_route": []})

    normal = asyncio.run(execute("normal"))["result"]
    repair = asyncio.run(execute("repair"))["result"]
    insufficient = asyncio.run(execute("insufficient"))["result"]
    assert normal["graph_nodes_executed"] == ["prepare", "initial_grounding", "synthesis", "validation", "final"]
    assert repair["graph_nodes_executed"] == [
        "prepare", "initial_grounding", "synthesis", "validation", "quality_repair",
        "revalidate_after_repair", "final",
    ]
    assert insufficient["graph_nodes_executed"] == [
        "prepare", "initial_grounding", "insufficient_evidence", "final",
    ]
    assert repair["node_model_calls"]["quality_repair"] == 1
    assert all("duration_ms" in row and "error" in row and "route_taken" in row
               for row in repair["graph_trace"])


def test_graph_construction_does_not_touch_lazy_model_providers():
    class ExplodingProvider:
        def __getattr__(self, name):
            raise AssertionError(f"model provider loaded during graph construction: {name}")

    graph = build_central_graph(CentralGraphDependencies(
        controller=FakeCentralController("normal"), model_provider=ExplodingProvider(), model_variant="base",
    ))
    assert graph.checkpointer_enabled is False


def test_langgraph_scope_and_central_isolation_are_static_invariants():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").casefold()
    direct_names = {
        line.split("=", 1)[0].split(">", 1)[0].strip()
        for line in requirements.splitlines() if line.strip() and not line.startswith("#")
    }
    assert not direct_names & {"langchain", "langsmith", "redis", "psycopg", "sqlalchemy"}
    production = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "app").rglob("*.py"))
    for forbidden in ("create_react_agent", "langchain.agents", "ToolNode", "MemorySaver",
                      "AsyncPostgresSaver", "PostgresSaver", "InMemoryStore"):
        assert forbidden not in production
    central = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "app/agents/central").rglob("*.py"))
    for forbidden in ("ResearchAgent", "EvidenceCriticAgent", "HistoryAnswererAgent",
                      "AgentOrchestrator", "build_three_llm_graph"):
        assert forbidden not in central
