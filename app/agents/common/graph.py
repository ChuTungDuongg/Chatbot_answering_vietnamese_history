"""Small, framework-independent contracts around LangGraph workflow execution."""
from __future__ import annotations

import hashlib
import inspect
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Generic, TypedDict, TypeVar

from app.telemetry import current_request_telemetry, log_event


GRAPH_VERSION = "1"


class CommonGraphState(TypedDict, total=False):
    """Fields shared by one request execution, never durable conversation state."""

    request_id: str | None
    conversation_id: str | None
    mode: str
    question: str
    attachments: list[dict[str, Any]]
    attachment_ids: tuple[str, ...]
    status: str
    errors: list[dict[str, Any]]
    final_answer: str
    sources: list[dict[str, Any]]
    result: dict[str, Any]
    graph_trace: list[dict[str, Any]]
    graph_route: list[str]
    next_route: str


@dataclass(frozen=True)
class GraphTopology:
    name: str
    version: str
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    conditional_routes: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()
    host_config_version: str = "1"

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "graph_name": self.name,
            "graph_version": self.version,
            "nodes": list(self.nodes),
            "edges": [list(edge) for edge in self.edges],
            "conditional_routes": [
                {"source": source, "routes": [list(route) for route in routes]}
                for source, routes in self.conditional_routes
            ],
            "host_config_version": self.host_config_version,
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.semantic_payload(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _telemetry_counters(state: dict[str, Any]) -> dict[str, int]:
    domain = state.get("central_state")
    if domain is not None:
        return {
            "model_calls": int(getattr(domain, "model_calls", 0)),
            "tool_calls": int(getattr(domain, "tool_calls", 0)),
            "input_tokens": int(getattr(domain, "input_tokens", 0)),
            "output_tokens": int(getattr(domain, "output_tokens", 0)),
        }
    telemetry = current_request_telemetry()
    if telemetry is None:
        return {name: 0 for name in ("model_calls", "tool_calls", "input_tokens", "output_tokens")}
    return {
        "model_calls": int(telemetry.total_llm_calls),
        "tool_calls": int(telemetry.tool_calls),
        "input_tokens": int(telemetry.total_input_tokens),
        "output_tokens": int(telemetry.total_output_tokens),
    }


def traced_node(
    *,
    graph_name: str,
    graph_version: str,
    node_name: str,
    function: Callable[[dict[str, Any]], dict[str, Any] | Awaitable[dict[str, Any]]],
) -> Callable[..., Any]:
    """Record stable node telemetry without persisting LangGraph internals."""

    def begin(state: dict[str, Any]) -> tuple[str, float, dict[str, int]]:
        return _utc_now(), time.perf_counter(), _telemetry_counters(state)

    def finish(
        state: dict[str, Any],
        update: dict[str, Any],
        started_at: str,
        started: float,
        before: dict[str, int],
    ) -> dict[str, Any]:
        ended_at = _utc_now()
        merged = dict(state)
        merged.update(update)
        after = _telemetry_counters(merged)
        elapsed_ms = (time.perf_counter() - started) * 1000
        trace = list(state.get("graph_trace") or [])
        model_calls = max(0, after["model_calls"] - before["model_calls"])
        tool_calls = max(0, after["tool_calls"] - before["tool_calls"])
        input_tokens = max(0, after["input_tokens"] - before["input_tokens"])
        output_tokens = max(0, after["output_tokens"] - before["output_tokens"])
        trace.append({
            "graph_name": graph_name,
            "graph_version": graph_version,
            "node": node_name,
            "node_name": node_name,
            "node_start": started_at,
            "node_end": ended_at,
            "duration_ms": elapsed_ms,
            "node_latency_ms": elapsed_ms,
            "model_calls": model_calls,
            "model_calls_delta": model_calls,
            "tool_calls": tool_calls,
            "tool_calls_delta": tool_calls,
            "input_tokens": input_tokens,
            "input_tokens_delta": input_tokens,
            "output_tokens": output_tokens,
            "output_tokens_delta": output_tokens,
            "error": None,
            "route_taken": update.get("next_route"),
        })
        update["graph_trace"] = trace
        log_event("GRAPH_NODE_COMPLETE", **trace[-1])
        return update

    def failed(started: float, exc: Exception) -> None:
        log_event(
            "GRAPH_NODE_ERROR",
            graph_name=graph_name,
            graph_version=graph_version,
            node_name=node_name,
            node_latency_ms=(time.perf_counter() - started) * 1000,
            error=f"{type(exc).__name__}: {exc}",
        )

    async def execute_async(state: dict[str, Any]) -> dict[str, Any]:
        started_at, started, before = begin(state)
        try:
            update = function(state)
            if inspect.isawaitable(update):
                update = await update
            update = dict(update or {})
        except Exception as exc:
            failed(started, exc)
            raise
        return finish(state, update, started_at, started, before)

    def execute_sync(state: dict[str, Any]) -> dict[str, Any]:
        started_at, started, before = begin(state)
        try:
            update = function(state)
            if inspect.isawaitable(update):
                raise TypeError(f"Node {node_name!r} returned an awaitable from a synchronous function")
            update = dict(update or {})
        except Exception as exc:
            failed(started, exc)
            raise
        return finish(state, update, started_at, started, before)

    execute = execute_async if inspect.iscoroutinefunction(function) else execute_sync
    execute.__name__ = node_name
    return execute


StateT = TypeVar("StateT", bound=dict[str, Any])


@dataclass
class WorkflowGraph(Generic[StateT]):
    """Compiled graph plus stable semantic metadata and normalized result trace."""

    compiled: Any
    topology: GraphTopology
    model_variant: str | None = None
    checkpointer_enabled: bool = False

    def get_graph(self, *args: Any, **kwargs: Any) -> Any:
        return self.compiled.get_graph(*args, **kwargs)

    @property
    def topology_fingerprint(self) -> str:
        return self.topology.fingerprint

    def _attach_metadata(self, state: StateT) -> StateT:
        result = state.get("result")
        if not isinstance(result, dict):
            return state
        trace = list(state.get("graph_trace") or [])
        route = list(state.get("graph_route") or [])
        metadata = {
            "graph_name": self.topology.name,
            "graph_version": self.topology.version,
            "graph_topology_fingerprint": self.topology.fingerprint,
            "graph_nodes_executed": [item["node"] for item in trace],
            "graph_route": route,
            "node_timings": {item["node"]: item["duration_ms"] for item in trace},
            "node_model_calls": {item["node"]: item["model_calls"] for item in trace},
            "node_tool_calls": {item["node"]: item["tool_calls"] for item in trace},
            "graph_trace": trace,
            "model_variant": self.model_variant,
        }
        result.update(metadata)
        compact_metadata = {key: value for key, value in metadata.items() if key != "graph_trace"}
        performance = dict(result.get("performance_debug") or {})
        performance.update(compact_metadata)
        result["performance_debug"] = performance
        if result.get("inference_mode") == "central":
            central = dict(result.get("central_debug") or {})
            central.update(compact_metadata)
            result["central_debug"] = central
        state["result"] = result
        return state

    def invoke(self, state: StateT, *, config: dict[str, Any] | None = None) -> StateT:
        output = self.compiled.invoke(state, config=config)
        return self._attach_metadata(output)

    async def ainvoke(self, state: StateT, *, config: dict[str, Any] | None = None) -> StateT:
        output = await self.compiled.ainvoke(state, config=config)
        return self._attach_metadata(output)


def extend_route(state: dict[str, Any], route: str) -> list[str]:
    return [*(state.get("graph_route") or []), route]
