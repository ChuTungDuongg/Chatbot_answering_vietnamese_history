"""Conservative execution of independent, read-only Central Agent tool calls.

Only distinct built-in searches can run concurrently. Fetch/read-after-search
chains, MCP tools, and unknown tools retain sequential execution so the planner
can observe their dependencies and request-scoped telemetry remains ordered.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar

Result = TypeVar("Result")

PARALLEL_SEARCH_TOOLS = frozenset({"search_history", "search_wikipedia", "search_web"})


def can_run_parallel(calls: Sequence[Any]) -> bool:
    names = [call.name for call in calls]
    return (len(names) > 1 and len(names) == len(set(names))
            and all(name in PARALLEL_SEARCH_TOOLS for name in names))


async def execute_tool_batch(
    calls: Sequence[Any], execute: Callable[[Any], Awaitable[Result]], *,
    cancel: threading.Event,
) -> tuple[list[tuple[Result, float]], str]:
    """Run one planner batch; return results/latencies in original call order."""
    async def measured(call: Any) -> tuple[Result, float]:
        if cancel.is_set():
            raise RuntimeError("Request cancelled")
        started = time.perf_counter_ns()
        result = await execute(call)
        if cancel.is_set():
            raise RuntimeError("Request cancelled")
        return result, (time.perf_counter_ns() - started) / 1e6

    if not can_run_parallel(calls):
        return [await measured(call) for call in calls], "sequential"

    tasks = [asyncio.create_task(measured(call)) for call in calls]
    try:
        results = await asyncio.gather(*tasks)
        if cancel.is_set():
            raise RuntimeError("Request cancelled")
        return results, "parallel"
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
