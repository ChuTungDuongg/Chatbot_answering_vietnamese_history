"""Request-local limits and cancellable waits for the planning/tool phase."""

import asyncio
from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True)
class AgentLimits:
    calls_per_round: int = 4
    total_calls: int = 12
    identical_calls: int = 1
    parallel_calls: int = 3
    deadline_seconds: float = 180
    tool_timeout_seconds: float = 30
    observation_chars: int = 8000
    total_observation_chars: int = 48000
    planning_output_tokens: int = 40000
    planning_input_tokens: int = 120000

    @classmethod
    def from_settings(cls, settings):
        return cls(calls_per_round=settings.central_max_tool_calls_per_round,
            total_calls=settings.central_max_total_tool_calls, identical_calls=settings.central_max_identical_tool_calls,
            parallel_calls=settings.central_max_parallel_tool_calls, deadline_seconds=settings.central_execution_deadline_seconds,
            tool_timeout_seconds=settings.central_tool_timeout_seconds, observation_chars=settings.central_max_observation_chars,
            total_observation_chars=settings.central_max_total_observation_chars,
            planning_output_tokens=settings.central_max_planning_output_tokens,
            planning_input_tokens=settings.central_max_planning_input_tokens)

    def __post_init__(self):
        for name in ("calls_per_round", "total_calls", "identical_calls", "parallel_calls",
                     "deadline_seconds", "tool_timeout_seconds", "observation_chars",
                     "total_observation_chars", "planning_output_tokens", "planning_input_tokens"):
            if getattr(self, name) <= 0:
                raise ValueError(f"Agent limit must be positive: {name}")
        if self.observation_chars < 256 or self.total_observation_chars < self.observation_chars:
            raise ValueError("Invalid agent observation budget")


class AgentDeadline(TimeoutError):
    pass


async def bounded_wait(awaitable, *, cancel: threading.Event, deadline: float, on_abort=None):
    """Cancel active async work promptly, without waiting forever on a bad transport.

    A sync worker cannot be killed. Its isolated context is discarded, its cancel
    flag is set, and registry resource locks remain held until the worker exits.
    """
    task = asyncio.ensure_future(awaitable)
    try:
        while not task.done():
            if cancel.is_set():
                raise asyncio.CancelledError()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AgentDeadline()
            await asyncio.wait({task}, timeout=min(.05, remaining))
        if cancel.is_set():
            raise asyncio.CancelledError()
        return task.result()
    except BaseException:
        if on_abort:
            on_abort()
        task.cancel()
        await asyncio.wait({task}, timeout=.2)
        def consume(done):
            if not done.cancelled():
                done.exception()
        if task.done():
            consume(task)
        else:
            task.add_done_callback(consume)
        raise
