"""Versioned, monotonic request trace without prompts or hidden reasoning."""

from __future__ import annotations

import contextvars
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


logger = logging.getLogger("app.telemetry")
_active: contextvars.ContextVar[RequestTrace | None] = contextvars.ContextVar("request_trace", default=None)


def log_event(event: str, **values: Any) -> None:
    logger.info(json.dumps({"schema_version": 1, "event": event, **values}, ensure_ascii=False, default=str))


@dataclass
class RequestTrace:
    request_id: str
    mode: str
    started_ns: int = field(default_factory=time.perf_counter_ns)
    marks_ns: dict[str, int] = field(default_factory=dict)
    domain_gate_result: str | None = None
    domain_gate_reason: str | None = None
    history_anchor: float | None = None
    ood_anchor: float | None = None
    domain_margin: float | None = None
    retrieval_skipped_due_to_ood: bool = False
    llm_calls_skipped_due_to_ood: bool = False

    def __post_init__(self) -> None:
        self.marks_ns["request_received"] = self.started_ns

    def mark(self, event: str, when_ns: int | None = None) -> None:
        self.marks_ns[event] = when_ns or time.perf_counter_ns()

    def offset_ms(self, event: str) -> float | None:
        value = self.marks_ns.get(event)
        return (value - self.started_ns) / 1e6 if value is not None else None

    def span_ms(self, start: str, end: str) -> float | None:
        if start not in self.marks_ns or end not in self.marks_ns:
            return None
        return (self.marks_ns[end] - self.marks_ns[start]) / 1e6

    def emit(self, *, metrics: dict[str, Any], model_id: str,
             model_revision: str | None, error: str | None = None) -> None:
        log_event("request_trace", request_id=self.request_id, mode=self.mode,
                  timestamp=datetime.now(timezone.utc).isoformat(), model_id=model_id,
                  model_revision=model_revision,
                  spans_ms={key: self.offset_ms(key) for key in self.marks_ns},
                  metrics=metrics, error=error)


def current_request_telemetry() -> RequestTrace | None:
    return _active.get()


def set_request_telemetry(trace: RequestTrace):
    return _active.set(trace)


def reset_request_telemetry(token) -> None:
    _active.reset(token)
