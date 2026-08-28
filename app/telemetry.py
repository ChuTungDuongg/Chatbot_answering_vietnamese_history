from __future__ import annotations

import contextvars
import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any


logger = logging.getLogger(__name__)


@dataclass
class GenerationMetric:
    adapter: str
    input_tokens: int
    output_tokens: int
    max_new_tokens: int
    lock_wait_ms: float = 0.0
    adapter_switch_ms: float = 0.0
    generation_ms: float = 0.0
    decode_ms: float = 0.0
    tokens_per_sec: float = 0.0
    peak_allocated_gb: float | None = None


@dataclass
class RequestTelemetry:
    request_id: str
    inference_mode: str | None = None
    deployment_id: str | None = None
    gpu: str | None = None
    started: float = field(default_factory=time.perf_counter)
    cold_start_included: bool = False
    failed_stage: str | None = None
    failure_code: str | None = None
    research_attempts: int = 0
    research_steps: int = 0
    research_ms: float = 0.0
    research_json_repairs: int = 0
    tool_calls: int = 0
    tool_calls_by_type: dict[str, int] = field(default_factory=dict)
    retrieval_ms: float = 0.0
    wikipedia_calls: int = 0
    wikipedia_ms: float = 0.0
    generic_web_calls: int = 0
    generic_web_ms: float = 0.0
    evidence_attempts: int = 0
    evidence_ms: float = 0.0
    evidence_recovery_used: bool = False
    evidence_repair_used: bool = False
    evidence_rebucket_attempted: bool = False
    evidence_rebucket_succeeded: bool = False
    evidence_rebucket_moved_claim_count: int = 0
    evidence_rebucket_destination_ids: list[str] = field(default_factory=list)
    evidence_final_validation_result: str | None = None
    history_ms: float = 0.0
    history_generation_calls: int = 0
    generation_metrics: list[GenerationMetric] = field(default_factory=list)

    def next_call_index(self) -> int:
        return len(self.generation_metrics) + 1

    def add_generation(self, metric: GenerationMetric) -> None:
        self.generation_metrics.append(metric)

    @property
    def total_llm_calls(self) -> int:
        return len(self.generation_metrics)

    @property
    def research_llm_calls(self) -> int:
        return sum(1 for item in self.generation_metrics if item.adapter == "research")

    @property
    def evidence_generation_calls(self) -> int:
        return sum(1 for item in self.generation_metrics if item.adapter == "evidence")

    @property
    def total_input_tokens(self) -> int:
        return sum(item.input_tokens for item in self.generation_metrics)

    @property
    def total_output_tokens(self) -> int:
        return sum(item.output_tokens for item in self.generation_metrics)

    @property
    def average_generation_tokens_per_sec(self) -> float:
        rates = [item.tokens_per_sec for item in self.generation_metrics if item.tokens_per_sec > 0]
        return float(statistics.mean(rates)) if rates else 0.0

    @property
    def peak_allocated_vram_gb(self) -> float | None:
        peaks = [item.peak_allocated_gb for item in self.generation_metrics if item.peak_allocated_gb is not None]
        return max(peaks) if peaks else None

    def summary(self, *, result: str) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "inference_mode": self.inference_mode,
            "deployment_id": self.deployment_id,
            "gpu": self.gpu,
            "result": result,
            "failed_stage": self.failed_stage,
            "failure_code": self.failure_code,
            "cold_start_included": self.cold_start_included,
            "total_ms": (time.perf_counter() - self.started) * 1000,
            "research_attempts": self.research_attempts,
            "research_steps": self.research_steps,
            "research_ms": self.research_ms,
            "research_llm_calls": self.research_llm_calls,
            "research_json_repairs": self.research_json_repairs,
            "tool_calls": self.tool_calls,
            "tool_calls_by_type": self.tool_calls_by_type,
            "retrieval_ms": self.retrieval_ms,
            "wikipedia_calls": self.wikipedia_calls,
            "wikipedia_ms": self.wikipedia_ms,
            "generic_web_calls": self.generic_web_calls,
            "generic_web_ms": self.generic_web_ms,
            "evidence_attempts": self.evidence_attempts,
            "evidence_ms": self.evidence_ms,
            "evidence_generation_calls": self.evidence_generation_calls,
            "evidence_recovery_used": self.evidence_recovery_used,
            "evidence_repair_used": self.evidence_repair_used,
            "evidence_rebucket_attempted": self.evidence_rebucket_attempted,
            "evidence_rebucket_succeeded": self.evidence_rebucket_succeeded,
            "evidence_rebucket_moved_claim_count": self.evidence_rebucket_moved_claim_count,
            "evidence_rebucket_destination_ids": self.evidence_rebucket_destination_ids,
            "evidence_final_validation_result": self.evidence_final_validation_result,
            "history_ms": self.history_ms,
            "history_generation_calls": self.history_generation_calls,
            "total_llm_calls": self.total_llm_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "average_generation_tokens_per_sec": self.average_generation_tokens_per_sec,
            "peak_allocated_vram_gb": self.peak_allocated_vram_gb,
        }


_current: contextvars.ContextVar[RequestTelemetry | None] = contextvars.ContextVar(
    "request_telemetry",
    default=None,
)


def current_request_telemetry() -> RequestTelemetry | None:
    return _current.get()


def set_request_telemetry(telemetry: RequestTelemetry | None) -> contextvars.Token:
    return _current.set(telemetry)


def reset_request_telemetry(token: contextvars.Token) -> None:
    _current.reset(token)


def log_event(name: str, **fields: Any) -> None:
    logger.info(name, extra={"event": name, **fields})
