"""Bounded, process-local serving metrics; never records user content or prompts."""

from __future__ import annotations

import math
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from typing import Any

from app.telemetry import log_event

MEASUREMENTS = (
    "e2e_ms", "final_answer_ttft_ms", "model_ttft_ms", "retrieval_ms",
    "generation_ms", "planning_model_ms", "tokens_per_second",
)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * pct
    lower = math.floor(index)
    upper = math.ceil(index)
    return round(values[lower] + (values[upper] - values[lower]) * (index - lower), 2)


class ServingMetrics:
    """A learning dashboard, not a durable/cluster-wide monitoring backend."""

    def __init__(self, max_samples: int = 500):
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._samples = defaultdict(lambda: deque(maxlen=max_samples))
        self._events = deque(maxlen=max_samples)
        self._totals = Counter()
        self._inflight = 0
        self._cache = {}

    def begin(self) -> None:
        with self._lock:
            self._inflight += 1

    def finish(self, *, mode: str, backend: str, outcome: str,
               metrics: dict[str, Any] | None = None,
               cache: dict[str, Any] | None = None) -> None:
        metrics = metrics or {}
        key = (mode if mode in {"hybrid", "central"} else "other",
               backend if backend in {"faiss", "qdrant"} else "unknown")
        sample = {
            name: float(metrics[name]) if isinstance(metrics.get(name), (int, float))
            and math.isfinite(float(metrics[name])) else None
            for name in MEASUREMENTS
        }
        sample["output_tokens"] = metrics.get("output_tokens")
        with self._lock:
            self._inflight = max(0, self._inflight - 1)
            self._totals[(key[0], key[1], outcome)] += 1
            self._samples[key].append(sample)
            self._events.append(time.monotonic())
            if cache:
                self._cache[key[0]] = cache
        log_event("serving_request", mode=key[0], retrieval_backend=key[1],
                  outcome=outcome, **sample)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            cutoff = time.monotonic() - 300
            summaries = []
            for (mode, backend), samples in sorted(self._samples.items()):
                items = list(samples)
                summary = {"mode": mode, "retrieval_backend": backend, "sample_count": len(items)}
                for name in MEASUREMENTS:
                    values = [item[name] for item in items if item[name] is not None]
                    summary[name] = {"count": len(values), "p50": _percentile(values, .5),
                                     "p95": _percentile(values, .95)}
                summaries.append(summary)
            snapshot = {
                "scope": "process_local", "durable": False,
                "uptime_seconds": round(time.monotonic() - self._started, 1),
                "inflight": self._inflight,
                "requests_last_5m": sum(ts >= cutoff for ts in self._events),
                "totals": [{"mode": m, "retrieval_backend": b, "outcome": o, "count": n}
                           for (m, b, o), n in sorted(self._totals.items())],
                "latency": summaries, "cache": dict(self._cache),
                "cache_notice": ("Hugging Face disk cache is not a per-request hit rate. "
                                 "vLLM prefix-cache hit rate is unknown unless exported by the engine."),
            }
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    snapshot["gpu_memory_bytes"] = {
                        "allocated": torch.cuda.memory_allocated(),
                        "reserved": torch.cuda.memory_reserved(),
                        "peak_allocated": torch.cuda.max_memory_allocated(),
                    }
            except (RuntimeError, AttributeError):
                pass
        return snapshot


serving_metrics = ServingMetrics()
