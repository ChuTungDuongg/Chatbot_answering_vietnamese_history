"""Bounded, process-local chat execution metrics. No content or identity fields."""
from __future__ import annotations

import asyncio
import math
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from typing import Any

from app.telemetry import log_event
from starlette.requests import ClientDisconnect

MEASUREMENTS = (
    "e2e_ms", "final_answer_ttft_ms", "model_ttft_ms", "retrieval_ms",
    "generation_ms", "planning_model_ms", "tokens_per_second",
)


def _number(value):
    if type(value) not in (int, float):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else None
    except OverflowError:
        return None


def _percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * pct
    lower, upper = math.floor(index), math.ceil(index)
    return round(values[lower] + (values[upper] - values[lower]) * (index - lower), 2)


def cache_snapshot(model):
    """Read only the explicit runtime metrics API; never load an engine."""
    engine = getattr(model, "inference_backend", "unknown")
    result = {"engine": engine if engine in {"vllm", "transformers"} else "unknown",
              "prefix_cache_enabled": None, "prefix_cache_hit_rate": None,
              "kv_cache_utilization": None, "status": "unavailable"}
    if engine == "vllm":
        # VLLMRuntime owns this property; unknown runtimes retain nulls.
        value = getattr(model, "cache_metrics", None)
        if isinstance(value, dict):
            result.update(sanitize_cache(value))
    return result


def sanitize_cache(cache):
    result = {"engine": cache.get("engine") if cache.get("engine") in
              {"vllm", "transformers"} else "unknown"}
    enabled = cache.get("prefix_cache_enabled")
    result["prefix_cache_enabled"] = enabled if type(enabled) is bool else None
    for name in ("prefix_cache_hit_rate", "kv_cache_utilization"):
        value = _number(cache.get(name))
        result[name] = value if value is not None and value <= 1 else None
    for name in ("prefix_cache_queries", "prefix_cache_hits", "last_observed_age_seconds"):
        result[name] = _number(cache.get(name))
    result["status"] = cache.get("status") if cache.get("status") in {
        "available", "unavailable", "not_loaded", "awaiting_stats", "stale"
    } else "unavailable"
    return result


def _memory():
    process = {"peak_rss_bytes": None, "status": "unavailable"}
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        process = {"peak_rss_bytes": value * (1 if sys.platform == "darwin" else 1024),
                   "status": "available"}
    except (ImportError, OSError, ValueError):
        pass
    gpu = None
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            # Do not initialize CUDA merely to answer an ops request.
            if torch.cuda.is_initialized():
                gpu = {"scope": "current_process_allocator",
                       "allocated": torch.cuda.memory_allocated(),
                       "reserved": torch.cuda.memory_reserved(),
                       "peak_allocated": torch.cuda.max_memory_allocated()}
        except (RuntimeError, AttributeError):
            pass
    return {"process_memory": process, "gpu_memory_bytes": gpu,
            "gpu_memory_notice": "Current process Torch allocator only; vLLM worker VRAM and device utilization require infrastructure metrics."}


class ServingMetrics:
    """Last N samples per fixed dimension; rolling traffic uses 1-second buckets."""

    def __init__(self, max_samples: int = 500):
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._samples = defaultdict(lambda: deque(maxlen=max_samples))
        self._buckets = deque(maxlen=301)
        self._totals = Counter()
        self._started_count = 0
        self._inflight = 0
        self._cache = {}
        self._cache_updated = {}
        self._http_totals = Counter()
        self._http_started = self._http_inflight = 0

    def begin(self):
        with self._lock:
            self._started_count += 1
            self._inflight += 1

    def finish(self, *, mode, backend, outcome, metrics=None, cache=None):
        metrics = metrics or {}
        key = (mode if mode in {"hybrid", "central"} else "other",
               backend if backend in {"faiss", "qdrant"} else "unknown",
               outcome if outcome in {"success", "error", "cancelled"} else "unknown")
        sample = {name: _number(metrics.get(name)) for name in MEASUREMENTS}
        tokens = _number(metrics.get("output_tokens"))
        sample["output_tokens"] = tokens if tokens is not None and tokens.is_integer() else None
        with self._lock:
            self._inflight = max(0, self._inflight - 1)
            self._totals[key] += 1
            self._samples[key].append(sample)
            second = int(time.monotonic())
            if not self._buckets or self._buckets[-1][0] != second:
                self._buckets.append([second, 0, 0, 0])
            bucket = self._buckets[-1]
            bucket[1] += 1
            if sample["output_tokens"] is not None:
                bucket[2] += sample["output_tokens"]
                bucket[3] += 1
            if cache:
                self._cache[key[0]] = sanitize_cache(cache)
                self._cache_updated[key[0]] = time.monotonic()
        log_event("serving_request", mode=key[0], retrieval_backend=key[1], outcome=key[2], **sample)

    def begin_http(self):
        with self._lock:
            self._http_started += 1
            self._http_inflight += 1

    def finish_http(self, route, status_class, outcome):
        with self._lock:
            self._http_inflight = max(0, self._http_inflight - 1)
            self._http_totals[(route, status_class, outcome)] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            cutoff = int(now) - 299
            buckets = [row for row in self._buckets if row[0] >= cutoff]
            count = sum(row[1] for row in buckets)
            tokens = sum(row[2] for row in buckets)
            known = sum(row[3] for row in buckets)
            seconds = min(300, max(0, now - self._started))
            summaries = []
            for (mode, backend, outcome), samples in sorted(self._samples.items()):
                items = list(samples)
                summary = {"mode": mode, "retrieval_backend": backend, "outcome": outcome,
                           "sample_count": len(items)}
                for name in MEASUREMENTS:
                    values = [item[name] for item in items if item[name] is not None]
                    summary[name] = {"count": len(values), "p50": _percentile(values, .5),
                                     "p95": _percentile(values, .95)}
                summaries.append(summary)
            caches = {}
            for mode, cache in self._cache.items():
                cache = dict(cache)
                age = cache.get("last_observed_age_seconds")
                if age is not None:
                    cache["last_observed_age_seconds"] = age + now - self._cache_updated[mode]
                    if cache["last_observed_age_seconds"] > 15:
                        cache.update(status="stale", kv_cache_utilization=None)
                caches[mode] = cache
            result = {
                "scope": "process_local", "durable": False,
                "request_scope": "accepted_chat_executions",
                "uptime_seconds": round(now - self._started, 1),
                "requests_started": self._started_count, "inflight": self._inflight,
                "requests_last_5m": count,
                "throughput": {"window_seconds": seconds, "bucket_seconds": 1,
                               "completed_requests_per_second": count / seconds if seconds else None,
                               "known_output_tokens_per_second": tokens / seconds if seconds and known else None,
                               "requests_with_known_output_tokens": known},
                "totals": [{"mode": m, "retrieval_backend": b, "outcome": o, "count": n}
                           for (m, b, o), n in sorted(self._totals.items())],
                "latency": summaries, "cache": caches,
                "http": {"requests_started": self._http_started, "inflight": self._http_inflight,
                         "totals": [{"route": r, "status_class": s, "outcome": o, "count": n}
                                    for (r, s, o), n in sorted(self._http_totals.items())]},
                "cache_notice": "Engine counters are token-based since last cache reset; not per-request cache hits. HF disk cache is unrelated.",
            }
        return {**result, **_memory()}


serving_metrics = ServingMetrics()


class RequestMetricsMiddleware:
    """Pure ASGI counters; never read/buffer bodies or change streaming frames."""
    ROUTES = {"/api/v1/chat": "chat", "/api/v1/chat/stream": "chat_stream",
              "/api/v1/retrieve": "retrieve"}

    def __init__(self, app, metrics=None):
        self.app = app
        self.metrics = metrics if metrics is not None else serving_metrics

    async def __call__(self, scope, receive, send):
        route = self.ROUTES.get(scope.get("path"))
        if scope["type"] != "http" or route is None:
            return await self.app(scope, receive, send)
        self.metrics.begin_http()
        status_class, outcome = "unknown", "error"
        terminal_body_sent = False

        async def counted_send(message):
            nonlocal status_class, outcome, terminal_body_sent
            if message["type"] == "http.response.start":
                code = message["status"] // 100
                status_class = f"{code}xx" if 1 <= code <= 5 else "unknown"
            try:
                await send(message)
                if message["type"] == "http.response.body" and not message.get("more_body", False):
                    terminal_body_sent = True
            except OSError:
                outcome = "cancelled"
                raise
        try:
            await self.app(scope, receive, counted_send)
            outcome = "complete" if terminal_body_sent else "cancelled"
        except (asyncio.CancelledError, GeneratorExit, ClientDisconnect):
            outcome = "cancelled"
            raise
        finally:
            if outcome == "error" and status_class == "unknown":
                status_class = "5xx"
            self.metrics.finish_http(route, status_class, outcome)
