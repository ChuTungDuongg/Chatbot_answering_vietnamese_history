"""Numeric vLLM 0.23 scheduler stats adapter; no prompt/token IDs retained."""
from __future__ import annotations

import math
import threading
import time


class CacheMetrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._engines = {}

    def record(self, stats, engine_index=0):
        # Current runtime uses one engine. Bound future engine dimensions too.
        if stats is None or type(engine_index) is not int or not 0 <= engine_index < 64:
            return
        prefix = getattr(stats, "prefix_cache_stats", None)
        queries, hits = getattr(prefix, "queries", None), getattr(prefix, "hits", None)
        usage = getattr(stats, "kv_cache_usage", None)
        valid_prefix = (type(queries) is int and type(hits) is int and 0 <= hits <= queries)
        valid_usage = (type(usage) in (int, float) and math.isfinite(usage) and 0 <= usage <= 1)
        if not valid_prefix and not valid_usage:
            return
        with self._lock:
            row = self._engines.setdefault(engine_index, {"queries": 0, "hits": 0,
                                                        "usage": None, "observed": None})
            if valid_prefix:
                if getattr(prefix, "reset", False) is True:
                    row["queries"] = row["hits"] = 0
                row["queries"] += queries
                row["hits"] += hits
            if valid_usage:
                row["usage"] = float(usage)
                row["observed"] = time.monotonic()

    def snapshot(self):
        with self._lock:
            rows = list(self._engines.values())
            queries = sum(row["queries"] for row in rows)
            hits = sum(row["hits"] for row in rows)
            observed = [row["observed"] for row in rows if row["observed"] is not None]
            age = time.monotonic() - min(observed) if observed else None
            fresh = age is not None and age <= 15
            usages = [row["usage"] for row in rows if row["usage"] is not None]
            return {"prefix_cache_queries": queries if rows else None,
                    "prefix_cache_hits": hits if rows else None,
                    "prefix_cache_hit_rate": hits / queries if queries else None,
                    # Maximum across engine ranks; current deployment has one.
                    "kv_cache_utilization": max(usages) if usages and fresh else None,
                    "last_observed_age_seconds": age,
                    "status": "available" if fresh else "stale" if rows else "awaiting_stats"}

    def logger_factory(self):
        # Import only inside the pinned Linux vLLM runtime.
        from vllm.v1.metrics.loggers import StatLoggerBase
        observer = self
        class CacheStatLogger(StatLoggerBase):
            def __init__(self, vllm_config, engine_index=0):
                self.engine_index = engine_index

            def record(self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0):
                observer.record(scheduler_stats, self.engine_index)

            def log_engine_initialized(self):
                pass
        return CacheStatLogger
