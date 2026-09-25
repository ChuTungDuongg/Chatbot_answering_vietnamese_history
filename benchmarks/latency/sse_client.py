"""Measure the real HTTP SSE endpoint with a monotonic client clock."""
from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable, Iterator
from uuid import uuid4


SERVER_METRICS = (
    "retrieval_ms", "prompt_build_ms", "generation_start_ms", "model_ttft_ms",
    "generation_ms", "input_tokens", "output_tokens", "tokens_per_second",
    "decode_tokens_per_second", "tpot_ms", "model_calls", "tool_calls",
    "tool_execution_ms", "tool_parse_failures", "action_rounds",
    "time_until_final_generation_ms", "final_answer_ttft_ms",
)
OVERLAPPING_SERVER_METRICS = {
    "first_status_event_ms": "server_first_status_event_ms",
    "e2e_ms": "server_e2e_ms",
    "itl_p50_ms": "server_itl_p50_ms",
    "itl_p95_ms": "server_itl_p95_ms",
    "itl_p99_ms": "server_itl_p99_ms",
}


@dataclass(frozen=True)
class SSEEvent:
    name: str
    data: dict[str, Any]
    observed_ns: int


def parse_sse(lines: Iterable[bytes], *, clock=time.perf_counter_ns) -> Iterator[SSEEvent]:
    """Parse SSE frames, including multiline data and comment keepalives."""
    name = "message"
    data: list[str] = []
    for raw in lines:
        line = raw.decode("utf-8-sig").rstrip("\r\n")
        if not line:
            if data:
                payload = json.loads("\n".join(data))
                if not isinstance(payload, dict):
                    raise ValueError("SSE data must be a JSON object")
                yield SSEEvent(name, payload, clock())
            name, data = "message", []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            name = value
        elif field == "data":
            data.append(value)
    if data:
        payload = json.loads("\n".join(data))
        if not isinstance(payload, dict):
            raise ValueError("SSE data must be a JSON object")
        yield SSEEvent(name, payload, clock())


def _ms(end_ns: int | None, start_ns: int) -> float | None:
    return (end_ns - start_ns) / 1_000_000 if end_ns is not None else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lo = math.floor(index)
    hi = math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def _json_request(url: str, payload: dict[str, Any], client_id: str, timeout: float):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json",
                 "X-Client-ID": client_id},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=timeout)


def measure_request(
    *, base_url: str, question: dict[str, Any], mode: str, client_id: str,
    timeout: float, phase: str, run_index: int, clock=time.perf_counter_ns,
) -> dict[str, Any]:
    """Create an isolated conversation, then time only the streamed chat request."""
    if mode not in {"hybrid", "central"}:
        raise ValueError("mode must be hybrid or central")
    client_request_id = str(uuid4())
    record: dict[str, Any] = {
        "schema_version": 1, "request_id": client_request_id, "server_request_id": None,
        "question_id": question["id"], "mode": mode, "phase": phase,
        "run_index": run_index, "cold_start": phase == "cold", "success": False,
        "http_status": None, "model_id": None, "model_revision": None,
        "ttfb_ms": None, "first_status_event_ms": None, "answer_ttft_ms": None,
        "e2e_ms": None, "itl_p50_ms": None, "itl_p95_ms": None,
        "itl_p99_ms": None, "inter_token_latency_ms": [],
        "answer": "", "sources": [], "cited_source_ids": [], "error": None,
        "generation_settings": None, "retrieval_settings": None,
        "tool_call_types": None,
        **{key: None for key in SERVER_METRICS},
        **{key: None for key in OVERLAPPING_SERVER_METRICS.values()},
    }
    root = base_url.rstrip("/")
    try:
        with _json_request(f"{root}/api/v1/conversations", {}, client_id, timeout) as response:
            conversation = json.load(response)
        conversation_id = conversation["id"]
        request = urllib.request.Request(
            f"{root}/api/v1/chat/stream",
            data=json.dumps({"conversation_id": conversation_id,
                             "question": question["question"], "mode": mode},
                            ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream",
                     "X-Client-ID": client_id, "X-Request-ID": client_request_id},
            method="POST",
        )
        start_ns = clock()
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_started_ns = clock()
            record["http_status"] = response.status
            record["server_request_id"] = response.headers.get("X-Request-ID")
            first_status_ns = first_answer_ns = done_ns = None
            answer_times: list[int] = []
            done: dict[str, Any] | None = None
            parts: list[str] = []
            for event in parse_sse(response, clock=clock):
                if event.name == "status" and first_status_ns is None:
                    first_status_ns = event.observed_ns
                elif event.name == "answer_delta":
                    delta = event.data.get("delta")
                    if not isinstance(delta, str):
                        raise ValueError("answer_delta.delta must be a string")
                    if delta:
                        if first_answer_ns is None:
                            first_answer_ns = event.observed_ns
                        answer_times.append(event.observed_ns)
                        parts.append(delta)
                elif event.name == "sources":
                    record["sources"] = event.data.get("items") or []
                    record["cited_source_ids"] = event.data.get("cited_source_ids") or []
                elif event.name == "error":
                    record["error"] = event.data
                elif event.name == "done":
                    done_ns, done = event.observed_ns, event.data
                    break
            record["answer"] = "".join(parts)
            record["ttfb_ms"] = _ms(response_started_ns, start_ns)
            record["first_status_event_ms"] = _ms(first_status_ns, start_ns)
            record["answer_ttft_ms"] = _ms(first_answer_ns, start_ns)
            record["e2e_ms"] = _ms(done_ns, start_ns)
            latencies = [(right - left) / 1_000_000 for left, right in zip(answer_times, answer_times[1:])]
            record["inter_token_latency_ms"] = latencies
            for name, p in (("itl_p50_ms", .5), ("itl_p95_ms", .95), ("itl_p99_ms", .99)):
                record[name] = _percentile(latencies, p)
            if done is None:
                record["error"] = record["error"] or {"type": "incomplete_stream", "message": "missing done event"}
            else:
                record["server_request_id"] = done.get("request_id") or record["server_request_id"]
                record["model_id"] = done.get("model_id")
                record["model_revision"] = done.get("model_revision")
                record["generation_settings"] = done.get("generation_settings")
                record["retrieval_settings"] = done.get("retrieval_settings")
                metrics = done.get("metrics") or {}
                if not isinstance(metrics, dict):
                    raise ValueError("done.metrics must be an object")
                for key in SERVER_METRICS:
                    value = metrics.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                        record[key] = value
                for server_key, raw_key in OVERLAPPING_SERVER_METRICS.items():
                    value = metrics.get(server_key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                        record[raw_key] = value
                if isinstance(metrics.get("tool_call_types"), list) and all(
                    isinstance(item, str) for item in metrics["tool_call_types"]
                ):
                    record["tool_call_types"] = metrics["tool_call_types"]
                if done.get("status") == "error" and record["error"] is None:
                    record["error"] = {"type": "server_error", "message": "done status was error"}
            record["success"] = record["http_status"] == 200 and done is not None and record["error"] is None
    except urllib.error.HTTPError as exc:
        record["http_status"] = exc.code
        record["error"] = {"type": "http_error", "message": exc.read(2048).decode("utf-8", errors="replace")}
    except Exception as exc:
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
    return record
