"""Auditable execution with explicit independence and observed reader inputs.

Dependencies are resolved by the next planner turn, never by substituting guessed
values into a call. Only admitted calls using already available inputs execute.
"""

import asyncio
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
import re
import threading
import time
from typing import Any
from urllib.parse import urlsplit

from app.central.budgets import AgentDeadline, AgentLimits, bounded_wait
from app.rag.backends import DenseBackendError
from app.models.tool_calls import DecodedFunctionCall
from app.mcp.schemas import MCPToolError
from app.tools.registry import ToolCallRecord, ToolExecutionContext, ToolRegistry


@dataclass
class ToolOutcome:
    call: DecodedFunctionCall
    output: Any
    record: ToolCallRecord
    context: ToolExecutionContext
    latency_ms: float = 0
    execution_mode: str = "rejected"
    batch: int = 0


def _url(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value)
        return value if parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password else None
    except ValueError:
        return None


def reader_input_observed(call, question, evidence):
    """Only source readers need an observed ID/URL, not exploratory searches."""
    if call.name == "fetch_wikipedia_page":
        key = str(call.arguments.get("page_id_or_title") or "").strip()
        language = call.arguments.get("language", "vi")
        # An exact title explicitly supplied in the question is already an input.
        if key and not key.isdigit() and key.casefold() in question.casefold():
            return True
        if key.isdigit() and key in re.findall(r"(?:page[_ -]?id|curid)\s*[=:]\s*(\d+)", question, flags=re.I):
            return True
        for chunk in evidence:
            metadata = chunk.get("metadata") or {}
            chunk_language = metadata.get("language") or str(chunk.get("chunk_id", "")).split("_")[1:2]
            if isinstance(chunk_language, list):
                chunk_language = chunk_language[0] if chunk_language else None
            if chunk_language != language:
                continue
            if key and key in {str(metadata.get("page_id", "")), str(chunk.get("title", ""))}:
                return True
        return False
    if call.name == "fetch_web_page":
        url = _url(call.arguments.get("url"))
        supplied = {value.rstrip(".,;!?)") for value in re.findall(r'https?://[^\s<>"\']+', question)}
        return bool(url and (url in supplied or any(c.get("url") == url for c in evidence)))
    return True


class AgentExecutor:
    """One instance per prepare(); no request state is stored on the runtime."""

    def __init__(self, registry: ToolRegistry, context: ToolExecutionContext, limits: AgentLimits,
                 *, cancel: threading.Event, question: str):
        self.registry, self.context, self.limits = registry, context, limits
        self.cancel, self.question = cancel, question
        self.deadline = time.monotonic() + limits.deadline_seconds
        self.attempts = Counter()
        self.calls = self.repeats_avoided = self.batch_count = 0
        self.wall_ms = 0.0
        self.descriptors = {row["name"]: row for row in registry.describe()}

    def error_record(self, call, error):
        descriptor = self.descriptors.get(call.name, {})
        return ToolCallRecord(call.name, {}, error=error,
            provider=descriptor.get("provider", "builtin"), server=descriptor.get("server"))

    def rejected(self, call, error):
        return ToolOutcome(call, None, self.error_record(call, error), self.context)

    async def execute(self, calls, *, independent, evidence, blocked=frozenset()):
        """Return outcomes in model order, regardless of completion order."""
        admitted, results = [], {}
        started = time.perf_counter_ns()
        for index, call in enumerate(calls):
            error = call.error
            if self.context.allowed_tools is not None and call.name not in self.context.allowed_tools:
                error = "tool_not_allowed"
            elif call.name in blocked:
                error = "external_evidence_required"
            elif index >= self.limits.calls_per_round:
                error = "round_tool_budget"
            elif self.calls >= self.limits.total_calls:
                error = "total_tool_budget"
            elif not reader_input_observed(call, self.question, evidence):
                error = "dependency_not_observed"
            canonical = call.arguments
            if not error:
                try:
                    validated = self.registry.validate_arguments(call.name, call.arguments)
                    canonical = validated.model_dump(mode="json") if hasattr(validated, "model_dump") else validated
                except Exception:
                    # Execution still uses the registry's existing validation and
                    # MCP failure policy; this pass only normalizes repeat keys.
                    pass
            key = hashlib.sha256(json.dumps([call.name, canonical], sort_keys=True,
                                           ensure_ascii=False).encode()).hexdigest()
            if not error and self.attempts[key] >= self.limits.identical_calls:
                error = "repeated_call"
                self.repeats_avoided += 1
            if error:
                results[index] = self.rejected(call, error)
            else:
                self.attempts[key] += 1
                self.calls += 1
                admitted.append((index, call))

        async def run(index, call, mode, batch):
            local_cancel = threading.Event()
            original_progress = self.context.progress
            def progress(event):
                if original_progress and not local_cancel.is_set() and not self.cancel.is_set():
                    original_progress(event)
            local = replace(self.context, retrieval_metrics={}, mcp_metrics=[], cancel=local_cancel,
                            progress=progress if original_progress else None)
            begin = time.perf_counter_ns()
            try:
                output, record = await bounded_wait(self.registry.call(call.name, call.arguments, context=local),
                    cancel=self.cancel,
                    deadline=min(self.deadline, time.monotonic() + self.limits.tool_timeout_seconds),
                    on_abort=local_cancel.set)
            except AgentDeadline:
                output, record = None, self.error_record(call, "tool_timeout")
                if record.provider == "mcp" and self.context.mcp_failure_policy == "fail":
                    raise MCPToolError("Công cụ MCP không hoàn tất; request yêu cầu dừng khi lỗi.") from None
                if self.context.progress and not self.cancel.is_set():
                    self.context.progress({"stage": f"tool:{call.name}", "state": "failed",
                                           "message": "Công cụ hết thời gian", "tool_call_id": call.id})
            except DenseBackendError:
                # Replanning may choose another permitted tool, but never change
                # the selected dense backend behind the user's back.
                output, record = None, self.error_record(call, "retrieval_unavailable")
            except asyncio.CancelledError:
                local_cancel.set()
                raise
            results[index] = ToolOutcome(call, output, record, local,
                (time.perf_counter_ns() - begin) / 1e6, mode, batch)

        async def flush(group):
            if not group:
                return
            self.batch_count += 1
            mode = "parallel" if len(group) > 1 else "sequential"
            tasks = [asyncio.create_task(run(index, call, mode, self.batch_count)) for index, call in group]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

        group, resources = [], set()
        try:
            for index, call in admitted:
                policy = self.registry.execution_policy(call.name)
                safe = index in independent and policy.read_only and policy.parallel_safe
                resource = policy.exclusive_resource
                if not safe or (resource and resource in resources) or len(group) >= self.limits.parallel_calls:
                    await flush(group)
                    group, resources = [], set()
                if not safe:
                    await flush([(index, call)])
                else:
                    group.append((index, call))
                    if resource:
                        resources.add(resource)
            await flush(group)
        finally:
            self.wall_ms += (time.perf_counter_ns() - started) / 1e6
        return [results[index] for index in range(len(calls))]


def merge_tool_metrics(target, local):
    """Called in model order after tasks finish; no workers share these containers."""
    if local is target:
        return
    if local.cancel and local.cancel.is_set():
        return
    if local.retrieval_metrics:
        target.retrieval_metrics["backend"] = local.retrieval_metrics.get("backend", target.retrieval_backend)
        timings = target.retrieval_metrics.setdefault("timings_ms", {})
        for name, value in local.retrieval_metrics.get("timings_ms", {}).items():
            timings[name] = timings.get(name, 0.0) + value
    target.mcp_metrics.extend(local.mcp_metrics)
