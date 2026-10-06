"""Central planning and tools followed by a genuinely streamed final answer."""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from app.models.tool_calls import HermesFunctionCallCodec
from app.models.base import ModelDone
from app.central.planning import (PLANNER_INSTRUCTION, evidence_chunks, final_after_tools,
                                  question_domain, terminal_evidence, planner_tool_schemas, COMPLETION_FIELD)
from app.rag.prompting import build_messages
from app.rag.schemas import PreparedAnswer
from app.tools.registry import ToolCallRecord, ToolExecutionContext, ToolRegistry
from app.tools.policy import tool_view
from app.rag.progress import StageProgress, ProgressCallback


class CentralRuntime:
    def __init__(self, *, model: Any, tools: ToolRegistry, max_action_rounds: int = 2,
                 action_max_new_tokens: int = 256, mcp_manager=None,
                 max_mcp_tools: int = 8, mcp_schema_budget: int = 16384):
        from app.config import CENTRAL_MODEL_ID

        if model.model_id != CENTRAL_MODEL_ID:
            raise ValueError("Central must use vanilla Qwen3-8B")
        self.model = model
        self.tools = tools
        self.max_action_rounds = max_action_rounds
        self.action_max_new_tokens = action_max_new_tokens
        self.codec = HermesFunctionCallCodec()
        self.mcp_manager = mcp_manager
        self.max_mcp_tools, self.mcp_schema_budget = max_mcp_tools, mcp_schema_budget

    def resolve_tools(self, steering=None, question=""):
        return tool_view(self.tools, steering=steering, question=question,
            mcp_capabilities=self.mcp_manager.capabilities() if self.mcp_manager else None,
            max_mcp_tools=self.max_mcp_tools, schema_budget=self.mcp_schema_budget)

    def _tool_schemas(self) -> list[dict[str, Any]]:
        return list(self.resolve_tools().schemas)

    async def prepare(self, question: str, top_k: int, history: list[dict[str, str]],
                      *, owner_id: str | None = None, conversation_id: str | None = None,
                      attachment_ids: tuple[str, ...] = (), trace: Any = None,
                      cancel: threading.Event | None = None,
                      retrieval_backend: str | None = None, progress: ProgressCallback | None = None,
                      steering=None, tool_view=None) -> PreparedAnswer:
        cancel = cancel or threading.Event()
        prepare_started = time.perf_counter_ns()
        if trace:
            trace.mark("prepare_started", prepare_started)
        view = tool_view or self.resolve_tools(steering, question)
        domain = question_domain(question)
        external_names = {name for name in view.allowed_names
                          if name not in {"search_history", "search_uploaded_documents"}}
        external_required = domain == "external" and bool(external_names)
        planner_schemas, controlled_tools = planner_tool_schemas([
            schema for schema in view.schemas
            if not (external_required and schema["function"]["name"] == "search_history")],
            mcp_schema_budget=self.mcp_schema_budget)
        messages: list[dict[str, Any]] = [{
            "role": "system",
            "content": PLANNER_INSTRUCTION,
        }]
        if domain == "external":
            messages[0]["content"] += " Câu hỏi hiện tại ở ngoài corpus Việt Nam; cần bằng chứng bên ngoài được phép."
        messages.extend({"role": item["role"], "content": str(item.get("content") or "")[:1500]}
                        for item in history[-4:] if item.get("role") in {"user", "assistant"})
        messages.append({"role": "user", "content": question})
        context = ToolExecutionContext(owner_id=owner_id, conversation_id=conversation_id,
                                       attachment_ids=attachment_ids or None,
                                       request_id=trace.request_id if trace else None,
                                       retrieval_backend=retrieval_backend, progress=progress,
                                       allowed_tools=view.allowed_names, cancel=cancel,
                                       mcp_failure_policy=view.failure_policy)
        stages = StageProgress(progress, retrieval_backend)
        contexts_by_id: dict[str, dict[str, Any]] = {}
        tool_records: list[dict[str, Any]] = []
        retrieval_ms = 0.0
        parse_failures = 0
        model_calls = 0
        rounds = 0
        planning_rounds: list[dict[str, Any]] = []
        exit_reason = "max_action_rounds"

        if trace:
            trace.mark("retrieval_started")
        for _ in range(self.max_action_rounds):
            if cancel.is_set():
                raise RuntimeError("Request cancelled")
            rounds += 1
            round_started = time.perf_counter_ns()
            with stages.track("tool_selection"):
                plan_text, plan_done = await self.model.generate(
                    messages, tools=planner_schemas, max_new_tokens=self.action_max_new_tokens,
                    cancel=cancel,
                )
            round_finished = time.perf_counter_ns()
            model_calls += 1
            parsed = self.codec.decode(plan_text)
            parse_failures += parsed.failures
            wall_ms = (round_finished - round_started) / 1e6
            model_metrics = plan_done.metrics if isinstance(plan_done, ModelDone) else {}
            model_ms = model_metrics.get("generation_ms")
            round_metrics = {"round": rounds, "wall_ms": wall_ms,
                             "started_ms": (round_started - (trace.started_ns if trace else prepare_started)) / 1e6,
                             "finished_ms": (round_finished - (trace.started_ns if trace else prepare_started)) / 1e6,
                             "model_ms": model_ms, "ttft_ms": model_metrics.get("model_ttft_ms"),
                             "preparation_and_queue_ms": max(0.0, wall_ms - model_ms) if model_ms is not None else None,
                             "input_tokens": model_metrics.get("input_tokens"),
                             "output_tokens": model_metrics.get("output_tokens"),
                             "finish_reason": model_metrics.get("finish_reason"),
                             "tools_requested": [call.name for call in parsed.tool_calls],
                             "parse_failures": parsed.failures, "early_exit": False}
            planning_rounds.append(round_metrics)
            if not parsed.tool_calls:
                exit_reason = "planner_no_tools"
                break
            messages.append({"role": "assistant", "content": plan_text})
            complete_evidence = parsed.failures == 0 and len(parsed.tool_calls) <= 4
            completion_declared = (final_after_tools(parsed.content) if "<plan_status>" in parsed.content else
                                   all(call.name in controlled_tools and call.arguments.get(COMPLETION_FIELD) is True
                                       for call in parsed.tool_calls))
            round_metrics["completion_declared"] = completion_declared
            external_evidence = any(c.get("source_kind") in {"wikipedia", "web", "mcp"}
                                    for c in contexts_by_id.values())
            local_evidence = any(c.get("source_kind", "history") == "history"
                                 for c in contexts_by_id.values())
            for call in parsed.tool_calls[:4]:
                if cancel.is_set():
                    raise RuntimeError("Request cancelled")
                started = time.perf_counter_ns()
                mcp_count = len(context.mcp_metrics)
                arguments = {key: value for key, value in call.arguments.items()
                             if not (call.name in controlled_tools and key == COMPLETION_FIELD)}
                if external_required and call.name == "search_history":
                    output, record = None, ToolCallRecord(call.name, {}, error="external_evidence_required")
                else:
                    output, record = await self.tools.call(call.name, arguments, context=context)
                elapsed = (time.perf_counter_ns() - started) / 1e6
                if call.name == "search_history":
                    retrieval_ms += elapsed
                tool_records.append({"name": call.name, "latency_ms": elapsed,
                                     "error": record.error, "result_count": record.result_count,
                                     "provider": record.provider, "server": record.server,
                                     **(context.mcp_metrics[-1] if len(context.mcp_metrics) > mcp_count else {})})
                chunks = evidence_chunks(output) if not record.error else []
                for chunk in chunks:
                    key = str(chunk["chunk_id"])
                    if key not in contexts_by_id or call.name in {"fetch_wikipedia_page", "fetch_page"}:
                        contexts_by_id[key] = chunk
                complete_evidence = complete_evidence and not record.error and terminal_evidence(call.name, output, view.allowed_names)
                external_evidence = external_evidence or (call.name in external_names and bool(chunks))
                local_evidence = local_evidence or (call.name == "search_history" and bool(chunks))
                observation = json.dumps(output if not record.error else {"success": False, "error": record.error},
                                         ensure_ascii=False, default=str)
                messages.append({"role": "tool", "name": call.name, "content": observation[:8000]})
            # No implicit "nonempty result means sufficient" shortcut: the planner
            # must declare full coverage and no pending dependencies, and actual
            # outputs must be successful, usable evidence. Missing/truncated plans,
            # snippets awaiting a reader, failures and mixed-domain gaps replan.
            if (complete_evidence and completion_declared
                    and model_metrics.get("finish_reason") != "length"
                    and (not external_required or external_evidence)
                    and (domain != "mixed" or (local_evidence and external_evidence))
                    and not attachment_ids):
                round_metrics["early_exit"] = True
                exit_reason = "evidence_sufficient"
                break

        if (domain != "external" and exit_reason != "evidence_sufficient" and "search_history" in view.allowed_names
                and not any(item["name"] == "search_history" for item in tool_records)):
            started = time.perf_counter_ns()
            output, record = await self.tools.call(
                "search_history", {"query": question, "top_k": top_k}, context=context
            )
            elapsed = (time.perf_counter_ns() - started) / 1e6
            retrieval_ms += elapsed
            tool_records.append({"name": "search_history", "latency_ms": elapsed,
                                 "error": record.error, "result_count": record.result_count, "provider": "builtin"})
            if not record.error:
                for chunk in evidence_chunks(output):
                    contexts_by_id.setdefault(str(chunk["chunk_id"]), chunk)

        if trace:
            trace.mark("retrieval_finished")
        selected = list(contexts_by_id.values())[:top_k]
        if any(chunk.get("source_kind") == "mcp" for chunk in contexts_by_id.values()):
            # MCP supplements local evidence instead of consuming its slots.
            selected = [c for c in contexts_by_id.values() if c.get("source_kind", "history") != "mcp"][:top_k]
            selected += [c for c in contexts_by_id.values() if c.get("source_kind") == "mcp"][:2]
        prompt_started = time.perf_counter_ns()
        with stages.track("prompt_preparation"):
            final_messages = build_messages(question, selected, history)
        prompt_build_ms = (time.perf_counter_ns() - prompt_started) / 1e6
        if trace:
            trace.mark("prompt_ready")
        planning = {"rounds": planning_rounds, "exit_reason": exit_reason, "domain": domain,
                    "total_wall_ms": sum(row["wall_ms"] for row in planning_rounds)}
        for name, field in (("total_model_ms", "model_ms"), ("total_ttft_ms", "ttft_ms"),
                            ("input_tokens", "input_tokens"), ("output_tokens", "output_tokens")):
            planning[name] = (sum(row[field] for row in planning_rounds)
                              if all(row[field] is not None for row in planning_rounds) else None)
        tool_ms = sum(row["latency_ms"] for row in tool_records)
        planning["orchestration_ms"] = max(0.0, (time.perf_counter_ns() - prepare_started) / 1e6
                                           - planning["total_wall_ms"] - tool_ms - prompt_build_ms)
        return PreparedAnswer(final_messages, selected,
                              {"final_context": selected, "query_variants": [question],
                               "retrieval_backend": context.retrieval_metrics.get("backend", retrieval_backend),
                               "timings_ms": context.retrieval_metrics.get("timings_ms", {}),
                               "history_retrieval_executed": bool(context.retrieval_metrics),
                               "steering": view.steering, "mcp": context.mcp_metrics},
                              retrieval_ms, prompt_build_ms, model_calls,
                              tool_records, parse_failures, rounds, planning)
