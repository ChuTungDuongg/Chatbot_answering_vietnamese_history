"""Central planning and tools followed by a genuinely streamed final answer."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import threading
import time
from typing import Any

from app.models.tool_calls import HermesFunctionCallCodec
from app.models.base import ModelDone
from app.prompts.loader import load_prompt
from app.central.budgets import AgentDeadline, AgentLimits, bounded_wait
from app.central.execution import AgentExecutor, merge_tool_metrics
from app.central.planning import (AGENTIC_PLANNER_INSTRUCTION, final_after_tools, tool_evidence, select_evidence,
                                  evidence_kind, question_domain, terminal_evidence, planner_tool_schemas, COMPLETION_FIELD, INDEPENDENT_FIELD)
from app.rag.prompting import build_messages
from app.rag.schemas import PreparedAnswer
from app.tools.registry import ToolExecutionContext, ToolRegistry
from app.tools.policy import tool_view
from app.rag.progress import StageProgress, ProgressCallback
from app.security.guardrails import guard_question, safe_contexts, safe_observation


EXTERNAL_DOMAIN_INSTRUCTION = load_prompt("central/external_domain")
GROUNDED_FINAL_INSTRUCTION = load_prompt("central/grounded_final")
EXECUTION_LIMIT_INSTRUCTION = load_prompt("central/execution_limit")
EMPTY_PLAN_RECOVERY = load_prompt("central/recover_empty_plan")


class CentralRuntime:
    def __init__(self, *, model: Any, tools: ToolRegistry, max_action_rounds: int = 4,
                 action_max_new_tokens: int = 256, mcp_manager=None,
                 max_mcp_tools: int = 8, mcp_schema_budget: int = 16384, limits: AgentLimits | None = None,
                 planning_reasoning_effort: str = "low"):
        from app.config import CENTRAL_MODEL_ID

        from app.models.openai_responses import OPENAI_MODELS, REASONING_EFFORTS
        if planning_reasoning_effort not in REASONING_EFFORTS | {"inherit"}:
            raise ValueError("Unsupported Central planning reasoning effort")
        self.planning_reasoning_effort = planning_reasoning_effort
        if not (model.model_id == CENTRAL_MODEL_ID or
                (getattr(model, "inference_backend", None) == "openai" and model.model_id in OPENAI_MODELS)):
            raise ValueError("Central must use vanilla Qwen3-8B")
        if not 0 <= max_action_rounds <= 8:
            raise ValueError("Central planning rounds must be between 0 and 8")
        self.limits = limits or AgentLimits()
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
        guard_question(question)
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
            "content": AGENTIC_PLANNER_INSTRUCTION,
        }]
        if domain == "external":
            messages[0]["content"] += EXTERNAL_DOMAIN_INSTRUCTION
        messages.extend({"role": item["role"], "content": str(item.get("content") or "")[:1500]}
                        for item in history[-4:] if item.get("role") in {"user", "assistant"})
        messages.append({"role": "user", "content": question})
        context = ToolExecutionContext(owner_id=owner_id, conversation_id=conversation_id,
                                       attachment_ids=attachment_ids or None,
                                       request_id=trace.request_id if trace else None,
                                       retrieval_backend=retrieval_backend, progress=progress,
                                       allowed_tools=view.allowed_names, cancel=cancel,
                                       mcp_failure_policy=view.failure_policy)
        executor = AgentExecutor(self.tools, context, self.limits, cancel=cancel, question=question)
        stages = StageProgress(progress, retrieval_backend)
        contexts_by_id: dict[str, dict[str, Any]] = {}
        tool_records: list[dict[str, Any]] = []
        evidence_groups, observed_evidence = [], []
        observation_chars = 0
        observation_error_chars = 0
        output_tokens = input_tokens = 0
        observation_exhausted = False
        retrieval_ms = 0.0
        parse_failures = 0
        model_calls = 0
        rounds = 0
        planning_rounds: list[dict[str, Any]] = []
        exit_reason = "max_action_rounds"
        native = getattr(self.model, "inference_backend", None) == "openai"
        final_effort = self.model.options.reasoning_effort if native else None
        planner_effort = (final_effort if self.planning_reasoning_effort == "inherit"
                          else self.planning_reasoning_effort) if native else None

        def ingest(outcomes, *, append_outputs=True):
            nonlocal retrieval_ms, observation_chars, observation_error_chars, observation_exhausted
            usable = True
            for outcome in outcomes:
                call, record = outcome.call, outcome.record
                merge_tool_metrics(context, outcome.context)
                if call.name == "search_history":
                    retrieval_ms += outcome.latency_ms
                tool_records.append({"name": call.name, "call_id": call.id, "round": rounds,
                    "latency_ms": outcome.latency_ms, "error": record.error,
                    "result_count": record.result_count, "provider": record.provider, "server": record.server,
                    "execution_mode": outcome.execution_mode, "batch": outcome.batch,
                    **(outcome.context.mcp_metrics[-1] if outcome.context is not context
                       and not outcome.context.cancel.is_set() and outcome.context.mcp_metrics else {})})
                chunks = safe_contexts(tool_evidence(call.name, outcome.output)) if not record.error else []
                group = []
                for chunk in chunks:
                    key = str(chunk["chunk_id"])
                    group.append(key)
                    if key not in contexts_by_id or call.name in {"fetch_wikipedia_page", "fetch_web_page"}:
                        contexts_by_id[key] = chunk
                if group:
                    evidence_groups.append(group)
                usable = usable and not record.error and terminal_evidence(call.name, chunks, view.allowed_names)
                remaining = self.limits.total_observation_chars - observation_chars
                if remaining < 256:
                    observation_exhausted = True
                    observation = safe_observation([], "observation_budget")
                    # Close every accepted native call even after the evidence
                    # budget. This fixed, bounded protocol overhead is separate.
                    observation_error_chars += len(observation)
                else:
                    observation = safe_observation(chunks, record.error,
                        max_chars=min(self.limits.observation_chars, remaining))
                    observation_chars += len(observation)
                    observed_evidence.extend(json.loads(observation).get("untrusted_evidence", []))
                if append_outputs:
                    if native:
                        messages.append({"type": "function_call_output", "call_id": call.id, "output": observation})
                    else:
                        messages.append({"role": "tool", "name": call.name, "content": observation})
            return usable

        if trace:
            trace.mark("retrieval_started")
        required_kinds = (("history", "external") if domain == "mixed" else
                          ("external",) if external_required else ()) + (("attachment",) if attachment_ids else ())
        def selected_evidence():
            return select_evidence(evidence_groups, contexts_by_id, top_k, required_kinds)

        def covers_scope(selected):
            return all(any(evidence_kind(row, kind) for row in selected) for kind in required_kinds)

        action_round_limit = self.max_action_rounds
        if native and not planner_schemas:
            exit_reason, action_round_limit = "no_tools_available", 0
        elif (native and domain == "external" and not external_names
              and not (attachment_ids and "search_uploaded_documents" in view.allowed_names)):
            exit_reason, action_round_limit = "no_external_tools", 0

        for _ in range(action_round_limit):
            if cancel.is_set():
                raise asyncio.CancelledError()
            if time.monotonic() >= executor.deadline:
                exit_reason = "execution_deadline"
                break
            if (output_tokens >= self.limits.planning_output_tokens or input_tokens >= self.limits.planning_input_tokens):
                exit_reason = "planning_token_budget"
                break
            rounds += 1
            model_calls += 1
            round_started = time.perf_counter_ns()
            planning_cancel = threading.Event()
            try:
                with stages.track("tool_selection"):
                    remaining_tokens = self.limits.planning_output_tokens - output_tokens
                    if native:
                        parsed, native_output, plan_done = await bounded_wait(self.model.plan(
                            messages, tools=planner_schemas, cancel=planning_cancel,
                            max_output_tokens=remaining_tokens, reasoning_effort=planner_effort,
                            tool_choice="auto" if contexts_by_id else "required"), cancel=cancel,
                            deadline=executor.deadline, on_abort=planning_cancel.set)
                    else:
                        plan_text, plan_done = await bounded_wait(self.model.generate(
                            messages, tools=planner_schemas,
                            max_new_tokens=min(self.action_max_new_tokens, remaining_tokens), cancel=planning_cancel),
                            cancel=cancel, deadline=executor.deadline, on_abort=planning_cancel.set)
                        parsed = self.codec.decode(plan_text)
            except AgentDeadline:
                planning_rounds.append({"round": rounds,
                    "wall_ms": (time.perf_counter_ns() - round_started) / 1e6,
                    "model_ms": None, "ttft_ms": None, "input_tokens": None, "output_tokens": None,
                    "reasoning_tokens": None, "cached_input_tokens": None, "early_exit": False,
                    "reasoning_effort": planner_effort,
                    "tools_requested": [], "error": "execution_deadline"})
                exit_reason = "execution_deadline"
                break
            # Refuse an unbounded/malformed transcript before accepting any call.
            if len(parsed.tool_calls) > 64:
                raise RuntimeError("Planner returned too many tool calls.")
            round_finished = time.perf_counter_ns()
            parse_failures += parsed.failures
            wall_ms = (round_finished - round_started) / 1e6
            model_metrics = plan_done.metrics if isinstance(plan_done, ModelDone) else {}
            model_ms = model_metrics.get("generation_ms")
            output_tokens += model_metrics.get("output_tokens") or 0
            input_tokens += model_metrics.get("input_tokens") or 0
            round_metrics = {"round": rounds, "wall_ms": wall_ms,
                             "reasoning_effort": planner_effort,
                             "started_ms": (round_started - (trace.started_ns if trace else prepare_started)) / 1e6,
                             "finished_ms": (round_finished - (trace.started_ns if trace else prepare_started)) / 1e6,
                             "model_ms": model_ms, "ttft_ms": model_metrics.get("model_ttft_ms"),
                             "preparation_and_queue_ms": max(0.0, wall_ms - model_ms) if model_ms is not None else None,
                             "input_tokens": model_metrics.get("input_tokens"),
                             "output_tokens": model_metrics.get("output_tokens"),
                             "reasoning_tokens": model_metrics.get("reasoning_tokens"),
                             "cached_input_tokens": model_metrics.get("cached_input_tokens"),
                             "finish_reason": model_metrics.get("finish_reason"),
                             "tools_requested": [call.name for call in parsed.tool_calls],
                             "parse_failures": parsed.failures, "early_exit": False}
            if native:
                round_metrics["tool_choice"] = "auto" if contexts_by_id else "required"
            planning_rounds.append(round_metrics)
            if not parsed.tool_calls:
                if native:
                    messages.extend(native_output)
                    refused = any(part.get("type") == "refusal" for item in native_output
                                  for part in item.get("content", []) if isinstance(part, dict))
                    if refused:
                        exit_reason = "planner_refused"
                        break
                    # One bounded repair for a premature first-round text response.
                    # Later stops after actual evidence retain their existing meaning.
                    if rounds == 1 and not contexts_by_id and rounds < action_round_limit:
                        round_metrics["replan_reason"] = "planner_no_tools_without_evidence"
                        messages.append({"role": "system", "content": EMPTY_PLAN_RECOVERY})
                        continue
                exit_reason = "planner_no_tools"
                break
            if native:
                messages.extend(native_output)
            else:
                messages.append({"role": "assistant", "content": plan_text})
            completion_declared = (final_after_tools(parsed.content) if "<plan_status>" in parsed.content else
                                   all(call.name in controlled_tools and call.arguments.get(COMPLETION_FIELD) is True
                                       for call in parsed.tool_calls))
            round_metrics["completion_declared"] = completion_declared
            calls, independent = [], set()
            for index, call in enumerate(parsed.tool_calls):
                error = call.error
                arguments = dict(call.arguments)
                if call.name in controlled_tools:
                    if any(key in arguments and type(arguments[key]) is not bool
                           for key in (COMPLETION_FIELD, INDEPENDENT_FIELD)):
                        error = "invalid_planner_control"
                    if arguments.pop(INDEPENDENT_FIELD, False) is True:
                        independent.add(index)
                    arguments.pop(COMPLETION_FIELD, None)
                calls.append(replace(call, arguments=arguments, error=error))
            outcomes = await executor.execute(calls, independent=independent, evidence=observed_evidence,
                blocked=frozenset({"search_history"}) if external_required else frozenset())
            complete_evidence = ingest(outcomes) and parsed.failures == 0
            round_metrics.update({"tool_calls": len(outcomes),
                "tools_executed": sum(row.execution_mode != "rejected" for row in outcomes),
                "execution_modes": [row.execution_mode for row in outcomes],
                "tool_errors": [row.record.error for row in outcomes if row.record.error],
                "evidence_chunks": len(contexts_by_id)})
            # No implicit "nonempty result means sufficient" shortcut: the planner
            # must declare full coverage and no pending dependencies, and actual
            # outputs must be successful, usable evidence. Missing/truncated plans,
            # snippets awaiting a reader, failures and mixed-domain gaps replan.
            if (complete_evidence and completion_declared and not observation_exhausted
                    and model_metrics.get("finish_reason") != "length"
                    and covers_scope(selected_evidence())):
                round_metrics["early_exit"] = True
                exit_reason = "evidence_sufficient"
                break
            round_metrics["replan_reason"] = ("tool_failure" if any(row.record.error for row in outcomes)
                else "evidence_incomplete" if not complete_evidence else "planner_requires_more")
            if all(row.record.error == "repeated_call" for row in outcomes):
                exit_reason = "repeated_calls"
                break
            if time.monotonic() >= executor.deadline:
                exit_reason = "execution_deadline"
                break
            if observation_exhausted:
                exit_reason = "observation_budget"
                break
            if executor.calls >= self.limits.total_calls:
                exit_reason = "total_tool_budget"
                break

        # Compatibility fallback uses the same permissions, deadline and call budgets.
        # It never changes the selected dense backend or supplies orphaned native outputs.
        fallback = None
        if exit_reason not in {"evidence_sufficient", "execution_deadline", "planning_token_budget",
                               "observation_budget", "total_tool_budget", "planner_refused"}:
            from app.models.tool_calls import DecodedFunctionCall
            if (domain != "external" and "search_history" in view.allowed_names
                    and not any(item["name"] == "search_history" for item in tool_records)):
                fallback = DecodedFunctionCall("fallback_history", "search_history", {"query": question, "top_k": top_k})
            elif (domain == "external" and "search_wikipedia" in view.allowed_names
                    and not any(item["name"] in {"search_wikipedia", "fetch_wikipedia_page"} for item in tool_records)):
                fallback = DecodedFunctionCall("fallback_wikipedia", "search_wikipedia", {"query": question, "top_k": min(top_k, 10)})
        if fallback and time.monotonic() < executor.deadline:
            ingest(await executor.execute([fallback], independent=set(), evidence=observed_evidence), append_outputs=False)
        if trace:
            trace.mark("retrieval_finished")
        selected = selected_evidence()
        if selected and not covers_scope(selected) and exit_reason != "planner_refused":
            exit_reason = "selected_evidence_incomplete"

        prompt_started = time.perf_counter_ns()
        with stages.track("prompt_preparation"):
            final_messages = build_messages(question, selected, history)
            if native and model_calls:
                # Keep the exact planner prefix and every returned reasoning
                # and function item, including the last round's tool outputs.
                # The final instruction limits claims to selected evidence.
                final_messages = [
                    *messages,
                    {"role": "system", "content":
                     GROUNDED_FINAL_INSTRUCTION
                     + final_messages[0]["content"]},
                    final_messages[-1],
                ]
            if (exit_reason not in {"evidence_sufficient", "planner_no_tools"}
                    or (exit_reason == "planner_no_tools" and any(row["error"] for row in tool_records))):
                if native and model_calls:
                    final_messages[-2]["content"] += " " + EXECUTION_LIMIT_INSTRUCTION
                else:
                    final_messages.insert(-1, {"role": "system", "content": EXECUTION_LIMIT_INSTRUCTION})
        prompt_build_ms = (time.perf_counter_ns() - prompt_started) / 1e6
        if trace:
            trace.mark("prompt_ready")
        planning = {"rounds": planning_rounds, "exit_reason": exit_reason, "domain": domain,
                    "effective_tools": sorted(view.allowed_names), "evidence_context_count": len(selected),
                    "empty_plan_recoveries": sum(row.get("replan_reason") == "planner_no_tools_without_evidence"
                                                 for row in planning_rounds),
                    "reasoning_effort": planner_effort, "final_reasoning_effort": final_effort,
                    "total_wall_ms": sum(row["wall_ms"] for row in planning_rounds),
                    "tools_executed": executor.calls, "repeats_avoided": executor.repeats_avoided,
                    "tool_wall_ms": executor.wall_ms, "observation_chars": observation_chars,
                    "observation_error_chars": observation_error_chars,
                    "observation_total_chars": observation_chars + observation_error_chars,
                    "parallel_batches": len({row["batch"] for row in tool_records if row["execution_mode"] == "parallel"}),
                    "replanning_events": max(0, len(planning_rounds) - 1),
                    "unknown_usage_rounds": sum(row["output_tokens"] is None for row in planning_rounds)}
        for name, field in (("total_model_ms", "model_ms"), ("total_ttft_ms", "ttft_ms"),
                            ("input_tokens", "input_tokens"), ("output_tokens", "output_tokens"),
                            ("reasoning_tokens", "reasoning_tokens"), ("cached_input_tokens", "cached_input_tokens")):
            planning[name] = (sum(row[field] for row in planning_rounds)
                              if all(row[field] is not None for row in planning_rounds) else None)
        planning["orchestration_ms"] = max(0.0, (time.perf_counter_ns() - prepare_started) / 1e6
                                           - planning["total_wall_ms"] - executor.wall_ms - prompt_build_ms)
        return PreparedAnswer(final_messages, selected,
                              {"final_context": selected, "query_variants": [question],
                               "retrieval_backend": context.retrieval_metrics.get("backend", retrieval_backend),
                               "timings_ms": context.retrieval_metrics.get("timings_ms", {}),
                               "history_retrieval_executed": bool(context.retrieval_metrics),
                               "steering": view.steering, "mcp": context.mcp_metrics},
                              retrieval_ms, prompt_build_ms, model_calls,
                              tool_records, parse_failures, rounds, planning)
