from __future__ import annotations

import asyncio
import json
import re
import time
from collections import Counter
from collections.abc import Callable
from typing import Any

from app.agents.central_model_runtime import CentralGeneration, CentralLLMBackend, CentralToolCall
from app.agents.central_prompt import CENTRAL_SYSTEM_PROMPT, is_analytical_question
from app.agents.central_tools import EXTERNAL_TOOLS, bounded_tool_arguments, normalize_tool_result, qwen_tool_schemas
from app.agents.config import CentralAgentConfig
from app.telemetry import current_request_telemetry
from app.tools.registry import ToolExecutionContext, ToolRegistry


INSUFFICIENT_EVIDENCE_ANSWER = (
    "Mình chưa tìm thấy đủ bằng chứng đáng tin cậy để trả lời câu hỏi này. "
    "Bạn có thể bổ sung tài liệu hoặc làm rõ giai đoạn, nhân vật hay sự kiện cần hỏi."
)
CITATION_RE = re.compile(r"\[([A-Za-z0-9_.:-]+)\]")


class CentralAgent:
    """One Qwen3-8B model that selects tools and writes its own final answer."""

    def __init__(
        self,
        *,
        model_runtime: CentralLLMBackend,
        tool_registry: ToolRegistry,
        config: CentralAgentConfig | None = None,
        has_uploaded_documents: Callable[[str, str], bool] | None = None,
    ):
        self.model_runtime = model_runtime
        self.tool_registry = tool_registry
        self.config = config or CentralAgentConfig()
        self.has_uploaded_documents = has_uploaded_documents
        self.max_history_messages = 6
        self.retrieval_history_messages = 4

    def _allowed_tools(self, owner_id: str | None, conversation_id: str | None) -> set[str]:
        names = set(self.tool_registry.names())
        allowed: set[str] = set()
        if self.config.enable_history and "search_history" in names:
            allowed.add("search_history")
        if self.config.enable_wikipedia:
            allowed.update(names & {"search_wikipedia", "fetch_wikipedia_page"})
        if self.config.enable_web:
            allowed.update(names & {"search_web", "fetch_web_page"})
        attachments_exist = False
        if (
            self.config.enable_documents
            and owner_id
            and conversation_id
            and self.has_uploaded_documents is not None
        ):
            try:
                attachments_exist = self.has_uploaded_documents(owner_id, conversation_id)
            except Exception:
                attachments_exist = False
        if attachments_exist and "search_uploaded_documents" in names:
            allowed.add("search_uploaded_documents")
        return allowed

    @staticmethod
    def _messages(question: str, history: list[dict[str, str]] | None) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": CENTRAL_SYSTEM_PROMPT}]
        for item in (history or [])[-6:]:
            role = str(item.get("role") or "")
            content = str(item.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": question})
        return messages

    @staticmethod
    def _assistant_tool_message(calls: tuple[CentralToolCall, ...]) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in calls
            ],
        }

    @staticmethod
    def _validated_citations(
        answer: str,
        source_by_id: dict[str, dict[str, Any]],
    ) -> tuple[str, list[str], list[str]]:
        mentioned = CITATION_RE.findall(answer)
        valid = list(dict.fromkeys(item for item in mentioned if item in source_by_id))
        invalid = list(dict.fromkeys(item for item in mentioned if item not in source_by_id))
        if invalid:
            invalid_set = set(invalid)
            answer = CITATION_RE.sub(lambda match: "" if match.group(1) in invalid_set else match.group(0), answer)
            answer = re.sub(r"[ \t]{2,}", " ", answer).strip()
        return answer, valid, invalid

    async def _run(
        self,
        *,
        question: str,
        history: list[dict[str, str]] | None,
        owner_id: str | None,
        conversation_id: str | None,
        request_id: str | None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        allowed_tools = self._allowed_tools(owner_id, conversation_id)
        schemas = qwen_tool_schemas(self.tool_registry, allowed_tools)
        messages = self._messages(question, history)
        context = ToolExecutionContext(
            owner_id=owner_id,
            conversation_id=conversation_id,
            request_id=request_id,
            session_id=request_id or conversation_id or "central",
        )
        seen_calls: set[str] = set()
        source_by_id: dict[str, dict[str, Any]] = {}
        traces: list[dict[str, Any]] = []
        tool_counts: Counter[str] = Counter()
        model_calls = 0
        generation_ms = 0.0
        tool_ms = 0.0
        input_tokens = 0
        output_tokens = 0
        observation_chars = 0
        external_results_count = 0
        final_answer = ""
        repair_attempted = False
        repair_used = False

        while model_calls < self.config.max_steps:
            generation: CentralGeneration = await asyncio.to_thread(
                self.model_runtime.generate,
                messages=messages,
                tools=schemas,
                max_new_tokens=self.config.max_new_tokens,
            )
            model_calls += 1
            generation_ms += generation.generation_ms
            input_tokens += generation.input_tokens
            output_tokens += generation.output_tokens
            if not generation.tool_calls:
                final_answer = generation.content.strip()
                if not final_answer and not repair_attempted and model_calls < self.config.max_steps:
                    repair_attempted = True
                    messages.append({
                        "role": "user",
                        "content": "Output trước rỗng hoặc sai protocol. Hãy trả lời hoặc phát đúng một tool call hợp lệ.",
                    })
                    continue
                break

            messages.append(self._assistant_tool_message(generation.tool_calls))
            prepared: list[tuple[CentralToolCall, dict[str, Any], str | None, int | None]] = []
            pending: list[tuple[str, dict[str, Any]]] = []
            for call in generation.tool_calls:
                arguments = bounded_tool_arguments(call.name, call.arguments, max_results=self.config.max_tool_results)
                signature = json.dumps([call.name, arguments], ensure_ascii=False, sort_keys=True)
                if signature in seen_calls:
                    prepared.append((call, arguments, "duplicate_tool_call_prevented", None))
                elif call.name not in allowed_tools:
                    prepared.append((call, arguments, "tool_not_available", None))
                else:
                    seen_calls.add(signature)
                    prepared.append((call, arguments, None, len(pending)))
                    pending.append((call.name, arguments))

            async def execute_tool(name: str, arguments: dict[str, Any]):
                call_started = time.perf_counter()
                result, record = await self.tool_registry.call(name, arguments, context=context)
                return result, record, (time.perf_counter() - call_started) * 1000

            executed = await asyncio.gather(*(
                execute_tool(name, arguments) for name, arguments in pending
            )) if pending else []
            if executed:
                tool_ms += max(item[2] for item in executed)

            for call, arguments, immediate_error, pending_index in prepared:
                sources: list[dict[str, Any]] = []
                if immediate_error is not None:
                    detail = (
                        immediate_error if immediate_error == "duplicate_tool_call_prevented"
                        else f"tool_not_available: {call.name}"
                    )
                    observation = json.dumps({"error": detail})
                    traces.append({"name": call.name, "arguments": arguments, "error": immediate_error})
                else:
                    assert pending_index is not None
                    result, record, elapsed = executed[pending_index]
                    tool_counts[call.name] += 1
                    remaining = max(0, self.config.observation_char_budget - observation_chars)
                    if record.error:
                        observation = json.dumps({"error": record.error}, ensure_ascii=False)
                    else:
                        observation, sources = normalize_tool_result(
                            call.name,
                            result,
                            max_results=self.config.max_tool_results,
                            char_budget=remaining,
                        )
                        for source in sources:
                            source_id = str(source["chunk_id"])
                            existing = source_by_id.get(source_id)
                            if existing is None or len(str(source.get("text") or "")) > len(str(existing.get("text") or "")):
                                source_by_id[source_id] = source
                        if call.name in EXTERNAL_TOOLS:
                            external_results_count += len(sources)
                    traces.append({
                        "name": call.name,
                        "arguments": arguments,
                        "result_count": record.result_count,
                        "error": record.error,
                        "latency_ms": elapsed,
                        "source_ids": [str(source["chunk_id"]) for source in sources],
                    })
                remaining_observation_chars = max(0, self.config.observation_char_budget - observation_chars)
                observation = observation[:remaining_observation_chars]
                observation_chars += len(observation)
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": observation})

        if (
            final_answer
            and is_analytical_question(question)
            and source_by_id
            and len(final_answer.split()) < 120
            and model_calls < self.config.max_steps
            and not repair_attempted
        ):
            repair_attempted = True
            messages.extend([
                {"role": "assistant", "content": final_answer},
                {
                    "role": "user",
                    "content": (
                        "Câu trả lời phân tích quá ngắn. Hãy viết lại sâu hơn từ đúng các bằng chứng đã có, "
                        "nêu nhiều chiều ý nghĩa/hệ quả phù hợp và giữ nguyên các ID trích dẫn hợp lệ."
                    ),
                },
            ])
            repaired = await asyncio.to_thread(
                self.model_runtime.generate,
                messages=messages,
                tools=schemas,
                max_new_tokens=self.config.max_new_tokens,
            )
            model_calls += 1
            generation_ms += repaired.generation_ms
            input_tokens += repaired.input_tokens
            output_tokens += repaired.output_tokens
            if repaired.content.strip() and not repaired.tool_calls:
                final_answer = repaired.content.strip()
                repair_used = True

        status = "ok" if final_answer else "insufficient_evidence"
        if not final_answer:
            final_answer = INSUFFICIENT_EVIDENCE_ANSWER
        final_answer, source_ids, invalid_source_ids = self._validated_citations(final_answer, source_by_id)
        elapsed_ms = (time.perf_counter() - started) * 1000
        telemetry = current_request_telemetry()
        if telemetry is not None:
            telemetry.tool_calls += sum(tool_counts.values())
            for name, count in tool_counts.items():
                telemetry.tool_calls_by_type[name] = telemetry.tool_calls_by_type.get(name, 0) + count
            telemetry.external_results_count += external_results_count
            telemetry.external_tools_called.extend(
                name for name in tool_counts if name in EXTERNAL_TOOLS and name not in telemetry.external_tools_called
            )
            telemetry.central_tool_ms += tool_ms
            telemetry.central_external_results_count += external_results_count

        provenance = {
            "mode": "central",
            "source": "central_qwen3_8b",
            "central_model_id": self.model_runtime.model_id,
            "central_adapter_loaded": bool(self.model_runtime.adapter_loaded),
            "model_placement": dict(getattr(self.model_runtime, "placement", {}) or {}),
            "central_model_calls": model_calls,
            "central_tool_calls": sum(tool_counts.values()),
            "central_tool_calls_by_type": dict(tool_counts),
            "central_external_results_count": external_results_count,
            "research_generation_calls": 0,
            "evidence_generation_calls": 0,
            "history_generation_calls": 0,
            "total_llm_calls": model_calls,
            "repair_generation_used": repair_used,
            "repair_generation_attempted": repair_attempted,
        }
        return {
            "question": question,
            "answer": final_answer,
            "status": status,
            "source_ids": source_ids,
            "source_chunks": list(source_by_id.values()),
            "model_source_ids": source_ids,
            "invalid_source_ids": invalid_source_ids,
            "retrieval": {
                "question": question,
                "final_context": list(source_by_id.values()),
                "tool_trace": [f"central:{item['name']}" for item in traces],
            },
            "analysis": {"question": question, "analytical": is_analytical_question(question)},
            "tool_trace": [f"central:{item['name']}" for item in traces],
            "central_debug": {"tools": traces, "allowed_tools": sorted(allowed_tools)},
            "agentic": True,
            "inference_mode": "central",
            "answer_provenance": provenance,
            "performance_debug": {
                "central_model_calls": model_calls,
                "central_tool_calls": sum(tool_counts.values()),
                "central_tool_calls_by_type": dict(tool_counts),
                "central_generation_ms": generation_ms,
                "central_tool_ms": tool_ms,
                "central_total_latency_ms": elapsed_ms,
                "central_input_tokens": input_tokens,
                "central_output_tokens": output_tokens,
                "central_external_results_count": external_results_count,
                "research_generation_calls": 0,
                "evidence_generation_calls": 0,
                "history_generation_calls": 0,
            },
            "latency_sec": elapsed_ms / 1000,
            "total_latency_sec": elapsed_ms / 1000,
        }

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        question = str(kwargs.get("question") or "").strip()
        try:
            return await asyncio.wait_for(
                self._run(
                    question=question,
                    history=kwargs.get("history"),
                    owner_id=kwargs.get("owner_id"),
                    conversation_id=kwargs.get("conversation_id"),
                    request_id=kwargs.get("request_id"),
                ),
                timeout=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            return {
                "question": question,
                "answer": INSUFFICIENT_EVIDENCE_ANSWER,
                "status": "insufficient_evidence",
                "source_ids": [],
                "source_chunks": [],
                "retrieval": {"question": question, "final_context": [], "tool_trace": []},
                "analysis": {"question": question},
                "tool_trace": [],
                "agentic": True,
                "inference_mode": "central",
                "answer_provenance": {
                    "mode": "central", "source": "central_timeout",
                    "research_generation_calls": 0, "evidence_generation_calls": 0,
                    "history_generation_calls": 0,
                },
            }

    def chat(
        self,
        question: str,
        final_k: int | None = None,
        history: list[dict[str, str]] | None = None,
        owner_id: str | None = None,
        conversation_id: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        del final_k
        return asyncio.run(self.run(
            question=question,
            history=history,
            owner_id=owner_id,
            conversation_id=conversation_id,
            request_id=request_id,
        ))
