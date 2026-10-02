"""Central planning and tools followed by a genuinely streamed final answer."""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from app.models.tool_calls import HermesFunctionCallCodec
from app.rag.prompting import build_messages
from app.rag.schemas import PreparedAnswer
from app.tools.registry import ToolExecutionContext, ToolRegistry


class CentralRuntime:
    def __init__(self, *, model: Any, tools: ToolRegistry, max_action_rounds: int = 2,
                 action_max_new_tokens: int = 256):
        from app.config import CENTRAL_MODEL_ID

        if model.model_id != CENTRAL_MODEL_ID:
            raise ValueError("Central must use vanilla Qwen3-8B")
        self.model = model
        self.tools = tools
        self.max_action_rounds = max_action_rounds
        self.action_max_new_tokens = action_max_new_tokens
        self.codec = HermesFunctionCallCodec()

    def _tool_schemas(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {
            "name": item["name"], "description": item["description"],
            "parameters": item["input_schema"],
        }} for item in self.tools.describe()]

    async def prepare(self, question: str, top_k: int, history: list[dict[str, str]],
                      *, owner_id: str | None = None, conversation_id: str | None = None,
                      attachment_ids: tuple[str, ...] = (), trace: Any = None,
                      cancel: threading.Event | None = None) -> PreparedAnswer:
        cancel = cancel or threading.Event()
        messages: list[dict[str, Any]] = [{
            "role": "system",
            "content": ("Bạn là trợ lý lịch sử Việt Nam. Chọn công cụ để tìm bằng chứng trước "
                        "khi trả lời. Ưu tiên search_history cho câu hỏi lịch sử. "
                        "Chỉ tạo tool_call hợp lệ; không trả lời cuối ở bước này."),
        }]
        messages.extend({"role": item["role"], "content": str(item.get("content") or "")[:1500]}
                        for item in history[-4:] if item.get("role") in {"user", "assistant"})
        messages.append({"role": "user", "content": question})
        context = ToolExecutionContext(owner_id=owner_id, conversation_id=conversation_id,
                                       attachment_ids=attachment_ids or None,
                                       request_id=trace.request_id if trace else None)
        contexts_by_id: dict[str, dict[str, Any]] = {}
        tool_records: list[dict[str, Any]] = []
        retrieval_ms = 0.0
        parse_failures = 0
        model_calls = 0
        rounds = 0

        if trace:
            trace.mark("retrieval_started")
        for _ in range(self.max_action_rounds):
            if cancel.is_set():
                raise RuntimeError("Request cancelled")
            rounds += 1
            plan_text, _ = await self.model.generate(
                messages, tools=self._tool_schemas(), max_new_tokens=self.action_max_new_tokens,
                cancel=cancel,
            )
            model_calls += 1
            parsed = self.codec.decode(plan_text)
            parse_failures += parsed.failures
            if not parsed.tool_calls:
                break
            messages.append({"role": "assistant", "content": plan_text})
            for call in parsed.tool_calls[:4]:
                if cancel.is_set():
                    raise RuntimeError("Request cancelled")
                started = time.perf_counter_ns()
                output, record = await self.tools.call(call.name, call.arguments, context=context)
                elapsed = (time.perf_counter_ns() - started) / 1e6
                if call.name == "search_history":
                    retrieval_ms += elapsed
                tool_records.append({"name": call.name, "latency_ms": elapsed,
                                     "error": record.error, "result_count": record.result_count})
                if isinstance(output, list):
                    for chunk in output:
                        if isinstance(chunk, dict) and chunk.get("chunk_id"):
                            contexts_by_id.setdefault(str(chunk["chunk_id"]), chunk)
                observation = json.dumps(output, ensure_ascii=False, default=str)
                messages.append({"role": "tool", "name": call.name, "content": observation[:8000]})

        if not any(item["name"] == "search_history" for item in tool_records):
            started = time.perf_counter_ns()
            output, record = await self.tools.call(
                "search_history", {"query": question, "top_k": top_k}, context=context
            )
            elapsed = (time.perf_counter_ns() - started) / 1e6
            retrieval_ms += elapsed
            tool_records.append({"name": "search_history", "latency_ms": elapsed,
                                 "error": record.error, "result_count": record.result_count})
            if isinstance(output, list):
                for chunk in output:
                    if isinstance(chunk, dict) and chunk.get("chunk_id"):
                        contexts_by_id.setdefault(str(chunk["chunk_id"]), chunk)

        if trace:
            trace.mark("retrieval_finished")
        selected = list(contexts_by_id.values())[:top_k]
        prompt_started = time.perf_counter_ns()
        final_messages = build_messages(question, selected, history)
        prompt_build_ms = (time.perf_counter_ns() - prompt_started) / 1e6
        if trace:
            trace.mark("prompt_ready")
        return PreparedAnswer(final_messages, selected,
                              {"final_context": selected, "query_variants": [question]},
                              retrieval_ms, prompt_build_ms, model_calls,
                              tool_records, parse_failures, rounds)
