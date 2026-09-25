"""Hybrid baseline: preserved retrieval followed by one vanilla 4B call."""

import asyncio
import time
from typing import Any

from app.rag.prompting import build_messages
from app.rag.schemas import PreparedAnswer
from app.rag.retriever import Retriever


class HybridRuntime:
    def __init__(self, retriever: Retriever, model: Any, attachment_retriever: Any = None):
        from app.config import HYBRID_MODEL_ID

        if model.model_id != HYBRID_MODEL_ID:
            raise ValueError("Hybrid must use vanilla Qwen3-4B-Instruct-2507")
        self.retriever, self.model = retriever, model
        self.attachment_retriever = attachment_retriever

    async def prepare(self, question: str, top_k: int, history: list[dict[str, str]],
                      *, trace: Any = None, owner_id: str | None = None,
                      conversation_id: str | None = None,
                      attachment_ids: tuple[str, ...] = (), **_: Any) -> PreparedAnswer:
        if trace:
            trace.mark("retrieval_started")
        retrieval_started = time.perf_counter_ns()
        retrieval = await asyncio.to_thread(self.retriever.retrieve, question, top_k)
        contexts = list(retrieval.get("final_context") or [])
        if attachment_ids and self.attachment_retriever is not None:
            attached = await asyncio.to_thread(
                self.attachment_retriever.retrieve, owner_id, conversation_id,
                question, top_k, attachment_ids,
            )
            contexts.extend({**chunk, "source_kind": "attachment"} for chunk in attached)
        retrieval_ms = (time.perf_counter_ns() - retrieval_started) / 1e6
        if trace:
            trace.mark("retrieval_finished")
        prompt_started = time.perf_counter_ns()
        messages = build_messages(question, contexts, history)
        prompt_build_ms = (time.perf_counter_ns() - prompt_started) / 1e6
        if trace:
            trace.mark("prompt_ready")
        return PreparedAnswer(messages, contexts, retrieval, retrieval_ms, prompt_build_ms)
