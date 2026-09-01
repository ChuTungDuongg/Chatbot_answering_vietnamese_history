from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import suppress
from types import SimpleNamespace

from app.agents.evidence_agent import EvidenceModelContractError
from app.api.routes import chat_stream
from app.chat.store import ConversationStore
from app.schemas import ChatRequest


class FakeService:
    loaded = True
    external_generation_backend = True
    model = None
    chunk_by_id = {}


class FakeRequest:
    def __init__(self, generator):
        self.app = SimpleNamespace(
            state=SimpleNamespace(
                rag_service=FakeService(),
                hybrid_runtime=generator,
            )
        )
        self.disconnected = False

    async def is_disconnected(self):
        return self.disconnected


class ContractFailingGenerator:
    max_history_messages = 6
    retrieval_history_messages = 4

    def __init__(self, *, delay_seconds: float = 0.0):
        self.delay_seconds = delay_seconds

    def chat(self, **kwargs):
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        raise EvidenceModelContractError(
            "claim under 'ev_01' is not grounded in that same evidence source",
            evidence_ids=["ev_01"],
            repair_attempted=True,
        )


def _store_with_conversation(tmp_path):
    owner_id = "test-client"
    store = ConversationStore(tmp_path / "chat.db")
    conversation = store.create_conversation(owner_id, title="Stream test")
    return store, owner_id, conversation["id"]


def _event_blocks(chunks: list[str]) -> list[tuple[str, dict]]:
    events = []
    for block in "".join(chunks).strip().split("\n\n"):
        event_name = ""
        data = None
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            if line.startswith("data:"):
                data = json.loads(line.removeprefix("data:").strip())
        if event_name and data is not None:
            events.append((event_name, data))
    return events


def test_stream_emits_typed_evidence_contract_error(caplog, tmp_path):
    async def run():
        store, owner_id, conversation_id = _store_with_conversation(tmp_path)
        payload = ChatRequest(
            conversation_id=conversation_id,
            question="Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?",
        )
        response = await chat_stream(
            payload,
            FakeRequest(ContractFailingGenerator()),
            owner_id,
            store,
        )
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)
        return _event_blocks(chunks)

    with caplog.at_level(logging.WARNING):
        events = asyncio.run(run())

    error = next(data for event, data in events if event == "error")
    assert error["type"] == "evidence_contract_error"
    assert error["stage"] == "evidence"
    assert error["code"] == "grounding_contract_failed"
    assert error["message"] == "Không thể hoàn tất câu trả lời do bước đánh giá bằng chứng thất bại."
    assert error["evidence_ids"] == ["ev_01"]
    assert error["repair_attempted"] is True
    assert error["validation_errors"] == []
    assert "Streaming evidence contract error:" in caplog.text
    assert "stage=evidence" in caplog.text
    assert "code=grounding_contract_failed" in caplog.text
    assert "repair_attempted=true" in caplog.text
    assert 'evidence_ids=["ev_01"]' in caplog.text
    assert "validation_errors=[]" in caplog.text
    assert events[-1][0] == "done"
    assert events[-1][1]["status"] == "error"


def test_stream_cancellation_consumes_background_task_exception(tmp_path):
    async def run():
        loop = asyncio.get_running_loop()
        unhandled = []
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _, context: unhandled.append(context))
        try:
            store, owner_id, conversation_id = _store_with_conversation(tmp_path)
            payload = ChatRequest(
                conversation_id=conversation_id,
                question="Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?",
            )
            response = await chat_stream(
                payload,
                FakeRequest(ContractFailingGenerator(delay_seconds=0.3)),
                owner_id,
                store,
            )
            iterator = response.body_iterator
            await iterator.__anext__()
            pending_read = asyncio.create_task(iterator.__anext__())
            await asyncio.sleep(0.05)
            pending_read.cancel()
            with suppress(asyncio.CancelledError):
                await pending_read
            with suppress(Exception):
                await iterator.aclose()
            await asyncio.sleep(0.4)
            return unhandled
        finally:
            loop.set_exception_handler(previous_handler)

    assert asyncio.run(run()) == []
