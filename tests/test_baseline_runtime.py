"""Baseline architecture and streaming tests using tiny fake model outputs."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from app.api.routes import _context_to_api, _model_metrics, _sources, chat_stream
from app.central.runtime import CentralRuntime
from app.chat.store import ConversationStore
from app.chat_modes import ChatMode, normalize_chat_mode
from app.config import CENTRAL_MODEL_ID, HYBRID_MODEL_ID
from app.models.base import ModelDelta, ModelDone
from app.models.qwen import QwenRuntime
from app.rag.hybrid_runtime import HybridRuntime
from app.rag.schemas import PreparedAnswer
from app.schemas import ChatRequest
from app.services.chat_mode_router import ChatModeRouter
from app.telemetry import RequestTrace
from app.tools.registry import ToolRegistry
from app.tools.local_search import SearchHistoryTool


class FakeModel:
    def __init__(self, model_id: str, *, fail: bool = False):
        self.model_id = model_id
        self.resolved_revision = "test-revision"
        self.generation_settings = {"do_sample": False, "enable_thinking": False,
                                    "adapter": None, "quantization": None}
        self.fail = fail
        self.calls = 0

    async def stream(self, messages, *, max_new_tokens, cancel):
        self.calls += 1
        started = time.perf_counter_ns()
        for delta in ("Chiến thắng ", "có ý nghĩa [c1]."):
            if cancel.is_set():
                return
            await asyncio.sleep(0)
            yield ModelDelta(delta, time.perf_counter_ns())
        if self.fail:
            raise RuntimeError("fake model failure")
        finished = time.perf_counter_ns()
        yield ModelDone(self.model_id, self.resolved_revision, started, started + 1,
                        finished, input_tokens=20, output_tokens=8)


class FakeRetriever:
    retrieval_config = {"final_context_k": 6}

    def __init__(self):
        self.calls = 0

    def retrieve(self, query: str, top_k: int):
        self.calls += 1
        return {"question": query, "final_context": [{"chunk_id": "c1", "source_id": "s1",
                "title": "Bạch Đằng", "text": "Chiến thắng năm 938."}],
                "query_variants": [query]}


def _app(tmp_path, *, fail=False):
    store = ConversationStore(tmp_path / "chat.sqlite3")
    retriever = FakeRetriever()
    model = FakeModel(HYBRID_MODEL_ID, fail=fail)
    hybrid = HybridRuntime(retriever, model)
    app = FastAPI()
    app.state.chat_store = store
    app.state.retriever = retriever
    app.state.chat_mode_router = ChatModeRouter(hybrid=hybrid, central=None)
    return app, store, model, retriever


async def _stream(app, store, *, fail=False):
    owner = "baseline-test-client"
    conversation = store.create_conversation(owner)

    async def connected():
        return False

    request = SimpleNamespace(app=app, is_disconnected=connected)
    payload = ChatRequest(conversation_id=conversation["id"], question="Ý nghĩa Bạch Đằng?", mode="hybrid")
    response = await chat_stream(payload, request, owner, store)
    body = "".join([frame async for frame in response.body_iterator])
    return conversation, response, _events(body)


def _events(body: str):
    events = []
    for frame in body.split("\n\n"):
        if not frame.strip():
            continue
        lines = frame.splitlines()
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")
        events.append((lines[0][7:], json.loads(lines[1][6:])))
    return events


def test_only_two_modes_and_model_ids():
    assert [mode.value for mode in ChatMode] == ["hybrid", "central"]
    for old in ("three_llm", "agentic_rag"):
        with pytest.raises(ValueError):
            normalize_chat_mode(old)
        with pytest.raises(ValidationError):
            ChatRequest.model_validate({"conversation_id": "00000000-0000-0000-0000-000000000001",
                                        "question": "Bạch Đằng?", "mode": old})
    retriever = FakeRetriever()
    assert HybridRuntime(retriever, FakeModel(HYBRID_MODEL_ID)).model.model_id == HYBRID_MODEL_ID
    assert CentralRuntime(model=FakeModel(CENTRAL_MODEL_ID), tools=ToolRegistry()).model.model_id == CENTRAL_MODEL_ID
    with pytest.raises(ValueError):
        HybridRuntime(retriever, FakeModel(CENTRAL_MODEL_ID))
    with pytest.raises(ValueError):
        CentralRuntime(model=FakeModel(HYBRID_MODEL_ID), tools=ToolRegistry())


def test_retrieval_does_not_call_model():
    retriever = FakeRetriever()
    model = FakeModel(HYBRID_MODEL_ID)
    prepared = asyncio.run(HybridRuntime(retriever, model).prepare("Bạch Đằng?", 6, []))
    assert prepared.contexts[0]["chunk_id"] == "c1"
    assert retriever.calls == 1
    assert model.calls == 0


def test_source_id_is_derived_without_changing_corpus_row():
    row = {"chunk_id": "c1", "metadata": {"source_sha1": "article-hash"}}
    assert _context_to_api(row, final_rank=2).source_id == "article-hash"
    assert _context_to_api(row, final_rank=2).display_index == 2
    assert "source_id" not in row


def test_source_id_citation_marks_the_retrieved_chunk():
    contexts = [{"chunk_id": "c1", "source_id": "s1", "title": "Bạch Đằng", "text": "Năm 938."}]
    sources, cited_ids = _sources("Chiến thắng [s1], năm [938].", contexts)
    assert cited_ids == ["c1"]
    assert sources[0]["cited"] is True
    assert contexts[0].get("cited") is None


def test_central_uses_same_history_retriever_as_tool():
    class PlanningModel(FakeModel):
        async def generate(self, messages, *, tools, max_new_tokens, cancel):
            self.calls += 1
            text = '<tool_call>{"name":"search_history","arguments":{"query":"Bạch Đằng","top_k":3}}</tool_call>'
            return text, None

    retriever = FakeRetriever()
    model = PlanningModel(CENTRAL_MODEL_ID)
    tools = ToolRegistry()
    tools.register(SearchHistoryTool(retriever))
    prepared = asyncio.run(CentralRuntime(model=model, tools=tools, max_action_rounds=1)
                           .prepare("Bạch Đằng?", 3, []))
    assert retriever.calls == 1
    assert model.calls == 1
    assert prepared.model_calls_before_final == 1
    assert prepared.tool_calls[0]["name"] == "search_history"
    assert prepared.contexts[0]["chunk_id"] == "c1"


def test_http_sse_deltas_reconstruct_stored_answer(tmp_path):
    app, store, model, _ = _app(tmp_path)
    conversation, response, events = asyncio.run(_stream(app, store))
    assert response.media_type == "text/event-stream"
    names = [name for name, _ in events]
    assert names[0] == "status"
    assert names.count("answer_delta") == 2
    assert names[-2:] == ["sources", "done"]
    answer = "".join(data["delta"] for name, data in events if name == "answer_delta")
    done = events[-1][1]
    saved = store.list_messages("baseline-test-client", conversation["id"])
    assert answer == saved[-1]["content"]
    assert done["model_id"] == HYBRID_MODEL_ID
    assert done["metrics"]["model_ttft_ms"] is not None
    assert done["metrics"]["retrieval_ms"] is not None
    assert done["metrics"]["model_calls"] == 1
    assert events[-2][1]["cited_source_ids"] == ["c1"]
    assert model.calls == 1


def test_model_failure_becomes_sse_error_without_assistant(tmp_path):
    app, store, _, _ = _app(tmp_path, fail=True)
    conversation, _, events = asyncio.run(_stream(app, store))
    assert [name for name, _ in events][-2:] == ["error", "done"]
    assert events[-1][1]["status"] == "error"
    assert len(store.list_messages("baseline-test-client", conversation["id"])) == 1


def test_metrics_use_monotonic_spans_and_leave_missing_null():
    trace = RequestTrace("req", "hybrid", started_ns=100)
    trace.mark("generation_started", 200)
    trace.mark("request_finished", 300)
    metrics = _model_metrics(None, trace, None, [])
    assert metrics["generation_start_ms"] == 0.0001
    assert metrics["model_ttft_ms"] is None
    assert metrics["retrieval_ms"] is None
    assert metrics["output_tokens"] is None


def test_qwen_worker_streams_and_stops_on_cancellation():
    import torch

    class TokenBatch(dict):
        def to(self, device):
            return self

    class TinyTokenizer:
        pad_token_id = 0
        eos_token_id = 99

        def apply_chat_template(self, messages, **kwargs):
            return "prompt"

        def __call__(self, prompt, **kwargs):
            return TokenBatch(input_ids=torch.tensor([[50]]))

        def decode(self, ids, **kwargs):
            return "".join({1: "Xin ", 2: "chào ", 3: "bạn. "}.get(int(i), "x ") for i in ids)

    class TinyModel:
        def __init__(self):
            self.stopped = threading.Event()

        def get_input_embeddings(self):
            return SimpleNamespace(weight=SimpleNamespace(device="cpu"))

        def generate(self, *, input_ids, streamer, stopping_criteria, **kwargs):
            streamer.put(input_ids)
            tokens = input_ids
            for index in range(30):
                value = (index % 3) + 1
                tokens = torch.cat((tokens, torch.tensor([[value]])), dim=1)
                streamer.put(torch.tensor([[value]]))
                if stopping_criteria(tokens, None):
                    break
                time.sleep(0.01)  # Simulated model decode work, never API pacing.
            streamer.end()
            self.stopped.set()

    runtime = QwenRuntime(model_id=HYBRID_MODEL_ID)
    runtime.tokenizer = TinyTokenizer()
    runtime.model = TinyModel()
    runtime.resolved_revision = "fake-revision"

    async def collect():
        cancel = threading.Event()
        stream = runtime.stream([{"role": "user", "content": "x"}], max_new_tokens=30, cancel=cancel)
        first = await anext(stream)
        assert isinstance(first, ModelDelta)
        assert not runtime.model.stopped.is_set()  # First delta precedes complete generation.
        await stream.aclose()
        assert cancel.is_set()
        assert runtime.model.stopped.wait(1.0)

    asyncio.run(collect())
