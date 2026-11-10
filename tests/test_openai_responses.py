import asyncio
import copy
import json
import threading
from types import SimpleNamespace as NS

import pytest

from app.models.base import ModelDelta, ModelDone
from app.models.openai_responses import OpenAIOptions, OpenAIProviderError, OpenAIResponsesRuntime
from app.services.openai_access import OpenAIAccessGuard


class Stream:
    def __init__(self, events):
        self.events = iter(events)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.events)
        except StopIteration:
            raise StopAsyncIteration

    async def close(self):
        self.closed = True


class Client:
    def __init__(self, *streams):
        self.streams = list(streams)
        self.requests = []
        self.responses = self

    async def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        return self.streams.pop(0)


def complete(output=(), **usage):
    return NS(type="response.completed", response=NS(status="completed", output=output,
        usage=NS(input_tokens=usage.get("input_tokens", 12), output_tokens=usage.get("output_tokens", 5),
                 input_tokens_details=NS(cached_tokens=3), output_tokens_details=NS(reasoning_tokens=2))))


def run(coro):
    return asyncio.run(coro)


def test_visible_stream_and_usage_are_mapped_without_reasoning():
    stream = Stream([NS(type="response.reasoning_summary_text.delta", delta="private"),
                     NS(type="response.output_text.delta", delta="Xin "),
                     NS(type="response.output_text.delta", delta="chào [c1]"),
                     complete()])
    client = Client(stream)
    runtime = OpenAIResponsesRuntime(client, OpenAIOptions("gpt-6.1-terra", "xhigh", 25000))

    async def collect():
        return [item async for item in runtime.stream([], max_new_tokens=10, cancel=threading.Event())]
    items = run(collect())
    assert [item.text for item in items if isinstance(item, ModelDelta)] == ["Xin ", "chào [c1]"]
    assert isinstance(items[-1], ModelDone)
    assert items[-1].reasoning_tokens == 2 and items[-1].cached_input_tokens == 3
    assert client.requests[0]["reasoning"] == {"effort": "xhigh"}
    assert client.requests[0]["max_output_tokens"] == 25000
    assert client.requests[0]["store"] is False and stream.closed


@pytest.mark.parametrize("kind", ["response.incomplete", "response.failed", "error", "eof"])
def test_failure_never_yields_completion(kind):
    stream = Stream([NS(type="response.output_text.delta", delta="partial")] +
                    ([] if kind == "eof" else [NS(type=kind)]))
    runtime = OpenAIResponsesRuntime(Client(stream), OpenAIOptions())

    async def collect():
        return [item async for item in runtime.stream([], max_new_tokens=1, cancel=threading.Event())]
    with pytest.raises(OpenAIProviderError):
        run(collect())
    assert stream.closed


def test_incomplete_budget_reports_available_usage_without_completion():
    response = NS(incomplete_details=NS(reason="max_output_tokens"),
                  usage=NS(input_tokens=9, output_tokens=25000,
                           input_tokens_details=NS(cached_tokens=2),
                           output_tokens_details=NS(reasoning_tokens=24900)))
    runtime = OpenAIResponsesRuntime(Client(Stream([NS(type="response.incomplete", response=response)])),
                                     OpenAIOptions())
    async def collect():
        return [item async for item in runtime.stream([], max_new_tokens=1, cancel=threading.Event())]
    with pytest.raises(OpenAIProviderError) as exc:
        run(collect())
    assert "budget" in str(exc.value)
    assert exc.value.metrics["reasoning_tokens"] == 24900
    assert exc.value.metrics["cached_input_tokens"] == 2


def test_native_calls_keep_ids_and_replay_reasoning_without_executing_malformed():
    output = [dict(type="reasoning", encrypted_content="opaque"),
              dict(type="function_call", call_id="c1", name="search_history", arguments='{"query":"Hue"}'),
              dict(type="function_call", call_id="c2", name="search_history", arguments="bad")]
    client = Client(Stream([complete(output)]))
    runtime = OpenAIResponsesRuntime(client, OpenAIOptions())
    parsed, replay, _ = run(runtime.plan([], tools=[{"function": {"name": "search_history",
        "description": "Search", "parameters": {"type": "object"}}}], cancel=threading.Event()))
    assert [(call.id, call.arguments) for call in parsed.tool_calls] == [("c1", {"query": "Hue"})]
    assert parsed.failures == 1 and replay == output
    assert client.requests[0]["tools"][0]["type"] == "function"


@pytest.mark.parametrize("model,effort", [("bad", "medium"), ("gpt-6-luna", "extreme")])
def test_options_reject_unsupported_values(model, effort):
    with pytest.raises(ValueError):
        OpenAIOptions(model, effort)


def test_cancel_closes_waiting_stream():
    class WaitingStream(Stream):
        async def __anext__(self):
            await asyncio.Event().wait()
    stream = WaitingStream([])
    cancel = threading.Event()
    runtime = OpenAIResponsesRuntime(Client(stream), OpenAIOptions())

    async def collect():
        async def cancel_soon():
            await asyncio.sleep(0.03)
            cancel.set()
        task = asyncio.create_task(cancel_soon())
        result = [item async for item in runtime.stream([], max_new_tokens=1, cancel=cancel)]
        await task
        return result
    assert run(collect()) == []
    assert stream.closed


def test_cancel_while_opening_upstream():
    class SlowClient:
        responses = None
        def __init__(self):
            self.responses = self
            self.cancelled = False
        async def create(self, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled = True
    client, cancel = SlowClient(), threading.Event()
    runtime = OpenAIResponsesRuntime(client, OpenAIOptions())
    async def collect():
        async def cancel_soon():
            await asyncio.sleep(0.03)
            cancel.set()
        task = asyncio.create_task(cancel_soon())
        result = [item async for item in runtime.stream([], max_new_tokens=1, cancel=cancel)]
        await task
        return result
    assert run(collect()) == [] and client.cancelled


def test_closing_consumer_immediately_closes_upstream():
    stream = Stream([NS(type="response.output_text.delta", delta="partial"), complete()])
    runtime = OpenAIResponsesRuntime(Client(stream), OpenAIOptions())

    async def stop():
        iterator = runtime.stream([], max_new_tokens=1, cancel=threading.Event())
        assert isinstance(await anext(iterator), ModelDelta)
        await iterator.aclose()
    run(stop())
    assert stream.closed


def test_independent_concurrent_request_options():
    first, second = Stream([complete()]), Stream([complete()])
    client = Client(first, second)
    one = OpenAIResponsesRuntime(client, OpenAIOptions("gpt-6-luna", "low"))
    two = OpenAIResponsesRuntime(client, OpenAIOptions("gpt-6.1-terra", "xhigh"))
    async def execute(model):
        return [item async for item in model.stream([], max_new_tokens=1, cancel=threading.Event())]
    async def both():
        return await asyncio.gather(execute(one), execute(two))
    run(both())
    assert {(item["model"], item["reasoning"]["effort"]) for item in client.requests} == {
        ("gpt-6-luna", "low"), ("gpt-6.1-terra", "xhigh")}


def test_access_guard_requires_secret_and_applies_global_quota():
    guard = OpenAIAccessGuard("x" * 32, hourly=1, daily=1, concurrent=1)
    assert not guard.authorized("Bearer wrong")
    assert guard.authorized("Bearer " + "x" * 32)
    lease = guard.acquire()
    assert lease
    assert not guard.acquire()
    guard.release(lease)
    assert not guard.acquire()
    guard.close()


def test_access_guard_shares_durable_limits_across_connections(tmp_path):
    database = tmp_path / "quota.sqlite3"
    first = OpenAIAccessGuard("x" * 32, hourly=1, daily=2, concurrent=1, database=database)
    second = OpenAIAccessGuard("x" * 32, hourly=1, daily=2, concurrent=1, database=database)
    lease = first.acquire()
    assert lease and not second.acquire()
    first.release(lease)
    first.close()
    assert not second.acquire()
    second.close()


def test_central_native_tool_round_replays_reasoning_and_matched_outputs():
    from app.central.runtime import CentralRuntime
    from tests.test_central_planning import registry
    calls = [dict(type="reasoning", encrypted_content="opaque"),
             dict(type="function_call", call_id="call-1", name="search_history",
                  arguments='{"query":"Bạch Đằng","top_k":3}')]
    client = Client(Stream([complete(calls)]), Stream([complete()]),
                    Stream([NS(type="response.output_text.delta", delta="Năm 938 [c1]."), complete()]))
    runtime = CentralRuntime(model=OpenAIResponsesRuntime(client, OpenAIOptions()),
                             tools=registry(external=False)[0], max_action_rounds=2)
    result = run(runtime.prepare("Ngô Quyền chiến thắng Bạch Đằng vào năm nào?", 3, []))
    assert result.tool_calls[0]["name"] == "search_history"
    assert result.contexts
    replay = client.requests[1]["input"]
    assert any(item.get("type") == "reasoning" for item in replay)
    assert any(item.get("type") == "function_call_output" and item["call_id"] == "call-1" for item in replay)
    assert any(item.get("type") == "function_call_output" and item["call_id"] == "call-1"
               for item in result.messages)
    assert "Bạch Đằng" in result.messages[-1]["content"]
    async def finish():
        return [item async for item in runtime.model.stream(
            result.messages, max_new_tokens=1, cancel=threading.Event())]
    assert run(finish())[0].text == "Năm 938 [c1]."
    assert any(item.get("call_id") == "call-1" for item in client.requests[2]["input"])
    assert "tools" not in client.requests[2]


def test_central_rejects_unapproved_native_tool_calls():
    from app.central.runtime import CentralRuntime
    from tests.test_central_planning import registry
    calls = [dict(type="function_call", call_id="bad", name="dangerous_tool", arguments="{}")]
    tools, _ = registry(external=False)
    client = Client(Stream([complete(calls)]), Stream([complete()]))
    result = run(CentralRuntime(model=OpenAIResponsesRuntime(client, OpenAIOptions()),
                                tools=tools, max_action_rounds=2).prepare("Bạch Đằng?", 3, []))
    assert result.tool_calls[0]["name"] == "dangerous_tool"
    assert result.tool_calls[0]["error"] == "tool_not_allowed"
    assert any(item.get("call_id") == "bad" and "tool_not_allowed" in item.get("output", "")
               for item in client.requests[1]["input"])


@pytest.mark.parametrize(("name", "category"), [
    ("APITimeoutError", "timeout"), ("RateLimitError", "rate_limit"),
    ("NotFoundError", "model_unavailable"), ("APIError", "upstream_error"),
])
def test_provider_errors_are_classified_and_redacted(name, category):
    error_type = type(name, (Exception,), {})
    class FailingClient:
        responses = None
        def __init__(self):
            self.responses = self
        async def create(self, **kwargs):
            raise error_type("secret API_KEY=sk-sensitive")
    runtime = OpenAIResponsesRuntime(FailingClient(), OpenAIOptions())
    async def collect():
        return [item async for item in runtime.stream([], max_new_tokens=1, cancel=threading.Event())]
    with pytest.raises(OpenAIProviderError) as exc:
        run(collect())
    assert exc.value.category == category
    assert "sk-sensitive" not in str(exc.value)


def test_schema_defaults_and_openai_options_validation():
    from pydantic import ValidationError
    from app.schemas import ChatRequest
    base = {"conversation_id": "b0eaee3b-302f-4f89-97c3-20b921e687a4", "question": "Lịch sử?"}
    assert ChatRequest(**base).provider == "local"
    for effort in ("low", "medium", "high", "xhigh"):
        assert ChatRequest(**base, provider="openai", model="gpt-6.1-terra",
                           reasoning_effort=effort).reasoning_effort == effort
    with pytest.raises(ValidationError):
        ChatRequest(**base, model="gpt-6-luna")
    with pytest.raises(ValidationError):
        ChatRequest(**base, provider="openai", model="gpt-unknown")


def test_citation_filter_handles_split_and_unknown_references():
    from app.api.routes import CitationFilter
    filt = CitationFilter([{"chunk_id": "c1", "source_id": "s1"}])
    assert filt.feed("Dẫn [c") == "Dẫn "
    assert filt.feed("1], sai [made") == "[c1], sai "
    assert filt.feed("up] và [s1].") == " và [s1]."
    assert filt.feed(" [unclosed", final=True) == " unclosed"


@pytest.mark.parametrize("model", ["gpt-6-luna", "gpt-6.1-terra"])
@pytest.mark.parametrize("failure", [False, True])
def test_hybrid_stream_uses_retrieval_and_persists_only_completed(tmp_path, model, failure):
    from app.api.routes import chat_stream
    from app.schemas import ChatRequest
    from tests.test_baseline_runtime import _app, _events
    app, store, _, retriever = _app(tmp_path)
    owner = "baseline-test-client"
    conversation = store.create_conversation(owner)
    terminal = NS(type="response.incomplete") if failure else complete()
    stream = Stream([NS(type="response.output_text.delta", delta="Năm 938 [c1] "),
                     NS(type="response.output_text.delta", delta="[invented]."), terminal])
    client = Client(stream)
    app.state.openai_client = client
    app.state.openai_guard = OpenAIAccessGuard("x" * 32, 10, 10, 1)
    payload = ChatRequest(conversation_id=conversation["id"], question="Ý nghĩa Bạch Đằng?",
                          provider="openai", model=model, reasoning_effort="high")
    async def connected():
        return False
    request = NS(app=app, is_disconnected=connected, headers={"Authorization": "Bearer " + "x" * 32})
    async def execute():
        response = await chat_stream(payload, request, owner, store)
        return _events("".join([frame async for frame in response.body_iterator]))
    events = run(execute())
    answer = "".join(data["delta"] for name, data in events if name == "answer_delta")
    assert answer == "Năm 938 [c1] ."
    assert retriever.calls == 1
    assert "Chiến thắng năm 938" in client.requests[0]["input"][-1]["content"]
    assert len(store.list_messages(owner, conversation["id"])) == (1 if failure else 2)
    assert events[-1][1]["status"] == ("error" if failure else "done")
    if not failure:
        assert next(data for name, data in events if name == "sources")["cited_source_ids"] == ["c1"]
        assert events[-1][1]["model_id"] == model


def test_hybrid_unauthorized_request_fails_before_inference(tmp_path):
    from fastapi import HTTPException
    from app.api.routes import chat_stream
    from app.schemas import ChatRequest
    from tests.test_baseline_runtime import _app
    app, store, _, retriever = _app(tmp_path)
    app.state.openai_client = Client()
    app.state.openai_guard = OpenAIAccessGuard("x" * 32, 10, 10, 1)
    conversation = store.create_conversation("owner")
    payload = ChatRequest(conversation_id=conversation["id"], question="Bạch Đằng?", provider="openai")
    with pytest.raises(HTTPException) as exc:
        run(chat_stream(payload, NS(app=app, headers={}), "owner", store))
    assert exc.value.status_code == 403 and retriever.calls == 0


def test_official_sdk_wire_stream_without_network():
    openai = pytest.importorskip("openai")
    httpx2 = pytest.importorskip("httpx2")
    captured = []
    events = [
        {"type": "response.output_text.delta", "sequence_number": 0, "item_id": "msg-1",
         "output_index": 0, "content_index": 0, "delta": "Chào [c1]", "logprobs": []},
        {"type": "response.completed", "sequence_number": 1,
         "response": {"id": "resp-test", "status": "completed", "output": [],
                      "usage": {"input_tokens": 10, "output_tokens": 4,
                                "input_tokens_details": {"cached_tokens": 2},
                                "output_tokens_details": {"reasoning_tokens": 1}}}},
    ]
    body = "".join("event: " + event["type"] + "\n" + "data: " +
                   json.dumps(event, ensure_ascii=False) + "\n\n" for event in events) + "data: [DONE]\n\n"

    def respond(request):
        captured.append(json.loads(request.content))
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

    async def execute():
        http = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
        client = openai.AsyncOpenAI(api_key="test-only", http_client=http, max_retries=0)
        try:
            runtime = OpenAIResponsesRuntime(client, OpenAIOptions("gpt-6-luna", "high"))
            return [event async for event in runtime.stream(
                [{"role": "user", "content": "Hello"}], max_new_tokens=3, cancel=threading.Event())]
        finally:
            await client.close()
    result = run(execute())
    assert result[0].text == "Chào [c1]"
    assert result[-1].reasoning_tokens == 1
    assert captured[0]["model"] == "gpt-6-luna"
    assert captured[0]["stream"] is True and captured[0]["store"] is False


def test_official_sdk_central_tool_continuation_without_network():
    openai = pytest.importorskip("openai")
    httpx2 = pytest.importorskip("httpx2")
    from app.central.runtime import CentralRuntime
    from tests.test_central_planning import registry
    captured = []
    output = [
        {"type": "reasoning", "id": "rs1", "summary": [], "encrypted_content": "opaque"},
        {"type": "function_call", "id": "fc1", "call_id": "call1", "name": "search_history",
         "arguments": '{"query":"Bạch Đằng","top_k":3}', "status": "completed"},
    ]
    def respond(request):
        captured.append(json.loads(request.content))
        response = {"id": "resp-test", "status": "completed", "output": output if len(captured) == 1 else [],
                    "usage": {"input_tokens": 10, "output_tokens": 4,
                              "input_tokens_details": {"cached_tokens": 2},
                              "output_tokens_details": {"reasoning_tokens": 1}}}
        event = {"type": "response.completed", "sequence_number": 1, "response": response}
        body = "event: response.completed\ndata: " + json.dumps(event, ensure_ascii=False) + "\n\ndata: [DONE]\n\n"
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())
    async def execute():
        client = openai.AsyncOpenAI(api_key="test-only",
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)), max_retries=0)
        try:
            runtime = CentralRuntime(model=OpenAIResponsesRuntime(client, OpenAIOptions()),
                                     tools=registry(external=False)[0], max_action_rounds=2)
            return await runtime.prepare("Bạch Đằng?", 3, [])
        finally:
            await client.close()
    prepared = run(execute())
    assert prepared.contexts
    assert captured[0]["tools"][0]["type"] == "function"
    assert any(item.get("type") == "reasoning" and item.get("encrypted_content") == "opaque"
               for item in captured[1]["input"])
    assert any(item.get("type") == "function_call_output" and item.get("call_id") == "call1"
               for item in captured[1]["input"])


def test_remote_only_startup_does_not_build_local_model(monkeypatch, tmp_path):
    import sys
    from app import main
    from app.config import Settings
    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.closed = False
        async def close(self):
            self.closed = True
    settings = Settings(_env_file=None, app_mode="full", enable_central_mode=False,
                        local_inference_enabled=False, openai_enabled=True,
                        openai_trusted_internal_only=True, openai_api_key="sdk-key",
                        openai_access_token="x" * 32, chat_database_path=tmp_path / "chat.sqlite3")
    monkeypatch.setattr(main, "settings", settings)
    monkeypatch.setitem(sys.modules, "openai", NS(AsyncOpenAI=FakeAsyncOpenAI))
    monkeypatch.setattr(main, "RAGService", lambda: NS(load=lambda: None, shutdown=lambda: None))
    monkeypatch.setattr(main, "HybridRetriever", lambda _: object())
    monkeypatch.setattr(main, "AttachmentService", lambda **_: object())
    monkeypatch.setattr(main, "TemporaryCorpusRetriever", lambda **_: object())
    monkeypatch.setattr(main, "build_model_runtime", lambda **_: pytest.fail("Local model was built"))
    app = NS(state=NS())
    async def start():
        async with main.lifespan(app):
            assert app.state.hybrid_runtime.model.inference_backend == "openai"
            client = app.state.openai_client
        assert client.closed
    run(start())
