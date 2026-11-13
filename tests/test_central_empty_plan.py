"""Regression for a completed Responses planner returning no calls or evidence."""
import asyncio
import json
import threading
from types import SimpleNamespace as NS

import pytest

from app.central.planning import COMPLETION_FIELD
from app.central.runtime import CentralRuntime
from app.mcp.schemas import ToolSteering
from app.models.openai_responses import OpenAIOptions, OpenAIResponsesRuntime
from app.tools.registry import ToolRegistry
from tests.test_central_planning import WikiFetch, WikiSearch, registry
from tests.test_openai_responses import Client, Stream, complete


def function(call_id, name, **arguments):
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": json.dumps(arguments)}


def native(client):
    return OpenAIResponsesRuntime(client, OpenAIOptions("gpt-6.1-terra", "xhigh"))


def test_empty_first_plan_recovers_with_approved_tools_and_native_call_ids():
    tools = ToolRegistry()
    tools.register(WikiSearch()); tools.register(WikiFetch())
    calls = [function("search", "search_wikipedia", query="Isaac Newton"),
             function("read", "fetch_wikipedia_page", page_id_or_title="1", **{COMPLETION_FIELD: True})]
    first = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Premature answer"}]}
    client = Client(Stream([complete([first])]), Stream([complete([calls[0]])]), Stream([complete([calls[1]])]))
    result = asyncio.run(CentralRuntime(model=native(client), tools=tools).prepare("Isaac Newton là ai?", 3, []))
    assert result.contexts and result.planning["exit_reason"] == "evidence_sufficient"
    assert result.action_rounds == 3
    assert result.planning["rounds"][0]["replan_reason"] == "planner_no_tools_without_evidence"
    assert [r["tool_choice"] for r in client.requests] == ["required", "required", "auto"]
    assert first in client.requests[1]["input"]
    outputs = [row for row in result.messages if row.get("type") == "function_call_output"]
    assert [row["call_id"] for row in outputs] == ["search", "read"]


def test_empty_plan_recovery_is_bounded_and_never_invents_evidence():
    tools = ToolRegistry(); tools.register(WikiFetch())
    client = Client(Stream([complete()]), Stream([complete()]))
    result = asyncio.run(CentralRuntime(model=native(client), tools=tools).prepare("Isaac Newton là ai?", 3, []))
    assert len(client.requests) == 2 and not result.contexts and not result.tool_calls
    assert result.planning["exit_reason"] == "planner_no_tools"


def test_denied_tools_skip_paid_planning_and_remain_denied():
    tools, retriever = registry()
    client = Client()
    result = asyncio.run(CentralRuntime(model=native(client), tools=tools).prepare("Bạch Đằng?", 3, [],
        steering=ToolSteering(allowed_tools=[])))
    assert not client.requests and retriever.calls == 0
    assert not result.contexts and not result.tool_calls and result.action_rounds == 0
    assert result.planning["exit_reason"] == "no_tools_available"


def test_world_question_without_external_permission_skips_paid_planning():
    tools, retriever = registry(external=False)
    client = Client()
    result = asyncio.run(CentralRuntime(model=native(client), tools=tools).prepare("Napoleon là ai?", 3, []))
    assert not client.requests and retriever.calls == 0 and not result.contexts
    assert result.planning["exit_reason"] == "no_external_tools"


def test_selected_attachment_cannot_bypass_missing_document_tool_permission():
    tools, retriever = registry(external=False)
    client = Client()
    result = asyncio.run(CentralRuntime(model=native(client), tools=tools).prepare("Napoleon?", 3, [],
        attachment_ids=("selected-file",), steering=ToolSteering(allowed_tools=["search_history"])))
    assert not client.requests and retriever.calls == 0 and not result.contexts
    assert result.planning["exit_reason"] == "no_external_tools"


def test_world_question_can_use_a_permitted_uploaded_document():
    from app.tools.attachment_search import SearchUploadedDocumentsTool
    class Documents:
        def retrieve(self, **kwargs):
            assert kwargs["owner_id"] == "owner" and kwargs["conversation_id"] == "chat"
            assert kwargs["attachment_ids"] == ("selected-file",)
            return [{"chunk_id": "document:1", "text": "Evidence from the owner's document."}]
    tools = ToolRegistry(); tools.register(SearchUploadedDocumentsTool(Documents()))
    client = Client(Stream([complete([function("document", "search_uploaded_documents", query="Napoleon",
                                            **{COMPLETION_FIELD: True})])]))
    result = asyncio.run(CentralRuntime(model=native(client), tools=tools).prepare("Napoleon?", 3, [],
        owner_id="owner", conversation_id="chat", attachment_ids=("selected-file",)))
    assert len(client.requests) == 1 and result.planning["exit_reason"] == "evidence_sufficient"
    assert result.contexts[0]["source_kind"] == "attachment"


def test_grounded_planner_can_stop_without_being_forced_to_call_again():
    tools, retriever = registry(external=False)
    client = Client(Stream([complete([function("history", "search_history", query="938")])]), Stream([complete()]))
    result = asyncio.run(CentralRuntime(model=native(client), tools=tools).prepare("Bạch Đằng?", 3, []))
    assert result.contexts and retriever.calls == 1
    assert [r["tool_choice"] for r in client.requests] == ["required", "auto"]


@pytest.mark.parametrize("bad", ["anything", "none"])
def test_invalid_planner_tool_choice_is_rejected_before_upstream(bad):
    client = Client()
    with pytest.raises(ValueError, match="tool choice"):
        asyncio.run(native(client).plan([], tools=[], cancel=threading.Event(), tool_choice=bad))
    assert not client.requests


def test_required_tool_choice_needs_an_allowed_schema():
    client = Client()
    with pytest.raises(ValueError, match="tool"):
        asyncio.run(native(client).plan([], tools=[], cancel=threading.Event(), tool_choice="required"))
    assert not client.requests


def test_stream_recovers_empty_planner_then_persists_only_cited_final_answer(tmp_path):
    from app.api.routes import chat_stream
    from app.schemas import ChatRequest
    from app.services.chat_mode_router import ChatModeRouter
    from app.services.openai_access import OpenAIAccessGuard
    from tests.test_baseline_runtime import _app, _events
    app, store, _, _ = _app(tmp_path)
    tools = ToolRegistry(); tools.register(WikiSearch()); tools.register(WikiFetch())
    client = Client(Stream([complete()]),
        Stream([complete([function("search", "search_wikipedia", query="Isaac Newton")])]),
        Stream([complete([function("read", "fetch_wikipedia_page", page_id_or_title="1", **{COMPLETION_FIELD: True})])]),
        Stream([NS(type="response.output_text.delta", delta="Nguồn xác nhận [wiki_vi_1]."), complete()]))
    app.state.chat_mode_router = ChatModeRouter(hybrid=None, central=CentralRuntime(model=native(client), tools=tools))
    app.state.openai_client = client
    app.state.openai_guard = OpenAIAccessGuard("x" * 32, 10, 10, 1)
    owner = "owner"; conversation = store.create_conversation(owner)
    payload = ChatRequest(conversation_id=conversation["id"], question="Isaac Newton là ai?", mode="central",
                          provider="openai", model="gpt-6.1-terra", reasoning_effort="xhigh")
    async def execute():
        request = NS(app=app, is_disconnected=lambda: asyncio.sleep(0, result=False),
                     headers={"Authorization": "Bearer " + "x" * 32})
        response = await chat_stream(payload, request, owner, store)
        return _events("".join([frame async for frame in response.body_iterator]))
    events = asyncio.run(execute())
    assert events[-1][1]["status"] == "done"
    assert not any(event == "error" for event, _ in events)
    assert len(store.list_messages(owner, conversation["id"])) == 2
    assert client.requests[-1]["reasoning"] == {"effort": "xhigh"}
    assert "tool_choice" not in client.requests[-1] and "tools" not in client.requests[-1]


def test_explicit_provider_refusal_is_not_retried_or_bypassed_with_fallback():
    tools, retriever = registry()
    output = [{"type": "message", "role": "assistant", "content": [{"type": "refusal", "refusal": "Refused"}]}]
    client = Client(Stream([complete(output)]))
    result = asyncio.run(CentralRuntime(model=native(client), tools=tools).prepare("Bạch Đằng?", 3, []))
    assert len(client.requests) == 1 and not result.contexts and retriever.calls == 0
    assert result.planning["exit_reason"] == "planner_refused"


def test_recovery_does_not_exceed_single_round_budget():
    tools = ToolRegistry(); tools.register(WikiFetch())
    client = Client(Stream([complete()]))
    result = asyncio.run(CentralRuntime(model=native(client), tools=tools, max_action_rounds=1).prepare("Newton?", 3, []))
    assert len(client.requests) == 1 and result.action_rounds == 1 and not result.contexts


def test_empty_plan_recovery_respects_actual_planning_token_budget():
    from app.central.budgets import AgentLimits
    tools = ToolRegistry(); tools.register(WikiFetch())
    client = Client(Stream([complete(output_tokens=5)]))
    result = asyncio.run(CentralRuntime(model=native(client), tools=tools,
        limits=AgentLimits(planning_output_tokens=5)).prepare("Newton?", 3, []))
    assert len(client.requests) == 1 and result.planning["exit_reason"] == "planning_token_budget"
    assert not result.contexts


def test_cancellation_during_empty_plan_recovery_closes_active_request():
    async def exercise():
        entered, stopped = asyncio.Event(), asyncio.Event()
        class SlowClient(Client):
            async def create(self, **kwargs):
                if self.requests:
                    self.requests.append(kwargs)
                    entered.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        stopped.set()
                return await super().create(**kwargs)
        tools = ToolRegistry(); tools.register(WikiFetch())
        client = SlowClient(Stream([complete()]))
        cancel = threading.Event()
        task = asyncio.create_task(CentralRuntime(model=native(client), tools=tools).prepare("Newton?", 3, [], cancel=cancel))
        await asyncio.wait_for(entered.wait(), 2)
        cancel.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert stopped.is_set() and len(client.requests) == 2
    asyncio.run(exercise())


def test_tool_choice_is_per_call_and_never_changes_shared_adapter():
    async def exercise():
        client = Client(Stream([complete()]), Stream([complete()]))
        model = native(client)
        schemas = [{"function": {"name": "search_history", "description": "Search", "parameters": {"type": "object"}}}]
        await asyncio.gather(*(model.plan([], tools=schemas, cancel=threading.Event(), tool_choice=choice)
                               for choice in ("auto", "required")))
        assert {request["tool_choice"] for request in client.requests} == {"auto", "required"}
        assert model.options.reasoning_effort == "xhigh"
        assert "tool_choice" not in model.generation_settings
    asyncio.run(exercise())


@pytest.mark.parametrize("question", ["Bạch Đằng?", "Vietnam cold war?"])
def test_refusal_after_evidence_stops_final_generation_and_persistence(tmp_path, question):
    from app.api.routes import chat_stream
    from app.schemas import ChatRequest
    from app.services.chat_mode_router import ChatModeRouter
    from app.services.openai_access import OpenAIAccessGuard
    from tests.test_baseline_runtime import _app, _events
    app, store, _, _ = _app(tmp_path)
    tools, _ = registry(external=False)
    refused = [{"type": "message", "role": "assistant", "content": [{"type": "refusal", "refusal": "Refused"}]}]
    client = Client(Stream([complete([function("history", "search_history", query="938")])]),
        Stream([complete(refused)]),
        Stream([NS(type="response.output_text.delta", delta="Should not be generated [c1]."), complete()]))
    app.state.chat_mode_router = ChatModeRouter(hybrid=None, central=CentralRuntime(model=native(client), tools=tools))
    app.state.openai_client = client
    app.state.openai_guard = OpenAIAccessGuard("x" * 32, 10, 10, 1)
    owner = "owner"; conversation = store.create_conversation(owner)
    payload = ChatRequest(conversation_id=conversation["id"], question=question, mode="central", provider="openai")
    async def execute():
        response = await chat_stream(payload, NS(app=app, headers={"Authorization": "Bearer " + "x" * 32},
            is_disconnected=lambda: asyncio.sleep(0, result=False)), owner, store)
        return _events("".join([frame async for frame in response.body_iterator]))
    events = asyncio.run(execute())
    assert events[-1][1]["status"] == "error"
    assert len(client.requests) == 2 and not any(event == "answer_delta" for event, _ in events)
    assert events[-1][1]["metrics"]["agent_exit_reason"] == "planner_refused"
    assert len(store.list_messages(owner, conversation["id"])) == 1


@pytest.mark.parametrize("scope", ["none", "world"])
def test_unavailable_tools_return_specific_sse_error_without_paid_call(tmp_path, scope):
    from app.api.routes import chat_stream
    from app.schemas import ChatRequest
    from app.services.chat_mode_router import ChatModeRouter
    from app.services.openai_access import OpenAIAccessGuard
    from tests.test_baseline_runtime import _app, _events
    app, store, _, _ = _app(tmp_path)
    tools, _ = registry(external=False)
    client = Client()
    app.state.chat_mode_router = ChatModeRouter(hybrid=None, central=CentralRuntime(model=native(client), tools=tools))
    app.state.openai_client = client
    app.state.openai_guard = OpenAIAccessGuard("x" * 32, 10, 10, 1)
    owner = "owner"; conversation = store.create_conversation(owner)
    payload = ChatRequest(conversation_id=conversation["id"], question="Napoleon?" if scope == "world" else "Bạch Đằng?",
        mode="central", provider="openai", steering=ToolSteering(allowed_tools=[]) if scope == "none" else None)
    async def execute():
        response = await chat_stream(payload, NS(app=app, headers={"Authorization": "Bearer " + "x" * 32},
            is_disconnected=lambda: asyncio.sleep(0, result=False)), owner, store)
        return _events("".join([frame async for frame in response.body_iterator]))
    events = asyncio.run(execute())
    assert not client.requests and events[-1][1]["status"] == "error"
    error = next(data for event, data in events if event == "error")
    assert ("cấp công cụ" if scope == "none" else "nguồn bên ngoài") in error["message"]
    metrics = events[-1][1]["metrics"]
    assert metrics["agent_exit_reason"] == ("no_tools_available" if scope == "none" else "no_external_tools")
    assert metrics["model_calls"] == metrics["evidence_context_count"] == 0
    assert len(store.list_messages(owner, conversation["id"])) == 1
