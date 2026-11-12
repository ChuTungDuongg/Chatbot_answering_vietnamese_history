"""Latency policies preserve request-selected final reasoning and grounding."""
import asyncio
import json
import threading

import pytest

from app.central.runtime import CentralRuntime
from app.config import Settings
from app.models.openai_responses import OpenAIOptions, OpenAIResponsesRuntime
from tests.test_central_planning import registry
from tests.test_openai_responses import Client, Stream, complete


def plan():
    return [{"type": "function_call", "call_id": "history", "name": "search_history",
        "arguments": '{"query":"938","__central_final_after_success":true}'}]


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh"])
def test_planning_reasoning_is_separate_from_selected_final_reasoning(effort):
    async def exercise():
        client = Client(Stream([complete(plan())]), Stream([complete()]))
        model = OpenAIResponsesRuntime(client, OpenAIOptions(reasoning_effort=effort))
        tools, _ = registry(external=False)
        prepared = await CentralRuntime(model=model, tools=tools).prepare("Bạch Đằng 938?", 3, [])
        assert client.requests[0]["reasoning"] == {"effort": "low"}
        assert prepared.planning["rounds"][0]["reasoning_effort"] == "low"
        assert prepared.planning["final_reasoning_effort"] == effort
        async for _ in model.stream(prepared.messages, max_new_tokens=1536, cancel=threading.Event()):
            pass
        assert client.requests[1]["reasoning"] == {"effort": effort}
        assert model.options.reasoning_effort == effort
        assert prepared.contexts and prepared.action_rounds == 1
    asyncio.run(exercise())


def test_inherit_planning_policy_preserves_legacy_effort():
    client = Client(Stream([complete(plan())]))
    model = OpenAIResponsesRuntime(client, OpenAIOptions(reasoning_effort="xhigh"))
    tools, _ = registry(external=False)
    asyncio.run(CentralRuntime(model=model, tools=tools, planning_reasoning_effort="inherit").prepare("Bạch Đằng?", 3, []))
    assert client.requests[0]["reasoning"] == {"effort": "xhigh"}


def test_concurrent_planning_overrides_cannot_mutate_shared_adapter():
    async def exercise():
        client = Client(Stream([complete()]), Stream([complete()]))
        model = OpenAIResponsesRuntime(client, OpenAIOptions(reasoning_effort="xhigh"))
        await asyncio.gather(*(model.plan([], tools=[], cancel=threading.Event(), reasoning_effort=effort)
                               for effort in ("low", "medium")))
        assert {r["reasoning"]["effort"] for r in client.requests} == {"low", "medium"}
        assert model.options.reasoning_effort == "xhigh"
    asyncio.run(exercise())


def test_planning_effort_settings_reject_unsupported_values():
    from pydantic import ValidationError
    assert Settings(_env_file=None).central_openai_planning_reasoning_effort == "low"
    with pytest.raises(ValidationError):
        Settings(_env_file=None, central_openai_planning_reasoning_effort="minimal")


def test_empty_planning_effort_is_rejected_before_upstream_request():
    client = Client(Stream([complete()]))
    model = OpenAIResponsesRuntime(client, OpenAIOptions())
    with pytest.raises(ValueError, match="reasoning effort"):
        asyncio.run(model.plan([], tools=[], cancel=threading.Event(), reasoning_effort=""))
    assert not client.requests


def test_multi_topic_research_batches_searches_then_observed_readers_in_two_rounds(monkeypatch):
    from app.tools.wikipedia import SearchWikipediaTool, FetchWikipediaPageTool
    searches, readers = threading.Barrier(2), threading.Barrier(3)
    def search(language, params):
        searches.wait(timeout=5)
        ids = (1, 2) if params["srsearch"] == "Waterloo Napoleon" else (3,)
        return {"query": {"search": [{"pageid": i, "title": f"History {i}", "snippet": "Source snippet"} for i in ids]}}
    def fetch(language, params):
        readers.wait(timeout=5)
        key = int(params["pageids"])
        return {"query": {"pages": [{"pageid": key, "title": f"History {key}", "extract": "Detailed historical evidence."}]}}
    monkeypatch.setattr("app.tools.builtin.search_wikipedia.tool._request_json", search)
    monkeypatch.setattr("app.tools.builtin.fetch_wikipedia_page.tool._request_json", fetch)
    def function(key, name, **args):
        return {"type": "function_call", "call_id": key, "name": name, "arguments": json.dumps(args)}
    first = [function(str(i), "search_wikipedia", query=query, language="en", __central_independent=True)
             for i, query in enumerate(("Waterloo Napoleon", "Hitler"))]
    second = [function(f"read{i}", "fetch_wikipedia_page", page_id_or_title=str(i), language="en",
                      __central_independent=True, __central_final_after_success=True) for i in (1, 2, 3)]
    tools, retriever = registry(external=False)
    tools.register(SearchWikipediaTool())
    tools.register(FetchWikipediaPageTool())
    client = Client(Stream([complete(first)]), Stream([complete(second)]))
    result = asyncio.run(CentralRuntime(model=OpenAIResponsesRuntime(client, OpenAIOptions(reasoning_effort="xhigh")),
        tools=tools).prepare("Vì sao Napoleon thua ở trận Waterloo? Napoleon có giống Hitler không?", 6, []))
    assert retriever.calls == 0 and result.action_rounds == 2
    assert result.planning["parallel_batches"] == 2 and result.planning["exit_reason"] == "evidence_sufficient"
    assert {row["chunk_id"] for row in result.contexts} == {"wiki_en_1", "wiki_en_2", "wiki_en_3"}
    assert [r["reasoning"]["effort"] for r in client.requests] == ["low", "low"]
    transcript = result.messages
    assert [m["call_id"] for m in transcript if m.get("type") == "function_call"] == [
        m["call_id"] for m in transcript if m.get("type") == "function_call_output"]
