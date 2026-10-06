"""Planner telemetry, grounded routing and conservative completion regressions."""

import asyncio
import json
import time

import pytest

from app.central.runtime import CentralRuntime
from app.config import CENTRAL_MODEL_ID
from app.models.base import ModelDone
from app.rag.prompting import SYSTEM_PROMPT
from app.telemetry import RequestTrace
from app.tools.local_search import SearchHistoryInput, SearchHistoryTool
from app.tools.registry import ToolRegistry
from app.tools.wikipedia import SearchWikipediaTool, FetchWikipediaPageTool
from tests.test_baseline_runtime import FakeModel, FakeRetriever


WORLD_QUERY = "tóm tắt lịch sử thế giới từ thế kỷ X tới hiện nay"
READY = '<plan_status>{"final_after_tools":true,"pending_tools":[]}</plan_status>'


def call(name, **arguments):
    return '<tool_call>' + json.dumps({"name": name, "arguments": arguments}) + '</tool_call>'


class Planner(FakeModel):
    def __init__(self, plans, *, finish_reason="stop"):
        super().__init__(CENTRAL_MODEL_ID)
        self.plans = plans
        self.finish_reason = finish_reason
        self.captures = []

    async def generate(self, messages, *, tools, max_new_tokens, cancel):
        self.captures.append((list(messages), tools))
        started = time.perf_counter_ns()
        await asyncio.sleep(0)
        first = time.perf_counter_ns()
        text = self.plans[len(self.captures) - 1]
        finished = time.perf_counter_ns()
        return text, ModelDone(self.model_id, "test-revision", started, first, finished,
            100, 20, max_new_tokens, self.finish_reason, self.finish_reason == "length",
            self.finish_reason == "length")


class WikiSearch(SearchWikipediaTool):
    def run(self, args):
        return [{"chunk_id": "wiki_vi_1", "source_kind": "wikipedia", "title": "Lịch sử thế giới",
                 "text": "Search snippet", "metadata": {"page_id": 1}}]


class WikiFetch(FetchWikipediaPageTool):
    def run(self, args):
        return {"chunk_id": "wiki_vi_1", "source_kind": "wikipedia", "title": "Lịch sử thế giới",
                "text": "Full external evidence about medieval and modern world history.",
                "url": "https://vi.wikipedia.org/?curid=1"}


def registry(*, external=True):
    tools = ToolRegistry()
    retriever = FakeRetriever()
    tools.register(SearchHistoryTool(retriever))
    if external:
        tools.register(WikiSearch())
        tools.register(WikiFetch())
    return tools, retriever


def test_successful_explicit_plan_exits_early_and_records_real_round_metadata():
    tools, retriever = registry()
    model = Planner([call("search_history", query="Bạch Đằng", top_k=3) + READY])
    trace = RequestTrace("r", "central")
    events = []
    prepared = asyncio.run(CentralRuntime(model=model, tools=tools).prepare(
        "Ngô Quyền chiến thắng Bạch Đằng vào năm nào?", 3, [], trace=trace, progress=events.append))
    assert retriever.calls == len(model.captures) == prepared.action_rounds == 1
    assert prepared.planning["exit_reason"] == "evidence_sufficient"
    row = prepared.planning["rounds"][0]
    assert row["round"] == 1 and row["early_exit"]
    assert row["wall_ms"] >= row["model_ms"] >= row["ttft_ms"] >= 0
    assert row["input_tokens"] == 100 and row["output_tokens"] == 20
    assert row["tools_requested"] == ["search_history"]
    assert prepared.planning["total_model_ms"] == row["model_ms"]
    assert row["finished_ms"] >= row["started_ms"]
    assert prepared.messages[0]["content"] == SYSTEM_PROMPT
    assert any(e["stage"] == "tool_selection" and e["state"] == "completed" for e in events)


def test_world_history_query_searches_external_then_fetches_grounded_page():
    tools, retriever = registry()
    class DomainPlanner(Planner):
        async def generate(self, messages, **kwargs):
            names = {item["function"]["name"] for item in kwargs["tools"]}
            assert "search_wikipedia" in names and "fetch_wikipedia_page" in names
            assert "search_history" not in names
            assert "corpus lịch sử Việt Nam" in messages[0]["content"]
            return await super().generate(messages, **kwargs)
    model = DomainPlanner([call("search_wikipedia", query=WORLD_QUERY) + READY,
                           call("fetch_wikipedia_page", page_id_or_title="1") + READY])
    runtime = CentralRuntime(model=model, tools=tools)
    prepared = asyncio.run(runtime.prepare(WORLD_QUERY, 3, []))
    assert retriever.calls == 0
    assert [row["name"] for row in prepared.tool_calls] == ["search_wikipedia", "fetch_wikipedia_page"]
    assert prepared.action_rounds == 2
    assert not prepared.planning["rounds"][0]["early_exit"]  # Snippet isn't full page evidence.
    assert prepared.contexts[0]["source_kind"] == "wikipedia"
    assert prepared.contexts[0]["text"].startswith("Full external evidence")
    assert "Full external evidence" in prepared.messages[-1]["content"]
    assert prepared.messages[0]["content"] == SYSTEM_PROMPT
    assert "không đoán" in SYSTEM_PROMPT


def test_world_query_hallucinated_local_call_is_rejected_without_local_fallback():
    tools, retriever = registry()
    model = Planner([call("search_history", query=WORLD_QUERY) + READY,
                     call("fetch_wikipedia_page", page_id_or_title="Lịch sử thế giới") + READY])
    prepared = asyncio.run(CentralRuntime(model=model, tools=tools).prepare(WORLD_QUERY, 3, []))
    assert retriever.calls == 0
    assert prepared.tool_calls[0]["error"] == "external_evidence_required"
    assert prepared.contexts and all(c["source_kind"] == "wikipedia" for c in prepared.contexts)


def test_world_query_without_permitted_external_sources_keeps_grounded_limitation():
    tools, retriever = registry(external=False)
    model = Planner([""])
    prepared = asyncio.run(CentralRuntime(model=model, tools=tools).prepare(WORLD_QUERY, 3, []))
    assert not prepared.contexts and retriever.calls == 0
    assert "Không có nguồn phù hợp" in prepared.messages[-1]["content"]
    assert prepared.messages[0]["content"] == SYSTEM_PROMPT


def test_vietnamese_history_still_prefers_local_corpus_and_keeps_tools_available():
    tools, retriever = registry()
    model = Planner([call("search_history", query="Lịch sử Việt Nam") + READY])
    prepared = asyncio.run(CentralRuntime(model=model, tools=tools).prepare("Tóm tắt lịch sử Việt Nam", 3, []))
    names = {item["function"]["name"] for item in model.captures[0][1]}
    assert names == {"search_history", "search_wikipedia", "fetch_wikipedia_page"}
    assert "ưu tiên công cụ này cho lịch sử Việt Nam" in model.captures[0][0][0]["content"]
    assert retriever.calls == 1 and prepared.tool_calls[0]["name"] == "search_history"


def test_mixed_domain_preserves_legitimate_second_tool_call_even_with_ready_signal():
    tools, retriever = registry()
    model = Planner([call("search_history", query="Việt Nam Chiến tranh Lạnh") + READY,
                     call("fetch_wikipedia_page", page_id_or_title="Chiến tranh Lạnh") + READY])
    prepared = asyncio.run(CentralRuntime(model=model, tools=tools).prepare(
        "Việt Nam chịu ảnh hưởng của Chiến tranh Lạnh thế giới như thế nào?", 3, []))
    assert prepared.action_rounds == 2 and retriever.calls == 1
    assert [row["name"] for row in prepared.tool_calls] == ["search_history", "fetch_wikipedia_page"]
    assert {c.get("source_kind", "history") for c in prepared.contexts} == {"history", "wikipedia"}
    assert not prepared.planning["rounds"][0]["early_exit"]


@pytest.mark.parametrize("status", ["", '<plan_status>{"final_after_tools":true,"pending_tools":["fetch_wikipedia_page"]}</plan_status>',
                                   '<plan_status>{"final_after_tools":true}</plan_status>', '<plan_status>invalid</plan_status>'])
def test_no_confirmed_completion_preserves_multi_round_planner(status):
    tools, _ = registry()
    model = Planner([call("search_history", query="Bạch Đằng") + status,
                     call("fetch_wikipedia_page", page_id_or_title="Bạch Đằng") + READY])
    prepared = asyncio.run(CentralRuntime(model=model, tools=tools).prepare("Bạch Đằng?", 3, []))
    assert prepared.action_rounds == 2 and len(prepared.tool_calls) == 2
    assert prepared.planning["input_tokens"] == 200 and prepared.planning["output_tokens"] == 40


@pytest.mark.parametrize("bad_result", ["empty", "error", "truncated", "no_text", "plan_length"])
def test_early_exit_requires_successful_usable_untruncated_evidence(bad_result):
    tools, _ = registry()
    class UnreliableHistory:
        name = "search_history"
        description = "Vietnamese history"
        input_schema = SearchHistoryInput
        def run(self, args):
            if bad_result == "error": raise RuntimeError("temporary failure")
            if bad_result == "empty": return []
            return [{"chunk_id": "c1", "text": "" if bad_result == "no_text" else "Evidence",
                     "metadata": {"truncated": bad_result == "truncated"}}]
    tools = ToolRegistry(); tools.register(UnreliableHistory()); tools.register(WikiFetch())
    model = Planner([call("search_history", query="Bạch Đằng") + READY,
                     call("fetch_wikipedia_page", page_id_or_title="Bạch Đằng") + READY],
                    finish_reason="length" if bad_result == "plan_length" else "stop")
    prepared = asyncio.run(CentralRuntime(model=model, tools=tools).prepare("Bạch Đằng?", 3, []))
    assert prepared.action_rounds == 2 and not prepared.planning["rounds"][0]["early_exit"]


def test_single_round_can_execute_multiple_tools_before_final():
    tools, _ = registry()
    model = Planner([call("search_history", query="Việt Nam") +
                     call("fetch_wikipedia_page", page_id_or_title="Chiến tranh Lạnh") + READY])
    prepared = asyncio.run(CentralRuntime(model=model, tools=tools).prepare(
        "Việt Nam chịu ảnh hưởng của Chiến tranh Lạnh thế giới như thế nào?", 3, []))
    assert prepared.action_rounds == 1 and len(prepared.tool_calls) == 2
    assert prepared.planning["rounds"][0]["tools_requested"] == ["search_history", "fetch_wikipedia_page"]


def test_completion_status_inside_hidden_reasoning_cannot_skip_a_round():
    tools, _ = registry()
    model = Planner(["<think>" + READY + "</think>" + call("search_history", query="Bạch Đằng"),
                     call("fetch_wikipedia_page", page_id_or_title="Bạch Đằng") + READY])
    prepared = asyncio.run(CentralRuntime(model=model, tools=tools).prepare("Bạch Đằng?", 3, []))
    assert prepared.action_rounds == 2 and not prepared.planning["rounds"][0]["early_exit"]


def test_schema_completion_is_optional_request_local_and_stripped_before_execution():
    from app.central.planning import COMPLETION_FIELD
    tools, retriever = registry()
    original_schemas = json.dumps(tools.describe(), sort_keys=True)
    model = Planner([call("search_history", query="Bạch Đằng", **{COMPLETION_FIELD: True})])
    prepared = asyncio.run(CentralRuntime(model=model, tools=tools).prepare("Bạch Đằng?", 3, []))
    assert prepared.planning["exit_reason"] == "evidence_sufficient"
    assert prepared.action_rounds == retriever.calls == 1
    schema = next(item["function"]["parameters"] for item in model.captures[0][1]
                  if item["function"]["name"] == "search_history")
    assert schema["properties"][COMPLETION_FIELD]["type"] == "boolean"
    assert COMPLETION_FIELD not in schema["required"]
    assert json.dumps(tools.describe(), sort_keys=True) == original_schemas
    assert COMPLETION_FIELD not in prepared.tool_calls[0]  # No planner control in executed args/trace.


def test_mcp_completion_flag_never_reaches_strict_remote_argument_validation():
    from app.central.planning import COMPLETION_FIELD
    from app.mcp.schemas import ToolSteering
    from app.mcp.adapters import tool_name
    from tests.test_mcp_integration import FakeClient, manager_fixture, result
    async def run():
        async def remote(name, arguments):
            assert arguments == {"query": WORLD_QUERY}
            return result()
        manager, tools, _ = await manager_fixture(client=FakeClient(behavior=remote))
        name = tool_name("research", "lookup")
        model = Planner([call(name, query=WORLD_QUERY, **{COMPLETION_FIELD: True})])
        try:
            prepared = await CentralRuntime(model=model, tools=tools, mcp_manager=manager).prepare(
                WORLD_QUERY, 3, [], steering=ToolSteering(mcp_enabled=True, allowed_mcp_servers=["research"]))
            assert prepared.action_rounds == 1 and prepared.planning["exit_reason"] == "evidence_sufficient"
            assert prepared.contexts[0]["source_kind"] == "mcp"
        finally:
            await manager.close()
    asyncio.run(run())


def test_planner_controls_preserve_remote_collision_and_schema_budget():
    from app.central.planning import COMPLETION_FIELD, planner_tool_schemas
    schema = {"type": "function", "function": {"name": "mcp__research__lookup", "description": "Research",
              "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}
    original_size = len(json.dumps(schema, ensure_ascii=False).encode())
    rows, controlled = planner_tool_schemas([schema], mcp_schema_budget=original_size)
    assert rows == [schema] and not controlled
    schema["function"]["parameters"]["properties"][COMPLETION_FIELD] = {"type": "string"}
    rows, controlled = planner_tool_schemas([schema])
    assert rows == [schema] and not controlled


def test_world_history_can_use_request_allowed_mcp_without_wikipedia():
    from app.mcp.schemas import ToolSteering
    from app.mcp.adapters import tool_name
    from tests.test_mcp_integration import manager_fixture
    async def run():
        manager, tools, client = await manager_fixture()
        name = tool_name("research", "lookup")
        model = Planner([call(name, query=WORLD_QUERY) + READY])
        runtime = CentralRuntime(model=model, tools=tools, mcp_manager=manager)
        try:
            prepared = await runtime.prepare(WORLD_QUERY, 3, [], steering=ToolSteering(
                mcp_enabled=True, allowed_mcp_servers=["research"]))
            assert client.calls == 1
            assert prepared.action_rounds == 1
            assert prepared.retrieval["steering"]["effective_tools"] == sorted([name, "search_history"])
            assert prepared.tool_calls[0]["name"] == name
            assert all(c["source_kind"] == "mcp" for c in prepared.contexts)
            assert "private" not in json.dumps(prepared.planning)
        finally:
            await manager.close()
    asyncio.run(run())
