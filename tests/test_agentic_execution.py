"""Deterministic harness tests: no real models, paid APIs, or network."""

import asyncio
from dataclasses import replace
import json
import threading
import time

import pytest

from app.central.budgets import AgentLimits
from app.central.runtime import CentralRuntime
from app.central.planning import COMPLETION_FIELD, INDEPENDENT_FIELD
from app.models.tool_calls import DecodedFunctionCall
from app.tools.local_search import SearchHistoryInput
from app.tools.registry import ToolExecutionContext, ToolExecutionPolicy, ToolRegistry
from app.security.guardrails import safe_observation
from tests.test_central_planning import Planner, READY, call, registry


def run(awaitable):
    return asyncio.run(awaitable)


class Search:
    name = "search_history"
    description = "Controlled retrieval"
    input_schema = SearchHistoryInput
    execution_policy = ToolExecutionPolicy(read_only=True, parallel_safe=True)

    def __init__(self, function):
        self.function = function

    async def run_with_context(self, arguments, context):
        return await self.function(arguments, context)


def tools_for(*tools):
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def chunk(key="c1", text="Evidence", **extra):
    return {"chunk_id": key, "text": text, **extra}


def test_explicit_independent_safe_searches_overlap_and_observations_keep_call_order():
    async def exercise():
        entered = set()
        both = asyncio.Event()
        async def first(args, context):
            entered.add("first")
            if len(entered) == 2: both.set()
            await asyncio.wait_for(both.wait(), .5)
            await asyncio.sleep(.02)
            context.retrieval_metrics.update(backend="qdrant", timings_ms={"embedding": 11})
            return [chunk("first")]
        async def second(args, context):
            entered.add("second")
            if len(entered) == 2: both.set()
            await asyncio.wait_for(both.wait(), .5)
            context.retrieval_metrics.update(backend="qdrant", timings_ms={"embedding": 7})
            return [chunk("second", source_kind="wikipedia")]
        history, wikipedia = Search(first), Search(second)
        wikipedia.name = "search_wikipedia"
        plan = call(history.name, query="A", **{INDEPENDENT_FIELD: True}) + call(wikipedia.name, query="B", **{INDEPENDENT_FIELD: True}) + READY
        model = Planner([plan])
        result = await CentralRuntime(model=model, tools=tools_for(history, wikipedia)).prepare("Sự kiện?", 4, [])
        assert [row["execution_mode"] for row in result.tool_calls] == ["parallel", "parallel"]
        assert [row["call_id"] for row in result.tool_calls] == ["call_0001", "call_0002"]
        assert result.retrieval["timings_ms"] == {"embedding": 18}
        assert result.planning["parallel_batches"] == 1
        # The rendezvous proves concurrency without a machine-dependent speed assertion.
        assert result.planning["tool_wall_ms"] >= max(row["latency_ms"] for row in result.tool_calls)
    run(exercise())


@pytest.mark.parametrize("explicit,safe", [(False, True), (True, False)])
def test_multiple_calls_alone_do_not_grant_parallel_execution(explicit, safe):
    async def exercise():
        active, maximum = 0, 0
        async def execute(args, context):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(.005)
            active -= 1
            return [chunk(args.query)]
        first, second = Search(execute), Search(execute)
        second.name = "search_wikipedia"
        first.execution_policy = second.execution_policy = ToolExecutionPolicy(read_only=True, parallel_safe=safe)
        model = Planner([call(first.name, query="A", **{INDEPENDENT_FIELD: explicit}) +
                         call(second.name, query="B", **{INDEPENDENT_FIELD: explicit}) + READY])
        result = await CentralRuntime(model=model, tools=tools_for(first, second)).prepare("Sự kiện?", 4, [])
        assert maximum == 1
        assert all(row["execution_mode"] == "sequential" for row in result.tool_calls)
    run(exercise())


def test_speculative_reader_is_rejected_then_rebuilt_from_observed_page_id():
    tools, _ = registry()
    model = Planner([
        call("search_wikipedia", query="Napoleon") + call("fetch_wikipedia_page", page_id_or_title="1") + READY,
        call("fetch_wikipedia_page", page_id_or_title="1") + READY,
    ])
    result = run(CentralRuntime(model=model, tools=tools).prepare("Lịch sử thế giới?", 3, []))
    assert [r["error"] for r in result.tool_calls] == [None, "dependency_not_observed", None]
    assert result.action_rounds == 2 and result.planning["exit_reason"] == "evidence_sufficient"
    assert "dependency_not_observed" in model.captures[1][0][-1]["content"]
    assert result.contexts[0]["text"].startswith("Full external evidence")


@pytest.mark.parametrize("failure", ["empty", "error", "truncated"])
def test_incomplete_or_failed_search_replans_with_refined_query(failure):
    queries = []
    async def execute(args, context):
        queries.append(args.query)
        if len(queries) == 1:
            if failure == "error": raise RuntimeError("private payload must not escape")
            return [] if failure == "empty" else [chunk(metadata={"truncated": True})]
        return [chunk("specific", "Evidence for the missing campaign")]
    model = Planner([call("search_history", query="broad") + READY,
                     call("search_history", query="missing campaign tactics") + READY])
    result = run(CentralRuntime(model=model, tools=tools_for(Search(execute))).prepare("So sánh các chiến thuật?", 4, []))
    assert queries == ["broad", "missing campaign tactics"]
    assert not result.planning["rounds"][0]["early_exit"]
    assert result.planning["exit_reason"] == "evidence_sufficient"
    assert "private payload" not in json.dumps(result.planning)


def test_nonempty_evidence_without_completion_does_not_skip_planning():
    tools, retriever = registry(external=False)
    model = Planner([call("search_history", query="first"), call("search_history", query="specific") + READY])
    result = run(CentralRuntime(model=model, tools=tools).prepare("Bạch Đằng?", 3, []))
    assert result.action_rounds == retriever.calls == 2


def test_identical_calls_are_not_reexecuted_and_final_answer_is_qualified():
    tools, retriever = registry(external=False)
    model = Planner([call("search_history", query="same"), call("search_history", query="same")])
    result = run(CentralRuntime(model=model, tools=tools).prepare("Bạch Đằng?", 3, []))
    assert retriever.calls == 1
    assert result.planning["repeats_avoided"] == 1
    assert result.planning["exit_reason"] == "repeated_calls"
    assert any("chưa xác minh" in message["content"] for message in result.messages if message.get("role") == "system")


def test_repeat_detection_normalizes_pydantic_defaults():
    tools, retriever = registry(external=False)
    model = Planner([call("search_history", query="same"), call("search_history", query="same", top_k=8)])
    result = run(CentralRuntime(model=model, tools=tools).prepare("Bạch Đằng?", 3, []))
    assert retriever.calls == 1 and result.planning["repeats_avoided"] == 1


def test_round_and_total_call_budgets_do_not_execute_excess_calls():
    async def execute(args, context):
        return [chunk(args.query)]
    model = Planner(["".join(call("search_history", query=str(i)) for i in range(5)),
                     call("search_history", query="remaining")])
    limits = AgentLimits(calls_per_round=2, total_calls=3)
    result = run(CentralRuntime(model=model, tools=tools_for(Search(execute)), limits=limits).prepare("Sự kiện?", 4, []))
    assert result.planning["tools_executed"] == 3
    assert [r["error"] for r in result.tool_calls[:5]] == [None, None, "round_tool_budget", "round_tool_budget", "round_tool_budget"]
    assert result.planning["exit_reason"] == "total_tool_budget"


def test_deadline_cancels_planner_and_can_build_a_qualified_final_prompt():
    async def exercise():
        stopped = asyncio.Event()
        class SlowPlanner(Planner):
            async def generate(self, *args, **kwargs):
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.set()
        result = await CentralRuntime(model=SlowPlanner([]), tools=ToolRegistry(),
            limits=AgentLimits(deadline_seconds=.03)).prepare("Sự kiện?", 3, [])
        assert stopped.is_set() and not result.contexts
        assert result.planning["exit_reason"] == "execution_deadline"
        assert "Không có nguồn phù hợp" in result.messages[-1]["content"]
    run(exercise())


def test_user_cancellation_stops_all_active_tool_tasks():
    async def exercise():
        active, stopped = set(), set()
        cancel = threading.Event()
        async def execute(args, context):
            active.add(args.query)
            if len(active) == 2: cancel.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.add(args.query)
        first, second = Search(execute), Search(execute)
        second.name = "search_wikipedia"
        model = Planner([call(first.name, query="A", **{INDEPENDENT_FIELD: True}) + call(second.name, query="B", **{INDEPENDENT_FIELD: True})])
        with pytest.raises(asyncio.CancelledError):
            await CentralRuntime(model=model, tools=tools_for(first, second)).prepare("Sự kiện?", 3, [], cancel=cancel)
        assert stopped == active == {"A", "B"}
    run(exercise())


def test_tool_timeout_does_not_merge_late_worker_metrics_or_emit_late_progress():
    async def exercise():
        release = threading.Event()
        late = []
        class SlowSync(Search):
            run_with_context = None
            def run(self, args):
                release.wait(.2)
                return [chunk("too_late")]
        tools = tools_for(SlowSync(None))
        model = Planner([call("search_history", query="slow"), ""])
        result = await CentralRuntime(model=model, tools=tools,
            limits=AgentLimits(tool_timeout_seconds=.02)).prepare("Sự kiện?", 3, [], progress=late.append)
        count = len(late)
        release.set()
        await asyncio.sleep(.03)
        assert not result.contexts and len(late) == count
        assert result.tool_calls[0]["error"] == "tool_timeout"
    run(exercise())


def test_observations_always_remain_valid_bounded_json():
    value = safe_observation([chunk(str(i), '"' * 10000, title="long" * 100, metadata={"truncated": True}) for i in range(20)], max_chars=256)
    assert len(value) <= 256
    assert json.loads(value)["observation_truncated"] is True


def test_independent_requests_cannot_share_metrics_or_tool_results():
    async def exercise():
        async def execute(args, context):
            await asyncio.sleep(.005)
            context.retrieval_metrics.update(backend=context.retrieval_backend,
                                            timings_ms={"embedding": 1 if args.query == "A" else 2})
            return [chunk(args.query)]
        tools = tools_for(Search(execute))
        async def ask(query, backend):
            return await CentralRuntime(model=Planner([call("search_history", query=query) + READY]), tools=tools).prepare(query, 3, [], retrieval_backend=backend)
        first, second = await asyncio.gather(ask("A", "faiss"), ask("B", "qdrant"))
        assert first.contexts[0]["chunk_id"] == "A" and second.contexts[0]["chunk_id"] == "B"
        assert first.retrieval["retrieval_backend"] == "faiss" and second.retrieval["retrieval_backend"] == "qdrant"
        assert first.retrieval["timings_ms"] == {"embedding": 1}
        assert second.retrieval["timings_ms"] == {"embedding": 2}
    run(exercise())


def test_refined_sources_survive_selection_and_web_sources_get_stable_ids():
    from app.central.planning import select_evidence, tool_evidence
    by_id = {str(i): chunk(str(i)) for i in range(12)}
    selected = select_evidence([[str(i) for i in range(10)], ["10", "11"]], by_id, 3)
    assert selected[0]["chunk_id"] == "10" and any(c["chunk_id"] == "0" for c in selected)
    source = {"url": "https://example.org/history", "text": "External evidence"}
    assert tool_evidence("search_web", [source])[0]["chunk_id"] == tool_evidence("fetch_web_page", source)[0]["chunk_id"]


def test_planning_token_budget_stops_before_extra_model_calls():
    tools, _ = registry(external=False)
    result = run(CentralRuntime(model=Planner([call("search_history", query="A")]), tools=tools,
        limits=AgentLimits(planning_output_tokens=20)).prepare("Bạch Đằng?", 3, []))
    assert result.model_calls_before_final == 1
    assert result.planning["exit_reason"] == "planning_token_budget"


def test_native_simple_fact_uses_one_plan_and_keeps_original_reasoning_prefix():
    from tests.test_openai_responses import Client, Stream, complete
    from app.models.openai_responses import OpenAIResponsesRuntime, OpenAIOptions
    tools, retriever = registry(external=False)
    output = [{"type": "reasoning", "encrypted_content": "opaque"},
              {"type": "function_call", "call_id": "native-1", "name": "search_history",
               "arguments": json.dumps({"query": "938", COMPLETION_FIELD: True})}]
    client = Client(Stream([complete(output)]))
    result = run(CentralRuntime(model=OpenAIResponsesRuntime(client, OpenAIOptions()), tools=tools).prepare("Bạch Đằng?", 3, []))
    assert result.action_rounds == retriever.calls == len(client.requests) == 1
    assert result.planning["exit_reason"] == "evidence_sufficient"
    assert result.messages[:len(client.requests[0]["input"])] == client.requests[0]["input"]
    assert result.messages[len(client.requests[0]["input"]):][:2] == output
    assert any(m.get("type") == "function_call_output" and m["call_id"] == "native-1" for m in result.messages)


def test_native_malformed_denied_and_over_budget_calls_all_receive_correlated_errors():
    from tests.test_openai_responses import Client, Stream, complete
    from app.models.openai_responses import OpenAIResponsesRuntime, OpenAIOptions
    tools, retriever = registry(external=False)
    output = [{"type": "function_call", "call_id": "malformed", "name": "search_history", "arguments": "bad"},
              {"type": "function_call", "call_id": "denied", "name": "dangerous_tool", "arguments": "{}"},
              {"type": "function_call", "call_id": "good", "name": "search_history", "arguments": '{"query":"938"}'},
              {"type": "function_call", "call_id": "excess", "name": "search_history", "arguments": '{"query":"939"}'}]
    client = Client(Stream([complete(output)]), Stream([complete()]))
    result = run(CentralRuntime(model=OpenAIResponsesRuntime(client, OpenAIOptions()), tools=tools,
        limits=AgentLimits(calls_per_round=3)).prepare("Bạch Đằng?", 3, []))
    outputs = [m for m in client.requests[1]["input"] if m.get("type") == "function_call_output"]
    assert [m["call_id"] for m in outputs] == ["malformed", "denied", "good", "excess"]
    assert [json.loads(m["output"]).get("error") for m in outputs] == ["invalid_arguments", "tool_not_allowed", None, "round_tool_budget"]
    assert retriever.calls == 1
    assert "bad" not in json.dumps(result.planning)


@pytest.mark.parametrize("bad", ["missing", "duplicate"])
def test_native_unmatchable_call_ids_fail_closed_before_execution(bad):
    from tests.test_openai_responses import Client, Stream, complete
    from app.models.openai_responses import OpenAIResponsesRuntime, OpenAIOptions, OpenAIProviderError
    tools, retriever = registry(external=False)
    output = [{"type": "function_call", "call_id": "id", "name": "search_history", "arguments": '{"query":"938"}'}]
    if bad == "missing":
        output[0].pop("call_id")
    else:
        output.append(dict(output[0]))
    with pytest.raises(OpenAIProviderError, match="invalid function-call envelope"):
        run(CentralRuntime(model=OpenAIResponsesRuntime(Client(Stream([complete(output)])), OpenAIOptions()), tools=tools).prepare("Bạch Đằng?", 3, []))
    assert retriever.calls == 0


def test_native_planning_output_ceiling_is_request_local_and_decreases_with_actual_usage():
    from tests.test_openai_responses import Client, Stream, complete
    from app.models.openai_responses import OpenAIResponsesRuntime, OpenAIOptions
    tools, _ = registry(external=False)
    output = [{"type": "function_call", "call_id": "one", "name": "search_history", "arguments": '{"query":"938"}'}]
    client = Client(Stream([complete(output, output_tokens=17)]), Stream([complete()]))
    model = OpenAIResponsesRuntime(client, OpenAIOptions(max_output_tokens=25000))
    result = run(CentralRuntime(model=model, tools=tools,
        limits=AgentLimits(planning_output_tokens=100)).prepare("Bạch Đằng?", 3, []))
    assert [r["max_output_tokens"] for r in client.requests] == [100, 83]
    assert model.options.max_output_tokens == 25000
    assert result.planning["output_tokens"] == 22


def test_mcp_is_sequential_even_when_model_and_remote_hint_request_parallelism():
    from app.mcp.schemas import ToolSteering
    from app.mcp.adapters import tool_name
    from tests.test_mcp_integration import FakeClient, manager_fixture, result
    async def exercise():
        active, maximum = 0, 0
        async def execute(name, args):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(.005)
            active -= 1
            return result()
        manager, tools, client = await manager_fixture(client=FakeClient(behavior=execute))
        name = tool_name("research", "lookup")
        try:
            plan = call(name, query="A", **{INDEPENDENT_FIELD: True}) + call(name, query="B", **{INDEPENDENT_FIELD: True}) + READY
            steering = ToolSteering(mcp_enabled=True, allowed_mcp_servers=["research"])
            runtime = CentralRuntime(model=Planner([plan, plan]), tools=tools, mcp_manager=manager)
            # The manager also serializes the same shared session across requests.
            await asyncio.gather(runtime.prepare("Lịch sử thế giới?", 3, [], steering=steering),
                                 runtime.prepare("Lịch sử thế giới?", 3, [], steering=steering))
            assert client.calls == 4 and maximum == 1
        finally:
            await manager.close()
    run(exercise())


def test_exclusive_retrieval_resource_is_not_parallelized_even_for_distinct_tools():
    async def exercise():
        active, peak = 0, 0
        async def execute(args, context):
            nonlocal active, peak
            active += 1; peak = max(active, peak)
            await asyncio.sleep(.005)
            active -= 1
            return [chunk(args.query)]
        first, second = Search(execute), Search(execute)
        second.name = "search_uploaded_documents"
        first.execution_policy = second.execution_policy = ToolExecutionPolicy(True, True, "retrieval")
        plan = call(first.name, query="A", **{INDEPENDENT_FIELD: True}) + call(second.name, query="B", **{INDEPENDENT_FIELD: True}) + READY
        result = await CentralRuntime(model=Planner([plan]), tools=tools_for(first, second)).prepare("Sự kiện?", 3, [])
        assert peak == 1 and result.planning["parallel_batches"] == 0
    run(exercise())


def test_argument_validation_and_invalid_control_values_do_not_reach_tools():
    tools, retriever = registry(external=False)
    model = Planner([call("search_history", query="invalid", top_k=21),
                     call("search_history", query="control", **{INDEPENDENT_FIELD: "true"}),
                     call("search_history", query="correct") + READY])
    result = run(CentralRuntime(model=model, tools=tools).prepare("Bạch Đằng?", 3, []))
    assert [r["error"] for r in result.tool_calls] == ["tool_failed", "invalid_planner_control", None]
    assert retriever.calls == 1


def test_zero_tools_steering_prevents_fallback_and_any_execution():
    from app.mcp.schemas import ToolSteering
    tools, retriever = registry(external=False)
    result = run(CentralRuntime(model=Planner([call("search_history", query="A"), ""]), tools=tools).prepare(
        "Bạch Đằng?", 3, [], steering=ToolSteering(allowed_tools=[])))
    assert retriever.calls == 0 and not result.contexts


def test_new_settings_validate_budgets_and_clone_into_remote_runtime(monkeypatch, tmp_path):
    from app.config import Settings
    from pydantic import ValidationError
    settings = Settings(_env_file=None, central_max_action_rounds=4, central_max_total_tool_calls=7)
    assert AgentLimits.from_settings(settings).total_calls == 7
    with pytest.raises(ValidationError, match="must cover one observation"):
        Settings(_env_file=None, central_max_total_observation_chars=256)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, central_max_tool_calls_per_round=0)


def test_backend_failure_is_observed_without_changing_retrieval_backend():
    from app.rag.backends import DenseBackendError
    requested = []
    async def execute(args, context):
        requested.append(context.retrieval_backend)
        if len(requested) == 1: raise DenseBackendError("private URL and credentials")
        return [chunk()]
    model = Planner([call("search_history", query="A"), call("search_history", query="refined") + READY])
    result = run(CentralRuntime(model=model, tools=tools_for(Search(execute))).prepare("Bạch Đằng?", 3, [], retrieval_backend="qdrant"))
    assert requested == ["qdrant", "qdrant"]
    assert result.tool_calls[0]["error"] == "retrieval_unavailable"
    assert "private URL" not in json.dumps(model.captures)


def test_sync_resource_gate_survives_timeout_and_prevents_cross_request_overlap():
    async def exercise():
        active = peak = 0
        mutex = threading.Lock()
        class Worker(Search):
            execution_policy = ToolExecutionPolicy(True, True, "retrieval")
            def run_with_context(self, args, context):
                nonlocal active, peak
                with mutex:
                    active += 1; peak = max(peak, active)
                time.sleep(.08 if args.query == "slow" else .005)
                with mutex: active -= 1
                context.retrieval_metrics["backend"] = args.query
                if context.progress: context.progress({"stage": "late", "state": "completed"})
                return [chunk(args.query)]
        tools = tools_for(Worker(None))
        first = await CentralRuntime(model=Planner([call("search_history", query="slow"), ""]), tools=tools,
            limits=AgentLimits(tool_timeout_seconds=.01)).prepare("Sự kiện?", 3, [])
        second = await CentralRuntime(model=Planner([call("search_history", query="fast") + READY]), tools=tools).prepare("Sự kiện?", 3, [])
        assert peak == 1 and not first.contexts and not first.retrieval["timings_ms"]
        assert second.contexts[0]["chunk_id"] == "fast"
        assert second.retrieval["retrieval_backend"] == "fast"
    run(exercise())


def test_reader_uses_exact_supplied_url_not_arbitrary_prefix():
    from app.central.execution import reader_input_observed
    assert reader_input_observed(DecodedFunctionCall("a", "fetch_web_page", {"url": "https://example.org/history"}),
                                 "Read https://example.org/history.", [])
    assert not reader_input_observed(DecodedFunctionCall("a", "fetch_web_page", {"url": "https://example.org"}),
                                     "Read https://example.org/history.", [])
    assert reader_input_observed(DecodedFunctionCall("a", "fetch_wikipedia_page", {"page_id_or_title": "77"}),
                                 "Read page_id=77", [])


def test_total_observation_budget_stops_replanning_without_orphaning_native_calls():
    from tests.test_openai_responses import Client, Stream, complete
    from app.models.openai_responses import OpenAIResponsesRuntime, OpenAIOptions
    tools, _ = registry(external=False)
    output = [{"type": "function_call", "call_id": str(index), "name": "search_history",
               "arguments": json.dumps({"query": str(index)})} for index in range(3)]
    client = Client(Stream([complete(output)]))
    result = run(CentralRuntime(model=OpenAIResponsesRuntime(client, OpenAIOptions()), tools=tools,
        limits=AgentLimits(observation_chars=256, total_observation_chars=256)).prepare("Bạch Đằng?", 3, []))
    assert result.planning["exit_reason"] == "observation_budget"
    assert len(client.requests) == 1
    outputs = [m for m in result.messages if m.get("type") == "function_call_output"]
    assert [m["call_id"] for m in outputs] == ["0", "1", "2"]
    assert all(len(m["output"]) <= 256 and isinstance(json.loads(m["output"]), dict) for m in outputs)
    assert result.planning["observation_total_chars"] == sum(len(m["output"]) for m in outputs)
    assert result.planning["observation_error_chars"] > 0


def test_web_provider_identity_is_not_confused_with_registry_provider_metadata():
    from app.tools.web_search import SearchWebTool
    class Provider:
        async def search(self, query, top_k):
            return [{"url": "https://example.org", "text": "Source"}]
    tools = tools_for(SearchWebTool(Provider()))
    _, record = run(tools.call("search_web", {"query": "history"}))
    assert record.provider == "builtin" and record.error is None


@pytest.mark.parametrize("failure_policy", ["fail", "continue"])
def test_central_timeout_preserves_mcp_failure_policy_and_identity(failure_policy):
    from app.mcp.schemas import MCPToolError, ToolSteering
    from app.mcp.adapters import tool_name
    from tests.test_mcp_integration import FakeClient, manager_fixture
    async def exercise():
        async def slow(name, arguments):
            await asyncio.sleep(2)
        manager, tools, client = await manager_fixture(client=FakeClient(behavior=slow))
        name = tool_name("research", "lookup")
        runtime = CentralRuntime(model=Planner([call(name, query="history"), ""]), tools=tools,
            mcp_manager=manager, limits=AgentLimits(tool_timeout_seconds=.01))
        steering = ToolSteering(mcp_enabled=True, allowed_mcp_servers=["research"],
                                mcp_failure_policy=failure_policy)
        try:
            request = runtime.prepare("Lịch sử thế giới?", 3, [], steering=steering)
            if failure_policy == "fail":
                with pytest.raises(MCPToolError):
                    await request
            else:
                prepared = await request
                record = prepared.tool_calls[0]
                assert record["error"] == "tool_timeout"
                assert record["provider"] == "mcp" and record["server"] == "research"
            assert client.cancelled == 1
        finally:
            await manager.close()
    run(exercise())


def test_selected_context_must_cover_required_domains_before_sufficiency():
    async def execute(args, context):
        return [chunk("vietnam", source_kind="history")]
    async def external(args, context):
        return [chunk(f"wiki_vi_{i}", source_kind="wikipedia") for i in range(4)]
    history, wikipedia = Search(execute), Search(external)
    wikipedia.name = "search_wikipedia"
    plan = call("search_history", query="Việt Nam") + call("search_wikipedia", query="Napoleon") + READY
    for top_k in (1, 2):
        model = Planner([plan, ""])
        result = run(CentralRuntime(model=model, tools=tools_for(history, wikipedia)).prepare(
            "So sánh Việt Nam và Napoleon", top_k, []))
        kinds = {c["source_kind"] for c in result.contexts}
        if top_k == 1:
            assert result.planning["exit_reason"] != "evidence_sufficient"
            assert any("chưa xác minh" in m.get("content", "") for m in result.messages if m.get("role") == "system")
        else:
            assert kinds == {"history", "wikipedia"}
            assert result.planning["exit_reason"] == "evidence_sufficient"
