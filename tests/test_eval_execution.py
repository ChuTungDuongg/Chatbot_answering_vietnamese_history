"""Benchmark execution uses the real chat runtimes with offline model/tool doubles."""

import asyncio
import json
import threading
import time
from copy import deepcopy
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from app.central.runtime import CentralRuntime
from app.config import CENTRAL_MODEL_ID, HYBRID_MODEL_ID, settings
from app.models.base import ModelDelta, ModelDone
from app.rag.hybrid_runtime import HybridRuntime
from app.services.chat_mode_router import ChatModeRouter
from app.services.openai_access import OpenAIAccessGuard
from app.tools.local_search import SearchHistoryTool
from app.tools.registry import ToolRegistry
from tests.test_baseline_runtime import FakeModel, FakeRetriever


class Planner(FakeModel):
    def __init__(self, *, unknown_usage=False, fail=False):
        super().__init__(CENTRAL_MODEL_ID, fail=fail)
        self.unknown_usage = unknown_usage
        self.planner_messages = None
        self.planner_tools = None

    async def generate(self, messages, *, tools, max_new_tokens, cancel):
        self.planner_messages, self.planner_tools = deepcopy(messages), deepcopy(tools)
        started = time.perf_counter_ns()
        text = '<tool_call>{"name":"search_history","arguments":{"query":"Bạch Đằng","top_k":3}}</tool_call>'
        text += '<plan_status>{"final_after_tools":true,"pending_tools":[]}</plan_status>'
        return text, ModelDone(
            self.model_id,
            "test-revision",
            started,
            started,
            time.perf_counter_ns(),
            None if self.unknown_usage else 10,
            3,
        )


def setup_app(monkeypatch, *, central=False, model=None, retriever=None):
    monkeypatch.setattr(settings, "local_inference_enabled", True)
    retriever = retriever or FakeRetriever()
    app = FastAPI()
    model = model or (Planner() if central else FakeModel(HYBRID_MODEL_ID))
    tools = ToolRegistry()
    tools.register(SearchHistoryTool(retriever))
    runtime = CentralRuntime(model=model, tools=tools) if central else HybridRuntime(retriever, model)
    app.state.retriever = retriever
    app.state.chat_mode_router = ChatModeRouter(
        hybrid=None if central else runtime, central=runtime if central else None
    )
    app.state.chat_store = SimpleNamespace(add_message=lambda *a: pytest.fail("benchmark wrote chat storage"))
    app.state.rag_service = SimpleNamespace(
        default_dense_backend="faiss",
        dense_retrievers={},
        corpus_sha256="actual-corpus",
        ordered_chunk_id_sha256="actual-order",
        manifest={"manifest_fingerprints": {"faiss": "actual-index"}},
    )
    experiment = {
        "id": "exp-1",
        "snapshot_id": "snapshot-1",
        "provider": "local",
        "model_id": model.model_id,
        "chat_mode": "central" if central else "hybrid",
        "reasoning_effort": None,
        "planning_reasoning_effort": "low" if central else None,
        "dense_backend": "faiss",
        "final_k": 3,
        "tool_allowlist": ["search_history"] if central else None,
    }
    return app, experiment, model, retriever, tools


def test_hybrid_execution_captures_actual_sources_without_chat_writes_or_reference_input(monkeypatch):
    from app.eval_studio.execution import ChatBenchmarkExecutor, freeze_configuration
    from app.eval_studio.v11 import RunInput

    app, experiment, model, retriever, _ = setup_app(monkeypatch)
    experiment["reference_answer"] = "REFERENCE MUST NEVER ENTER MODEL"
    frozen = freeze_configuration(app, experiment, {})
    run = asyncio.run(ChatBenchmarkExecutor(app).execute("Ý nghĩa Bạch Đằng?", frozen, threading.Event()))
    RunInput.model_validate(
        {
            "experiment_id": "exp-1",
            "question_id": "q1",
            **{k: v for k, v in run.items() if k != "diagnostics"},
        }
    )
    assert run["completion_status"] == "completed"
    assert run["answer"] == "Chiến thắng có ý nghĩa [c1]."
    assert run["evidence"][0]["excerpt"] == "Chiến thắng năm 938."
    assert run["evidence"][0]["selected"] is True
    assert run["evidence"][0]["cited"] is True
    assert run["trace"]["cited_chunk_ids"] == ["c1"]
    assert run["input_tokens"] == 20 and run["output_tokens"] == 8
    assert run["cost"] is None and retriever.calls == 1
    assert "REFERENCE MUST" not in json.dumps(model.messages)
    assert "REFERENCE MUST" not in json.dumps(frozen)
    assert run["diagnostics"]["capture"]["final_context"] is True


def test_configuration_rejects_prompt_schema_and_runtime_drift(monkeypatch):
    from app.eval_studio.execution import freeze_configuration, verify_configuration
    from app.rag import prompting

    app, experiment, model, _, tools = setup_app(monkeypatch, central=True)
    frozen = freeze_configuration(app, experiment, {})
    assert frozen["runtime_provenance"]["corpus_hash"] == "actual-corpus"
    assert frozen["runtime_provenance"]["retrieval_index_hash"] is not None
    assert "shared/grounded_system" in frozen["prompts"]
    assert "central/independent_parameter" in frozen["prompts"]
    verify_configuration(app, frozen)
    old = model.resolved_revision
    model.resolved_revision = "different-model"
    with pytest.raises(ValueError, match="drift"):
        verify_configuration(app, frozen)
    model.resolved_revision = old
    tools._descriptors["search_history"]["description"] += " changed"
    with pytest.raises(ValueError, match="drift"):
        verify_configuration(app, frozen)
    tools._descriptors["search_history"]["description"] = frozen["tool_definitions"][0]["description"]
    monkeypatch.setattr(prompting, "LIVE_SYSTEM_PROMPT", "changed production prompt")
    with pytest.raises(ValueError, match="drift"):
        verify_configuration(app, frozen)


def test_central_variants_are_scoped_and_usage_includes_planning(monkeypatch):
    from app.eval_studio.execution import ChatBenchmarkExecutor, freeze_configuration
    from app.rag.prompting import LIVE_SYSTEM_PROMPT

    app, experiment, model, _, tools = setup_app(monkeypatch, central=True)
    baseline = deepcopy(tools.describe())
    frozen = freeze_configuration(
        app,
        experiment,
        {
            "prompt_overrides": {
                "shared/grounded_system": "VARIANT GROUNDING",
                "central/planner": "VARIANT PLANNER",
                "central/completion_parameter": "VARIANT CONTROL",
            },
            "tool_descriptions": {"search_history": "VARIANT TOOL"},
        },
    )
    run = asyncio.run(ChatBenchmarkExecutor(app).execute("Ý nghĩa Bạch Đằng?", frozen, threading.Event()))
    assert run["completion_status"] == "completed"
    assert model.planner_messages[0]["content"].startswith("VARIANT PLANNER")
    assert "VARIANT GROUNDING" in model.messages[0]["content"]
    schema = model.planner_tools[0]["function"]
    assert schema["description"] == "VARIANT TOOL"
    assert (
        schema["parameters"]["properties"]["__central_final_after_success"]["description"]
        == "VARIANT CONTROL"
    )
    assert tools.describe() == baseline
    assert run["input_tokens"] == 30 and run["output_tokens"] == 11
    call = run["diagnostics"]["tool_calls"][0]
    assert call["arguments_valid"] is True and len(call["argument_fingerprint"]) == 64
    assert call["status"] == "completed" and call["result_count"] == 1
    assert "Bạch Đằng" not in json.dumps(run["diagnostics"])
    assert run["diagnostics"]["planning"]["exit_reason"] == "evidence_sufficient"
    original = asyncio.run(
        app.state.chat_mode_router.runtime_for("central").prepare("Ý nghĩa Bạch Đằng?", 3, [])
    )
    assert original.messages[0]["content"] == LIVE_SYSTEM_PROMPT
    assert "VARIANT" not in model.planner_messages[0]["content"]


def test_unknown_planning_usage_keeps_total_unknown(monkeypatch):
    from app.eval_studio.execution import ChatBenchmarkExecutor, freeze_configuration

    app, experiment, _, _, _ = setup_app(monkeypatch, central=True, model=Planner(unknown_usage=True))
    run = asyncio.run(
        ChatBenchmarkExecutor(app).execute(
            "Ý nghĩa Bạch Đằng?", freeze_configuration(app, experiment, {}), threading.Event()
        )
    )
    assert run["input_tokens"] is None
    assert run["output_tokens"] == 11
    assert run["diagnostics"]["usage"]["final_input_tokens"] == 20


def test_configuration_rejects_unknown_resources_models_and_missing_settings(monkeypatch):
    from app.eval_studio.execution import freeze_configuration

    app, experiment, _, _, _ = setup_app(monkeypatch)
    for override in (
        {"prompt_overrides": {"offline/no_rag": "bad"}},
        {"tool_descriptions": {"unknown": "bad"}},
    ):
        with pytest.raises(ValueError):
            freeze_configuration(app, experiment, override)
    for key, value in (("model_id", "made-up"), ("provider", None), ("chat_mode", None), ("final_k", None)):
        with pytest.raises(ValueError):
            freeze_configuration(app, {**experiment, key: value}, {})


@pytest.mark.parametrize("model_id", ["gpt-6-luna", "gpt-6.1-terra"])
def test_native_planner_keeps_replay_private_and_captures_rejected_argument_validation(monkeypatch, model_id):
    from app.eval_studio.execution import ChatBenchmarkExecutor, freeze_configuration
    from tests.test_openai_responses import Client, Stream, complete

    app, experiment, _, _, _ = setup_app(monkeypatch, central=True)
    app.state.chat_mode_router.runtime_for("central").max_action_rounds = 1
    output = [
        {"type": "reasoning", "encrypted_content": "PRIVATE REPLAY MUST NEVER PERSIST"},
        {"type": "function_call", "call_id": "bad", "name": "search_history", "arguments": "not-json"},
        {
            "type": "function_call",
            "call_id": "good",
            "name": "search_history",
            "arguments": json.dumps(
                {
                    "query": "secret query never persist",
                    "top_k": 3,
                    "__central_final_after_success": True,
                    "__central_independent": False,
                }
            ),
        },
    ]
    client = Client(
        Stream([complete(output)]),
        Stream(
            [
                SimpleNamespace(
                    type="response.output_text.delta",
                    delta="Đáp án [c1] [invented].<think>PRIVATE REASONING</think>",
                ),
                complete(),
            ]
        ),
    )
    app.state.openai_client = client
    guard = app.state.openai_guard = OpenAIAccessGuard("test-owner-token", 100, 100, 1)
    experiment.update(provider="openai", model_id=model_id, reasoning_effort="high")
    try:
        frozen = freeze_configuration(
            app, experiment, {"tool_descriptions": {"search_history": "NATIVE VARIANT"}}
        )
        run = asyncio.run(ChatBenchmarkExecutor(app).execute("Ý nghĩa Bạch Đằng?", frozen, threading.Event()))
        assert run["completion_status"] == "completed" and run["answer"] == "Đáp án [c1] ."
        bad, good = run["diagnostics"]["tool_calls"]
        assert bad["arguments_valid"] is False and bad["status"] == "rejected"
        assert bad["error_code"] == "invalid_arguments"
        assert good["arguments_valid"] is True and good["status"] == "completed"
        assert len(good["argument_fingerprint"]) == 64
        assert run["input_tokens"] == 24 and run["output_tokens"] == 10
        serialized = json.dumps(run)
        assert "PRIVATE" not in serialized and "secret query" not in serialized
        assert "encrypted_content" not in serialized and 'arguments":' not in serialized
        assert client.requests[0]["reasoning"] == {"effort": "low"}
        assert client.requests[1]["reasoning"] == {"effort": "high"}
        assert client.requests[0]["tools"][0]["description"] == "NATIVE VARIANT"
        assert any(item.get("encrypted_content") for item in client.requests[1]["input"])
        lease = guard.acquire()
        assert lease
        guard.release(lease)
    finally:
        guard.close()


def test_cancel_waiting_openai_releases_provider_slot_and_closes_stream(monkeypatch):
    from app.eval_studio.execution import ChatBenchmarkExecutor, freeze_configuration
    from tests.test_openai_responses import Client, Stream

    class WaitingStream(Stream):
        async def __anext__(self):
            entered.set()
            await asyncio.Event().wait()

    app, experiment, _, _, _ = setup_app(monkeypatch)
    entered = threading.Event()
    stream = WaitingStream([])
    app.state.openai_client = Client(stream)
    guard = app.state.openai_guard = OpenAIAccessGuard("owner", 100, 100, 1)
    experiment.update(provider="openai", model_id="gpt-6-luna", reasoning_effort="medium")
    frozen = freeze_configuration(app, experiment, {})
    cancel = threading.Event()

    async def execute():
        task = asyncio.create_task(ChatBenchmarkExecutor(app).execute("Ý nghĩa Bạch Đằng?", frozen, cancel))
        while not entered.is_set():
            await asyncio.sleep(0.005)
        assert guard.acquire() is None
        cancel.set()
        return await asyncio.wait_for(task, 2)

    try:
        run = asyncio.run(execute())
        assert run["completion_status"] == "cancelled" and run["error_reason"] == "cancelled"
        assert run["answer"] == "" and run["input_tokens"] is None and stream.closed
        assert run["diagnostics"]["usage"]["final_calls"] == 1
        lease = guard.acquire()
        assert lease
        guard.release(lease)
    finally:
        guard.close()


def test_failure_preserves_safe_evidence_and_negative_scores_without_fabricated_citations(monkeypatch):
    from app.eval_studio.execution import ChatBenchmarkExecutor, freeze_configuration

    class ScoredRetriever(FakeRetriever):
        def retrieve(self, *args, **kwargs):
            result = super().retrieve(*args, **kwargs)
            row = result["final_context"][0]
            row.update(
                url="https://history.example/source?token=private-secret#hidden",
                best_dense_score=-0.25,
                reranker_score=-2,
                text="Full selected evidence sk-proj-abcdefghijklmnopq and https://example.com/?token=hidden",
            )
            result["candidates20"] = [{"chunk_id": "c2", "text": "An actual discarded candidate."}]
            return result

    app, experiment, _, _, _ = setup_app(
        monkeypatch, model=FakeModel(HYBRID_MODEL_ID, fail=True), retriever=ScoredRetriever()
    )
    run = asyncio.run(
        ChatBenchmarkExecutor(app).execute(
            "Ý nghĩa Bạch Đằng?", freeze_configuration(app, experiment, {}), threading.Event()
        )
    )
    assert run["completion_status"] == "failed" and run["answer"] == ""
    assert run["input_tokens"] is None and run["output_tokens"] is None
    rows = {row["chunk_id"]: row for row in run["evidence"]}
    assert rows["c1"]["selected"] is True and rows["c2"]["selected"] is False
    assert rows["c1"]["cited"] is None and rows["c1"]["final_rank"] == 1
    assert rows["c1"]["dense_score"] == -0.25 and rows["c1"]["reranker_score"] == -2
    assert rows["c1"]["source_url"] == "https://history.example/source"
    assert "private-secret" not in json.dumps(run) and "sk-proj" not in json.dumps(run)
    assert "?token=" not in json.dumps(run)


def test_failed_planning_usage_is_unknown_and_failure_details_are_sanitized(monkeypatch):
    from app.eval_studio.execution import ChatBenchmarkExecutor, freeze_configuration
    from tests.test_openai_responses import Stream, Client

    app, experiment, _, _, _ = setup_app(monkeypatch, central=True)
    app.state.openai_client = Client(Stream([SimpleNamespace(type="response.failed")]))
    guard = app.state.openai_guard = OpenAIAccessGuard("owner", 100, 100, 1)
    experiment.update(provider="openai", model_id="gpt-6-luna", reasoning_effort="medium")
    try:
        run = asyncio.run(
            ChatBenchmarkExecutor(app).execute(
                "Ý nghĩa Bạch Đằng?", freeze_configuration(app, experiment, {}), threading.Event()
            )
        )
        assert run["completion_status"] == "failed" and run["error_reason"] == "upstream_error"
        assert run["diagnostics"]["usage"]["planning_calls"] == 1
        assert run["diagnostics"]["usage"]["planning_input_tokens"] is None
        assert run["diagnostics"]["planning"]["exit_reason"] is None
        assert run["diagnostics"]["capture"]["final_context"] is False
        assert run["diagnostics"]["capture"]["tool_trace"] is False
        assert run["tool_calls"] is None
    finally:
        guard.close()


def test_mcp_metadata_requires_strict_policy_and_cannot_silently_conflict(monkeypatch):
    from app.eval_studio.execution import freeze_configuration

    app, experiment, _, _, _ = setup_app(monkeypatch, central=True)
    with pytest.raises(ValueError, match="MCP"):
        freeze_configuration(app, {**experiment, "mcp_config": "server metadata only"}, {})
    with pytest.raises(ValueError, match="MCP"):
        freeze_configuration(
            app, {**experiment, "mcp_config": '{"mcp_enabled":true}'}, {"mcp_enabled": False}
        )
    frozen = freeze_configuration(app, {**experiment, "mcp_config": '{"mcp_enabled":false}'}, {})
    assert frozen["declared_mcp_config"]["mcp_enabled"] is False
    assert frozen["steering"]["mcp_enabled"] is False


def test_invalid_nonfinite_native_arguments_are_observed_without_changing_execution(monkeypatch):
    from app.eval_studio.execution import ChatBenchmarkExecutor, freeze_configuration
    from tests.test_openai_responses import Client, Stream, complete

    app, experiment, _, _, _ = setup_app(monkeypatch, central=True)
    app.state.chat_mode_router.runtime_for("central").max_action_rounds = 1
    app.state.openai_client = Client(
        Stream(
            [
                complete(
                    [
                        {
                            "type": "function_call",
                            "call_id": "nonfinite",
                            "name": "search_history",
                            "arguments": '{"query":NaN,"top_k":3}',
                        }
                    ]
                )
            ]
        )
    )
    guard = app.state.openai_guard = OpenAIAccessGuard("owner", 100, 100, 1)
    experiment.update(provider="openai", model_id="gpt-6-luna", reasoning_effort="medium")
    try:
        run = asyncio.run(
            ChatBenchmarkExecutor(app).execute(
                "Ý nghĩa Bạch Đằng?", freeze_configuration(app, experiment, {}), threading.Event()
            )
        )
        assert run["completion_status"] == "failed" and run["error_reason"] == "GuardrailError"
        call = run["diagnostics"]["tool_calls"][0]
        assert call["arguments_valid"] is False and call["status"] == "failed"
        assert len(call["argument_fingerprint"]) == 64
    finally:
        guard.close()


def test_production_benchmark_api_freezes_real_runtime_and_rejects_duplicate_start(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.eval_studio.benchmark_api import router
    from app.eval_studio.store import EvalStore
    from app.eval_studio.v11 import StudioV11
    from tests.test_openai_responses import Client, Stream

    entered = threading.Event()

    class WaitingStream(Stream):
        async def __anext__(self):
            entered.set()
            await asyncio.Event().wait()

    app, configuration, _, _, _ = setup_app(monkeypatch)
    store = EvalStore(tmp_path / "production-api.sqlite")
    question = store.questions()[0]
    store.save_question(
        {**question, "question": "Ý nghĩa Bạch Đằng?", "reference_answer": "REFERENCE NOT SENT"}
    )
    store.snapshot("native-api")
    studio = StudioV11(store)
    experiment = studio.create_experiment(
        {
            "name": "Native API fixture",
            "snapshot_id": "native-api",
            "provider": "openai",
            "model_id": "gpt-6-luna",
            "reasoning_effort": "medium",
            "chat_mode": "hybrid",
            "dense_backend": "faiss",
            "final_k": 3,
        }
    )
    app.state.eval_store, app.state.eval_v11 = store, studio
    stream = WaitingStream([])
    client_model = app.state.openai_client = Client(stream)
    guard = app.state.openai_guard = OpenAIAccessGuard("owner", 100, 100, 1)
    monkeypatch.setattr(settings, "eval_studio_enabled", True)

    @app.middleware("http")
    async def owner(request, call_next):
        request.state.owner_authenticated = True
        return await call_next(request)

    app.include_router(router)
    try:
        with TestClient(app) as client:
            created = client.post(
                "/api/v1/eval/benchmarks",
                json={
                    "experiment_id": experiment["id"],
                    "question_ids": [question["id"]],
                    "timeout_seconds": 3,
                    "requests_per_minute": 600,
                },
            )
            assert created.status_code == 201, created.text
            frozen = created.json()["configuration"]
            assert frozen["provider"] == "openai" and frozen["model_id"] == "gpt-6-luna"
            assert not client_model.requests
            path = f"/api/v1/eval/benchmarks/{created.json()['id']}"
            assert client.post(path + "/start", json={"confirm_live": True}).status_code == 200
            assert entered.wait(1)
            assert client.post(path + "/start", json={"confirm_live": True}).status_code == 409
            assert len(client_model.requests) == 1
            assert "REFERENCE NOT SENT" not in json.dumps(client_model.requests)
            assert client.post(path + "/cancel").status_code == 200
            for _ in range(100):
                detail = client.get(path).json()
                if detail["status"] == "cancelled":
                    break
                time.sleep(0.01)
            assert detail["status"] == "cancelled" and stream.closed
    finally:
        guard.close()


def test_external_timeout_releases_native_slot_without_setting_job_cancel(monkeypatch):
    from app.eval_studio.execution import ChatBenchmarkExecutor, freeze_configuration
    from tests.test_openai_responses import Client, Stream

    class WaitingStream(Stream):
        async def __anext__(self):
            entered.set()
            await asyncio.Event().wait()

    app, experiment, _, _, _ = setup_app(monkeypatch)
    entered = threading.Event()
    stream = WaitingStream([])
    app.state.openai_client = Client(stream)
    guard = app.state.openai_guard = OpenAIAccessGuard("owner", 100, 100, 1)
    experiment.update(provider="openai", model_id="gpt-6-luna", reasoning_effort="medium")
    frozen = freeze_configuration(app, experiment, {})
    cancel = threading.Event()

    async def execute():
        task = asyncio.create_task(ChatBenchmarkExecutor(app).execute("Ý nghĩa Bạch Đằng?", frozen, cancel))
        while not entered.is_set():
            await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(execute())
        assert not cancel.is_set()
        assert stream.closed
        lease = guard.acquire()
        assert lease
        guard.release(lease)
    finally:
        guard.close()


def test_failed_mcp_tool_stays_failed_after_diagnostic_finalization():
    from pydantic import BaseModel
    from app.eval_studio.execution import _Capture, _Registry
    from app.eval_studio.behavior import evaluate_behavior

    class Arguments(BaseModel):
        query: str

    class FailingMCP:
        name = "mcp_fixture_search"
        description = "Offline failing fixture"
        input_schema = Arguments
        descriptor = {
            "name": name,
            "description": description,
            "input_schema": Arguments.model_json_schema(),
            "provider": "mcp",
        }

        async def run(self, arguments):
            raise RuntimeError("Private upstream error that must never be stored")

    registry, capture = ToolRegistry(), _Capture()
    registry.register(FailingMCP())
    wrapped = _Registry(registry, {"tool_definitions": registry.describe()}, capture)
    result, record = asyncio.run(wrapped.call("mcp_fixture_search", {"query": "fixture"}))
    assert result is None and record.error == "MCP tool failed"
    capture.prepared = SimpleNamespace(tool_calls=[{"name": record.name, "error": record.error}])
    capture.finalize_tools()
    assert capture.tools[0]["status"] == "failed"
    assert capture.tools[0]["error_code"] == "tool_failed"
    metrics = evaluate_behavior(
        {}, {"answer": "Fixture"}, [], {"capture": {"tool_trace": True}, "tool_calls": capture.tools}
    )
    assert metrics["tools"]["success_rate"] == 0
    assert "Private upstream" not in json.dumps(capture.tools)


def test_missing_evidence_provenance_is_not_invented():
    from app.eval_studio.execution import _Capture, _evidence

    capture = _Capture()
    capture.retrieved = [{"chunk_id": "external-fixture", "text": "Captured text"}]
    assert _evidence(capture, [])[0]["source_kind"] is None


def test_overlapping_cooperative_and_timeout_cancellation_finishes_provider_cleanup(monkeypatch):
    from app.eval_studio.execution import ChatBenchmarkExecutor, freeze_configuration
    from tests.test_openai_responses import Client, Stream

    entered, closing = threading.Event(), threading.Event()

    class SlowClosingStream(Stream):
        async def __anext__(self):
            entered.set()
            await asyncio.Event().wait()

        async def close(self):
            closing.set()
            await asyncio.sleep(0.1)
            self.closed = True

    app, experiment, _, _, _ = setup_app(monkeypatch)
    stream = SlowClosingStream([])
    app.state.openai_client = Client(stream)
    guard = app.state.openai_guard = OpenAIAccessGuard("owner", 100, 100, 1)
    experiment.update(provider="openai", model_id="gpt-6-luna", reasoning_effort="medium")
    frozen = freeze_configuration(app, experiment, {})
    cancel = threading.Event()

    async def exercise():
        task = asyncio.create_task(ChatBenchmarkExecutor(app).execute("Ý nghĩa Bạch Đằng?", frozen, cancel))
        while not entered.is_set():
            await asyncio.sleep(0.005)
        cancel.set()
        while not closing.is_set():
            await asyncio.sleep(0.005)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    try:
        asyncio.run(exercise())
        assert stream.closed
        lease = guard.acquire()
        assert lease
        guard.release(lease)
    finally:
        guard.close()
