"""Actual sequence termination and live vanilla/SFT budget/trace coverage."""

import asyncio
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from app.config import CENTRAL_MODEL_ID, HYBRID_MODEL_ID, Settings
from app.models.base import ModelDelta, ModelDone
from app.models.qwen import QwenRuntime
from tests.test_baseline_runtime import _app, _events


@pytest.mark.parametrize("override", [None, "1024", "2048"])
def test_hybrid_default_and_environment_override(monkeypatch, override):
    monkeypatch.delenv("HYBRID_MAX_NEW_TOKENS", raising=False)
    if override is not None:
        monkeypatch.setenv("HYBRID_MAX_NEW_TOKENS", override)
    cfg = Settings(_env_file=None)
    assert cfg.hybrid_max_new_tokens == (int(override) if override else 1536)
    assert cfg.central_final_max_new_tokens == 1536
    assert not cfg.do_sample and not cfg.enable_thinking
    assert cfg.model_temperature == .7 and cfg.model_top_p == 1.0


@pytest.mark.parametrize("variant", ["vanilla", "sft"])
@pytest.mark.parametrize("budget", [1536, 1024, 2048])
@pytest.mark.parametrize("endpoint", ["chat", "chat/stream"])
def test_hybrid_live_budget_and_completed_trace_match_persisted_trace(monkeypatch, tmp_path, variant, budget, endpoint):
    import app.api.routes as routes
    app, store, model, _ = _app(tmp_path)
    monkeypatch.setenv("HYBRID_MAX_NEW_TOKENS", str(budget))
    cfg = Settings(_env_file=None)
    monkeypatch.setattr(routes, "settings", cfg)
    model.model_variant = variant
    model.adapter_attached = variant == "sft"
    received = []
    async def stream(messages, *, max_new_tokens, cancel):
        received.append(max_new_tokens)
        started = time.perf_counter_ns()
        yield ModelDelta("Answer [c1].", time.perf_counter_ns())
        yield ModelDone(HYBRID_MODEL_ID, "test-revision", started, started + 1, time.perf_counter_ns(),
                        20, 430, max_new_tokens, "stop", False, False)
    model.stream = stream
    owner = "budget-test"
    conversation = store.create_conversation(owner)
    with TestClient(app) as client:
        response = client.post("/api/v1/" + endpoint, headers={"X-Client-ID": owner}, json={
            "conversation_id": conversation["id"], "question": "Bạch Đằng năm 938?", "debug": True})
    assert response.status_code == 200
    if endpoint == "chat":
        debug = response.json()["debug"]
        latency = response.json()["latency_ms"]
    else:
        events = _events(response.text)
        debug = next(data for name, data in events if name == "debug_trace")
        done = events[-1][1]
        latency = done["latency_ms"]
        assert events[0][0] == "status"
        assert next(data for name, data in events if name == "answer_delta")["delta"] == "Answer [c1]."
        assert done["generation_settings"]["max_new_tokens"] == budget
        assert done["metrics"] == debug["performance"]
    assert received == [cfg.hybrid_max_new_tokens]
    assert debug["generation"]["settings"]["max_new_tokens"] == budget
    assert debug["performance"]["max_new_tokens"] == budget
    assert debug["generation"]["finish_reason"] == "stop"
    assert debug["generation"]["hit_max_new_tokens"] is False and debug["generation"]["truncated"] is False
    assert debug["performance"]["e2e_ms"] == latency and latency > 0
    saved = store.list_messages(owner, conversation["id"])[-1]
    assert saved["debug_trace"] == debug and saved["content"] == "Answer [c1]."


@pytest.mark.parametrize("model_id", [HYBRID_MODEL_ID, CENTRAL_MODEL_ID])
@pytest.mark.parametrize("tokens,budget,other_stop,expose_sequences,reason", [
    ([1, 2, 99], 6, False, True, "stop"),
    ([1, 2], 2, False, True, "length"),
    ([1, 99], 2, False, True, "stop"),
    ([1], 6, False, True, None),
    ([1, 2], 2, False, False, None),
    ([1, 2], 2, True, True, None),
])
def test_qwen_actual_termination_without_guessing(model_id, tokens, budget, other_stop, expose_sequences, reason):
    import torch
    class Batch(dict):
        def to(self, device): return self
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 99
        def apply_chat_template(self, messages, **kwargs): return "prompt"
        def __call__(self, *args, **kwargs): return Batch(input_ids=torch.tensor([[50]]))
        def decode(self, ids, **kwargs): return "".join("text " for token in ids if int(token) != 99)
    class Model:
        generation_config = SimpleNamespace(max_time=1 if other_stop else None, stop_strings=None)
        def get_input_embeddings(self): return SimpleNamespace(weight=SimpleNamespace(device="cpu"))
        def generate(self, *, input_ids, streamer, stopping_criteria, **kwargs):
            streamer.put(input_ids)
            result = input_ids
            for token in tokens:
                result = torch.cat((result, torch.tensor([[token]])), dim=1)
                streamer.put(torch.tensor([[token]]))
                if stopping_criteria(result, None): break
            streamer.end()
            return result if expose_sequences else None
    runtime = QwenRuntime(model_id=model_id)
    runtime.model, runtime.tokenizer = Model(), Tokenizer()
    text, done = asyncio.run(runtime.generate([{"role": "user", "content": "x"}], max_new_tokens=budget))
    assert text
    assert done.output_tokens == len(tokens)
    assert done.max_new_tokens == budget and done.finish_reason == reason
    expected = None if reason is None else reason == "length"
    assert done.hit_max_new_tokens is expected and done.truncated is expected
    assert done.metrics["finish_reason"] == reason
    assert done.metrics["model_ttft_ms"] is not None


def test_completed_trace_update_is_owner_and_conversation_scoped(tmp_path):
    _, store, _, _ = _app(tmp_path)
    conversation = store.create_conversation("owner")
    another = store.create_conversation("owner")
    message = store.add_message("owner", conversation["id"], "assistant", "Answer", [])
    for owner, chat in [("other-owner", conversation["id"]), ("owner", another["id"])]:
        with pytest.raises(LookupError):
            store.update_message_debug_trace(owner, chat, message["id"], {"performance": {"e2e_ms": 5}})
    assert store.get_message("owner", conversation["id"], message["id"])["debug_trace"] is None
    store.update_message_debug_trace("owner", conversation["id"], message["id"], {"performance": {"e2e_ms": 5}})
    assert store.get_message("owner", conversation["id"], message["id"])["debug_trace"]["performance"]["e2e_ms"] == 5


@pytest.mark.parametrize("variant", ["vanilla", "sft"])
@pytest.mark.parametrize("budget", [1536, 2048])
def test_baseline_metadata_reports_actual_shared_hybrid_budget(monkeypatch, tmp_path, variant, budget):
    import app.services.metadata as metadata
    app, _, model, _ = _app(tmp_path)
    app.state.hybrid_runtime = app.state.chat_mode_router.runtime_for("hybrid")
    model.model_variant = variant
    model.adapter_attached = variant == "sft"
    monkeypatch.setenv("HYBRID_MAX_NEW_TOKENS", str(budget))
    monkeypatch.setattr(metadata, "settings", Settings(_env_file=None))
    monkeypatch.setattr(metadata, "_hash_file", lambda _: None)
    monkeypatch.setattr(metadata, "_hash_index", lambda _: None)
    monkeypatch.setattr(metadata, "_hardware", lambda: {})
    monkeypatch.setattr(metadata, "_git_commit", lambda: None)
    result = metadata.build_baseline_metadata("hybrid", app)
    assert result["generation_settings"]["max_new_tokens"] == budget
    assert result["model_variant"] == variant


def test_central_completed_sse_planning_budget_termination_and_latency_breakdown(tmp_path):
    from app.central.runtime import CentralRuntime
    from app.services.chat_mode_router import ChatModeRouter
    from tests.test_central_planning import Planner, READY, call, registry
    app, store, _, _ = _app(tmp_path)
    tools, _ = registry()
    model = Planner([call("search_history", query="Bạch Đằng") + READY])
    async def final_stream(messages, *, max_new_tokens, cancel):
        assert max_new_tokens == 1536
        started = time.perf_counter_ns()
        yield ModelDelta("Answer [c1].", time.perf_counter_ns())
        yield ModelDone(CENTRAL_MODEL_ID, "test-revision", started, started + 1, time.perf_counter_ns(),
                        100, max_new_tokens, max_new_tokens, "length", True, True)
    model.stream = final_stream
    central = CentralRuntime(model=model, tools=tools)
    app.state.chat_mode_router = ChatModeRouter(hybrid=None, central=central)
    owner = "central-trace"
    conversation = store.create_conversation(owner)
    with TestClient(app) as client:
        response = client.post("/api/v1/chat/stream", headers={"X-Client-ID": owner}, json={
            "conversation_id": conversation["id"], "question": "Bạch Đằng?", "mode": "central", "debug": True})
    events = _events(response.text)
    assert events[-1][1]["status"] == "done"
    debug = next(data for name, data in events if name == "debug_trace")
    metrics = debug["performance"]
    assert metrics["e2e_ms"] > 0 and metrics["model_calls"] == 2
    assert metrics["planning_model_ms"] >= metrics["planning_model_ttft_ms"] >= 0
    assert metrics["planning_input_tokens"] == 100 and metrics["planning_output_tokens"] == 20
    assert metrics["finish_reason"] == "length" and metrics["hit_max_new_tokens"] and metrics["truncated"]
    assert debug["generation"]["settings"]["max_new_tokens"] == 1536
    assert debug["planning"]["rounds"][0]["early_exit"]
    assert sum(metrics["pre_final_breakdown_ms"].values()) == pytest.approx(metrics["generation_start_ms"], abs=.01)
    assert metrics["itl_p50_ms"] is None  # Single delta, retain existing null semantics.
    assert store.list_messages(owner, conversation["id"])[-1]["debug_trace"] == debug
