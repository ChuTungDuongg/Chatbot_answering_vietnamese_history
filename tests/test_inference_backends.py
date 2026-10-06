"""Engine contract/parity checks without GPUs, model downloads or vLLM installed."""
import asyncio
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import pytest
from pydantic import ValidationError

from app.config import HYBRID_MODEL_ID, Settings
from app.models.base import ModelDelta, ModelDone, chat_prompt
from app.models.factory import build_model_runtime
from app.models.qwen import QwenRuntime
from app.models.vllm import VLLMRuntime


def test_default_factory_and_invalid_backend():
    settings = Settings(_env_file=None)
    assert settings.inference_backend == "transformers"
    assert isinstance(build_model_runtime(settings=settings, model_id=HYBRID_MODEL_ID), QwenRuntime)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, inference_backend="invalid")
    with pytest.raises(ValueError, match="Unsupported"):
        build_model_runtime(settings=SimpleNamespace(inference_backend="invalid"), model_id=HYBRID_MODEL_ID)


def test_vllm_requires_isolation_and_rejects_unverified_sft():
    with pytest.raises(ValidationError, match="isolated"):
        Settings(_env_file=None, app_mode="full", inference_backend="vllm")
    settings = Settings(_env_file=None, app_mode="full", inference_backend="vllm", enable_central_mode=False)
    assert isinstance(build_model_runtime(settings=settings, model_id=HYBRID_MODEL_ID), VLLMRuntime)
    with pytest.raises(NotImplementedError, match="SFT/LoRA"):
        VLLMRuntime(model_id=HYBRID_MODEL_ID, adapter_path="adapter")


@pytest.fixture
def fake_vllm(monkeypatch):
    captures = {"loads": 0, "aborts": [], "closed": 0, "hang": False, "reason": "stop"}
    class Tokenizer:
        pad_token_id, eos_token_id, eos_token = 151643, 151645, "EOS"
        def apply_chat_template(self, messages, **options):
            captures["template"] = (messages, options)
            return "rendered prompt"
        def __call__(self, prompt, **kwargs):
            assert kwargs == {"add_special_tokens": False}
            captures["prompt"] = prompt
            return {"input_ids": [11, 12]}
    tokenizer = Tokenizer()
    class Engine:
        def __init__(self, args):
            captures["loads"] += 1
            captures["args"] = args
            self.vllm_config = SimpleNamespace(
                cache_config=SimpleNamespace(enable_prefix_caching=args.enable_prefix_caching,
                                             gpu_memory_utilization=args.gpu_memory_utilization),
                scheduler_config=SimpleNamespace(max_num_seqs=args.max_num_seqs),
                model_config=SimpleNamespace(max_model_len=262144, dtype="torch.bfloat16",
                                             enforce_eager=args.enforce_eager,
                                             generation_config=args.generation_config, quantization=None),
                parallel_config=SimpleNamespace(tensor_parallel_size=1))
        async def generate(self, prompt, params, request_id):
            captures.update(prompt_tokens=prompt["prompt_token_ids"], sampling=params, request_id=request_id)
            if captures["hang"]:
                await asyncio.sleep(100)
            now = time.monotonic()
            stats = SimpleNamespace(scheduled_ts=now, first_token_ts=now + .001, last_token_ts=now + .002)
            yield SimpleNamespace(finished=False, metrics=None, outputs=[SimpleNamespace(text="", token_ids=[21])])
            yield SimpleNamespace(finished=True, metrics=stats, outputs=[SimpleNamespace(
                text="Đáp án", token_ids=[151645], finish_reason=captures["reason"])])
        async def abort(self, request_id): captures["aborts"].append(request_id)
        def shutdown(self): captures["closed"] += 1
    modules = {
        "vllm": {"SamplingParams": lambda **kwargs: SimpleNamespace(**kwargs), "TokensPrompt": lambda **kwargs: kwargs},
        "vllm.sampling_params": {"RequestOutputKind": SimpleNamespace(DELTA="delta")},
        "vllm.engine.arg_utils": {"AsyncEngineArgs": lambda **kwargs: SimpleNamespace(**kwargs)},
        "vllm.v1.engine.async_llm": {"AsyncLLM": SimpleNamespace(from_engine_args=lambda args: Engine(args))},
        "transformers": {"AutoConfig": SimpleNamespace(from_pretrained=lambda *a, **kw: SimpleNamespace(_commit_hash="revision-sha")),
                         "AutoTokenizer": SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer)},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    import app.models.vllm as module
    original = module.importlib.metadata.version
    monkeypatch.setattr(module.importlib.metadata, "version", lambda package: "0.23.0" if package == "vllm" else original(package))
    return captures, tokenizer


@pytest.mark.parametrize("reason,max_tokens,hit,truncated", [("stop", 2, False, False), ("length", 2, True, True), ("length", 10, False, True)])
def test_vllm_stream_contract_timing_tokens_and_termination(fake_vllm, reason, max_tokens, hit, truncated):
    captures, _ = fake_vllm
    captures["reason"] = reason
    runtime = VLLMRuntime(model_id=HYBRID_MODEL_ID)
    async def run():
        events = [event async for event in runtime.stream([{"role": "user", "content": "Q"}],
                                                         max_new_tokens=max_tokens, cancel=threading.Event())]
        assert len(events) == 2 and isinstance(events[0], ModelDelta) and isinstance(events[1], ModelDone)
        done = events[1]
        assert done.input_tokens == 2 and done.output_tokens == 2
        assert done.model_revision == "revision-sha"
        assert done.finish_reason == reason and done.hit_max_new_tokens is hit and done.truncated is truncated
        assert done.started_ns <= done.first_token_ns <= done.finished_ns
        assert done.metrics["model_timing_observer"] == "vllm_engine_core_monotonic"
        assert done.metrics["tpot_ms"] == pytest.approx(1.0, abs=.001)
        assert captures["sampling"].temperature == 0 and captures["sampling"].top_p == 1
        assert captures["sampling"].output_kind == "delta"
        assert runtime.is_loaded
        await runtime.aclose()
        assert not runtime.is_loaded and captures["closed"] == 1
    asyncio.run(run())


def test_vllm_honors_hermes_tools_and_effective_config(fake_vllm):
    captures, tokenizer = fake_vllm
    messages = [{"role": "user", "content": "Q"}]
    tools = [{"type": "function", "function": {"name": "search_history"}}]
    expected = chat_prompt(tokenizer, messages, tools=tools, enable_thinking=False)
    async def run():
        runtime = VLLMRuntime(model_id=HYBRID_MODEL_ID, enable_prefix_caching=True, max_num_seqs=4)
        await asyncio.gather(runtime.aload(), runtime.aload())
        assert captures["loads"] == 1
        answer, done = await runtime.generate(messages, tools=tools, max_new_tokens=20)
        assert answer == "Đáp án" and done.output_tokens == 2
        assert captures["prompt"] == expected
        assert captures["template"] == (messages, {"tokenize": False, "add_generation_prompt": True,
                                                 "enable_thinking": False, "tools": tools})
        settings = runtime.engine_metadata
        assert settings["inference_backend"] == "vllm" and settings["inference_engine_version"] == "0.23.0"
        assert settings["inference_engine_config"]["enable_prefix_caching"] is True
        assert settings["inference_engine_config"]["max_num_seqs"] == 4
        assert settings["inference_engine_config"]["max_model_len"] == 262144
        assert captures["args"].quantization is None and captures["args"].generation_config == "vllm"
        await runtime.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("cancel_task", [False, True])
def test_vllm_cancellation_aborts_pending_engine_request(fake_vllm, cancel_task):
    captures, _ = fake_vllm
    captures["hang"] = True
    async def run():
        runtime = VLLMRuntime(model_id=HYBRID_MODEL_ID)
        cancel = threading.Event()
        async def consume():
            return [event async for event in runtime.stream([], max_new_tokens=10, cancel=cancel)]
        task = asyncio.create_task(consume())
        while "request_id" not in captures:
            await asyncio.sleep(.001)
        if cancel_task:
            task.cancel()
            with pytest.raises(asyncio.CancelledError): await task
        else:
            cancel.set()
            assert await task == []
        assert captures["aborts"] == [captures["request_id"]]
        await runtime.aclose()
    asyncio.run(run())


def test_pre_cancelled_vllm_does_not_load_or_allocate(fake_vllm):
    captures, _ = fake_vllm
    async def run():
        cancel = threading.Event(); cancel.set()
        runtime = VLLMRuntime(model_id=HYBRID_MODEL_ID)
        assert [event async for event in runtime.stream([], max_new_tokens=10, cancel=cancel)] == []
        assert captures["loads"] == 0
    asyncio.run(run())


def test_baseline_metadata_reports_backend_identity(monkeypatch):
    from app.services import metadata
    model = QwenRuntime(model_id=HYBRID_MODEL_ID)
    monkeypatch.setattr(metadata, "_hash_file", lambda _: "corpus")
    monkeypatch.setattr(metadata, "_hash_index", lambda _: "index")
    monkeypatch.setattr(metadata, "_hardware", lambda: {})
    app = SimpleNamespace(state=SimpleNamespace(hybrid_runtime=SimpleNamespace(model=model), retriever=None))
    result = metadata.build_baseline_metadata("hybrid", app)
    assert result["inference_backend"] == "transformers"
    assert result["inference_engine_version"]
    assert result["generation_settings"]["do_sample"] is False
