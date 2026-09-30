"""Small variant and startup checks without downloading Qwen model weights."""

import asyncio
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from app.chat_modes import ChatMode
from app.config import HYBRID_MODEL_ID, Settings
from app.models.qwen import QwenRuntime


@pytest.mark.parametrize("variant", ["vanilla", "sft"])
def test_full_startup_loads_selected_hybrid_once(monkeypatch, tmp_path, variant):
    main = importlib.import_module("app.main")
    adapter = tmp_path / "adapter"
    if variant == "sft":
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    settings = Settings(_env_file=None, app_mode="full", enable_central_mode=False,
                        runtime_loading_strategy="lazy", model_variant=variant,
                        model_adapter_path=adapter if variant == "sft" else None)
    monkeypatch.setattr(main, "settings", settings)
    monkeypatch.setattr(main, "RAGService", lambda: SimpleNamespace(load=lambda: None,
                                                                    shutdown=lambda: None))
    monkeypatch.setattr(main, "ConversationStore", lambda _: object())
    monkeypatch.setattr(main, "HybridRetriever", lambda _: object())
    monkeypatch.setattr(main, "AttachmentService", lambda **_: object())
    monkeypatch.setattr(main, "TemporaryCorpusRetriever", lambda **_: object())

    class FakeQwen:
        def __init__(self, *, model_id, adapter_path, **_):
            self.model_id = model_id
            self.adapter_path = adapter_path
            self.model = None
            self.loads = 0

        def load(self):
            self.loads += 1
            self.model = SimpleNamespace(peft_config={"default": {}} if self.adapter_path else None)

    monkeypatch.setattr(main, "QwenRuntime", FakeQwen)
    app = SimpleNamespace(state=SimpleNamespace())

    async def start():
        async with main.lifespan(app):
            runtime = app.state.chat_mode_router.runtime_for(ChatMode.HYBRID)
            assert runtime is app.state.hybrid_runtime
            assert runtime.model.loads == 1
            assert runtime.model.model_id == HYBRID_MODEL_ID
            assert runtime.model.adapter_path == (adapter if variant == "sft" else None)
            assert bool(runtime.model.model.peft_config) == (variant == "sft")

    asyncio.run(start())


def test_qwen_load_attaches_peft_to_same_model_used_by_stream(monkeypatch, tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    calls = []

    class FakeBase:
        config = SimpleNamespace(_commit_hash="base-revision")

        def to(self, device):
            return self

        def eval(self):
            return self

    class FakeAdapter(FakeBase):
        peft_config = {"default": object()}

        def __init__(self, base):
            self.base = base

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda model_id, **kwargs:
        calls.append(("tokenizer", model_id)) or SimpleNamespace(pad_token_id=0))
    transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=lambda model_id, **kwargs:
        calls.append(("base", model_id)) or FakeBase())
    peft = ModuleType("peft")
    peft.PeftConfig = SimpleNamespace(from_pretrained=lambda path, **kwargs:
        SimpleNamespace(base_model_name_or_path=HYBRID_MODEL_ID))
    peft.PeftModel = SimpleNamespace(from_pretrained=lambda base, path, **kwargs:
        calls.append(("adapter", path)) or FakeAdapter(base))
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "peft", peft)

    vanilla = QwenRuntime(model_id=HYBRID_MODEL_ID, dtype="float32")
    vanilla.load()
    assert isinstance(vanilla.model, FakeBase)
    assert not hasattr(vanilla.model, "peft_config")
    assert calls == [("tokenizer", HYBRID_MODEL_ID), ("base", HYBRID_MODEL_ID)]

    calls.clear()
    sft = QwenRuntime(model_id=HYBRID_MODEL_ID, adapter_path=adapter, dtype="float32")
    sft.load()
    sft.load()
    assert isinstance(sft.model, FakeAdapter)
    assert sft.model.peft_config
    assert calls == [("tokenizer", HYBRID_MODEL_ID), ("base", HYBRID_MODEL_ID),
                     ("adapter", str(adapter))]


def test_sft_missing_weights_fails_before_model_load(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="adapter_model.safetensors"):
        QwenRuntime(model_id=HYBRID_MODEL_ID, adapter_path=adapter)


def test_full_startup_rejects_incomplete_sft_adapter(monkeypatch, tmp_path):
    main = importlib.import_module("app.main")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(main, "settings", Settings(_env_file=None, app_mode="full",
                        enable_central_mode=False, model_variant="sft", model_adapter_path=adapter))
    monkeypatch.setattr(main, "RAGService", lambda: SimpleNamespace(load=lambda: None,
                                                                    shutdown=lambda: None))
    monkeypatch.setattr(main, "ConversationStore", lambda _: object())
    monkeypatch.setattr(main, "HybridRetriever", lambda _: object())
    monkeypatch.setattr(main, "AttachmentService", lambda **_: object())
    monkeypatch.setattr(main, "TemporaryCorpusRetriever", lambda **_: object())

    async def start():
        async with main.lifespan(SimpleNamespace(state=SimpleNamespace())):
            pytest.fail("startup accepted an incomplete SFT adapter")

    with pytest.raises(FileNotFoundError, match="adapter_model.safetensors"):
        asyncio.run(start())
