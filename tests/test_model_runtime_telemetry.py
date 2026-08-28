from __future__ import annotations

import torch

from app.agents.model_runtime import SharedAgentModelRuntime
from app.telemetry import RequestTelemetry, reset_request_telemetry, set_request_telemetry


class FakeInputs(dict):
    def to(self, _device):
        return self


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.outputs = ["{}"]

    def __call__(self, prompt, return_tensors, add_special_tokens):
        del prompt, return_tensors, add_special_tokens
        return FakeInputs({"input_ids": torch.tensor([[10, 11, 12]])})

    def decode(self, _tokens, skip_special_tokens):
        del _tokens, skip_special_tokens
        return self.outputs.pop(0)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        del tokenize, add_generation_prompt
        return "\n".join(item["content"] for item in messages)


class FakeEmbeddings:
    weight = torch.empty(1, device="cpu")


class FakeModel:
    hf_device_map = {"model": 0}

    def __init__(self):
        self.adapters = []

    def set_adapter(self, adapter):
        self.adapters.append(adapter)

    def generate(self, **_kwargs):
        return torch.tensor([[10, 11, 12, 13, 14]])

    def get_input_embeddings(self):
        return FakeEmbeddings()


def _runtime(tokenizer):
    runtime = SharedAgentModelRuntime.__new__(SharedAgentModelRuntime)
    runtime.tokenizer = tokenizer
    runtime.model = FakeModel()
    runtime.adapters = frozenset({"research", "evidence", "history"})
    runtime._lock = __import__("threading").RLock()
    return runtime


def test_actual_model_generate_counting():
    telemetry = RequestTelemetry(request_id="req-1")
    token = set_request_telemetry(telemetry)
    try:
        output = _runtime(FakeTokenizer()).generate_prompt(adapter="research", prompt="hello")
    finally:
        reset_request_telemetry(token)

    assert output == "{}"
    assert telemetry.total_llm_calls == 1
    assert telemetry.research_llm_calls == 1
    assert telemetry.total_input_tokens == 3
    assert telemetry.total_output_tokens == 2


def test_research_json_repair_counting():
    tokenizer = FakeTokenizer()
    tokenizer.outputs = ["not json", '{"ok": true}']
    runtime = _runtime(tokenizer)
    telemetry = RequestTelemetry(request_id="req-2")
    token = set_request_telemetry(telemetry)
    try:
        value = runtime.generate_json(
            adapter="research",
            messages=[{"role": "user", "content": "return json"}],
            repair=True,
        )
    finally:
        reset_request_telemetry(token)

    assert value == {"ok": True}
    assert telemetry.total_llm_calls == 2
    assert telemetry.research_json_repairs == 1
