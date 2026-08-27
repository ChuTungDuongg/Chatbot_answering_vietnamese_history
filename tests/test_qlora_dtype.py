import pytest
import sys
from types import SimpleNamespace

from training.common.qlora import QLoRASettings, build_bnb_config, resolve_precision


def test_bf16_controls_bnb_compute_dtype():
    value = resolve_precision(bf16=True, fp16=False, bf16_supported=True)
    assert (value.bf16, value.fp16, value.compute_dtype) == (True, False, "bfloat16")


def test_fp16_controls_bnb_compute_dtype():
    value = resolve_precision(bf16=False, fp16=True, bf16_supported=False)
    assert (value.bf16, value.fp16, value.compute_dtype) == (False, True, "float16")


def test_auto_uses_hardware_and_explicit_mismatch_is_rejected():
    assert resolve_precision(bf16=None, fp16=None, bf16_supported=True).compute_dtype == "bfloat16"
    assert resolve_precision(bf16=None, fp16=None, bf16_supported=False).compute_dtype == "float16"
    assert resolve_precision(bf16=False, fp16=False, bf16_supported=False).compute_dtype == "float32"
    with pytest.raises(ValueError, match="does not match"):
        resolve_precision(bf16=False, fp16=True, bnb_compute_dtype="bfloat16", bf16_supported=True)


def test_bnb_config_receives_torch_dtype(monkeypatch):
    import torch

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(BitsAndBytesConfig=FakeConfig))
    bf16 = build_bnb_config(QLoRASettings(bnb_4bit_compute_dtype="bfloat16"))
    fp16 = build_bnb_config(QLoRASettings(bnb_4bit_compute_dtype="float16"))
    assert bf16.kwargs["bnb_4bit_compute_dtype"] is torch.bfloat16
    assert fp16.kwargs["bnb_4bit_compute_dtype"] is torch.float16
