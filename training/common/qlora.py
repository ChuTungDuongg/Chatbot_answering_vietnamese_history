from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class LoRASettings:
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: tuple[str, ...] = field(default_factory=lambda: DEFAULT_LORA_TARGETS)


@dataclass(frozen=True)
class QLoRASettings:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"


def build_bnb_config(settings: QLoRASettings):
    import torch
    from transformers import BitsAndBytesConfig

    dtype = getattr(torch, settings.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=settings.load_in_4bit,
        bnb_4bit_quant_type=settings.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=settings.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=dtype,
    )


def build_lora_config(settings: LoRASettings):
    from peft import LoraConfig

    return LoraConfig(
        r=settings.r,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(settings.target_modules),
    )

