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
    bias: str = "none"
    target_modules: tuple[str, ...] = field(default_factory=lambda: DEFAULT_LORA_TARGETS)


@dataclass(frozen=True)
class QLoRASettings:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"


@dataclass(frozen=True)
class PrecisionSettings:
    compute_dtype: str
    bf16: bool
    fp16: bool


def resolve_precision(
    *,
    bf16: bool | None,
    fp16: bool | None,
    bnb_compute_dtype: str = "auto",
    bf16_supported: bool | None = None,
) -> PrecisionSettings:
    """Resolve Trainer and bitsandbytes precision from one contract.

    With no explicit trainer flags, auto uses BF16 only on supported hardware and
    otherwise FP16. Explicit flags always win. Disabling both selects FP32.
    """
    if bf16 is True and fp16 is True:
        raise ValueError("BF16 and FP16 cannot both be enabled")
    if bf16_supported is None:
        try:
            import torch

            bf16_supported = bool(
                torch.cuda.is_available()
                and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            )
        except (ImportError, RuntimeError):
            bf16_supported = False

    flags_unspecified = bf16 is None and fp16 is None
    if flags_unspecified:
        resolved_bf16 = bool(bf16_supported)
        resolved_fp16 = not resolved_bf16
    else:
        resolved_bf16 = bf16 is True
        resolved_fp16 = fp16 is True

    trainer_dtype = "bfloat16" if resolved_bf16 else "float16" if resolved_fp16 else "float32"
    requested = trainer_dtype if bnb_compute_dtype == "auto" else bnb_compute_dtype
    if requested != trainer_dtype:
        raise ValueError(
            f"BitsAndBytes compute dtype {requested} does not match Trainer dtype {trainer_dtype}. "
            "Use --bnb-compute-dtype auto or matching --bf16/--fp16 flags."
        )
    if requested == "bfloat16" and not bf16_supported:
        raise ValueError("BF16 was requested but the current CUDA device does not support BF16; use --no-bf16 --fp16.")
    return PrecisionSettings(compute_dtype=requested, bf16=resolved_bf16, fp16=resolved_fp16)


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
        bias=settings.bias,
        task_type="CAUSAL_LM",
        target_modules=list(settings.target_modules),
    )
