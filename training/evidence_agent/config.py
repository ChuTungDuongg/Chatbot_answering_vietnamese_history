from __future__ import annotations

from dataclasses import dataclass

from training.common.qlora import LoRASettings
from app.agents.common.model_registry import SHARED_BASE_MODEL_ID


@dataclass(frozen=True)
class EvidenceAgentConfig:
    model_id: str = SHARED_BASE_MODEL_ID
    max_length: int = 4096
    epochs: int = 3
    train_batch_size: int = 4
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    logging_steps: int = 25
    eval_steps: int = 100
    save_steps: int = 100
    lora: LoRASettings = LoRASettings(r=16, alpha=32)



