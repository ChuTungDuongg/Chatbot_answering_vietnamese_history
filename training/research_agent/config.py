from __future__ import annotations

from dataclasses import dataclass

from training.common.qlora import LoRASettings


@dataclass(frozen=True)
class ResearchAgentConfig:
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_length: int = 4096
    epochs: int = 3
    train_batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    logging_steps: int = 25
    eval_steps: int = 100
    save_steps: int = 100
    lora: LoRASettings = LoRASettings()



