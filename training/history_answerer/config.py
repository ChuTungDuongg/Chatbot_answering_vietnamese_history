from __future__ import annotations

from dataclasses import dataclass

from training.common.qlora import LoRASettings


BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


@dataclass(frozen=True)
class Phase1Config:
    model_id: str = BASE_MODEL_ID
    max_length: int = 1024
    max_samples: int | None = 100_000
    train_ratio: float = 0.90
    eval_ratio: float = 0.05
    epochs: int = 1
    train_batch_size: int = 4
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    eval_steps: int = 500
    save_steps: int = 500
    early_stopping_patience: int = 3
    logging_steps: int = 50
    lora: LoRASettings = LoRASettings()


@dataclass(frozen=True)
class Phase6Config:
    model_id: str = BASE_MODEL_ID
    max_length: int = 4096
    train_ratio: float = 0.90
    eval_ratio: float = 0.05
    epochs: int = 5
    train_batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1.5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    early_stopping_patience: int = 4
    source_line_loss_weight: float = 1.6
    answer_loss_weight: float = 1.0
    lora: LoRASettings = LoRASettings()



