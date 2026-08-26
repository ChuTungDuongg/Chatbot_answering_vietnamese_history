from __future__ import annotations

from pathlib import Path
from typing import Any


class JsonlMetricsCallback:
    """Persist Trainer logs with lightweight GPU memory telemetry."""

    def __init__(self, output_dir: str | Path):
        self.path = Path(output_dir) / "training_log.jsonl"

    def on_log(self, args, state, control, logs=None, **kwargs):
        import json

        payload = {
            "step": state.global_step,
            "epoch": state.epoch,
            **(logs or {}),
            **{f"gpu_{key}": value for key, value in summarize_gpu().items()},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def on_save(self, args, state, control, **kwargs):
        self.on_log(
            args,
            state,
            control,
            {"checkpoint": str(Path(args.output_dir) / f"checkpoint-{state.global_step}")},
        )


def build_training_arguments(
    *,
    output_dir: str | Path,
    epochs: float,
    train_batch_size: int,
    eval_batch_size: int,
    grad_accum_steps: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    logging_steps: int,
    eval_steps: int,
    save_steps: int,
    bf16: bool = True,
    fp16: bool = False,
    report_to: str = "none",
    gradient_checkpointing: bool = True,
    seed: int = 42,
):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=bf16,
        fp16=fp16,
        report_to=[] if report_to == "none" else [report_to],
        remove_unused_columns=False,
        gradient_checkpointing=gradient_checkpointing,
        seed=seed,
        data_seed=seed,
    )


def build_metrics_callback(output_dir: str | Path):
    from transformers import TrainerCallback

    class MetricsCallback(JsonlMetricsCallback, TrainerCallback):
        pass

    return MetricsCallback(output_dir)


def summarize_gpu() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda": False}
        return {
            "cuda": True,
            "device": torch.cuda.get_device_name(0),
            "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
        }
    except Exception as exc:
        return {"cuda": False, "error": str(exc)}
