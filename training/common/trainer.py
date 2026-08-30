from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonlMetricsCallback:
    """Persist inexpensive Trainer logs plus timing and CUDA peak telemetry."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        effective_batch_size: int | None = None,
        mean_train_tokens: float | None = None,
    ):
        self.path = Path(output_dir) / "training_log.jsonl"
        self.effective_batch_size = effective_batch_size
        self.mean_train_tokens = mean_train_tokens
        self.started_at = time.monotonic()

    def on_train_begin(self, args, state, control, **kwargs):
        self.started_at = time.monotonic()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except (ImportError, RuntimeError):
            pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        elapsed = max(0.0, time.monotonic() - self.started_at)
        max_steps = int(getattr(state, "max_steps", 0) or 0)
        step = int(getattr(state, "global_step", 0) or 0)
        remaining = None
        if step > 0 and max_steps > step:
            remaining = elapsed / step * (max_steps - step)
        tokens_processed = None
        if self.effective_batch_size and self.mean_train_tokens is not None:
            tokens_processed = int(step * self.effective_batch_size * self.mean_train_tokens)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "epoch": getattr(state, "epoch", None),
            "elapsed_seconds": round(elapsed, 3),
            "estimated_remaining_seconds": round(remaining, 3) if remaining is not None else None,
            "effective_batch_size": self.effective_batch_size,
            "tokens_processed_estimate": tokens_processed,
            **dict(logs or {}),
            **{f"gpu_{key}": value for key, value in summarize_gpu().items()},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

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
    """Backward-compatible shared defaults for the other training CLIs."""
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


def build_metrics_callback(
    output_dir: str | Path,
    *,
    effective_batch_size: int | None = None,
    mean_train_tokens: float | None = None,
):
    from transformers import TrainerCallback

    class MetricsCallback(JsonlMetricsCallback, TrainerCallback):
        pass

    return MetricsCallback(
        output_dir,
        effective_batch_size=effective_batch_size,
        mean_train_tokens=mean_train_tokens,
    )


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
            "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        }
    except Exception as exc:
        return {"cuda": False, "error": str(exc)}
