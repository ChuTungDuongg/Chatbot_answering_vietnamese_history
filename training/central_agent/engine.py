from __future__ import annotations

import argparse
import math
import os
import shutil
from pathlib import Path
from typing import Any

from training.common.qlora import PrecisionSettings
from training.trajectory_dataset.io_utils import atomic_write_json
from training.trajectory_dataset.preprocess import build_canonical_sft_example

from .config import build_lora_settings, build_qlora_settings, effective_train_batch_size
from .constants import ADAPTER_FILES, CHECKPOINT_PATTERN
from .data import DatasetSplit, ResolvedPaths


def load_tokenizer(tokenizer_id: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def create_training_arguments(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    precision: PrecisionSettings,
):
    from transformers import TrainingArguments

    data_seed = args.seed if args.data_seed is None else args.data_seed
    return TrainingArguments(
        output_dir=str(output_dir), run_name=args.run_name,
        learning_rate=args.learning_rate, num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm,
        warmup_ratio=0.0 if args.warmup_steps is not None else args.warmup_ratio,
        warmup_steps=0 if args.warmup_steps is None else args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type, optim=args.optim,
        adam_beta1=args.adam_beta1, adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        logging_strategy=args.logging_strategy, logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy, eval_steps=args.eval_steps,
        save_strategy=args.save_strategy, save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        bf16=precision.bf16, fp16=precision.fp16, tf32=args.tf32,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
        dataloader_persistent_workers=args.dataloader_persistent_workers,
        group_by_length=args.group_by_length,
        seed=args.seed, data_seed=data_seed, full_determinism=args.full_determinism,
        report_to=[] if args.report_to == "none" else [args.report_to],
        remove_unused_columns=False,
    )


def _tokenized_dataset(tokenizer: Any, rows: list[dict[str, Any]], max_seq_length: int):
    from datasets import Dataset

    def tokenize(row: dict[str, Any]) -> dict[str, list[int]]:
        return build_canonical_sft_example(tokenizer, row, max_length=max_seq_length)

    dataset = Dataset.from_list(rows)
    return dataset.map(
        tokenize, remove_columns=dataset.column_names,
        desc="Canonical assistant-action tokenization",
    )


def trainable_parameter_summary(model: Any) -> dict[str, int | float]:
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = model.get_nb_trainable_parameters()
    else:
        parameters = list(model.parameters())
        trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in parameters)
    return {
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "trainable_percentage": round(100.0 * trainable / total, 6) if total else 0.0,
    }


def copy_adapter_artifacts(source: Path, destination: Path) -> bool:
    if not (source / "adapter_config.json").is_file():
        return False
    if not any((source / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        return False
    destination.mkdir(parents=True, exist_ok=True)
    for name in ADAPTER_FILES:
        target = destination / name
        if target.exists():
            target.unlink()
        source_file = source / name
        if source_file.is_file():
            shutil.copy2(source_file, target)
    return True


def adapter_artifacts_exist(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() and any(
        (path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    )


def _clear_adapter_artifacts(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name in ADAPTER_FILES:
        candidate = path / name
        if candidate.is_file():
            candidate.unlink()


class _FinalAdapterCaptureMixin:
    """Save the last optimized PEFT state immediately before HF reloads the best model."""

    def __init__(self, *args: Any, final_adapter_output: Path, **kwargs: Any) -> None:
        self._final_adapter_output = Path(final_adapter_output)
        self._final_adapter_saved = False
        self._final_adapter_global_step: int | None = None
        self._final_adapter_source: str | None = None
        super().__init__(*args, **kwargs)

    def _capture_final_adapter(self, source: str) -> None:
        step = int(self.state.global_step)
        should_save = bool(getattr(self.args, "should_save", True))
        if should_save:
            _clear_adapter_artifacts(self._final_adapter_output)
        self.save_model(str(self._final_adapter_output), _internal_call=True)
        if should_save and not adapter_artifacts_exist(self._final_adapter_output):
            raise RuntimeError(
                f"Trainer did not save valid adapter artifacts at final global step {step}: "
                f"{self._final_adapter_output}"
            )
        self._final_adapter_saved = True
        self._final_adapter_global_step = step
        self._final_adapter_source = source

    def _load_best_model(self):
        # Transformers 4.57.6 invokes this inside _inner_training_loop before
        # on_train_end and before returning from train(). At this exact point,
        # self.model still contains the last optimizer update.
        self._capture_final_adapter("in_memory_before_best_model_reload")
        return super()._load_best_model()


def final_state_preserving_trainer_class(base_trainer: type) -> type:
    """Create the lazy Trainer subclass without importing Transformers at module import."""

    class FinalStatePreservingTrainer(_FinalAdapterCaptureMixin, base_trainer):
        pass

    return FinalStatePreservingTrainer


def _checkpoint_step(checkpoint: Path | None) -> int | None:
    if checkpoint is None:
        return None
    match = CHECKPOINT_PATTERN.fullmatch(checkpoint.name)
    return int(match.group(1)) if match else None


def finalize_adapter_artifacts(
    trainer: Any,
    output_dir: Path,
    *,
    load_best_model_at_end: bool,
) -> dict[str, Any]:
    """Materialize and describe true-final and best adapter artifacts."""
    final_adapter = output_dir / "final_adapter"
    final_step = int(trainer.state.global_step)
    captured = bool(getattr(trainer, "_final_adapter_saved", False))
    captured_step = getattr(trainer, "_final_adapter_global_step", None)
    final_source = getattr(trainer, "_final_adapter_source", None)
    best_value = getattr(trainer.state, "best_model_checkpoint", None)
    best_checkpoint = Path(best_value).resolve() if best_value else None

    if not captured:
        if load_best_model_at_end and best_checkpoint is not None:
            raise RuntimeError(
                "true final adapter was not captured before best-model reload; refusing to save "
                "trainer.model as final_adapter"
            )
        _clear_adapter_artifacts(final_adapter)
        trainer.save_model(str(final_adapter), _internal_call=True)
        captured = True
        captured_step = final_step
        final_source = "in_memory_after_train_without_best_model_reload"
    if captured_step != final_step:
        raise RuntimeError(
            f"final adapter step mismatch: captured {captured_step}, Trainer completed {final_step}"
        )
    if bool(getattr(trainer.args, "should_save", True)) and not adapter_artifacts_exist(final_adapter):
        raise RuntimeError(f"final adapter artifacts are incomplete: {final_adapter}")

    best_adapter = output_dir / "best_adapter"
    best_step = _checkpoint_step(best_checkpoint)
    if best_checkpoint is not None:
        if not copy_adapter_artifacts(best_checkpoint, best_adapter):
            raise RuntimeError(f"best checkpoint has no copyable adapter artifacts: {best_checkpoint}")
        best_source = str(best_checkpoint)
    else:
        if not copy_adapter_artifacts(final_adapter, best_adapter):
            raise RuntimeError("no best checkpoint exists and final adapter could not seed best_adapter")
        best_source = "final_adapter_fallback_no_best_checkpoint"

    return {
        "status": "complete",
        "final_global_step": final_step,
        "final_adapter_source": str(final_source),
        "final_adapter_path": str(final_adapter.resolve()),
        "best_global_step": best_step,
        "best_adapter_source": best_source,
        "best_adapter_path": str(best_adapter.resolve()),
        "load_best_model_at_end": bool(load_best_model_at_end),
    }


def print_adapter_metadata(metadata: dict[str, Any]) -> None:
    print(f"FINAL_GLOBAL_STEP={metadata['final_global_step']}", flush=True)
    print(f"FINAL_ADAPTER_SOURCE={metadata['final_adapter_source']}", flush=True)
    best_step = metadata["best_global_step"]
    print(f"BEST_GLOBAL_STEP={best_step if best_step is not None else 'NONE'}", flush=True)
    print(f"BEST_ADAPTER_SOURCE={metadata['best_adapter_source']}", flush=True)


def _save_json_metrics(path: Path, metrics: dict[str, Any]) -> None:
    atomic_write_json(path, {key: value for key, value in metrics.items() if value is not None})


def with_perplexity(metrics: dict[str, Any], loss_key: str) -> dict[str, Any]:
    result = dict(metrics)
    loss = result.get(loss_key)
    if isinstance(loss, (int, float)) and math.isfinite(loss):
        try:
            perplexity = math.exp(float(loss))
        except OverflowError:
            perplexity = math.inf
        if math.isfinite(perplexity):
            result["perplexity"] = perplexity
    return result


def load_model(args: argparse.Namespace, precision: PrecisionSettings, attention_implementation: str):
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM

    from training.common.qlora import build_bnb_config, build_lora_config

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "attn_implementation": attention_implementation,
    }
    if args.load_in_4bit:
        kwargs["quantization_config"] = build_bnb_config(build_qlora_settings(args, precision))
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **kwargs)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing,
        )
    model = get_peft_model(model, build_lora_config(build_lora_settings(args)))
    model.config.use_cache = False
    return model


def train(
    args: argparse.Namespace,
    paths: ResolvedPaths,
    splits: dict[str, DatasetSplit],
    tokenizer: Any,
    preflight: dict[str, Any],
    precision: PrecisionSettings,
    attention_implementation: str,
    manifest: dict[str, Any],
    resume_checkpoint: Path | None,
) -> None:
    from transformers import EarlyStoppingCallback, Trainer

    from training.common.sft import AssistantOnlyCollator
    from training.common.trainer import build_metrics_callback

    if args.report_to == "wandb":
        if args.wandb_project:
            os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
    train_dataset = _tokenized_dataset(tokenizer, splits["train"].selected_rows, args.max_seq_length)
    validation_dataset = _tokenized_dataset(
        tokenizer, splits["validation"].selected_rows, args.max_seq_length,
    )
    test_dataset = (
        _tokenized_dataset(tokenizer, splits["test"].selected_rows, args.max_seq_length)
        if "test" in splits and splits["test"].selected_rows else None
    )
    model = load_model(args, precision, attention_implementation)
    parameter_summary = trainable_parameter_summary(model)
    print(parameter_summary)
    manifest["parameters"] = parameter_summary
    atomic_write_json(paths.output_dir / "run_manifest.json", manifest)
    callbacks: list[Any] = [build_metrics_callback(
        paths.output_dir,
        effective_batch_size=effective_train_batch_size(args),
        mean_train_tokens=preflight["splits"]["train"]["tokens"]["mean"],
    )]
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        ))
    FinalStatePreservingTrainer = final_state_preserving_trainer_class(Trainer)
    trainer = FinalStatePreservingTrainer(
        model=model,
        args=create_training_arguments(args, output_dir=paths.output_dir, precision=precision),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=AssistantOnlyCollator(tokenizer.pad_token_id),
        callbacks=callbacks,
        final_adapter_output=paths.output_dir / "final_adapter",
    )
    manifest["adapter_artifacts"] = {
        "status": "training_in_progress",
        "final_global_step": None,
        "final_adapter_source": None,
        "best_global_step": None,
        "best_adapter_source": None,
    }
    atomic_write_json(paths.output_dir / "run_manifest.json", manifest)
    train_result = trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None,
    )
    _save_json_metrics(paths.output_dir / "train_metrics.json", dict(train_result.metrics))
    adapter_metadata = finalize_adapter_artifacts(
        trainer,
        paths.output_dir,
        load_best_model_at_end=args.load_best_model_at_end,
    )
    manifest["adapter_artifacts"] = adapter_metadata
    atomic_write_json(paths.output_dir / "run_manifest.json", manifest)
    print_adapter_metadata(adapter_metadata)
    tokenizer.save_pretrained(paths.output_dir / "tokenizer")
    validation_metrics = with_perplexity(trainer.evaluate(validation_dataset), "eval_loss")
    _save_json_metrics(paths.output_dir / "validation_metrics.json", validation_metrics)
    if args.evaluate_test_after_train and test_dataset is not None:
        test_metrics = with_perplexity(
            trainer.evaluate(test_dataset, metric_key_prefix="test"), "test_loss",
        )
        _save_json_metrics(paths.output_dir / "test_metrics.json", test_metrics)
