"""Train a Qwen3-4B PEFT adapter from the frozen TRAIN-only SFT messages.

Heavy training dependencies are imported only after validation and --dry-run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from app.config import HYBRID_MODEL_ID
from app.rag.response_modes import MODE_INSTRUCTIONS

LOG = logging.getLogger(__name__)
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SFT_LOSS_OPTIONS = {"assistant_only_loss": False, "completion_only_loss": True}
MODEL_SELECTION = {"metric_for_best_model": "eval_loss", "greater_is_better": False}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sft(path: Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            mode = row.get("response_mode")
            key = (str(row.get("canonical_id")), str(mode))
            if (not isinstance(messages, list) or len(messages) != 3 or
                [item.get("role") for item in messages] != ["system", "user", "assistant"] or
                mode not in MODE_INSTRUCTIONS or MODE_INSTRUCTIONS[mode] not in messages[0]["content"] or
                not all(isinstance(item.get("content"), str) and item["content"].strip() for item in messages)):
                raise ValueError(f"Invalid mode-conditioned SFT row {path}:{line_number}")
            if key in seen:
                raise ValueError(f"Duplicate canonical ID/response mode: {key}")
            seen.add(key)
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"Empty SFT file: {path}")
    return rows


def to_prompt_completion(rows: list[dict]) -> list[dict]:
    """Convert frozen SFT messages in memory; keep system/mode text unchanged."""
    converted = []
    for row in rows:
        messages = row["messages"]
        if len(messages) != 3 or [item.get("role") for item in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"Expected system/user/assistant SFT messages: {row.get('id')}")
        converted.append({"prompt": messages[:-1], "completion": [messages[-1]]})
    return converted


def validate_prompt_lengths(rows: Iterable[dict], tokenizer, max_seq_length: int) -> None:
    """Reject prompts that leave no room for the assistant completion."""
    for row in rows:
        prompt_tokens = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=True,
                                                      add_generation_prompt=True, enable_thinking=False)
        if len(prompt_tokens) >= max_seq_length - 32:
            raise ValueError(f"SFT prompt exceeds --max-seq-length before assistant answer: {row.get('id')}")


def assert_split_isolation(train: list[dict], validation: list[dict], split_manifest: dict | None) -> None:
    train_ids = {row["canonical_id"] for row in train}
    validation_ids = {row["canonical_id"] for row in validation}
    if train_ids & validation_ids:
        raise ValueError("SFT train/validation canonical IDs overlap")
    if split_manifest:
        groups = split_manifest.get("ids") or split_manifest.get("split_ids")
        if groups:
            if not train_ids <= set(groups["train"]) or not validation_ids <= set(groups["validation"]):
                raise ValueError("SFT rows disagree with frozen split IDs")
            if (train_ids | validation_ids) & set(groups["test"]):
                raise ValueError("TEST ID found in SFT input")


def effective_batch(per_device: int, accumulation: int, devices: int) -> int:
    if min(per_device, accumulation, devices) < 1:
        raise ValueError("Batch, accumulation and device count must be positive")
    return per_device * accumulation * devices


def lora_kwargs(args: argparse.Namespace) -> dict:
    return {"r": args.lora_r, "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout, "target_modules": list(LORA_TARGETS),
            "bias": "none", "task_type": "CAUSAL_LM"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=HYBRID_MODEL_ID)
    parser.add_argument("--model-revision")
    parser.add_argument("--train-file", type=Path, default=Path("training/datasets/vn_history_rag_sft_v1/train_sft.jsonl"))
    parser.add_argument("--validation-file", type=Path, default=Path("training/datasets/vn_history_rag_sft_v1/validation_sft.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--compute-dtype", choices=("auto", "bfloat16", "float16"), default="auto")
    parser.add_argument("--resume-from-checkpoint", help="latest or a checkpoint directory")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true", help="GPU smoke: cap to 16/8 samples and one step")
    return parser


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _resume_path(value: str | None, checkpoint_root: Path) -> Path | None:
    if not value:
        return None
    if value == "latest":
        choices = sorted(checkpoint_root.glob("checkpoint-*"),
                         key=lambda path: int(path.name.split("-")[-1]) if path.name.split("-")[-1].isdigit() else -1)
        if not choices:
            return None
        return choices[-1]
    path = Path(value)
    if not path.is_dir() or not (path / "trainer_state.json").is_file():
        raise FileNotFoundError(f"Invalid Trainer checkpoint: {path}")
    return path


def prepare(args: argparse.Namespace) -> tuple[list[dict], list[dict], dict, Path | None]:
    if args.model_id != HYBRID_MODEL_ID:
        raise ValueError(f"This frozen experiment requires {HYBRID_MODEL_ID}")
    if args.fast_dev_run:
        args.max_train_samples = min(args.max_train_samples or 16, 16)
        args.max_eval_samples = min(args.max_eval_samples or 8, 8)
    if any(value is not None and value < 1 for value in (args.max_train_samples, args.max_eval_samples)):
        raise ValueError("Sample caps must be positive")
    for name in ("epochs", "learning_rate", "max_seq_length", "lora_r", "eval_steps", "save_steps", "logging_steps",
                 "save_total_limit", "early_stopping_patience"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.early_stopping_threshold < 0:
        raise ValueError("early_stopping_threshold must be nonnegative")
    if args.eval_steps != args.save_steps:
        raise ValueError("save_steps must equal eval_steps for best-checkpoint selection and prompt early stopping")
    train = read_sft(args.train_file, args.max_train_samples)
    validation = read_sft(args.validation_file, args.max_eval_samples)
    sft_manifest_path = args.train_file.parent / "manifest.json"
    sft_manifest = json.loads(sft_manifest_path.read_text(encoding="utf-8")) if sft_manifest_path.exists() else None
    split_path = args.train_file.parent / "split_manifest.json"
    if not split_path.is_file() and args.train_file.resolve().is_relative_to(Path.cwd().resolve()):
        split_path = Path("evaluation/datasets/v1_silver/splits/v1_seed42/split_manifest.json")
    split = json.loads(split_path.read_text(encoding="utf-8")) if split_path.exists() else None
    if split is None:
        raise FileNotFoundError("Frozen split_manifest.json must be beside SFT data or in the local repository")
    if sft_manifest and sft_manifest.get("split_manifest_sha256") != sha256(split_path):
        raise ValueError("SFT/split manifest SHA mismatch")
    if sft_manifest:
        for name, path in (("train_sft_sha256", args.train_file), ("validation_sft_sha256", args.validation_file)):
            if sft_manifest.get(name) != sha256(path):
                raise ValueError(f"SFT manifest {name} mismatch")
    assert_split_isolation(train, validation, split)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
              if key not in {"dry_run", "resume_from_checkpoint"}}
    manifest = {"schema_version": 2, "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
                "model_id": args.model_id, "model_revision": args.model_revision,
                "train_sha256": sha256(args.train_file), "validation_sha256": sha256(args.validation_file),
                "sft_manifest_sha256": sha256(sft_manifest_path) if sft_manifest else None,
                "split_manifest_sha256": sha256(split_path) if split else (sft_manifest or {}).get("split_manifest_sha256"),
                "corpus_sha256": (sft_manifest or {}).get("corpus_sha256"),
                "seed": args.seed, "lora": lora_kwargs(args), "config": config,
                **MODEL_SELECTION, "best_model_checkpoint": None, "best_metric": None,
                "early_stopping_patience": args.early_stopping_patience,
                "early_stopping_threshold": args.early_stopping_threshold,
                "final_adapter_source": "not_saved_dry_run" if args.dry_run else "pending_training",
                "train_rows": len(train), "validation_rows": len(validation),
                "platform": platform.platform()}
    manifest_path = args.output_dir / "manifest.json"
    resume = _resume_path(args.resume_from_checkpoint, args.output_dir / "checkpoints")
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("status") == "complete":
            raise FileExistsError(f"Completed training run exists: {args.output_dir}")
        if not args.resume_from_checkpoint:
            raise FileExistsError(f"Run exists; use --resume-from-checkpoint latest: {args.output_dir}")
        for key in ("model_id", "model_revision", "train_sha256", "validation_sha256", "sft_manifest_sha256",
                    "split_manifest_sha256", "config", "metric_for_best_model", "greater_is_better",
                    "early_stopping_patience", "early_stopping_threshold"):
            if old.get(key) != manifest.get(key):
                raise ValueError(f"Resume provenance mismatch: {key}")
        manifest = old
        if resume is None:
            manifest["restarted_without_checkpoint_at"] = datetime.now(timezone.utc).isoformat()
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Nonempty output directory without run manifest: {args.output_dir}")
    elif args.resume_from_checkpoint:
        raise ValueError("Checkpoint resume requires an existing run manifest")
    return train, validation, manifest, resume


def run(args: argparse.Namespace) -> dict:
    train, validation, manifest, resume = prepare(args)
    try:
        import torch
        devices = max(1, torch.cuda.device_count())
    except ImportError:
        devices = 1
    batch = effective_batch(args.per_device_train_batch_size, args.gradient_accumulation_steps, devices)
    steps = 1 if args.fast_dev_run else math.ceil(len(train) / batch * args.epochs)
    print(f"[train] rows={len(train)} validation={len(validation)} devices={devices} effective_batch={batch} estimated_steps={steps}")
    if args.dry_run:
        print(json.dumps({"dry_run": True, "output_dir": str(args.output_dir), "manifest": manifest}, ensure_ascii=True, indent=2))
        return manifest
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("4-bit QLoRA training requires CUDA; use --dry-run here or run on Colab L4/A100")
    try:
        import bitsandbytes  # noqa: F401
        import peft
        import transformers
        import trl
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                                  EarlyStoppingCallback, TrainerCallback, set_seed)
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("Install requirements-training.txt in a CUDA environment") from exc
    dtype = (torch.bfloat16 if args.compute_dtype == "bfloat16" or
             args.compute_dtype == "auto" and torch.cuda.is_bf16_supported() else torch.float16)
    if args.compute_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected GPU does not support bfloat16; choose --compute-dtype float16 or auto")
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest["libraries"] = {"torch": torch.__version__, "transformers": transformers.__version__,
                              "trl": trl.__version__, "peft": peft.__version__}
    manifest["gpu"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    manifest["compute_dtype"] = str(dtype)
    _atomic_json(args.output_dir / "manifest.json", manifest)
    model_args = {"revision": args.model_revision, "trust_remote_code": True}
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, **model_args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # A truncated prompt with no assistant tokens cannot teach the response mode.
    validate_prompt_lengths((*train, *validation), tokenizer, args.max_seq_length)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, **model_args, device_map="auto", dtype=dtype,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
    resolved = getattr(model.config, "_commit_hash", None) or args.model_revision
    if manifest.get("resolved_model_revision") and resolved != manifest["resolved_model_revision"]:
        raise RuntimeError("Resolved model revision changed since previous training checkpoint")
    manifest["resolved_model_revision"] = resolved
    if args.gradient_checkpointing:
        model.config.use_cache = False
    _atomic_json(args.output_dir / "manifest.json", manifest)
    cadence = 1 if args.fast_dev_run else args.eval_steps
    config = SFTConfig(
        output_dir=str(args.output_dir / "checkpoints"), num_train_epochs=args.epochs,
        max_steps=1 if args.fast_dev_run else -1, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_seq_length, packing=args.packing, **SFT_LOSS_OPTIONS,
        gradient_checkpointing=args.gradient_checkpointing, bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16, seed=args.seed, logging_steps=args.logging_steps,
        eval_strategy="steps", eval_steps=cadence, save_strategy="steps",
        save_steps=cadence, save_total_limit=args.save_total_limit,
        load_best_model_at_end=True, **MODEL_SELECTION, logging_first_step=True,
        dataloader_num_workers=args.dataloader_num_workers, report_to="none")
    class Progress(TrainerCallback):
        start = time.monotonic()
        last_step = 0
        last_time = start
        learning_rate = None
        def on_log(self, tr_args, state, control, logs=None, **kwargs):
            now = time.monotonic()
            elapsed = now - self.start
            step = int(state.global_step)
            rate = step / elapsed if elapsed else 0.0
            logs = logs or {}
            if "learning_rate" in logs:
                self.learning_rate = logs["learning_rate"]
            info = {"time": datetime.now(timezone.utc).isoformat(), "step": step,
                    "epoch": state.epoch, "elapsed_s": round(elapsed, 1),
                    "step_time_s": round((now - self.last_time) / max(step - self.last_step, 1), 2),
                    "eta_s": round((state.max_steps - step) / rate) if rate else None,
                    "examples_per_s": round(rate * batch, 2),
                    "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), **logs}
            if "eval_loss" in logs:
                best = state.best_metric
                improved = best is None or logs["eval_loss"] < best
                info.update(event="evaluation", eval_loss=logs["eval_loss"],
                            best_metric_so_far=logs["eval_loss"] if improved else best,
                            best_model_checkpoint=(str(Path(tr_args.output_dir) / f"checkpoint-{step}")
                                                   if improved else state.best_model_checkpoint),
                            learning_rate=self.learning_rate)
            with (args.output_dir / "training_log.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(info, ensure_ascii=False, default=str) + "\n")
                stream.flush()
            LOG.info("step=%s/%s epoch=%s loss=%s eval_loss=%s examples/s=%s ETA=%ss",
                     step, state.max_steps, state.epoch, info.get("loss"), info.get("eval_loss"),
                     info["examples_per_s"], info["eta_s"])
            self.last_step, self.last_time = step, now
        def on_save(self, tr_args, state, control, **kwargs):
            LOG.info("checkpoint saved: step %s", state.global_step)
    trainer = SFTTrainer(
        model=model, args=config, processing_class=tokenizer,
        train_dataset=Dataset.from_list(to_prompt_completion(train)),
        eval_dataset=Dataset.from_list(to_prompt_completion(validation)),
        peft_config=LoraConfig(**lora_kwargs(args)),
        callbacks=[Progress(), EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold)])
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    best_checkpoint = trainer.state.best_model_checkpoint
    if best_checkpoint and not Path(best_checkpoint).is_dir():
        raise RuntimeError(f"Best validation checkpoint was not retained: {best_checkpoint}")
    if bool(best_checkpoint) != (trainer.state.best_metric is not None):
        raise RuntimeError("Trainer best-checkpoint and best-metric state disagree")
    manifest["best_model_checkpoint"] = best_checkpoint
    manifest["best_metric"] = trainer.state.best_metric
    manifest["final_adapter_source"] = ("best_validation_checkpoint" if best_checkpoint
                                        else "final_training_state_no_validation_checkpoint")
    # Trainer.train reloads the best checkpoint before returning when load_best_model_at_end=True.
    trainer.save_model(str(args.output_dir / "adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "adapter"))
    trainer.save_state()
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["global_step"] = trainer.state.global_step
    _atomic_json(args.output_dir / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run(build_parser().parse_args(argv))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        LOG.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
