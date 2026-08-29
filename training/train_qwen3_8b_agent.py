from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.common.qlora import LoRASettings, QLoRASettings, resolve_precision
from training.trajectory_dataset.io_utils import read_jsonl
from training.trajectory_dataset.validate import validate_rows


DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QLoRA SFT for one future central Qwen3-8B history/tool agent.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--train-file", default="training/trajectory_dataset/outputs/final/train.jsonl")
    parser.add_argument("--validation-file", default="training/trajectory_dataset/outputs/final/validation.jsonl")
    parser.add_argument("--output-dir", default="outputs/qwen3-8b-central-history-agent")
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--bnb-compute-dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--report-to", choices=("none", "wandb"), default="none")
    parser.add_argument("--dry-run", action="store_true", help="Validate config/data without loading a tokenizer or model.")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if min(args.per_device_train_batch_size, args.per_device_eval_batch_size, args.gradient_accumulation_steps) < 1:
        raise ValueError("batch sizes and gradient accumulation must be positive")
    if args.max_seq_length < 64:
        raise ValueError("--max-seq-length must be at least 64")
    if args.lora_r < 1 or args.lora_alpha < 1 or not 0 <= args.lora_dropout < 1:
        raise ValueError("invalid LoRA settings")
    if args.save_total_limit < 1:
        raise ValueError("--save-total-limit must be positive")
    if args.bf16 is True and args.fp16 is True:
        raise ValueError("--bf16 and --fp16 cannot both be enabled")


def _load_valid(path: str | Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    result = validate_rows(rows)
    if result.rejected:
        reasons = [item["reason"] for item in result.rejected[:5]]
        raise ValueError(f"canonical dataset validation failed for {path}: {reasons}")
    if not result.valid:
        raise ValueError(f"canonical dataset is empty: {path}")
    return result.valid


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    train_rows = _load_valid(args.train_file)
    validation_rows = _load_valid(args.validation_file)
    summary = {
        "model_id": args.model_id,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "output_dir": args.output_dir,
        "dry_run": args.dry_run,
        "loss_contract": {
            "assistant_tool_calls": "trained",
            "assistant_final_answers": "trained",
            "system_user_tool_observations": "masked",
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    from datasets import Dataset
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    from training.common.qlora import build_bnb_config, build_lora_config
    from training.common.sft import AssistantOnlyCollator
    from training.common.trainer import build_metrics_callback
    from training.trajectory_dataset.preprocess import build_canonical_sft_example

    precision = resolve_precision(
        bf16=args.bf16,
        fp16=args.fp16,
        bnb_compute_dtype=args.bnb_compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=build_bnb_config(
            replace(QLoRASettings(), bnb_4bit_compute_dtype=precision.compute_dtype)
        ),
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        build_lora_config(
            LoRASettings(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
        ),
    )
    model.config.use_cache = False

    def tokenize(row: dict[str, Any]) -> dict[str, list[int]]:
        return build_canonical_sft_example(tokenizer, row, max_length=args.max_seq_length)

    train_dataset = Dataset.from_list(train_rows).map(tokenize, remove_columns=list(train_rows[0]))
    validation_dataset = Dataset.from_list(validation_rows).map(tokenize, remove_columns=list(validation_rows[0]))
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=precision.bf16,
        fp16=precision.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        data_seed=args.seed,
        report_to=[] if args.report_to == "none" else [args.report_to],
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=AssistantOnlyCollator(tokenizer.pad_token_id),
        callbacks=[build_metrics_callback(args.output_dir)],
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
