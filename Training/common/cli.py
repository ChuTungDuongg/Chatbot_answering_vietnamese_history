from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Any

from training.common.qlora import LoRASettings


def add_training_arguments(parser: argparse.ArgumentParser, config: Any) -> None:
    """Add the shared, user-overridable QLoRA training flags."""
    parser.add_argument("--epochs", type=float, default=config.epochs)
    parser.add_argument("--batch-size", type=int, default=config.train_batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=config.eval_batch_size)
    parser.add_argument(
        "--gradient-accumulation-steps",
        "--grad-accum-steps",
        dest="gradient_accumulation_steps",
        type=int,
        default=config.gradient_accumulation_steps,
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        dest="learning_rate",
        type=float,
        default=config.learning_rate,
    )
    parser.add_argument("--weight-decay", type=float, default=config.weight_decay)
    parser.add_argument("--warmup-ratio", type=float, default=config.warmup_ratio)
    parser.add_argument("--max-length", type=int, default=config.max_length)
    parser.add_argument("--lora-r", type=int, default=config.lora.r)
    parser.add_argument("--lora-alpha", type=int, default=config.lora.alpha)
    parser.add_argument("--lora-dropout", type=float, default=config.lora.dropout)
    parser.add_argument("--save-steps", type=int, default=config.save_steps)
    parser.add_argument("--eval-steps", type=int, default=config.eval_steps)
    parser.add_argument("--logging-steps", type=int, default=config.logging_steps)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--report-to", choices=("none", "wandb"), default="none")


def lora_settings_from_args(args: argparse.Namespace, base: LoRASettings) -> LoRASettings:
    return replace(base, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)


def validate_training_arguments(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("Batch sizes must be at least 1.")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be at least 1.")
    if args.max_length < 64:
        raise ValueError("--max-length must be at least 64.")
    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of --bf16 or --fp16.")
