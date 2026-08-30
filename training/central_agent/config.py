from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from training.common.qlora import DEFAULT_LORA_TARGETS, LoRASettings, PrecisionSettings, QLoRASettings

from .constants import DEFAULT_MODEL_ID, DEFAULT_RUN_NAME


def build_parser(*, defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Production Qwen3-8B NF4 QLoRA training for the canonical central history/tool agent.",
    )
    parser.add_argument("--config", help="JSON config; precedence is defaults < config < explicit CLI flags.")
    parser.add_argument(
        "--dump-config", nargs="?", const="-", metavar="PATH",
        help="Print resolved CLI config, or save it to PATH, without loading tokenizer/model.",
    )
    parser.add_argument("--print-recommended-config", action="store_true")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--tokenizer-id", default=None)
    parser.add_argument("--drive-root", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--validation-file", default=None)
    parser.add_argument("--test-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument(
        "--lr-scheduler-type",
        choices=("linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"),
        default="cosine",
    )
    parser.add_argument("--optim", choices=("paged_adamw_8bit", "adamw_torch"), default="paged_adamw_8bit")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1.0e-8)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bnb-4bit-quant-type", choices=("nf4", "fp4"), default="nf4")
    parser.add_argument("--bnb-use-double-quant", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--bnb-compute-dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto",
    )
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-bias", choices=("none", "all", "lora_only"), default="none")
    parser.add_argument("--lora-target-modules", default=",".join(DEFAULT_LORA_TARGETS))
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--attn-implementation", choices=("auto", "flash_attention_2", "sdpa", "eager"), default="auto",
    )
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--dataloader-pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataloader-persistent-workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--group-by-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logging-strategy", choices=("steps", "epoch", "no"), default="steps")
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-strategy", choices=("steps", "epoch", "no"), default="steps")
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-strategy", choices=("steps", "epoch", "no"), default="steps")
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--load-best-model-at-end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metric-for-best-model", default="eval_loss")
    parser.add_argument("--greater-is-better", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--allow-resume-data-mismatch", action="store_true")
    parser.add_argument("--report-to", choices=("none", "wandb"), default="none")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--full-determinism", action="store_true")
    parser.add_argument("--evaluate-test-after-train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--min-free-gpu-gb", type=float, default=None)
    parser.add_argument("--allow-cpu-training", action="store_true", help=argparse.SUPPRESS)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Paths/config/schema only; no tokenizer or model.")
    modes.add_argument(
        "--preflight-only", action="store_true",
        help="Load tokenizer and run exact canonical token/supervision audit; no model weights.",
    )
    if defaults:
        parser.set_defaults(**defaults)
    return parser


def _load_config_defaults(path: str | Path, valid_keys: set[str]) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON config {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("config JSON must contain one object")
    normalized = {str(key).replace("-", "_"): item for key, item in value.items()}
    unknown = sorted(set(normalized) - valid_keys)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    return normalized


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    base = build_parser()
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config")
    preliminary, _ = pre_parser.parse_known_args(raw)
    defaults: dict[str, Any] = {}
    if preliminary.config:
        valid = {action.dest for action in base._actions if action.dest != "help"}
        defaults = _load_config_defaults(preliminary.config, valid)
    return build_parser(defaults=defaults).parse_args(raw)


def parse_lora_targets(value: str | Iterable[str]) -> tuple[str, ...]:
    raw = value.split(",") if isinstance(value, str) else list(value)
    targets = tuple(dict.fromkeys(str(item).strip() for item in raw if str(item).strip()))
    if not targets:
        raise ValueError("--lora-target-modules must contain at least one module")
    if any(not re.fullmatch(r"[A-Za-z0-9_.]+", item) for item in targets):
        raise ValueError("LoRA target modules may contain only letters, digits, underscores, and dots")
    return targets


def validate_args(args: argparse.Namespace) -> None:
    if any(value < 1 for value in (
        args.per_device_train_batch_size, args.per_device_eval_batch_size,
        args.gradient_accumulation_steps,
    )):
        raise ValueError("batch sizes and gradient accumulation must be positive")
    if args.max_seq_length < 64:
        raise ValueError("--max-seq-length must be at least 64")
    if args.learning_rate <= 0 or args.num_train_epochs <= 0:
        raise ValueError("learning rate and epochs must be positive")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("--max-steps must be -1 or positive")
    if args.weight_decay < 0 or args.max_grad_norm <= 0:
        raise ValueError("weight decay must be non-negative and max grad norm positive")
    if not 0 <= args.warmup_ratio <= 1 or (args.warmup_steps is not None and args.warmup_steps < 0):
        raise ValueError("warmup ratio must be in [0,1] and warmup steps non-negative")
    if args.lora_r < 1 or args.lora_alpha < 1 or not 0 <= args.lora_dropout < 1:
        raise ValueError("invalid LoRA settings")
    parse_lora_targets(args.lora_target_modules)
    if args.bf16 is True and args.fp16 is True:
        raise ValueError("--bf16 and --fp16 cannot both be enabled")
    if args.save_total_limit < 1:
        raise ValueError("--save-total-limit must be positive")
    for name in ("logging_steps", "eval_steps", "save_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.load_best_model_at_end:
        if args.eval_strategy == "no" or args.save_strategy == "no":
            raise ValueError("load_best_model_at_end requires evaluation and saving")
        if args.eval_strategy != args.save_strategy:
            raise ValueError("load_best_model_at_end requires matching eval/save strategies")
        if args.eval_strategy == "steps" and args.save_steps % args.eval_steps:
            raise ValueError("with best-model loading, --save-steps must be a multiple of --eval-steps")
    if args.early_stopping_patience > 0 and (
        args.eval_strategy == "no" or not args.load_best_model_at_end
    ):
        raise ValueError("early stopping requires evaluation and --load-best-model-at-end")
    if args.early_stopping_threshold < 0:
        raise ValueError("early stopping threshold must be non-negative")
    if args.dataloader_num_workers < 0:
        raise ValueError("dataloader workers cannot be negative")
    if args.dataloader_persistent_workers and args.dataloader_num_workers == 0:
        raise ValueError("persistent dataloader workers require --dataloader-num-workers > 0")
    if args.min_free_gpu_gb is not None and args.min_free_gpu_gb < 0:
        raise ValueError("--min-free-gpu-gb cannot be negative")
    for name in ("max_train_samples", "max_validation_samples", "max_test_samples"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not args.run_name.strip() or any(part in args.run_name for part in ("/", "\\", "..")):
        raise ValueError("--run-name must be a safe single directory name")


def world_size() -> int:
    try:
        return max(1, int(os.environ.get("WORLD_SIZE", "1")))
    except ValueError:
        return 1


def effective_train_batch_size(args: argparse.Namespace, *, distributed_world_size: int | None = None) -> int:
    size = world_size() if distributed_world_size is None else distributed_world_size
    return args.per_device_train_batch_size * args.gradient_accumulation_steps * size


def build_qlora_settings(args: argparse.Namespace, precision: PrecisionSettings) -> QLoRASettings:
    return QLoRASettings(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_use_double_quant,
        bnb_4bit_compute_dtype=precision.compute_dtype,
    )


def build_lora_settings(args: argparse.Namespace) -> LoRASettings:
    return LoRASettings(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        bias=args.lora_bias,
        target_modules=parse_lora_targets(args.lora_target_modules),
    )


def safe_cli_arguments(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        lowered = key.casefold()
        if lowered.endswith("_token") or any(term in lowered for term in ("password", "secret", "api_key", "access_key")):
            result[key] = "<redacted>"
        else:
            result[key] = value
    return result


def dump_config(args: argparse.Namespace) -> None:
    encoded = json.dumps(safe_cli_arguments(args), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.dump_config == "-":
        print(encoded, end="")
    else:
        target = Path(args.dump_config).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(encoded, encoding="utf-8")
        print(f"CONFIG_WRITTEN={target}")
