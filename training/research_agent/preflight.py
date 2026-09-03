from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import statistics
from pathlib import Path
from typing import Any

from training.common.jsonl import read_jsonl
from training.research_agent.validate_dataset import validate_rows
from app.agents.common.model_registry import SHARED_BASE_MODEL_ID
from training.common.datasets import split_rows
from training.common.sft import assistant_only_token_stats


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _summary(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values) if values else 0,
        "mean": statistics.mean(values) if values else 0.0,
        "max": max(values) if values else 0,
    }


def tokenization_preflight(rows: list[dict[str, Any]], tokenizer: Any, *, max_length: int) -> dict[str, Any]:
    splits = split_rows(
        rows,
        seed=42,
        train_ratio=0.88,
        eval_ratio=0.06,
        group_key="group_id",
        stratify_key="trajectory_class",
    )
    report: dict[str, Any] = {}
    all_errors: list[str] = []
    for name in ("train", "eval", "test"):
        prompt: list[int] = []
        assistant: list[int] = []
        sequence: list[int] = []
        truncated = zero = 0
        errors: list[str] = []
        split_rows_value = getattr(splits, name)
        for index, row in enumerate(split_rows_value, 1):
            try:
                stats = assistant_only_token_stats(tokenizer, row["messages"], max_length=max_length)
                prompt.append(stats.prompt_tokens)
                assistant.append(stats.assistant_tokens)
                sequence.append(stats.sequence_tokens)
                truncated += int(stats.truncated)
                zero += int(stats.assistant_tokens_kept == 0)
            except (TypeError, ValueError) as exc:
                errors.append(f"{name} row {index}: {exc}")
        if zero:
            errors.append(f"{name}: {zero} zero-supervised rows")
        report[name] = {
            "rows": len(split_rows_value),
            "prompt_tokens": _summary(prompt),
            "assistant_tokens": _summary(assistant),
            "sequence_tokens": _summary(sequence),
            "overlength_rows": truncated,
            "zero_supervised_rows": zero,
            "errors": errors,
        }
        all_errors.extend(errors)
    return {"valid": not all_errors, "splits": report, "errors": all_errors}


def collect_preflight(
    dataset: str | None = None,
    *,
    tokenizer_id: str | None = None,
    max_length: int = 4096,
) -> dict[str, Any]:
    import torch

    cuda = bool(torch.cuda.is_available())
    gpu = torch.cuda.get_device_name(0) if cuda else "none"
    bf16 = bool(cuda and getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    report: dict[str, Any] = {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": cuda,
        "gpu": gpu,
        "bf16_supported": bf16,
        "bitsandbytes_version": _version("bitsandbytes"),
        "transformers_version": _version("transformers"),
        "peft_version": _version("peft"),
        "recommended_dtype": "bfloat16" if bf16 else "float16",
        "recommended_starting_batch_size": 1 if "T4" in gpu.upper() or not cuda else 2,
    }
    if dataset:
        path = Path(dataset)
        report["dataset_path"] = str(path.resolve())
        try:
            rows = read_jsonl(path)
            report["dataset_validation"] = validate_rows(rows)
            if tokenizer_id:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
                report["tokenizer_id"] = tokenizer_id
                report["tokenization"] = tokenization_preflight(rows, tokenizer, max_length=max_length)
        except (OSError, ValueError) as exc:
            report["dataset_validation"] = {"valid": False, "errors": [str(exc)]}
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightweight Colab preflight; never loads the Qwen model.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--tokenizer-id", default=SHARED_BASE_MODEL_ID)
    parser.add_argument("--max-length", type=int, default=4096)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = collect_preflight(
        args.dataset,
        tokenizer_id=args.tokenizer_id if args.dataset else None,
        max_length=args.max_length,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    valid = report.get("dataset_validation", {}).get("valid", True)
    valid = valid and report.get("tokenization", {}).get("valid", True)
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
