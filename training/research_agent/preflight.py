from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

from training.common.jsonl import read_jsonl
from training.research_agent.validate_dataset import validate_rows


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def collect_preflight(dataset: str | None = None) -> dict[str, Any]:
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
            report["dataset_validation"] = validate_rows(read_jsonl(path))
        except (OSError, ValueError) as exc:
            report["dataset_validation"] = {"valid": False, "errors": [str(exc)]}
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightweight Colab preflight; never loads the Qwen model.")
    parser.add_argument("--dataset", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = collect_preflight(args.dataset)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("dataset_validation", {}).get("valid", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
