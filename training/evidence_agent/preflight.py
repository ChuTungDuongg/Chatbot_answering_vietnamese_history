from __future__ import annotations

import argparse
import json
from typing import Any

from training.common.datasets import split_rows, split_statistics
from training.common.jsonl import read_jsonl
from training.evidence_agent.config import EvidenceAgentConfig
from training.evidence_agent.sft import prepare_evidence_split
from training.evidence_agent.validate_dataset import validate_rows


def collect_tokenization_preflight(
    dataset: str,
    *,
    tokenizer_id: str,
    max_length: int,
    seed: int = 42,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Validate and tokenize all splits without loading model weights."""
    rows = read_jsonl(dataset)
    validation = validate_rows(rows)
    if not validation["valid"]:
        raise ValueError(f"dataset validation failed: {validation['errors'][:5]}")
    splits = split_rows(
        rows,
        seed=seed,
        max_samples=max_samples,
        group_key="group_id",
        stratify_key="behavior",
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    tokenization = {}
    for name in ("train", "eval", "test"):
        _, tokenization[name] = prepare_evidence_split(
            tokenizer,
            getattr(splits, name),
            max_length=max_length,
            split_name=name,
        )
    return {
        "valid": True,
        "dataset": dataset,
        "tokenizer_id": tokenizer_id,
        "max_length": max_length,
        "splits": split_statistics(splits),
        "tokenization": tokenization,
    }


def build_parser() -> argparse.ArgumentParser:
    cfg = EvidenceAgentConfig()
    parser = argparse.ArgumentParser(
        description="Evidence Agent tokenization preflight; loads a tokenizer but never Qwen model weights."
    )
    parser.add_argument("--dataset", default="datasets/evidence_agent/train.jsonl")
    parser.add_argument("--tokenizer-id", default=cfg.model_id)
    parser.add_argument("--max-length", type=int, default=cfg.max_length)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = collect_tokenization_preflight(
            args.dataset,
            tokenizer_id=args.tokenizer_id,
            max_length=args.max_length,
            seed=args.seed,
            max_samples=args.max_samples,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        report = {"valid": False, "errors": [str(exc)]}
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
