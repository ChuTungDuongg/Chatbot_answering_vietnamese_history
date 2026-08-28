"""Legacy compatibility CLI for reproducing the retired Phase 1 merge flow."""

from __future__ import annotations

import argparse

from training.history_answerer.config import LEGACY_BASE_MODEL_ID
from training.history_answerer.merge_adapter import merge_lora_adapter


def merge_phase1_adapter(*, model_id, phase1_adapter, output_dir, dtype="bfloat16", trust_remote_code=True):
    return merge_lora_adapter(
        model_id=model_id,
        adapter=phase1_adapter,
        output_dir=output_dir,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Legacy: merge a Phase 1 LoRA adapter.")
    parser.add_argument("--model-id", default=LEGACY_BASE_MODEL_ID)
    parser.add_argument("--phase1-adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    merge_lora_adapter(
        model_id=args.model_id,
        adapter=args.phase1_adapter,
        output_dir=args.output_dir,
        dtype=args.dtype,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
