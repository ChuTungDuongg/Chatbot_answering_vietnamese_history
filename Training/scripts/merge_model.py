from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model for deployment.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from training.history_answerer.merge_phase1 import merge_phase1_adapter

    merge_phase1_adapter(
        model_id=args.base_model,
        phase1_adapter=args.adapter,
        output_dir=args.output_dir,
        dtype=args.dtype,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



