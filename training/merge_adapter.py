"""Explicitly merge a completed QLoRA adapter into an unquantized Qwen3-4B model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import HYBRID_MODEL_ID


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=HYBRID_MODEL_ID)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    args = parser.parse_args(argv)
    if args.model_id != HYBRID_MODEL_ID:
        parser.error(f"Expected {HYBRID_MODEL_ID}")
    if not (args.adapter_path / "adapter_config.json").is_file():
        parser.error("Adapter config is missing")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory already contains files")
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    config = PeftConfig.from_pretrained(str(args.adapter_path), local_files_only=True)
    if config.base_model_name_or_path != args.model_id:
        parser.error("Adapter base model does not match --model-id")
    base = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=getattr(torch, args.dtype),
                                                device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, str(args.adapter_path), is_trainable=False,
                                     local_files_only=True).merge_and_unload()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    (args.output_dir / "merge_manifest.json").write_text(json.dumps({"model_id": args.model_id,
        "adapter_path": str(args.adapter_path.resolve()), "dtype": args.dtype,
        "resolved_model_revision": getattr(base.config, "_commit_hash", None)}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
