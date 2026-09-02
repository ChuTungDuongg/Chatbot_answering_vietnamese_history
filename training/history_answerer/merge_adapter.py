from __future__ import annotations

import argparse
from pathlib import Path

from training.history_answerer.config import BASE_MODEL_ID
from app.agents.common.model_registry import adapter_declared_base


def merge_lora_adapter(
    *,
    model_id: str,
    adapter: str | Path,
    output_dir: str | Path,
    dtype: str = "bfloat16",
    trust_remote_code: bool = True,
) -> Path:
    declared_base = adapter_declared_base(adapter)
    if declared_base != model_id:
        raise ValueError(
            f"cannot merge adapter trained for {declared_base!r} into base {model_id!r}"
        )
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    merged.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its vanilla base model.")
    parser.add_argument("--model-id", default=BASE_MODEL_ID)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = merge_lora_adapter(
        model_id=args.model_id,
        adapter=args.adapter,
        output_dir=args.output_dir,
        dtype=args.dtype,
    )
    print(f"Merged LoRA adapter into {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
