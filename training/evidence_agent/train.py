from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.common.cli import add_training_arguments, lora_settings_from_args, validate_training_arguments
from training.common.datasets import split_rows, split_statistics
from training.common.jsonl import read_jsonl
from training.evidence_agent.config import EvidenceAgentConfig
from training.evidence_agent.sft import prepare_evidence_split
from training.evidence_agent.validate_dataset import validate_rows
from app.agents.model_registry import validate_role_adapter


def build_parser() -> argparse.ArgumentParser:
    cfg = EvidenceAgentConfig()
    parser = argparse.ArgumentParser(description="QLoRA SFT for the canonical Evidence Critic/Compressor contract.")
    parser.add_argument("--dataset", default="datasets/evidence_agent/train.jsonl")
    parser.add_argument("--output-dir", default="artifacts/training/evidence_agent/qwen3_critic_v2")
    parser.add_argument("--model-id", default=cfg.model_id)
    parser.add_argument(
        "--init-adapter",
        default=None,
        help="Start a new run from learned PEFT adapter weights with a fresh optimizer/scheduler.",
    )
    add_training_arguments(parser, cfg)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def initialize_peft_adapter(model, args: argparse.Namespace, cfg: EvidenceAgentConfig):
    """Attach either an existing trainable adapter or a fresh LoRA adapter."""
    if args.init_adapter:
        from peft import PeftModel

        return PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    from peft import get_peft_model
    from training.common.qlora import build_lora_config

    return get_peft_model(model, build_lora_config(lora_settings_from_args(args, cfg.lora)))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_training_arguments(args)
    if args.init_adapter and args.resume_from_checkpoint:
        raise ValueError("--init-adapter starts a new run and cannot be combined with --resume-from-checkpoint")
    if args.init_adapter:
        validate_role_adapter("evidence", args.init_adapter)
    rows = read_jsonl(args.dataset)
    validation = validate_rows(rows)
    if not validation["valid"]:
        raise ValueError(f"dataset validation failed: {validation['errors'][:5]}")
    splits = split_rows(
        rows,
        seed=args.seed,
        max_samples=args.max_samples,
        group_key="group_id",
        stratify_key="behavior",
    )
    statistics = split_statistics(splits)
    split_groups = [
        {str(row["group_id"]) for row in getattr(splits, name)}
        for name in ("train", "eval", "test")
    ]
    assert split_groups[0].isdisjoint(split_groups[1])
    assert split_groups[0].isdisjoint(split_groups[2])
    assert split_groups[1].isdisjoint(split_groups[2])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prepared: dict[str, list[dict[str, list[int]]]] = {}
    token_statistics = {}
    for name in ("train", "eval", "test"):
        prepared[name], token_statistics[name] = prepare_evidence_split(
            tokenizer,
            getattr(splits, name),
            max_length=args.max_length,
            split_name=name,
        )

    manifest = {
        "statistics": statistics,
        "tokenization": token_statistics,
        "groups": {
            name: sorted({str(row["group_id"]) for row in getattr(splits, name)})
            for name in ("train", "eval", "test")
        },
    }
    manifest_path = Path(args.output_dir) / "split_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "rows": len(rows),
        "model_id": args.model_id,
        "batch_size": args.batch_size,
        "init_adapter": args.init_adapter,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "splits": statistics,
        "tokenization": token_statistics,
    }, ensure_ascii=False, sort_keys=True))
    if args.dry_run:
        return 0

    from datasets import Dataset
    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, Trainer

    from training.common.qlora import QLoRASettings, build_bnb_config
    from training.common.sft import AssistantOnlyCollator
    from training.common.trainer import build_metrics_callback, build_training_arguments

    cfg = EvidenceAgentConfig()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=build_bnb_config(QLoRASettings()),
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = initialize_peft_adapter(model, args, cfg)
    model.config.use_cache = False

    train_ds = Dataset.from_list(prepared["train"])
    eval_ds = Dataset.from_list(prepared["eval"])
    trainer = Trainer(
        model=model,
        args=build_training_arguments(
            output_dir=args.output_dir,
            epochs=args.epochs,
            train_batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            grad_accum_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            bf16=args.bf16,
            fp16=args.fp16,
            gradient_checkpointing=args.gradient_checkpointing,
            seed=args.seed,
            report_to=args.report_to,
        ),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=AssistantOnlyCollator(tokenizer.pad_token_id),
        callbacks=[build_metrics_callback(args.output_dir)],
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
