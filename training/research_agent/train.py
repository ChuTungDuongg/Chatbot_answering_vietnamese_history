from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from training.common.cli import add_training_arguments, lora_settings_from_args, validate_training_arguments
from training.common.datasets import load_messages, split_rows, split_statistics
from training.common.qlora import QLoRASettings, resolve_precision
from training.research_agent.config import ResearchAgentConfig
from training.research_agent.validate_dataset import validate_rows
from app.agents.model_registry import validate_role_adapter


def build_parser() -> argparse.ArgumentParser:
    cfg = ResearchAgentConfig()
    parser = argparse.ArgumentParser(description="QLoRA SFT for the Research/Tool Agent.")
    parser.add_argument("--dataset", default="datasets/research_agent/history_trajectories.jsonl")
    parser.add_argument("--output-dir", default="outputs/research-agent-history-policy")
    parser.add_argument("--model-id", default=cfg.model_id)
    parser.add_argument(
        "--init-adapter",
        default=None,
        help="Start a corrective run from existing Research PEFT weights with fresh optimizer state.",
    )
    add_training_arguments(parser, cfg, auto_precision=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_training_arguments(args)
    if args.init_adapter and args.resume_from_checkpoint:
        raise ValueError("--init-adapter cannot be combined with --resume-from-checkpoint")
    if args.init_adapter:
        validate_role_adapter("research", args.init_adapter)
    rows = load_messages(args.dataset)
    validation = validate_rows(rows)
    if not validation["valid"]:
        raise ValueError(f"dataset validation failed: {validation['errors'][:5]}")
    precision = resolve_precision(
        bf16=args.bf16,
        fp16=args.fp16,
        bnb_compute_dtype=args.bnb_compute_dtype,
    )
    splits = split_rows(
        rows,
        seed=args.seed,
        max_samples=args.max_samples,
        train_ratio=0.88,
        eval_ratio=0.06,
        group_key="group_id",
        stratify_key="trajectory_class",
    )
    statistics = split_statistics(splits)
    group_sets = [
        {str(row["group_id"]) for row in getattr(splits, name)}
        for name in ("train", "eval", "test")
    ]
    assert group_sets[0].isdisjoint(group_sets[1])
    assert group_sets[0].isdisjoint(group_sets[2])
    assert group_sets[1].isdisjoint(group_sets[2])
    from transformers import AutoTokenizer
    from training.research_agent.preflight import tokenization_preflight

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenization = tokenization_preflight(rows, tokenizer, max_length=args.max_length)
    if not tokenization["valid"]:
        raise ValueError(f"Research tokenization preflight failed: {tokenization['errors'][:5]}")
    print({
        "rows": len(rows),
        "model_id": args.model_id,
        "batch_size": args.batch_size,
        "trainer_dtype": "bfloat16" if precision.bf16 else "float16" if precision.fp16 else "float32",
        "bnb_compute_dtype": precision.compute_dtype,
        "init_adapter": args.init_adapter,
        "splits": statistics,
        "tokenization": tokenization["splits"],
    })
    manifest = {
        "statistics": statistics,
        "groups": {
            name: sorted({str(row["group_id"]) for row in getattr(splits, name)})
            for name in ("train", "eval", "test")
        },
    }
    manifest_path = Path(args.output_dir) / "split_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if args.dry_run:
        return 0

    from datasets import Dataset
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM

    from training.common.qlora import build_bnb_config, build_lora_config
    from training.common.sft import AssistantOnlyCollator, build_assistant_only_example
    from training.common.trainer import build_metrics_callback, build_training_arguments
    from transformers import Trainer

    cfg = ResearchAgentConfig()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=build_bnb_config(replace(QLoRASettings(), bnb_4bit_compute_dtype=precision.compute_dtype)),
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, build_lora_config(lora_settings_from_args(args, cfg.lora)))
    model.config.use_cache = False

    def tokenize(row):
        return build_assistant_only_example(tokenizer, row["messages"], max_length=args.max_length)

    train_ds = Dataset.from_list(splits.train).map(tokenize, remove_columns=list(splits.train[0].keys()))
    eval_ds = Dataset.from_list(splits.eval).map(tokenize, remove_columns=list(splits.eval[0].keys()))
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
            bf16=precision.bf16,
            fp16=precision.fp16,
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



