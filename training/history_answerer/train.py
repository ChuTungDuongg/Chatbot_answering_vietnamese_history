from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.common.cli import add_training_arguments, lora_settings_from_args, validate_training_arguments
from training.common.datasets import first_user_assistant, load_messages, split_rows
from training.common.qlora import QLoRASettings, build_bnb_config, build_lora_config
from training.common.trainer import build_metrics_callback, build_training_arguments, summarize_gpu
from training.history_answerer.config import Phase6Config


def build_parser() -> argparse.ArgumentParser:
    cfg = Phase6Config()
    parser = argparse.ArgumentParser(
        description="Fresh grounded QLoRA for the History Answerer on the shared vanilla Qwen3 base."
    )
    parser.add_argument(
        "--dataset-messages",
        "--dataset",
        dest="dataset_messages",
        default="datasets/history_answerer/train.jsonl",
    )
    parser.add_argument("--dataset-chunks", default="training/Dataset/merged_jsonl/all_chunk_id.jsonl")
    parser.add_argument("--output-dir", default="outputs/history-answerer-full")
    parser.add_argument("--model-id", default=cfg.model_id)
    add_training_arguments(parser, cfg)
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset/splits without loading the model.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Phase6Config()
    validate_training_arguments(args)
    rows = load_messages(args.dataset_messages)
    from training.history_answerer.validate_dataset import validate_rows

    validation = validate_rows(rows)
    if not validation["valid"]:
        raise ValueError(f"History dataset validation failed: {validation['errors'][:5]}")
    if args.dataset_chunks and not Path(args.dataset_chunks).is_file():
        raise FileNotFoundError(args.dataset_chunks)
    splits = split_rows(
        rows,
        seed=args.seed,
        max_samples=args.max_samples,
        group_key="group_id",
        stratify_key="type",
        train_ratio=cfg.train_ratio,
        eval_ratio=cfg.eval_ratio,
    )

    from transformers import AutoTokenizer

    from training.history_answerer.loss import build_rag_training_example_with_stats
    from training.history_answerer.preflight import tokenization_report

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenization = {
        name: tokenization_report(
            getattr(splits, name), tokenizer, max_length=args.max_length, split_name=name
        )
        for name in ("train", "eval", "test")
    }
    token_errors = [error for report in tokenization.values() for error in report["errors"]]
    if token_errors:
        raise ValueError(f"History tokenization preflight failed: {token_errors[:5]}")
    summary = {
        "rows": len(rows),
        "model_id": args.model_id,
        "fresh_adapter": True,
        "splits": {
            name: {
                "rows": len(getattr(splits, name)),
                "types": {
                    row_type: sum(row.get("type") == row_type for row in getattr(splits, name))
                    for row_type in sorted({str(row.get("type")) for row in getattr(splits, name)})
                },
            }
            for name in tokenization
        },
        "tokenization": tokenization,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "preflight_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        return 0

    from datasets import Dataset
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, EarlyStoppingCallback

    from training.history_answerer.loss import (
        WeightedDataCollator,
        build_rag_training_example,
        weighted_trainer_class,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=build_bnb_config(QLoRASettings()),
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, build_lora_config(lora_settings_from_args(args, Phase6Config().lora)))
    model.config.use_cache = False
    print({"gpu": summarize_gpu()})

    def tokenize(row):
        user_text, assistant_text = first_user_assistant(row)
        return build_rag_training_example(tokenizer, user_text, assistant_text, max_length=args.max_length)

    train_ds = Dataset.from_list(splits.train).map(tokenize, remove_columns=list(splits.train[0].keys()))
    eval_ds = Dataset.from_list(splits.eval).map(tokenize, remove_columns=list(splits.eval[0].keys()))
    train_args = build_training_arguments(
        output_dir=output_dir,
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
    )
    trainer_cls = weighted_trainer_class()
    trainer = trainer_cls(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=WeightedDataCollator(tokenizer.pad_token_id),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=Phase6Config().early_stopping_patience),
            build_metrics_callback(output_dir),
        ],
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "adapter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



