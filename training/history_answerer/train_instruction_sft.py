from __future__ import annotations

import argparse

from training.common.cli import add_training_arguments, lora_settings_from_args, validate_training_arguments
from training.common.datasets import first_user_assistant, load_messages, split_rows
from training.history_answerer.config import Phase1Config


def build_parser() -> argparse.ArgumentParser:
    cfg = Phase1Config()
    parser = argparse.ArgumentParser(description="Legacy Phase 1 instruction SFT for the Qwen2.5 baseline.")
    parser.add_argument("--dataset", default="Dataset/merged_jsonl/all_messages.jsonl")
    parser.add_argument("--output-dir", default="artifacts/training/history_answerer/phase1_sft")
    parser.add_argument("--model-id", default=cfg.model_id)
    add_training_arguments(parser, cfg)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_training_arguments(args)
    rows = load_messages(args.dataset)
    splits = split_rows(rows, seed=args.seed, max_samples=args.max_samples or Phase1Config().max_samples)
    print({"rows": len(rows), "train": len(splits.train), "eval": len(splits.eval), "test": len(splits.test)})
    if args.dry_run:
        return 0

    from datasets import Dataset
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from training.common.qlora import QLoRASettings, build_bnb_config, build_lora_config
    from training.common.trainer import build_metrics_callback, build_training_arguments
    from training.history_answerer.loss import (
        WeightedDataCollator,
        build_instruction_training_example,
        weighted_trainer_class,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=build_bnb_config(QLoRASettings()),
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, build_lora_config(lora_settings_from_args(args, Phase1Config().lora)))
    model.config.use_cache = False

    def tokenize(row):
        user_text, assistant_text = first_user_assistant(row)
        return build_instruction_training_example(tokenizer, user_text, assistant_text, max_length=args.max_length)

    train_ds = Dataset.from_list(splits.train).map(tokenize, remove_columns=list(splits.train[0].keys()))
    eval_ds = Dataset.from_list(splits.eval).map(tokenize, remove_columns=list(splits.eval[0].keys()))
    trainer = weighted_trainer_class()(
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
        data_collator=WeightedDataCollator(tokenizer.pad_token_id),
        callbacks=[build_metrics_callback(args.output_dir)],
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



