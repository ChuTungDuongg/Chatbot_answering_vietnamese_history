from __future__ import annotations

import json
import sys

from training.common.qlora import resolve_precision
from training.central.normalization.validation import require_v2_trajectory
from training.trajectory_dataset.io_utils import atomic_write_json

from .config import dump_config, effective_train_batch_size, parse_args, validate_args
from .data import dataset_summary, load_datasets, resolve_paths, run_preflight
from .engine import load_tokenizer, train
from .runtime import (
    build_run_manifest,
    disk_snapshot,
    find_latest_checkpoint,
    hardware_snapshot,
    read_manifest_for_checkpoint,
    recommended_config,
    resolve_attention_implementation,
    resolve_resume_checkpoint,
    validate_resume_compatibility,
)


def _print_paths(paths) -> None:
    print(f"DRIVE_ROOT={paths.drive_root if paths.drive_root is not None else 'NONE'}")
    print(f"TRAIN_FILE={paths.train_file}")
    print(f"VALIDATION_FILE={paths.validation_file}")
    print(f"TEST_FILE={paths.test_file if paths.test_file is not None else 'NONE'}")
    print(f"OUTPUT_DIR={paths.output_dir}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    if args.print_recommended_config:
        print(json.dumps(recommended_config(hardware_snapshot()), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.dump_config is not None:
        dump_config(args)
        return 0
    paths = resolve_paths(args)
    _print_paths(paths)
    splits = load_datasets(paths, args)
    for split in splits.values():
        for row in split.rows:
            require_v2_trajectory(row)
    print(json.dumps({"datasets": dataset_summary(splits)}, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"EFFECTIVE_TRAIN_BATCH_SIZE={effective_train_batch_size(args)}")
    print(json.dumps({"disk": disk_snapshot(paths.output_dir)}, ensure_ascii=False, sort_keys=True))
    if args.dry_run:
        print("DRY_RUN=PASS")
        return 0
    tokenizer = load_tokenizer(args.tokenizer_id or args.model_id)
    preflight = run_preflight(tokenizer, splits, max_seq_length=args.max_seq_length)
    print(json.dumps({"preflight": preflight}, ensure_ascii=False, indent=2, sort_keys=True))
    if not preflight["supervision_invariants_ok"]:
        raise ValueError("canonical tokenizer preflight failed supervision invariants; refusing training")
    if args.preflight_only:
        print("PREFLIGHT_ONLY=PASS")
        return 0
    hardware = hardware_snapshot()
    print(json.dumps({"hardware": hardware}, ensure_ascii=False, indent=2, sort_keys=True))
    if not hardware["cuda_available"] and not args.allow_cpu_training:
        raise ValueError("CUDA is unavailable; refusing Qwen3-8B QLoRA training on CPU")
    if args.min_free_gpu_gb is not None:
        free = hardware.get("gpu_free_vram_gb")
        if free is None or float(free) < args.min_free_gpu_gb:
            raise ValueError(f"free GPU memory {free} GB is below --min-free-gpu-gb {args.min_free_gpu_gb}")
    precision = resolve_precision(
        bf16=args.bf16,
        fp16=args.fp16,
        bnb_compute_dtype=args.bnb_compute_dtype,
        bf16_supported=bool(hardware.get("bf16_supported")),
    )
    attention = resolve_attention_implementation(args.attn_implementation)
    resume_checkpoint = resolve_resume_checkpoint(args, paths.output_dir)
    manifest = build_run_manifest(
        args, paths, splits,
        precision=precision, attention_implementation=attention,
        hardware=hardware, resume_source=resume_checkpoint,
    )
    if resume_checkpoint is not None:
        validate_resume_compatibility(
            read_manifest_for_checkpoint(resume_checkpoint), manifest,
            allow_data_mismatch=args.allow_resume_data_mismatch,
        )
    atomic_write_json(paths.output_dir / "run_manifest.json", manifest)
    try:
        train(
            args, paths, splits, tokenizer, preflight, precision,
            attention, manifest, resume_checkpoint,
        )
    except KeyboardInterrupt:
        latest = find_latest_checkpoint(paths.output_dir)
        print(f"TRAINING_INTERRUPTED=YES\nLATEST_SAFE_CHECKPOINT={latest if latest else 'NONE'}", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
