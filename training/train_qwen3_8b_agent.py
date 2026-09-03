"""Backward-compatible CLI for the modular central-agent training package.

The implementation lives in ``training.central.train`` so configuration,
dataset/preflight, runtime/resume, and GPU training concerns can be tested and
debugged independently. Existing one-command invocations remain unchanged.
"""

from __future__ import annotations

from training.central.train.cli import main
from training.central.train.config import (
    build_lora_settings,
    build_parser,
    build_qlora_settings,
    effective_train_batch_size,
    parse_args,
    parse_lora_targets,
    safe_cli_arguments as _safe_cli_arguments,
    validate_args,
)
from training.central.train.constants import DEFAULT_MODEL_ID, MANIFEST_SCHEMA_VERSION
from training.central.train.data import (
    audit_tokenized_split,
    load_datasets,
    resolve_paths,
    sha256_file,
)
from training.central.train.engine import create_training_arguments, load_model as _load_model, load_tokenizer
from training.central.train.runtime import (
    build_run_manifest,
    checkpoint_is_valid,
    find_latest_checkpoint,
    resolve_resume_checkpoint,
    validate_resume_compatibility,
)


__all__ = [
    "DEFAULT_MODEL_ID",
    "MANIFEST_SCHEMA_VERSION",
    "_load_model",
    "_safe_cli_arguments",
    "audit_tokenized_split",
    "build_lora_settings",
    "build_parser",
    "build_qlora_settings",
    "build_run_manifest",
    "checkpoint_is_valid",
    "create_training_arguments",
    "effective_train_batch_size",
    "find_latest_checkpoint",
    "load_datasets",
    "load_tokenizer",
    "main",
    "parse_args",
    "parse_lora_targets",
    "resolve_paths",
    "resolve_resume_checkpoint",
    "sha256_file",
    "validate_args",
    "validate_resume_compatibility",
]


if __name__ == "__main__":
    raise SystemExit(main())
