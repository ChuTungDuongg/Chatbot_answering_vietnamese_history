from __future__ import annotations

import re


DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"
DEFAULT_RUN_NAME = "qwen3-8b-agent-v1"
DEFAULT_TRAIN_FILE = "training/trajectory_dataset/outputs/final/train.jsonl"
DEFAULT_VALIDATION_FILE = "training/trajectory_dataset/outputs/final/validation.jsonl"
MANIFEST_SCHEMA_VERSION = "qwen3-central-agent-run-v1"
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")
ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "README.md",
)
