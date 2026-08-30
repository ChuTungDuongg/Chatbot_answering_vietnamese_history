from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.common.qlora import PrecisionSettings

from .config import (
    build_lora_settings,
    build_qlora_settings,
    effective_train_batch_size,
    safe_cli_arguments,
    world_size,
)
from .constants import CHECKPOINT_PATTERN, MANIFEST_SCHEMA_VERSION
from .data import DatasetSplit, ResolvedPaths, dataset_summary


def hardware_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(), "platform": platform.platform(),
        "cuda_available": False, "gpu_name": None, "gpu_total_vram_gb": None,
        "gpu_free_vram_gb": None, "compute_capability": None,
        "bf16_supported": False, "cuda_version": None,
    }
    try:
        import torch

        result["torch_version"] = torch.__version__
        result["cuda_version"] = torch.version.cuda
        result["cuda_available"] = bool(torch.cuda.is_available())
        if result["cuda_available"]:
            properties = torch.cuda.get_device_properties(0)
            result["gpu_name"] = torch.cuda.get_device_name(0)
            result["gpu_total_vram_gb"] = round(properties.total_memory / 1024**3, 3)
            result["compute_capability"] = list(torch.cuda.get_device_capability(0))
            result["bf16_supported"] = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
            try:
                free_bytes, _ = torch.cuda.mem_get_info(0)
                result["gpu_free_vram_gb"] = round(free_bytes / 1024**3, 3)
            except RuntimeError:
                pass
    except (ImportError, RuntimeError) as exc:
        result["torch_error"] = str(exc)
    for package in ("transformers", "peft", "bitsandbytes", "datasets", "accelerate"):
        try:
            result[f"{package}_version"] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[f"{package}_version"] = None
    return result


def disk_snapshot(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    free_gb = round(usage.free / 1024**3, 3)
    return {
        "filesystem_root": path.anchor or str(path),
        "free_gb": free_gb,
        "total_gb": round(usage.total / 1024**3, 3),
        "low_space_warning": free_gb < 10.0,
    }


def resolve_attention_implementation(requested: str) -> str:
    if requested == "auto":
        return "sdpa"
    if requested == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        raise ValueError(
            "flash_attention_2 was explicitly requested but flash-attn is unavailable. "
            "Install a CUDA/PyTorch-compatible flash-attn build or use --attn-implementation sdpa."
        )
    return requested


def checkpoint_is_valid(path: Path) -> bool:
    adapter = (path / "adapter_model.safetensors").is_file() or (path / "adapter_model.bin").is_file()
    return all((path / name).is_file() for name in (
        "trainer_state.json", "optimizer.pt", "scheduler.pt", "adapter_config.json",
    )) and adapter


def valid_checkpoints(output_dir: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    if not output_dir.exists():
        return found
    for candidate in output_dir.iterdir():
        match = CHECKPOINT_PATTERN.fullmatch(candidate.name)
        if match and candidate.is_dir() and checkpoint_is_valid(candidate):
            found.append((int(match.group(1)), candidate.resolve()))
    return sorted(found)


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = valid_checkpoints(output_dir)
    return checkpoints[-1][1] if checkpoints else None


def resolve_resume_checkpoint(args: argparse.Namespace, output_dir: Path) -> Path | None:
    if args.resume_from_checkpoint:
        explicit = Path(args.resume_from_checkpoint).expanduser()
        if not explicit.is_absolute():
            explicit = explicit.resolve()
        if not checkpoint_is_valid(explicit):
            raise ValueError(f"explicit resume checkpoint is missing required Trainer/adapter state: {explicit}")
        if explicit.parent.resolve() != output_dir.resolve():
            raise ValueError(
                f"explicit resume checkpoint is outside this run output directory: {explicit.parent} != {output_dir}"
            )
        print(f"AUTO_RESUME_FOUND=EXPLICIT:{explicit}")
        return explicit
    if args.auto_resume:
        latest = find_latest_checkpoint(output_dir)
        print(f"AUTO_RESUME_FOUND={latest if latest is not None else 'NONE'}")
        return latest
    return None


def read_manifest_for_checkpoint(checkpoint: Path) -> dict[str, Any]:
    manifest_path = checkpoint.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"resume run manifest is missing: {manifest_path}")
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read resume manifest {manifest_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"resume manifest must be an object: {manifest_path}")
    return value


def validate_resume_compatibility(
    previous: dict[str, Any], current: dict[str, Any], *, allow_data_mismatch: bool,
) -> None:
    for key in ("run_name", "model_id", "max_seq_length"):
        if previous.get(key) != current.get(key):
            raise ValueError(f"unsafe resume mismatch for {key}: {previous.get(key)!r} != {current.get(key)!r}")
    for key in ("qlora", "lora"):
        if previous.get(key) != current.get(key):
            raise ValueError(f"unsafe resume mismatch for {key}")
    previous_files = previous.get("datasets") if isinstance(previous.get("datasets"), dict) else {}
    current_files = current.get("datasets") if isinstance(current.get("datasets"), dict) else {}
    mismatches = [
        name for name in ("train", "validation")
        if (previous_files.get(name) or {}).get("sha256") != (current_files.get(name) or {}).get("sha256")
    ]
    if mismatches and not allow_data_mismatch:
        raise ValueError(
            f"resume dataset fingerprint mismatch for {mismatches}; use --allow-resume-data-mismatch only intentionally"
        )
    previous_args = previous.get("cli_arguments") if isinstance(previous.get("cli_arguments"), dict) else {}
    current_args = current.get("cli_arguments") if isinstance(current.get("cli_arguments"), dict) else {}
    unsafe_settings = (
        "learning_rate", "per_device_train_batch_size", "gradient_accumulation_steps",
        "optim", "lr_scheduler_type", "warmup_ratio", "warmup_steps",
    )
    changed = [key for key in unsafe_settings if previous_args.get(key) != current_args.get(key)]
    if changed:
        raise ValueError(f"unsafe resume training-setting mismatch: {changed}")


def git_snapshot(workdir: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=workdir, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=workdir, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
        return {"commit_sha": commit, "dirty_working_tree": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit_sha": None, "dirty_working_tree": None}


def build_run_manifest(
    args: argparse.Namespace,
    paths: ResolvedPaths,
    splits: dict[str, DatasetSplit],
    *,
    precision: PrecisionSettings,
    attention_implementation: str,
    hardware: dict[str, Any],
    resume_source: Path | None,
) -> dict[str, Any]:
    lora = build_lora_settings(args)
    lora_manifest = asdict(lora)
    lora_manifest["target_modules"] = list(lora.target_modules)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_name": args.run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "tokenizer_id": args.tokenizer_id or args.model_id,
        "datasets": dataset_summary(splits),
        "max_seq_length": args.max_seq_length,
        "cli_arguments": safe_cli_arguments(args),
        "precision": asdict(precision),
        "attention_implementation": attention_implementation,
        "qlora": asdict(build_qlora_settings(args, precision)),
        "lora": lora_manifest,
        "effective_train_batch_size": effective_train_batch_size(args),
        "world_size": world_size(),
        "hardware": hardware,
        "disk": disk_snapshot(paths.output_dir),
        "git": git_snapshot(Path(__file__).resolve().parents[2]),
        "resume_source": str(resume_source) if resume_source else None,
        "output_directory": str(paths.output_dir),
        "loss_contract": {
            "assistant_tool_calls": "trained",
            "assistant_final_answers": "trained",
            "system_user_tool_observations": "masked",
            "preprocessor": "training.trajectory_dataset.preprocess.build_canonical_sft_example",
        },
    }


def recommended_config(hardware: dict[str, Any]) -> dict[str, Any]:
    vram = float(hardware.get("gpu_total_vram_gb") or 0.0)
    name = str(hardware.get("gpu_name") or "unknown")
    if vram >= 70:
        profile, batch, accumulation = "A100_80GB", 4, 4
    elif vram >= 35:
        profile, batch, accumulation = "A100_40GB", 2, 8
    else:
        profile, batch, accumulation = "L4_24GB_or_smaller", 1, 16
    return {
        "informational_only": True,
        "detected_gpu": name,
        "detected_vram_gb": vram or None,
        "profile": profile,
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_use_double_quant": True,
        "max_seq_length": 4096,
        "per_device_train_batch_size": batch,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": accumulation,
        "gradient_checkpointing": True,
        "effective_batch_size_single_gpu": batch * accumulation,
        "note": "Measure memory on the actual model/data; this command never changes training arguments.",
    }
