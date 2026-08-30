from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.trajectory_dataset.io_utils import read_jsonl
from training.trajectory_dataset.preprocess import IGNORE_INDEX, analyze_truncation, build_canonical_sft_example
from training.trajectory_dataset.validate import validate_rows

from .constants import DEFAULT_TRAIN_FILE, DEFAULT_VALIDATION_FILE


@dataclass(frozen=True)
class ResolvedPaths:
    drive_root: Path | None
    dataset_root: Path | None
    train_file: Path
    validation_file: Path
    test_file: Path | None
    output_dir: Path


@dataclass
class DatasetSplit:
    name: str
    path: Path
    rows: list[dict[str, Any]]
    selected_rows: list[dict[str, Any]]
    sha256: str
    source_distribution: dict[str, int]
    task_distribution: dict[str, int]


def _writable_directory(path: Path, *, create: bool) -> None:
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"directory does not exist: {path}")
    try:
        descriptor, probe = tempfile.mkstemp(prefix=".qwen3-write-probe-", dir=path)
        os.close(descriptor)
        Path(probe).unlink()
    except OSError as exc:
        raise ValueError(f"directory is not writable: {path}: {exc}") from exc


def _resolve_under(base: Path | None, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


def resolve_paths(args: argparse.Namespace, *, verify_writable: bool = True) -> ResolvedPaths:
    drive_root = Path(args.drive_root).expanduser().resolve() if args.drive_root else None
    if drive_root is not None:
        _writable_directory(drive_root, create=False)
    dataset_root = _resolve_under(drive_root, args.dataset_root) if args.dataset_root else None
    if dataset_root is not None and (not dataset_root.exists() or not dataset_root.is_dir()):
        raise ValueError(f"dataset root does not exist: {dataset_root}")

    def dataset_file(explicit: str | None, default_name: str, legacy_default: str | None) -> Path | None:
        if explicit:
            return _resolve_under(dataset_root, explicit)
        if dataset_root is not None:
            return (dataset_root / default_name).resolve()
        return Path(legacy_default).resolve() if legacy_default else None

    train_file = dataset_file(args.train_file, "train.jsonl", DEFAULT_TRAIN_FILE)
    validation_file = dataset_file(args.validation_file, "validation.jsonl", DEFAULT_VALIDATION_FILE)
    test_file = dataset_file(args.test_file, "test.jsonl", None)
    if args.test_file is None and test_file is not None and not test_file.exists():
        test_file = None
    if args.output_dir:
        output_dir = _resolve_under(drive_root, args.output_dir)
    elif drive_root is not None:
        output_dir = (drive_root / "training_runs" / args.run_name).resolve()
    else:
        output_dir = (Path("outputs") / args.run_name).resolve()
    if verify_writable:
        _writable_directory(output_dir, create=True)
    assert train_file is not None and validation_file is not None
    return ResolvedPaths(
        drive_root=drive_root, dataset_root=dataset_root,
        train_file=train_file, validation_file=validation_file,
        test_file=test_file, output_dir=output_dir,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_limit(rows: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    if limit is None or limit >= len(rows):
        return list(rows)
    indices = sorted(random.Random(seed).sample(range(len(rows)), limit))
    return [rows[index] for index in indices]


def _distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key) or "unknown") for row in rows).items()))


def load_dataset_split(
    name: str,
    path: Path,
    *,
    sample_limit: int | None,
    seed: int,
    required_nonempty: bool,
) -> DatasetSplit:
    if not path.exists() or not path.is_file():
        raise ValueError(f"{name} dataset file does not exist: {path}")
    rows = read_jsonl(path)
    validation = validate_rows(rows)
    if validation.rejected:
        reasons = [item["reason"] for item in validation.rejected[:5]]
        raise ValueError(
            f"canonical dataset validation failed for {name} ({len(validation.rejected)} invalid rows) "
            f"at {path}: {reasons}"
        )
    if required_nonempty and not validation.valid:
        raise ValueError(f"canonical {name} dataset is empty: {path}")
    selected = deterministic_limit(validation.valid, sample_limit, seed)
    return DatasetSplit(
        name=name, path=path, rows=validation.valid, selected_rows=selected,
        sha256=sha256_file(path),
        source_distribution=_distribution(validation.valid, "source_dataset"),
        task_distribution=_distribution(validation.valid, "task_type"),
    )


def _explicit_groups(row: dict[str, Any]) -> set[str]:
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    values = provenance.get("source_groups")
    if isinstance(values, list):
        groups = {str(value) for value in values if value not in (None, "")}
        if groups:
            return groups
    for value in (
        provenance.get("source_group"), provenance.get("source_document_id"),
        provenance.get("article_id"), row.get("group_id"),
    ):
        if value not in (None, ""):
            return {str(value)}
    return set()


def verify_no_group_leakage(splits: dict[str, DatasetSplit]) -> None:
    groups = {
        name: {group for row in split.rows for group in _explicit_groups(row)}
        for name, split in splits.items()
    }
    names = sorted(groups)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = groups[left] & groups[right]
            if overlap:
                raise ValueError(f"source-group leakage between {left} and {right}: {sorted(overlap)[:10]}")


def load_datasets(paths: ResolvedPaths, args: argparse.Namespace) -> dict[str, DatasetSplit]:
    data_seed = args.seed if args.data_seed is None else args.data_seed
    splits = {
        "train": load_dataset_split(
            "train", paths.train_file, sample_limit=args.max_train_samples,
            seed=data_seed, required_nonempty=True,
        ),
        "validation": load_dataset_split(
            "validation", paths.validation_file, sample_limit=args.max_validation_samples,
            seed=data_seed + 1, required_nonempty=True,
        ),
    }
    if paths.test_file is not None:
        splits["test"] = load_dataset_split(
            "test", paths.test_file, sample_limit=args.max_test_samples,
            seed=data_seed + 2, required_nonempty=False,
        )
    if args.evaluate_test_after_train and "test" not in splits:
        raise ValueError("--evaluate-test-after-train requires --test-file or <dataset-root>/test.jsonl")
    verify_no_group_leakage(splits)
    return splits


def dataset_summary(splits: dict[str, DatasetSplit]) -> dict[str, Any]:
    return {
        name: {
            "path": str(split.path), "rows": len(split.rows),
            "selected_rows": len(split.selected_rows), "sha256": split.sha256,
            "source_dataset_distribution": split.source_distribution,
            "task_type_distribution": split.task_distribution,
        }
        for name, split in splits.items()
    }


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))]


def audit_tokenized_split(tokenizer: Any, rows: list[dict[str, Any]], *, max_seq_length: int) -> dict[str, Any]:
    lengths: list[int] = []
    initial_user_lost = assistant_lost = tool_call_lost = final_lost = all_lost = 0
    zero_supervised = preprocessing_errors = 0
    error_examples: list[str] = []
    for index, row in enumerate(rows):
        try:
            truncation = analyze_truncation(tokenizer, row, max_length=max_seq_length)
            lengths.append(int(truncation["total_tokens"]))
            initial_user_lost += int(bool(truncation["initial_user_lost"]))
            assistant_lost += int(int(truncation["lost_assistant_targets"]) > 0)
            tool_call_lost += int(int(truncation["lost_tool_call_targets"]) > 0)
            final_lost += int(bool(truncation["final_assistant_lost"]))
            all_lost += int(bool(truncation["all_assistant_supervision_lost"]))
            feature = build_canonical_sft_example(tokenizer, row, max_length=max_seq_length)
            if not any(label != IGNORE_INDEX for label in feature["labels"]):
                zero_supervised += 1
        except Exception as exc:
            preprocessing_errors += 1
            message = str(exc)
            if "zero" in message.casefold() and "assistant" in message.casefold():
                zero_supervised += 1
            if len(error_examples) < 5:
                error_examples.append(f"row {index} ({row.get('id')}): {message}")
    return {
        "rows": len(rows),
        "tokens": {
            "max": max(lengths, default=0), "p50": _percentile(lengths, 0.50),
            "p95": _percentile(lengths, 0.95), "p99": _percentile(lengths, 0.99),
            "mean": round(sum(lengths) / len(lengths), 3) if lengths else 0.0,
        },
        "rows_over_max_seq_length": sum(value > max_seq_length for value in lengths),
        "rows_initial_user_lost": initial_user_lost,
        "rows_any_assistant_supervision_lost": assistant_lost,
        "rows_any_tool_call_supervision_lost": tool_call_lost,
        "rows_final_assistant_supervision_lost": final_lost,
        "rows_all_assistant_supervision_lost": all_lost,
        "rows_zero_supervised_tokens": zero_supervised,
        "preprocessing_errors": preprocessing_errors,
        "error_examples": error_examples,
        "supervision_invariants_ok": not any((
            initial_user_lost, assistant_lost, tool_call_lost, final_lost,
            all_lost, zero_supervised, preprocessing_errors,
        )),
    }


def run_preflight(tokenizer: Any, splits: dict[str, DatasetSplit], *, max_seq_length: int) -> dict[str, Any]:
    reports = {
        name: audit_tokenized_split(tokenizer, split.selected_rows, max_seq_length=max_seq_length)
        for name, split in splits.items()
    }
    return {
        "max_seq_length": max_seq_length,
        "splits": reports,
        "supervision_invariants_ok": all(report["supervision_invariants_ok"] for report in reports.values()),
    }
