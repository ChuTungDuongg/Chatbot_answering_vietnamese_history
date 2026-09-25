"""Verify historical data and index bytes against a preservation snapshot.

This command only reads files. It never regenerates a corpus or an index and
prints its report to stdout so a caller can save it outside the data tree.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOTS = ("Dataset", "training/Dataset", "artifacts", "data", "datasets", "evaluation/datasets")


def _within_repo(repo_root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Manifest path is not repository-relative: {relative_path}")
    resolved = (repo_root / candidate).resolve()
    if not resolved.is_relative_to(repo_root.resolve()):
        raise ValueError(f"Manifest path resolves outside repository: {relative_path}")
    return resolved


def fingerprint(path: Path, *, count_lines: bool = False) -> dict[str, int | str]:
    """Stream a file once to obtain its size, SHA-256, and optional line count."""
    digest = hashlib.sha256()
    size = 0
    physical_lines = 0
    last_byte = None
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
            if count_lines:
                physical_lines += chunk.count(b"\n")
                last_byte = chunk[-1]
    result: dict[str, int | str] = {"size_bytes": size, "sha256": digest.hexdigest()}
    if count_lines:
        result["physical_lines"] = physical_lines + int(last_byte is not None and last_byte != 10)
    return result


def _length_summary(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None, "p95": None}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 2),
        "median": statistics.median(ordered),
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
    }


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def diagnose_corpus(corpus_path: Path, *, near_empty_chars: int = 40) -> dict[str, Any]:
    """Stream JSONL diagnostics without changing or normalizing its records."""
    if near_empty_chars < 1:
        raise ValueError("near_empty_chars must be positive")
    chunk_ids: Counter[str] = Counter()
    text_hashes: Counter[str] = Counter()
    source_types: Counter[str] = Counter()
    source_labels: Counter[str] = Counter()
    hf_datasets: Counter[str] = Counter()
    char_lengths: list[int] = []
    token_estimates: list[int] = []
    counters: Counter[str] = Counter()
    malformed_line_numbers: list[int] = []
    digest = hashlib.sha256()
    total_bytes = 0
    physical_lines = 0

    with corpus_path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            digest.update(raw_line)
            total_bytes += len(raw_line)
            physical_lines += 1
            if not raw_line.strip():
                counters["blank_lines"] += 1
                continue
            try:
                row = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                counters["malformed_json_rows"] += 1
                if len(malformed_line_numbers) < 20:
                    malformed_line_numbers.append(line_number)
                continue
            if not isinstance(row, dict):
                counters["non_object_rows"] += 1
                continue

            counters["chunks"] += 1
            chunk_id = row.get("chunk_id")
            if _present(chunk_id):
                chunk_ids[str(chunk_id).strip()] += 1
            else:
                counters["missing_chunk_ids"] += 1

            text = row.get("text")
            if not isinstance(text, str):
                counters["non_string_text_rows"] += 1
                text = ""
            trimmed_text = text.strip()
            if not trimmed_text:
                counters["empty_text_rows"] += 1
            else:
                text_hashes[hashlib.sha256(text.encode("utf-8")).hexdigest()] += 1
                counters["nonempty_text_rows"] += 1
            char_length = len(text)
            char_lengths.append(char_length)
            token_estimates.append(len(text.split()))
            if len(trimmed_text) < near_empty_chars:
                counters["near_empty_chunks"] += 1

            if not _present(row.get("title")):
                counters["missing_titles"] += 1
            metadata = row.get("metadata")
            if not isinstance(metadata, dict) or not metadata:
                counters["missing_source_metadata"] += 1
                metadata = {}

            explicit_source_id = any(_present(row.get(key)) or _present(metadata.get(key)) for key in ("source_id", "document_id", "doc_id"))
            inferred_source_id = _present(metadata.get("source_sha1")) or (
                _present(row.get("hf_dataset")) and row.get("raw_record_index") is not None
            )
            if explicit_source_id:
                counters["explicit_source_id_rows"] += 1
            elif inferred_source_id:
                counters["inferred_source_id_rows"] += 1
            else:
                counters["missing_source_ids"] += 1

            source_type = str(row.get("source_type") or "unknown").strip() or "unknown"
            source_types[source_type] += 1
            source_label = str(row.get("source") or "unknown").strip() or "unknown"
            source_labels[source_label] += 1
            hf_dataset = row.get("hf_dataset")
            if _present(hf_dataset):
                hf_datasets[str(hf_dataset)] += 1
            url_expected = any(term in source_type.lower() for term in ("web", "wiki", "page", "site", "url")) or source_label.startswith(("http://", "https://"))
            if url_expected:
                counters["url_expected_rows"] += 1
                if not _present(row.get("url")):
                    counters["missing_expected_urls"] += 1

            years_present = _present(metadata.get("years")) or _present(row.get("year"))
            dates_present = _present(metadata.get("dates")) or _present(row.get("date"))
            if years_present:
                counters["year_metadata_rows"] += 1
            if dates_present:
                counters["date_metadata_rows"] += 1
            if years_present or dates_present:
                counters["year_or_date_metadata_rows"] += 1

    chunks = counters["chunks"]
    duplicate_id_counts = [count for count in chunk_ids.values() if count > 1]
    duplicate_text_rows = sum(count - 1 for count in text_hashes.values() if count > 1)
    nonempty_text_rows = counters["nonempty_text_rows"]
    return {
        "schema_version": 1,
        "mode": "corpus_diagnostics",
        "corpus": str(corpus_path),
        "size_bytes": total_bytes,
        "sha256": digest.hexdigest(),
        "physical_lines": physical_lines,
        "chunk_count": chunks,
        "blank_lines": counters["blank_lines"],
        "malformed_json_rows": counters["malformed_json_rows"],
        "malformed_json_line_examples": malformed_line_numbers,
        "non_object_rows": counters["non_object_rows"],
        "chunk_ids": {
            "unique_count": len(chunk_ids),
            "missing_count": counters["missing_chunk_ids"],
            "duplicate_id_count": len(duplicate_id_counts),
            "duplicate_rows": sum(count - 1 for count in duplicate_id_counts),
        },
        "text": {
            "non_string_rows": counters["non_string_text_rows"],
            "empty_rows": counters["empty_text_rows"],
            "unique_nonempty_texts": len(text_hashes),
            "exact_duplicate_rows": duplicate_text_rows,
            "exact_duplicate_text_rate": round(duplicate_text_rows / nonempty_text_rows, 6) if nonempty_text_rows else None,
            "near_empty_chunks": counters["near_empty_chunks"],
            "near_empty_threshold_chars": near_empty_chars,
        },
        "missing_titles": counters["missing_titles"],
        "missing_source_metadata": counters["missing_source_metadata"],
        "source_ids": {
            "explicit_rows": counters["explicit_source_id_rows"],
            "inferred_rows": counters["inferred_source_id_rows"],
            "missing_rows": counters["missing_source_ids"],
            "inference_note": "metadata.source_sha1 or hf_dataset with raw_record_index counts as an inferred source identifier",
        },
        "urls": {
            "expected_rows": counters["url_expected_rows"],
            "missing_where_expected": counters["missing_expected_urls"],
            "expectation_note": "URL expected for web/wiki/page/site/url source types or URL-shaped source labels",
        },
        "length_distribution": {
            "characters": _length_summary(char_lengths),
            "whitespace_token_estimate": _length_summary(token_estimates),
            "token_note": "Whitespace-delimited estimate; no model tokenizer is loaded",
        },
        "year_date_metadata_coverage": {
            "year_rows": counters["year_metadata_rows"],
            "date_rows": counters["date_metadata_rows"],
            "either_rows": counters["year_or_date_metadata_rows"],
            "either_rate": round(counters["year_or_date_metadata_rows"] / chunks, 6) if chunks else None,
        },
        "source_distribution": {
            "source_types": dict(sorted(source_types.items())),
            "distinct_source_labels": len(source_labels),
            "top_source_label_counts": [count for _, count in source_labels.most_common(20)],
            "hf_datasets": dict(sorted(hf_datasets.items())),
        },
    }


def audit_manifest(manifest_path: Path, repo_root: Path = REPO_ROOT, *, all_files: bool = False) -> dict[str, Any]:
    """Compare a snapshot with current files without writing to the repository."""
    repo_root = repo_root.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inventory = manifest["files"]
    if not isinstance(inventory, list):
        raise ValueError("Manifest 'files' must be a list")
    expected_files = [
        entry for entry in inventory
        if all_files or entry.get("preservation_tier", "protected") == "protected"
    ]
    inventory_paths = [entry["path"] for entry in inventory]
    if len(inventory_paths) != len(set(inventory_paths)):
        raise ValueError("Duplicate manifest paths")
    inventory_path_set = set(inventory_paths)

    missing: list[str] = []
    modified: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    unchanged = 0
    checked_bytes = 0

    for entry in expected_files:
        relative_path = entry["path"]
        if not isinstance(relative_path, str):
            raise ValueError("Manifest file path must be a string")
        path = _within_repo(repo_root, relative_path)
        if not path.is_file():
            missing.append(relative_path)
            continue
        try:
            actual = fingerprint(path, count_lines="physical_lines" in entry)
        except OSError as exc:
            errors.append({"path": relative_path, "error": type(exc).__name__})
            continue
        checked_bytes += int(actual["size_bytes"])
        differences = {
            key: {"expected": entry[key], "actual": actual[key]}
            for key in ("size_bytes", "sha256", "physical_lines")
            if key in entry and actual[key] != entry[key]
        }
        if differences:
            modified.append({"path": relative_path, "differences": differences})
        else:
            unchanged += 1

    unexpected: list[str] = []
    for root in manifest.get("roots", []):
        root_path = _within_repo(repo_root, root)
        if not root_path.is_dir():
            continue
        for path in root_path.rglob("*"):
            if path.is_file():
                relative_path = path.relative_to(repo_root).as_posix()
                if relative_path not in inventory_path_set:
                    unexpected.append(relative_path)

    chunk_counts = {
        entry["path"]: entry["jsonl_record_count_estimate"]
        for entry in expected_files
        if "/corpus/" in entry["path"] and "jsonl_record_count_estimate" in entry
    }
    index_counts = {
        entry["path"]: entry["declared_index_count"]
        for entry in expected_files
        if "declared_index_count" in entry
    }
    return {
        "manifest": str(manifest_path),
        "snapshot_utc": manifest.get("snapshot_utc"),
        "scope": "all_files" if all_files else "protected",
        "inventory_files": len(inventory),
        "expected_files": len(expected_files),
        "unchanged_files": unchanged,
        "checked_bytes": checked_bytes,
        "missing_files": missing,
        "modified_files": modified,
        "read_errors": errors,
        "unexpected_files": sorted(set(unexpected)),
        "corpus_chunk_counts": chunk_counts,
        "index_declared_counts": index_counts,
        "preserved": not (missing or modified or errors),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only corpus diagnostics or SHA-256 preservation audit.")
    parser.add_argument("--corpus", type=Path, help="JSONL corpus to inspect without changing it.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path outside corpus/data/index directories.")
    parser.add_argument("--near-empty-chars", type=int, default=40, help="Character threshold for a near-empty chunk (default: 40).")
    parser.add_argument("--manifest", type=Path, default=Path("docs/corpus_preservation_manifest.json"))
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--all-files", action="store_true", help="Verify mutable and model files recorded for inventory review too.")
    parser.add_argument("--strict-new", action="store_true", help="Also fail if new files appear under snapshotted roots.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output and not args.corpus:
        raise ValueError("--output requires --corpus; preservation reports print to stdout")
    if args.corpus:
        corpus_path = args.corpus.resolve()
        report = diagnose_corpus(corpus_path, near_empty_chars=args.near_empty_chars)
        if args.output:
            output_path = args.output.resolve()
            if output_path == corpus_path or output_path.is_relative_to(corpus_path.parent):
                raise ValueError("Report output must be outside the corpus directory")
            repo_root = args.repo_root.resolve()
            if any(output_path.is_relative_to(repo_root / root) for root in DATA_ROOTS):
                raise ValueError("Report output must be outside repository data and artifact directories")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    manifest_path = args.manifest if args.manifest.is_absolute() else args.repo_root / args.manifest
    report = audit_manifest(manifest_path, args.repo_root, all_files=args.all_files)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["preserved"] and (not args.strict_new or not report["unexpected_files"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
