"""Progressive V1 build; stages can be verified and reused after interruption."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sqlite3
import sys
from typing import Any, Callable
from urllib.parse import quote

from scripts.corpus_v1 import PIPELINE_VERSION, SCHEMA_VERSION
from scripts.corpus_v1.chunking import chunk_text
from scripts.corpus_v1.history_filter import YEAR, classify
from scripts.corpus_v1.normalize import normalize_text
from scripts.corpus_v1.provenance import atomic_json, canonical, digest_file, git_sha, stage
from scripts.corpus_v1.schema import sha256_text, stable_id
from scripts.corpus_v1.source import detect_fields, iter_records, load_source


V0_ROOTS = ("vn_history_deployment", "vn_history_modal")


def _safe_output(root: Path) -> Path:
    root = root.resolve()
    if any(part.casefold() in V0_ROOTS for part in root.parts):
        raise ValueError("Corpus V1 output cannot be inside a protected V0 artifact root")
    if "corpus_v1" not in {part.casefold() for part in root.parts}:
        raise ValueError("Output path must contain a corpus_v1 directory")
    return root


def _rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed checkpoint {path}:{number}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"Non-object checkpoint {path}:{number}")
            yield row


def _write(stream, row: dict[str, Any]) -> None:
    stream.write(canonical(row) + "\n")


def _copy_atomic(source: Path, destination: Path) -> None:
    if destination.exists():
        if digest_file(source) != digest_file(destination):
            raise RuntimeError(f"Existing final output differs: {destination}")
        return
    temp = destination.with_name(destination.name + ".partial")
    if temp.exists():
        raise RuntimeError(f"Incomplete final output: {temp}")
    with source.open("rb") as reader, temp.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
    os.replace(temp, destination)


def build(config: dict[str, Any], output: Path, *, resume: bool = False,
          source_factory: Callable[[], Any] | None = None,
          token_counter: Callable[[str], int] | None = None) -> dict[str, Any]:
    root = _safe_output(output)
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError(f"Output exists: {root}; use --resume or another output")
    root.mkdir(parents=True, exist_ok=True)
    cfg = {"schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION,
           "dataset_id": config["dataset_id"], "dataset_config": config.get("dataset_config"),
           "dataset_revision": config.get("dataset_revision"), "split": config.get("split", "train"),
           "streaming": config.get("streaming", True), "cache_dir": config.get("cache_dir"),
           "fields": config.get("fields", {}), "chunk_tokens": config.get("chunk_tokens", 384),
           "chunk_overlap": config.get("chunk_overlap", 48),
           "tokenizer_id": config.get("tokenizer_id", "intfloat/multilingual-e5-base"),
           "filter": "broad_high_recall_v1", "normalization": "unicode_nfc_whitespace_v1",
           "dedup": "normalized_exact_sha256_v1", "include_review": True, "sample_seed": 1729}
    if not 16 <= cfg["chunk_tokens"] or not 0 <= cfg["chunk_overlap"] < cfg["chunk_tokens"]:
        raise ValueError("Invalid chunk budget/overlap")
    fingerprint = hashlib.sha256(canonical(cfg).encode("utf-8")).hexdigest()
    old_config = root / "config.json"
    if old_config.exists():
        if json.loads(old_config.read_text(encoding="utf-8")) != cfg:
            raise RuntimeError("Build configuration differs; use a new output directory")
    else:
        atomic_json(old_config, cfg)
    source_meta_path = root / "source_manifest.json"
    if source_meta_path.exists():
        source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
        if source_meta.get("config_fingerprint") != fingerprint:
            raise RuntimeError("Source manifest configuration mismatch")
    else:
        source_meta = {"dataset_id": cfg["dataset_id"], "dataset_config": cfg["dataset_config"],
                       "dataset_revision": cfg["dataset_revision"], "split": cfg["split"],
                       "streaming": cfg["streaming"], "config_fingerprint": fingerprint,
                       "hf_fingerprint": None}
        atomic_json(source_meta_path, source_meta)

    normalized = root / "intermediate/10_normalized_documents.jsonl"

    def make_normalized(temp: Path) -> dict[str, Any]:
        dataset = source_factory() if source_factory else load_source(cfg)
        source_meta["hf_fingerprint"] = getattr(dataset, "_fingerprint", None)
        iterator = iter_records(dataset)
        try:
            first_index, first = next(iterator)
        except StopIteration:
            first_index, first = None, None
        names = list(getattr(dataset, "features", None) or (first or {}).keys())
        fields = detect_fields(names, cfg["fields"])
        source_meta["field_mapping"] = fields
        atomic_json(source_meta_path, source_meta)
        count = 0
        with temp.open("w", encoding="utf-8", newline="\n") as writer:
            for index, raw in ([] if first is None else [(first_index, first)]):
                count += _normalize_record(writer, index, raw, cfg, fields)
            for index, raw in iterator:
                count += _normalize_record(writer, index, raw, cfg, fields)
        return {"count": count}

    s10 = stage(root, "10_normalized_documents", fingerprint, resume, make_normalized)
    filtered = root / "intermediate/20_filtered_documents.jsonl"

    def make_filtered(temp: Path) -> dict[str, Any]:
        counts: Counter[str] = Counter()
        with temp.open("w", encoding="utf-8", newline="\n") as writer:
            for row in _rows(normalized):
                score, decision, reasons = classify(row["title"], row["text"])
                row.update(historical_relevance_score=score, historical_filter_decision=decision,
                           historical_filter_reasons=reasons)
                counts[decision] += 1
                _write(writer, row)
        return {"count": sum(counts.values()), "decisions": dict(counts)}

    s20 = stage(root, "20_filtered_documents", fingerprint, resume, make_filtered)
    deduped = root / "intermediate/30_deduplicated_documents.jsonl"

    def make_dedup(temp: Path) -> dict[str, Any]:
        db_path = root / "intermediate/30_seen.sqlite3.partial"
        if db_path.exists():
            db_path.unlink()  # scratch database, never a completed stage
        connection = sqlite3.connect(db_path)
        connection.execute("CREATE TABLE seen (hash TEXT PRIMARY KEY)")
        kept = duplicates = 0
        try:
            with temp.open("w", encoding="utf-8", newline="\n") as writer:
                for row in _rows(filtered):
                    if row["historical_filter_decision"] == "DROP":
                        continue
                    digest = sha256_text(row["text"])
                    try:
                        connection.execute("INSERT INTO seen VALUES (?)", (digest,))
                    except sqlite3.IntegrityError:
                        duplicates += 1
                        continue
                    _write(writer, row)
                    kept += 1
            connection.commit()
        finally:
            connection.close()
            db_path.unlink(missing_ok=True)
        return {"count": kept, "duplicate_document_count": duplicates}

    s30 = stage(root, "30_deduplicated_documents", fingerprint, resume, make_dedup)
    chunks = root / "intermediate/40_chunks.jsonl"

    def make_chunks(temp: Path) -> dict[str, Any]:
        counter = token_counter
        if counter is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_id"], use_fast=True,
                                                      cache_dir=cfg["cache_dir"])
            counter = lambda value: len(tokenizer.encode(value, add_special_tokens=False))
        db_path = root / "intermediate/40_seen.sqlite3.partial"
        if db_path.exists():
            db_path.unlink()
        connection = sqlite3.connect(db_path)
        connection.execute("CREATE TABLE seen (hash TEXT PRIMARY KEY)")
        count = duplicates = 0
        try:
            with temp.open("w", encoding="utf-8", newline="\n") as writer:
                for doc in _rows(deduped):
                    for index, (section, value, tokens) in enumerate(chunk_text(
                            doc["text"], counter, cfg["chunk_tokens"], cfg["chunk_overlap"])):
                        digest = sha256_text(value)
                        try:
                            connection.execute("INSERT INTO seen VALUES (?)", (digest,))
                        except sqlite3.IntegrityError:
                            duplicates += 1
                            continue
                        row = {"schema_version": 1,
                               "chunk_id": stable_id("chk", doc["document_id"], index, section, digest),
                               "document_id": doc["document_id"], "source_id": doc["source_id"],
                               "source_type": "wikipedia", "title": doc["title"],
                               "section": section, "url": doc["url"], "text": value, "language": "vi",
                               "chunk_index": index, "token_count": tokens, "char_count": len(value),
                               "years": sorted(set(YEAR.findall(value))),
                               "historical_relevance_score": doc["historical_relevance_score"],
                               "metadata": {"raw_record_index": doc["raw_record_index"],
                                            "historical_filter_decision": doc["historical_filter_decision"],
                                            "source_metadata": doc["source_metadata"]}}
                        _write(writer, row)
                        count += 1
            connection.commit()
        finally:
            connection.close()
            db_path.unlink(missing_ok=True)
        return {"count": count, "duplicate_chunk_count": duplicates}

    s40 = stage(root, "40_chunks", fingerprint, resume, make_chunks)
    _copy_atomic(deduped, root / "documents.jsonl")
    _copy_atomic(chunks, root / "chunks.jsonl")
    stats = _stats(root, s10, s20, s30, s40)
    atomic_json(root / "stats.json", stats)
    filter_audit = _filter_audit(filtered)
    atomic_json(root / "filter_audit.json", filter_audit)
    for decision, label in (("KEEP", "kept"), ("REVIEW", "review"), ("DROP", "dropped")):
        sample_path = root / "samples" / (label + ".jsonl")
        sample_path.parent.mkdir(exist_ok=True)
        temp = sample_path.with_name(sample_path.name + ".partial")
        with temp.open("w", encoding="utf-8", newline="\n") as writer:
            for row in filter_audit["examples"][decision]:
                _write(writer, row)
        os.replace(temp, sample_path)
    files = ["config.json", "source_manifest.json", "documents.jsonl", "chunks.jsonl",
             "stats.json", "filter_audit.json", "samples/kept.jsonl",
             "samples/review.jsonl", "samples/dropped.jsonl"]
    hashes = {name: digest_file(root / name) for name in files}
    atomic_json(root / "hashes.json", hashes)
    manifest = {"schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION,
                "git_sha": git_sha(), "built_at_utc": datetime.now(timezone.utc).isoformat(),
                "config_fingerprint": fingerprint, "source": source_meta,
                "document_count": s30["count"], "chunk_count": s40["count"],
                "filter_counts": s20["decisions"],
                "duplicate_document_count": s30["duplicate_document_count"],
                "duplicate_chunk_count": s40["duplicate_chunk_count"],
                "chunk_tokens": cfg["chunk_tokens"], "chunk_overlap": cfg["chunk_overlap"],
                "normalization": cfg["normalization"], "dedup": cfg["dedup"],
                "hashes": hashes, "software_versions": _versions()}
    atomic_json(root / "manifest.json", manifest)
    return manifest


def _normalize_record(writer, index: int, raw: dict[str, Any], config: dict[str, Any],
                      fields: dict[str, str | None]) -> int:
    title = normalize_text(str(raw.get(fields["title"], "") or ""))
    original = str(raw.get(fields["text"], "") or "")
    text = normalize_text(original)
    source_key = str(raw.get(fields["id"], "") or index) if fields["id"] else str(index)
    source_id = stable_id("src", config["dataset_id"], config.get("dataset_config"),
                          config.get("dataset_revision"), config["split"], source_key)
    metadata = {key: value for key, value in raw.items() if key not in set(filter(None, fields.values()))
                and isinstance(value, (str, int, float, bool, type(None)))}
    row = {"schema_version": 1, "document_id": stable_id("doc", source_id),
           "source_id": source_id, "source_type": "wikipedia", "title": title,
           "url": (str(raw.get(fields["url"], "") or "") if fields["url"] else "")
                  or ("https://vi.wikipedia.org/wiki/" + quote(title.replace(" ", "_")) if title else ""),
           "language": "vi", "text": text, "raw_record_index": index,
           "raw_sha256": sha256_text(original), "source_metadata": metadata,
           "historical_relevance_score": None, "historical_filter_decision": None,
           "historical_filter_reasons": []}
    _write(writer, row)
    return 1


def _summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {key: None for key in ("min", "mean", "median", "p90", "p95", "max")}
    import statistics
    ordered = sorted(values)
    return {"min": ordered[0], "mean": round(statistics.fmean(ordered), 2),
            "median": statistics.median(ordered), "p90": ordered[int(.9 * (len(ordered) - 1))],
            "p95": ordered[int(.95 * (len(ordered) - 1))], "max": ordered[-1]}


def _stats(root: Path, *stages: dict[str, Any]) -> dict[str, Any]:
    chars, tokens = [], []
    for row in _rows(root / "chunks.jsonl"):
        chars.append(row["char_count"])
        tokens.append(row["token_count"])
    return {"stage_counts": {s["stage"]: s["count"] for s in stages},
            "chunk_characters": _summary(chars), "chunk_tokens": _summary(tokens),
            "duplicate_document_count": stages[2]["duplicate_document_count"],
            "duplicate_chunk_count": stages[3]["duplicate_chunk_count"]}


def _filter_audit(path: Path) -> dict[str, Any]:
    randomizer = random.Random(1729)
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    years: Counter[str] = Counter()
    samples: dict[str, list[dict[str, Any]]] = {key: [] for key in ("KEEP", "REVIEW", "DROP")}
    for row in _rows(path):
        decision = row["historical_filter_decision"]
        counts[decision] += 1
        reasons.update(row["historical_filter_reasons"])
        if YEAR.search(row["text"]):
            years[decision] += 1
        sample = {"document_id": row["document_id"], "title": row["title"],
                  "decision": decision, "reasons": row["historical_filter_reasons"],
                  "text_excerpt": row["text"][:300]}
        bucket = samples[decision]
        if len(bucket) < 12:
            bucket.append(sample)
        else:
            position = randomizer.randrange(counts[decision])
            if position < 12:
                bucket[position] = sample
    total = sum(counts.values())
    return {"total_documents": total, "counts": dict(counts),
            "rates": {key: round(counts[key] / total, 6) if total else 0
                      for key in ("KEEP", "REVIEW", "DROP")},
            "top_reasons": reasons.most_common(20), "year_coverage": dict(years),
            "examples": samples, "sample_seed": 1729,
            "note": "Deterministic engineering heuristic; score is not calibrated."}


def _versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version
    result: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in ("datasets", "transformers", "tokenizers"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result
