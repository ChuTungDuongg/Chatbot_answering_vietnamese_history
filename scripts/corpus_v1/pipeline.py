"""Sharded, resumable Wikipedia corpus builder independent of FastAPI."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import chain, islice
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time
from typing import Any, Callable, Iterable

from scripts.corpus_v1 import PIPELINE_VERSION, SCHEMA_VERSION
from scripts.corpus_v1.chunking import CHUNKER_VERSION, chunk_text
from scripts.corpus_v1.history_filter import FILTER_VERSION, YEAR, classify
from scripts.corpus_v1.normalize import normalize_text
from scripts.corpus_v1.progress import ProgressReporter
from scripts.corpus_v1.provenance import atomic_json, canonical, digest_file, git_sha
from scripts.corpus_v1.schema import sha256_text, stable_id
from scripts.corpus_v1.shards import KINDS, commit_shard, completed_shards, shard_file
from scripts.corpus_v1.source import (SHA_PATTERN, article_url, detect_fields, load_source,
                                      resolve_source, selected_splits)
from scripts.corpus_v1.stats import rows, summarize


V0_ROOTS = ("vn_history_deployment", "vn_history_modal")


def _safe_output(root: Path) -> Path:
    root = root.resolve()
    names = {part.casefold() for part in root.parts}
    if names.intersection(V0_ROOTS):
        raise ValueError("Corpus V1 output cannot be inside a protected V0 artifact root")
    if "corpus_v1" not in names:
        raise ValueError("Output path must contain a corpus_v1 directory")
    return root


def _request(config: dict[str, Any]) -> dict[str, Any]:
    limit = config.get("max_records_per_split")
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError("max_records_per_split must be a positive integer")
    shard_size = config.get("shard_size", 10000)
    if not isinstance(shard_size, int) or shard_size < 1:
        raise ValueError("shard_size must be a positive integer")
    chunk_tokens = config.get("chunk_tokens", 384)
    chunk_overlap = config.get("chunk_overlap", 48)
    if not 16 <= chunk_tokens or not 0 <= chunk_overlap < chunk_tokens:
        raise ValueError("Require chunk_tokens >= 16 and 0 <= overlap < chunk_tokens")
    return {"dataset_id": config["dataset_id"], "dataset_config": config.get("dataset_config"),
            "requested_revision": config.get("requested_revision") or config.get("dataset_revision") or "main",
            "split": config.get("split", "train"), "streaming": config.get("streaming", True),
            "cache_dir": config.get("cache_dir"), "source_kind": config.get("source_kind", "wikipedia"),
            "language": config.get("language", "vi"), "fields": config.get("fields", {}),
            "metadata_fields": config.get("metadata_fields", []),
            "chunk_tokens": chunk_tokens, "chunk_overlap": chunk_overlap,
            "tokenizer_id": config.get("tokenizer_id", "intfloat/multilingual-e5-base"),
            "tokenizer_requested_revision": config.get("tokenizer_requested_revision", "main"),
            "max_records_per_split": limit, "shard_size": shard_size,
            "offline_fixture_dir": config.get("offline_fixture_dir")}


def _tokenizer_sha(request: dict[str, Any], injected_counter: bool) -> str | None:
    if injected_counter:
        return None
    from huggingface_hub import HfApi
    try:
        info = HfApi().model_info(request["tokenizer_id"], revision=request["tokenizer_requested_revision"])
    except Exception as exc:
        raise RuntimeError(f"Cannot pin tokenizer {request['tokenizer_id']}: {exc}") from exc
    sha = str(info.sha or "").lower()
    if not SHA_PATTERN.fullmatch(sha):
        raise RuntimeError("Tokenizer revision did not resolve to a commit SHA")
    return sha


def _new_configuration(request: dict[str, Any], source_info: dict[str, Any],
                       injected_counter: bool) -> dict[str, Any]:
    sha = str(source_info.get("resolved_revision_sha") or "").lower()
    if not SHA_PATTERN.fullmatch(sha):
        raise RuntimeError("Build requires an immutable Hugging Face dataset commit SHA")
    splits = selected_splits(request, source_info)
    names = source_info.get("features") or list(filter(None, request["fields"].values()))
    fields = detect_fields(names, request["fields"])
    return {"schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION,
            "request_options": request, "dataset_id": request["dataset_id"],
            "dataset_config": request["dataset_config"],
            "requested_revision": request["requested_revision"], "resolved_revision_sha": sha,
            "source_splits": splits, "expected_split_sizes": source_info.get("expected_split_sizes", {}),
            "field_mapping": fields, "source_kind": request["source_kind"],
            "language": request["language"], "metadata_fields": request["metadata_fields"],
            "streaming": request["streaming"], "cache_dir": request["cache_dir"],
            "chunk_tokens": request["chunk_tokens"], "chunk_overlap": request["chunk_overlap"],
            "tokenizer_id": request["tokenizer_id"],
            "tokenizer_revision_sha": _tokenizer_sha(request, injected_counter),
            "filter_version": FILTER_VERSION, "chunker_version": CHUNKER_VERSION,
            "normalization_version": "unicode_nfc_whitespace_v1",
            "dedup_version": "global_exact_sha256_v2", "include_review": True,
            "sample_seed": 1729, "build_scope": "pilot" if request["max_records_per_split"] else "full",
            "record_limit_per_split": request["max_records_per_split"],
            "shard_size": request["shard_size"]}


def _source_manifest(info: dict[str, Any], cfg: dict[str, Any], fingerprint: str,
                     observed: dict[str, int], complete: bool,
                     created_at: str) -> dict[str, Any]:
    return {**info, "schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION,
            "git_sha": git_sha(), "config_fingerprint": fingerprint,
            "source_splits": cfg["source_splits"], "field_mapping": cfg["field_mapping"],
            "filter_version": cfg["filter_version"], "chunker_version": cfg["chunker_version"],
            "normalization_version": cfg["normalization_version"], "dedup_version": cfg["dedup_version"],
            "build_scope": cfg["build_scope"], "record_limit_per_split": cfg["record_limit_per_split"],
            "observed_split_sizes": observed, "source_complete": complete,
            "created_at_utc": created_at}


def _source_iterator(dataset: Any, offset: int):
    if offset and callable(getattr(dataset, "skip", None)):
        return iter(dataset.skip(offset))
    return islice(iter(dataset), offset, None)


def _seed_seen(connection: sqlite3.Connection, root: Path,
               shards: dict[str, list[dict[str, Any]]]) -> None:
    for split, parts in shards.items():
        for part in range(len(parts)):
            for row in rows(shard_file(root, split, part, "records")):
                try:
                    connection.execute("INSERT INTO source_ids VALUES (?)", (row["source_id"],))
                except sqlite3.IntegrityError as exc:
                    raise RuntimeError("Duplicate source ID in completed shards") from exc
            for kind, table in (("documents", "document_texts"), ("chunks", "chunk_texts")):
                for row in rows(shard_file(root, split, part, kind)):
                    try:
                        connection.execute(f"INSERT INTO {table} VALUES (?)", (sha256_text(row["text"]),))
                    except sqlite3.IntegrityError as exc:
                        raise RuntimeError(f"Duplicate {kind} text in completed shards") from exc
        connection.commit()


def _normalize_record(raw: dict[str, Any], cfg: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    fields = cfg["field_mapping"]
    title = normalize_text(str(raw.get(fields["title"], "") or ""))
    original = str(raw.get(fields["text"], "") or "")
    article_id = str(raw.get(fields["id"], "") or index) if fields["id"] else str(index)
    source_id = stable_id("src", cfg["dataset_id"], cfg["resolved_revision_sha"], split, article_id)
    metadata = {key: raw.get(key) for key in cfg["metadata_fields"] if key in raw}
    for key, value in raw.items():
        if (key not in set(filter(None, fields.values())) and key not in metadata
                and isinstance(value, (str, int, float, bool, type(None)))):
            metadata[key] = value
    return {"schema_version": SCHEMA_VERSION, "document_id": stable_id("doc", source_id),
            "source_id": source_id, "source_type": "wikipedia", "source_dataset_id": cfg["dataset_id"],
            "source_article_id": article_id, "source_split": split,
            "source_revision_sha": cfg["resolved_revision_sha"], "title": title,
            "url": article_url(raw, fields, cfg["source_kind"]), "language": cfg["language"],
            "text": normalize_text(original), "raw_record_index": index,
            "raw_sha256": sha256_text(original), "source_metadata": metadata,
            "historical_relevance_score": None, "historical_filter_decision": None,
            "historical_filter_reasons": []}


def _write(writer, row: dict[str, Any]) -> None:
    writer.write(canonical(row) + "\n")


def _process_shard(raw_rows: Iterable[dict[str, Any]], cfg: dict[str, Any], split: str,
                   start: int, connection: sqlite3.Connection,
                   counter: Callable[[str], int], scratch: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    files = {kind: scratch / (kind + ".jsonl") for kind in KINDS}
    counts: Counter[str] = Counter()
    with files["records"].open("w", encoding="utf-8", newline="\n") as records, \
         files["documents"].open("w", encoding="utf-8", newline="\n") as documents, \
         files["chunks"].open("w", encoding="utf-8", newline="\n") as chunks:
        for offset, raw in enumerate(raw_rows):
            if not isinstance(raw, dict):
                raise RuntimeError(f"Non-object source row {split}:{start + offset}")
            doc = _normalize_record(raw, cfg, split, start + offset)
            try:
                connection.execute("INSERT INTO source_ids VALUES (?)", (doc["source_id"],))
            except sqlite3.IntegrityError as exc:
                raise RuntimeError(f"Duplicate source article ID at {split}:{start + offset}") from exc
            category = doc["source_metadata"].get("main_category")
            score, decision, reasons = classify(doc["title"], doc["text"], category)
            doc.update(historical_relevance_score=score, historical_filter_decision=decision,
                       historical_filter_reasons=reasons)
            counts["source_count"] += 1
            counts[decision] += 1
            record = {"source_id": doc["source_id"], "source_split": split, "title": doc["title"],
                      "main_category": category, "quality_score": doc["source_metadata"].get("quality_score"),
                      "wikidata_id": doc["source_metadata"].get("wikidata_id"),
                      "decision": decision, "reasons": reasons, "char_count": len(doc["text"]),
                      "has_year_or_period": bool(YEAR.search(doc["text"])),
                      "text_excerpt": doc["text"][:300]}
            _write(records, record)
            if decision == "DROP":
                continue
            try:
                connection.execute("INSERT INTO document_texts VALUES (?)", (sha256_text(doc["text"]),))
            except sqlite3.IntegrityError:
                counts["duplicate_document_count"] += 1
                continue
            _write(documents, doc)
            counts["document_count"] += 1
            for chunk_index, (section, value, tokens) in enumerate(chunk_text(
                    doc["text"], counter, cfg["chunk_tokens"], cfg["chunk_overlap"])):
                if tokens > cfg["chunk_tokens"]:
                    raise RuntimeError("Chunker emitted a chunk above token budget")
                digest = sha256_text(value)
                try:
                    connection.execute("INSERT INTO chunk_texts VALUES (?)", (digest,))
                except sqlite3.IntegrityError:
                    counts["duplicate_chunk_count"] += 1
                    continue
                chunk = {"schema_version": SCHEMA_VERSION,
                         "chunk_id": stable_id("chk", doc["document_id"], chunk_index, section, digest),
                         "document_id": doc["document_id"], "source_id": doc["source_id"],
                         "source_type": "wikipedia", "source_dataset_id": doc["source_dataset_id"],
                         "source_article_id": doc["source_article_id"], "source_split": split,
                         "source_revision_sha": doc["source_revision_sha"],
                         "title": doc["title"], "section": section, "url": doc["url"],
                         "text": value, "language": doc["language"], "chunk_index": chunk_index,
                         "token_count": tokens, "char_count": len(value),
                         "years": sorted(set(YEAR.findall(value))),
                         "historical_relevance_score": score,
                         "metadata": {"raw_record_index": doc["raw_record_index"],
                                      "historical_filter_decision": decision,
                                      "source_metadata": doc["source_metadata"]}}
                _write(chunks, chunk)
                counts["chunk_count"] += 1
    return files, {"source_count": counts["source_count"], "document_count": counts["document_count"],
                   "chunk_count": counts["chunk_count"], "filter_counts":
                   {key: counts[key] for key in ("KEEP", "REVIEW", "DROP")},
                   "duplicate_document_count": counts["duplicate_document_count"],
                   "duplicate_chunk_count": counts["duplicate_chunk_count"]}


def _aggregate(root: Path, shards: dict[str, list[dict[str, Any]]], kind: str) -> None:
    destination = root / f"{kind}.jsonl"
    temporary = destination.with_name(destination.name + ".partial")
    with temporary.open("wb") as writer:
        for split, parts in shards.items():
            for part in range(len(parts)):
                with shard_file(root, split, part, kind).open("rb") as reader:
                    shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
    if destination.exists():
        if digest_file(destination) != digest_file(temporary):
            raise RuntimeError(f"Existing final output differs: {destination}")
        temporary.unlink()
    else:
        os.replace(temporary, destination)


def _versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version
    import sys
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for name in ("datasets", "huggingface_hub", "transformers", "tokenizers"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def build(config: dict[str, Any], output: Path, *, resume: bool = False,
          source_factory: Callable[[str], Any] | None = None,
          token_counter: Callable[[str], int] | None = None,
          source_info: dict[str, Any] | None = None,
          progress: ProgressReporter | None = None) -> dict[str, Any]:
    reporter = progress if progress is not None else ProgressReporter()
    try:
        return _build(config, output, resume=resume, source_factory=source_factory,
                      token_counter=token_counter, source_info=source_info, reporter=reporter)
    except Exception as exc:
        reporter.error(output, exc)
        raise


def _build(config: dict[str, Any], output: Path, *, resume: bool,
           source_factory: Callable[[str], Any] | None,
           token_counter: Callable[[str], int] | None,
           source_info: dict[str, Any] | None,
           reporter: ProgressReporter) -> dict[str, Any]:
    root = _safe_output(output)
    request = _request(config)
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError(f"Output exists: {root}; use --resume or another output")
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    source_path = root / "source_manifest.json"
    if config_path.exists():
        reporter.phase("loading saved build configuration")
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        if not resume or cfg.get("request_options") != request or cfg.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("Build configuration differs; use a new output directory")
        if source_path.exists():
            source_meta = json.loads(source_path.read_text(encoding="utf-8"))
            info = {key: value for key, value in source_meta.items() if key not in
                    ("observed_split_sizes", "source_complete", "created_at_utc", "config_fingerprint")}
        else:
            if (root / "intermediate" / "shards").exists():
                raise RuntimeError("Missing source manifest alongside shard checkpoints")
            # A disconnect between config and source-manifest writes is recoverable.
            info = source_info or resolve_source({**request, "requested_revision": cfg["resolved_revision_sha"]})
            if info.get("resolved_revision_sha") != cfg["resolved_revision_sha"]:
                raise RuntimeError("Recovered source revision differs from saved configuration")
            info["requested_revision"] = cfg["requested_revision"]
            source_meta = {}
    else:
        if source_path.exists():
            raise RuntimeError("Source manifest exists without config.json")
        reporter.phase(f"resolving source dataset={request['dataset_id']} requested_revision={request['requested_revision']}")
        info = source_info or resolve_source(request)
        reporter.phase("resolving tokenizer revision and creating build configuration")
        cfg = _new_configuration(request, info, token_counter is not None)
        atomic_json(config_path, cfg)
        source_meta = {}
    fingerprint = hashlib.sha256(canonical(cfg).encode("utf-8")).hexdigest()
    created_at = source_meta.get("created_at_utc") or datetime.now(timezone.utc).isoformat()
    scratch_parent = Path(config.get("scratch_dir") or tempfile.gettempdir()).expanduser().resolve()
    reporter.startup(cfg, root, scratch_parent, resume=resume)
    if resume:
        reporter.resume_validating()
    shards = {}
    for split in cfg["source_splits"]:
        shards[split] = completed_shards(root, split, fingerprint, resume=resume,
                                         on_orphan_cleared=reporter.orphan_cleared if resume else None)
        if resume:
            reporter.resume_split(split, len(shards[split]),
                                  sum(meta["source_count"] for meta in shards[split]))
    observed = {split: sum(meta["source_count"] for meta in parts) for split, parts in shards.items()}
    if resume:
        reporter.overall(observed)
    if source_meta.get("config_fingerprint") not in (None, fingerprint):
        raise RuntimeError("Source manifest configuration mismatch")
    source_meta = _source_manifest(info, cfg, fingerprint, observed, False, created_at)
    final_path = root / "manifest.json"
    if final_path.exists():
        final = json.loads(final_path.read_text(encoding="utf-8"))
        if final.get("config_fingerprint") != fingerprint:
            raise RuntimeError("Final manifest configuration mismatch")
        reporter.phase("validating finalized corpus hashes")
        if all((root / name).is_file() and digest_file(root / name) == digest
               for name, digest in final.get("hashes", {}).items()):
            reporter.done(final, root, already_complete=True)
            return final
        raise RuntimeError("Final corpus output hash mismatch")
    atomic_json(source_path, source_meta)

    started = time.monotonic()
    processed_this_run: Counter[str] = Counter()
    scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="corpus-v1-", dir=scratch_parent) as scratch_name:
        scratch = Path(scratch_name)
        connection = sqlite3.connect(scratch / "seen.sqlite3")
        try:
            for table in ("source_ids", "document_texts", "chunk_texts"):
                connection.execute(f"CREATE TABLE {table} (hash TEXT PRIMARY KEY)")
            if resume:
                reporter.dedup(restored=False)
            _seed_seen(connection, root, shards)
            if resume:
                reporter.dedup(restored=True)
            counter = token_counter
            if counter is None:
                reporter.phase("loading pinned tokenizer")
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    cfg["tokenizer_id"], revision=cfg["tokenizer_revision_sha"],
                    cache_dir=cfg["cache_dir"], use_fast=True)
                counter = lambda value: len(tokenizer.encode(value, add_special_tokens=False))
            for split in cfg["source_splits"]:
                expected = cfg["expected_split_sizes"].get(split)
                limit = cfg["record_limit_per_split"]
                reporter.split_start(split, observed[split], expected, resume=resume)
                if limit is not None and observed[split] >= limit:
                    reporter.split_done(split, observed[split], expected, limit, exhausted=False)
                    continue
                if limit is None and expected is not None and observed[split] >= expected:
                    reporter.split_done(split, observed[split], expected, limit, exhausted=False)
                    continue
                if resume and observed[split]:
                    reporter.line(f"[resume] continuing {split} at row={observed[split]}")
                dataset = source_factory(split) if source_factory else load_source(cfg, split)
                dataset_fingerprint = getattr(dataset, "_fingerprint", None)
                if isinstance(dataset_fingerprint, str) and dataset_fingerprint:
                    info.setdefault("hf_fingerprints_by_split", {})[split] = dataset_fingerprint
                iterator = _source_iterator(dataset, observed[split])
                exhausted = False
                while True:
                    if limit is not None and observed[split] >= limit:
                        break
                    first = next(iterator, None)
                    if first is None:
                        exhausted = True
                        break
                    size = cfg["shard_size"] if limit is None else min(cfg["shard_size"], limit - observed[split])
                    block = chain((first,), islice(iterator, size - 1))
                    part = len(shards[split])
                    shard_started = reporter.shard_start(split, part + 1, observed[split], size)
                    files, counts = _process_shard(block, cfg, split, observed[split], connection,
                                                   counter, scratch)
                    if counts["source_count"] == 0:
                        raise RuntimeError("Empty shard after a nonempty source row")
                    meta = commit_shard(root, split, part, fingerprint, files, counts, observed[split])
                    connection.commit()
                    shards[split].append(meta)
                    observed[split] += counts["source_count"]
                    for key in ("source_count", "document_count", "chunk_count"):
                        processed_this_run[key] += counts[key]
                    atomic_json(source_path, _source_manifest(info, cfg, fingerprint, observed, False, created_at))
                    reporter.shard_done(split, part + 1, counts, shard_started, observed)
                    if counts["source_count"] < size:
                        exhausted = True
                        break
                reporter.split_done(split, observed[split], expected, limit, exhausted=exhausted)
            if cfg["build_scope"] == "full":
                for split in cfg["source_splits"]:
                    expected = cfg["expected_split_sizes"].get(split)
                    if expected is not None and observed[split] != expected:
                        raise RuntimeError(f"Incomplete source split {split}: observed {observed[split]}, expected {expected}")
        finally:
            connection.close()
    complete = cfg["build_scope"] == "full"
    source_meta = _source_manifest(info, cfg, fingerprint, observed, complete, created_at)
    reporter.finalize("writing source_manifest.json")
    atomic_json(source_path, source_meta)
    reporter.finalize("writing source_manifest.json", completed=True)
    reporter.finalize("aggregating documents")
    _aggregate(root, shards, "documents")
    reporter.finalize("aggregating documents", completed=True)
    reporter.finalize("aggregating chunks")
    _aggregate(root, shards, "chunks")
    reporter.finalize("aggregating chunks", completed=True)
    reporter.finalize("generating stats and filter audit")
    stats, filter_audit = summarize(root, shards, time.monotonic() - started, processed_this_run)
    atomic_json(root / "stats.json", stats)
    atomic_json(root / "filter_audit.json", filter_audit)
    reporter.finalize("generating stats and filter audit", completed=True)
    reporter.finalize("generating samples")
    sample_dir = root / "samples"
    sample_dir.mkdir(exist_ok=True)
    for decision, label in (("KEEP", "kept"), ("REVIEW", "review"), ("DROP", "dropped")):
        path = sample_dir / f"{label}.jsonl"
        temporary = path.with_name(path.name + ".partial")
        with temporary.open("w", encoding="utf-8", newline="\n") as writer:
            for row in filter_audit["examples"][decision]:
                _write(writer, row)
        os.replace(temporary, path)
    reporter.finalize("generating samples", completed=True)
    files = ["config.json", "source_manifest.json", "documents.jsonl", "chunks.jsonl",
             "stats.json", "filter_audit.json", "samples/kept.jsonl",
             "samples/review.jsonl", "samples/dropped.jsonl"]
    reporter.finalize("calculating hashes")
    hashes = {name: digest_file(root / name) for name in files}
    atomic_json(root / "hashes.json", hashes)
    reporter.finalize("calculating hashes", completed=True)
    manifest = {"schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION,
                "git_sha": git_sha(), "built_at_utc": datetime.now(timezone.utc).isoformat(),
                "config_fingerprint": fingerprint, "dataset_id": cfg["dataset_id"],
                "dataset_config": cfg["dataset_config"],
                "requested_revision": cfg["requested_revision"],
                "resolved_revision_sha": cfg["resolved_revision_sha"],
                "source_splits": cfg["source_splits"],
                "expected_split_sizes": cfg["expected_split_sizes"],
                "observed_split_sizes": observed, "source_complete": complete,
                "build_scope": cfg["build_scope"],
                "record_limit_per_split": cfg["record_limit_per_split"],
                "license": info.get("license"), "dataset_card_url": info.get("dataset_card_url"),
                "field_mapping": cfg["field_mapping"], "filter_version": cfg["filter_version"],
                "chunker_version": cfg["chunker_version"],
                "normalization_version": cfg["normalization_version"],
                "dedup_version": cfg["dedup_version"], "shard_size": cfg["shard_size"],
                "document_count": stats["document_count"], "chunk_count": stats["chunk_count"],
                "filter_counts": filter_audit["counts"],
                "duplicate_document_count": stats["duplicate_document_count"],
                "duplicate_chunk_count": stats["duplicate_chunk_count"],
                "chunk_tokens": cfg["chunk_tokens"], "chunk_overlap": cfg["chunk_overlap"],
                "tokenizer_id": cfg["tokenizer_id"],
                "tokenizer_revision_sha": cfg["tokenizer_revision_sha"],
                "source": source_meta, "hashes": hashes, "software_versions": _versions()}
    reporter.finalize("writing manifest.json")
    atomic_json(final_path, manifest)
    reporter.finalize("writing manifest.json", completed=True)
    reporter.done(manifest, root)
    return manifest
