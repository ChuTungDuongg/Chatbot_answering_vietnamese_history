"""Bounded-memory summaries for UVW pilot and full corpus runs."""

from __future__ import annotations

from collections import Counter
import heapq
import json
from pathlib import Path
import random
from typing import Any, Iterator

from scripts.corpus_v1.shards import shard_file


def rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed JSONL {path}:{number}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"Non-object JSONL {path}:{number}")
            yield row


def _percentile(hist: Counter[int], fraction: float) -> int | None:
    count = sum(hist.values())
    if not count:
        return None
    target = 1 + int((count - 1) * fraction)
    seen = 0
    for value, frequency in sorted(hist.items()):
        seen += frequency
        if seen >= target:
            return value
    return None


def distribution(hist: Counter[int]) -> dict[str, int | float | None]:
    count = sum(hist.values())
    return {"count": count, "min": min(hist) if hist else None,
            "mean": round(sum(value * frequency for value, frequency in hist.items()) / count, 2) if count else None,
            "median": _percentile(hist, .5), "p90": _percentile(hist, .9),
            "p95": _percentile(hist, .95), "max": max(hist) if hist else None}


def summarize(root: Path, manifests: dict[str, list[dict[str, Any]]],
              elapsed_seconds: float, processed_this_run: Counter[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    decision_counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    by_category: dict[str, Counter[str]] = {}
    quality: Counter[str] = Counter()
    source_lengths: Counter[int] = Counter()
    document_lengths: Counter[int] = Counter()
    chunk_lengths: Counter[int] = Counter()
    chunk_tokens: Counter[int] = Counter()
    missing: Counter[str] = Counter()
    year_coverage: Counter[str] = Counter()
    source_rows_by_split: Counter[str] = Counter()
    reservoir: dict[str, list[dict[str, Any]]] = {key: [] for key in ("KEEP", "REVIEW", "DROP")}
    randomizer = random.Random(1729)
    largest_documents: list[tuple[int, str]] = []
    largest_chunks: list[tuple[int, str]] = []
    document_count = chunk_count = 0
    duplicate_documents = duplicate_chunks = 0
    shard_bytes = 0
    for split, parts in manifests.items():
        for part, meta in enumerate(parts):
            source_rows_by_split[split] += meta["source_count"]
            duplicate_documents += meta["duplicate_document_count"]
            duplicate_chunks += meta["duplicate_chunk_count"]
            shard_bytes += sum(meta["size_bytes"].values())
            for row in rows(shard_file(root, split, part, "records")):
                decision = row["decision"]
                decision_counts[decision] += 1
                reasons.update(row["reasons"])
                category = row.get("main_category") or "<missing>"
                categories[category] += 1
                by_category.setdefault(category, Counter())[decision] += 1
                quality[str(row.get("quality_score")) if row.get("quality_score") is not None else "<missing>"] += 1
                source_lengths[row["char_count"]] += 1
                if not row.get("main_category"):
                    missing["main_category"] += 1
                if not row.get("wikidata_id"):
                    missing["wikidata_id"] += 1
                if not row.get("title"):
                    missing["title"] += 1
                if row.get("has_year_or_period"):
                    year_coverage["source_rows"] += 1
                bucket = reservoir[decision]
                sample = {key: row[key] for key in
                          ("source_id", "source_split", "title", "main_category", "quality_score",
                           "decision", "reasons", "text_excerpt")}
                if len(bucket) < 12:
                    bucket.append(sample)
                else:
                    position = randomizer.randrange(decision_counts[decision])
                    if position < 12:
                        bucket[position] = sample
            for row in rows(shard_file(root, split, part, "documents")):
                document_count += 1
                size = len(row["text"])
                document_lengths[size] += 1
                if not row.get("url"):
                    missing["document_url"] += 1
                item = (size, str(row["document_id"]))
                if len(largest_documents) < 10:
                    heapq.heappush(largest_documents, item)
                elif item > largest_documents[0]:
                    heapq.heapreplace(largest_documents, item)
            for row in rows(shard_file(root, split, part, "chunks")):
                chunk_count += 1
                chunk_lengths[row["char_count"]] += 1
                chunk_tokens[row["token_count"]] += 1
                if row.get("years"):
                    year_coverage["chunks"] += 1
                item = (row["char_count"], str(row["chunk_id"]))
                if len(largest_chunks) < 10:
                    heapq.heappush(largest_chunks, item)
                elif item > largest_chunks[0]:
                    heapq.heapreplace(largest_chunks, item)
    total = sum(decision_counts.values())
    top_categories = categories.most_common(30)
    filter_audit = {"total_documents": total,
                    "counts": {key: decision_counts[key] for key in reservoir},
                    "rates": {key: round(decision_counts[key] / total, 6) if total else 0 for key in reservoir},
                    "top_reasons": reasons.most_common(30),
                    "top_main_categories": top_categories,
                    "decision_by_main_category": {category: dict(by_category[category]) for category, _ in top_categories},
                    "missing_main_category_rate": round(missing["main_category"] / total, 6) if total else 0,
                    "wikidata_id_coverage": round(1 - missing["wikidata_id"] / total, 6) if total else 0,
                    "quality_score_distribution": dict(sorted(quality.items())),
                    "year_or_period_source_rows": year_coverage["source_rows"],
                    "examples": reservoir, "sample_seed": 1729,
                    "note": "Deterministic heuristic; score is not a calibrated probability. First-N pilots may be order-biased."}
    stats = {"source_rows_by_split": dict(source_rows_by_split),
             "document_count": document_count, "chunk_count": chunk_count,
             "duplicate_document_count": duplicate_documents, "duplicate_chunk_count": duplicate_chunks,
             "source_characters": distribution(source_lengths),
             "document_characters": distribution(document_lengths),
             "chunk_characters": distribution(chunk_lengths), "chunk_tokens": distribution(chunk_tokens),
             "year_or_period_chunk_count": year_coverage["chunks"],
             "missing_fields": dict(missing),
             "largest_documents": [{"document_id": key, "char_count": size}
                                   for size, key in sorted(largest_documents, reverse=True)],
             "largest_chunks": [{"chunk_id": key, "char_count": size}
                                for size, key in sorted(largest_chunks, reverse=True)],
             "shard_bytes": shard_bytes,
             "final_document_bytes": (root / "documents.jsonl").stat().st_size,
             "final_chunk_bytes": (root / "chunks.jsonl").stat().st_size,
             "approx_generated_bytes": shard_bytes + (root / "documents.jsonl").stat().st_size
                                     + (root / "chunks.jsonl").stat().st_size,
             "elapsed_seconds": round(elapsed_seconds, 2),
             "processed_this_run": dict(processed_this_run),
             "source_rows_per_second": round(processed_this_run["source_count"] / elapsed_seconds, 2) if elapsed_seconds else None,
             "documents_per_second": round(processed_this_run["document_count"] / elapsed_seconds, 2) if elapsed_seconds else None,
             "chunks_per_second": round(processed_this_run["chunk_count"] / elapsed_seconds, 2) if elapsed_seconds else None}
    return stats, filter_audit
