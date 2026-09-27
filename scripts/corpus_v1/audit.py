"""Read-only diagnostics for a completed V1 directory or chunks JSONL."""

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.corpus_v1.history_filter import YEAR
from scripts.corpus_v1.stats import distribution
from scripts.corpus_v1.provenance import digest_file


def audit(corpus: Path) -> dict[str, Any]:
    chunks_path = corpus / "chunks.jsonl" if corpus.is_dir() else corpus
    root = chunks_path.parent
    docs_path = root / "documents.jsonl"
    counters: Counter[str] = Counter()
    sources: dict[str, Counter[str]] = {"document": Counter(), "chunk": Counter()}
    decisions: dict[str, Counter[str]] = {"document": Counter(), "chunk": Counter()}
    ids: dict[str, set[str]] = {"document": set(), "chunk": set()}
    texts: dict[str, set[str]] = {"document": set(), "chunk": set()}
    char_lengths: dict[str, Counter[int]] = {"document": Counter(), "chunk": Counter()}
    token_lengths: dict[str, Counter[int]] = {"document": Counter(), "chunk": Counter()}
    splits: dict[str, Counter[str]] = {"document": Counter(), "chunk": Counter()}
    quality: Counter[str] = Counter()
    largest: dict[str, list[dict[str, Any]]] = {"document": [], "chunk": []}
    malformed: list[str] = []

    for kind, path in (("document", docs_path), ("chunk", chunks_path)):
        if not path.exists():
            malformed.append(f"missing file: {path}")
            continue
        with path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("row is not an object")
                except (json.JSONDecodeError, ValueError) as exc:
                    counters["malformed_rows"] += 1
                    if len(malformed) < 30:
                        malformed.append(f"{path}:{line_no}: {exc}")
                    continue
                counters[f"{kind}_count"] += 1
                identifier = row.get(f"{kind}_id")
                if identifier is not None and not isinstance(identifier, (str, int, float)):
                    identifier = None
                if not identifier:
                    counters[f"missing_{kind}_ids"] += 1
                elif identifier in ids[kind]:
                    counters[f"duplicate_{kind}_ids"] += 1
                else:
                    ids[kind].add(identifier)
                value = row.get("text")
                if not isinstance(value, str) or not value.strip():
                    counters[f"missing_{kind}_text"] += 1
                    value = ""
                digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
                if value and digest in texts[kind]:
                    counters[f"exact_duplicate_{kind}_text"] += 1
                texts[kind].add(digest)
                if len(value.strip()) < 40:
                    counters[f"near_empty_{kind}s"] += 1
                if not row.get("title"):
                    counters[f"missing_{kind}_titles"] += 1
                if not row.get("url"):
                    counters[f"missing_{kind}_urls"] += 1
                if not row.get("source_id"):
                    counters[f"missing_{kind}_source_ids"] += 1
                if not row.get("document_id"):
                    counters[f"missing_{kind}_document_ids"] += 1
                if YEAR.search(value):
                    counters[f"{kind}_with_year_or_period"] += 1
                sources[kind][str(row.get("source_type") or "missing")] += 1
                splits[kind][str(row.get("source_split") or "missing")] += 1
                metadata = row.get("metadata")
                decision = row.get("historical_filter_decision") or (
                    metadata.get("historical_filter_decision") if isinstance(metadata, dict) else None)
                decisions[kind][str(decision or "missing")] += 1
                char_lengths[kind][len(value)] += 1
                try:
                    token_lengths[kind][int(row.get("token_count") or len(value.split()))] += 1
                except (ValueError, TypeError):
                    counters[f"invalid_{kind}_token_counts"] += 1
                    token_lengths[kind][len(value.split())] += 1
                if kind == "document":
                    source_metadata = row.get("source_metadata") or {}
                    if isinstance(source_metadata, dict):
                        value_quality = source_metadata.get("quality_score")
                        quality[str(value_quality) if value_quality is not None else "<missing>"] += 1
                largest[kind].append({"id": identifier, "characters": len(value)})
                largest[kind].sort(key=lambda item: item["characters"], reverse=True)
                del largest[kind][10:]
    hashes = {}
    hash_path = root / "hashes.json"
    if hash_path.exists():
        expected = json.loads(hash_path.read_text(encoding="utf-8"))
        hashes = {name: {"expected": digest, "actual": digest_file(root / name)
                          if (root / name).is_file() else None}
                  for name, digest in expected.items()}
    required = ["malformed_rows"]
    for kind in ("document", "chunk"):
        required += [f"{kind}_count", f"missing_{kind}_ids", f"duplicate_{kind}_ids",
                     f"missing_{kind}_text", f"exact_duplicate_{kind}_text",
                     f"near_empty_{kind}s", f"missing_{kind}_titles",
                     f"missing_{kind}_urls", f"missing_{kind}_source_ids",
                     f"missing_{kind}_document_ids", f"{kind}_with_year_or_period",
                     f"invalid_{kind}_token_counts"]
    return {"corpus": str(chunks_path),
            "counts": {key: counters[key] for key in required},
            "unique_document_ids": len(ids["document"]), "unique_chunk_ids": len(ids["chunk"]),
            "source_distribution": {kind: dict(value) for kind, value in sources.items()},
            "source_split_distribution": {kind: dict(value) for kind, value in splits.items()},
            "quality_score_distribution": dict(sorted(quality.items())),
            "filter_decision_distribution": {kind: dict(value) for kind, value in decisions.items()},
            "characters": {kind: distribution(values) for kind, values in char_lengths.items()},
            "estimated_tokens": {kind: distribution(values) for kind, values in token_lengths.items()},
            "largest": largest, "malformed_examples": malformed,
            "hash_mismatches": [name for name, pair in hashes.items() if pair["expected"] != pair["actual"]]}
