"""Binary relevance metrics over ranked chunk and distinct source IDs."""
from __future__ import annotations

import math
from typing import Any


KS = (1, 3, 5, 10)


def _distinct_ids(rows: list[dict[str, Any]], key: str) -> list[str]:
    seen: set[str] = set()
    ranked: list[str] = []
    for row in rows:
        value = row.get(key)
        if value is not None:
            item = str(value)
            if item and item not in seen:
                seen.add(item)
                ranked.append(item)
    return ranked


def _score(ranked: list[str], gold: list[str] | None, k: int) -> dict[str, float | None]:
    names = ("hit_rate", "recall", "precision", "mrr", "ndcg")
    if gold is None:
        return {name: None for name in names}
    relevant = set(gold)
    top = ranked[:k]
    hits = [index for index, item in enumerate(top, 1) if item in relevant]
    if not relevant:
        return {"hit_rate": 0.0, "recall": None, "precision": 0.0,
                "mrr": None, "ndcg": None}
    dcg = sum(1 / math.log2(index + 1) for index in hits)
    ideal = sum(1 / math.log2(index + 1) for index in range(1, min(k, len(relevant)) + 1))
    return {"hit_rate": float(bool(hits)), "recall": len(hits) / len(relevant),
            "precision": len(hits) / k, "mrr": 1 / hits[0] if hits else 0.0,
            "ndcg": dcg / ideal}


def score_retrieval(rows: list[dict[str, Any]], *, relevant_chunk_ids: list[str] | None,
                    relevant_source_ids: list[str] | None) -> dict[str, dict[str, float | None]]:
    """Source-level ranking collapses repeated chunks by first source occurrence."""
    chunks = _distinct_ids(rows, "chunk_id")
    sources = _distinct_ids(rows, "source_id")
    # Retrieved rows without their ID metadata cannot be checked against gold IDs.
    if rows and not chunks:
        relevant_chunk_ids = None
    if rows and not sources:
        relevant_source_ids = None
    return {
        "chunk": {f"{metric}@{k}": value for k in KS
                  for metric, value in _score(chunks, relevant_chunk_ids, k).items()},
        "source": {f"{metric}@{k}": value for k in KS
                   for metric, value in _score(sources, relevant_source_ids, k).items()},
    }
