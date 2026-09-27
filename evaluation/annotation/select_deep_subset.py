"""Select a balanced deep-QA review queue from accepted retrieval annotations."""

from __future__ import annotations

import argparse
import hashlib
import os
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.annotation.workspace import Workspace, dumps


def retrieval_behavior(candidate: dict[str, Any]) -> str:
    if candidate.get("answerable") is False or candidate.get("in_domain") is False:
        return "negative_edge"
    gold = {chunk for chunk, decision in candidate.get("chunk_decisions", {}).items()
            if decision == "relevant"}
    evidence = candidate.get("evidence") or []
    by_stage = {stage: [item["chunk_id"] for item in sorted(evidence, key=lambda row: row["rank"])
                        if item["stage"] == stage][:10]
                for stage in ("dense", "bm25", "reranked")}
    final = bool(gold.intersection(by_stage["reranked"][:5]))
    dense = bool(gold.intersection(by_stage["dense"]))
    bm25 = bool(gold.intersection(by_stage["bm25"]))
    if not final:
        return "final_miss"
    if dense and not bm25:
        return "dense_only_hit"
    if bm25 and not dense:
        return "bm25_only_hit"
    return "shared_hit"


def choose(rows: list[dict[str, Any]], limit: int, *, seed: int = 2026) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row["review_status"] == "accepted" and
                row["retrieval_reviewed"] and row["deep_status"] == "not_selected"]
    selected: list[dict[str, Any]] = []
    counts = {name: Counter() for name in ("category", "difficulty", "behavior", "question_type")}
    remaining = eligible[:]
    while remaining and len(selected) < limit:
        def priority(row: dict[str, Any]):
            category = row["category"]
            difficulty = row.get("difficulty") or "unset"
            behavior = retrieval_behavior(row)
            question_type = row.get("question_type") or "unset"
            balance = (3 / (1 + counts["category"][category]) +
                       2 / (1 + counts["difficulty"][difficulty]) +
                       2 / (1 + counts["behavior"][behavior]) +
                       1 / (1 + counts["question_type"][question_type]))
            tie = hashlib.sha256(f"{seed}:{row['candidate_id']}".encode()).hexdigest()
            return balance, tie
        choice = max(remaining, key=priority)
        remaining.remove(choice)
        selected.append(choice)
        counts["category"][choice["category"]] += 1
        counts["difficulty"][choice.get("difficulty") or "unset"] += 1
        counts["behavior"][retrieval_behavior(choice)] += 1
        counts["question_type"][choice.get("question_type") or "unset"] += 1
    return selected


def select(workspace: Workspace, output: Path, *, limit: int, seed: int = 2026) -> int:
    if output.exists():
        raise FileExistsError(output)
    rows = [workspace.get(item["candidate_id"]) for item in workspace.all_candidates()]
    chosen = choose(rows, limit, seed=seed)
    if len(chosen) < limit:
        raise ValueError(f"Only {len(chosen)} accepted retrieval records remain for a {limit}-question deep queue")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("x", encoding="utf-8") as handle:
        for row in chosen:
            handle.write(dumps({"candidate_id": row["candidate_id"], "question": row["question"],
                                "category": row["category"], "difficulty": row.get("difficulty"),
                                "retrieval_behavior": retrieval_behavior(row),
                                "deep_status": "pending"}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    workspace.select_deep([row["candidate_id"] for row in chosen])
    os.replace(temporary, output)
    return len(chosen)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("evaluation/annotation/workspace"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/annotation/workspace/deep_queue.jsonl"))
    parser.add_argument("--limit", type=int, default=125)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive")
    workspace = Workspace(args.workspace)
    try:
        count = select(workspace, args.output, limit=args.limit, seed=args.seed)
        print(f"selected {count} questions for human deep-QA review: {args.output}")
        return 0
    finally:
        workspace.close()


if __name__ == "__main__":
    raise SystemExit(main())
