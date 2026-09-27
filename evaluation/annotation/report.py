"""Coverage and concentration report for the local annotation workspace."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.annotation.policy import duplicate_pairs
from evaluation.annotation.workspace import Workspace


def coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row["review_status"] == "accepted"]
    exact, near = duplicate_pairs(rows)
    fields = {"chunk_gold": lambda row: bool(any(v == "relevant" for v in row.get("chunk_decisions", {}).values())),
              "source_gold": lambda row: bool(any(v == "relevant" for v in row.get("source_decisions", {}).values())),
              "gold_answer": lambda row: bool(row.get("gold_answer")),
              "required_facts": lambda row: bool(row.get("required_facts")),
              "citation_labels": lambda row: row.get("gold_citation_source_ids") is not None}
    return {"total_candidates": len(rows),
            "review_status": dict(Counter(row["review_status"] for row in rows)),
            "deep_status": dict(Counter(row["deep_status"] for row in rows)),
            "category_distribution": dict(Counter(row["category"] for row in rows)),
            "accepted_category_distribution": dict(Counter(row["category"] for row in accepted)),
            "difficulty_distribution": dict(Counter(str(row.get("difficulty") or "unset") for row in rows)),
            "answerable_distribution": dict(Counter(str(row.get("answerable")) for row in accepted)),
            "in_domain_distribution": dict(Counter(str(row.get("in_domain")) for row in accepted)),
            "accepted_label_coverage": {name: sum(bool(predicate(row)) for row in accepted)
                                        for name, predicate in fields.items()},
            "evidence_complete": sum(bool(row["evidence_complete"]) for row in rows),
            "duplicate_warnings": {"exact": exact, "near": near},
            "top_source_concentration": Counter(row.get("origin_source_id") for row in rows
                                                if row.get("origin_source_id")).most_common(10),
            "top_title_concentration": Counter(row.get("origin_title") for row in rows
                                               if row.get("origin_title")).most_common(10)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("evaluation/annotation/workspace"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    workspace = Workspace(args.workspace)
    try:
        result = coverage(workspace.all_candidates())
    finally:
        workspace.close()
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
