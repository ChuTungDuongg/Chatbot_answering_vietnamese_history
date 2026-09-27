"""Validate accepted gold labels against Corpus V1 IDs; never repair silently."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.annotation.corpus import CorpusLookup
from evaluation.annotation.policy import CATEGORIES, REVIEW_POLICY_VERSION, duplicate_pairs
from evaluation.annotation.records import gold_record
from evaluation.annotation.workspace import Workspace
from evaluation.schema import Question


def validate_records(records: list[dict[str, Any]], lookup: CorpusLookup, *,
                     tier: str = "retrieval") -> dict[str, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    seen = set()
    for row in records:
        identifier = str(row.get("id") or "<missing>")
        if identifier in seen:
            errors.append(f"{identifier}: duplicate ID")
        seen.add(identifier)
        try:
            Question.model_validate(row)
        except Exception as exc:
            errors.append(f"{identifier}: incompatible Question schema ({type(exc).__name__})")
            continue
        if row.get("review_status") != "accepted" or row.get("retrieval_reviewed") is not True:
            errors.append(f"{identifier}: retrieval review is not accepted")
        if not str(row.get("question") or "").strip():
            errors.append(f"{identifier}: empty question")
        if row.get("category") not in CATEGORIES:
            errors.append(f"{identifier}: invalid category")
        if row.get("answerable") is None or row.get("in_domain") is None:
            errors.append(f"{identifier}: answerable and in_domain require human labels")
        if row.get("answerable") is True and row.get("in_domain") is False:
            errors.append(f"{identifier}: out-of-domain question cannot be answerable")
        reviewed_at, version = row.get("reviewed_at"), row.get("review_version")
        try:
            if not reviewed_at:
                raise ValueError()
            datetime.fromisoformat(reviewed_at)
        except (TypeError, ValueError):
            errors.append(f"{identifier}: invalid reviewed_at")
        if type(version) is not int or version < 1:
            errors.append(f"{identifier}: invalid review_version")
        if row.get("review_policy_version") != REVIEW_POLICY_VERSION:
            errors.append(f"{identifier}: stale or missing review policy version")
        chunks = row.get("relevant_chunk_ids")
        sources = row.get("relevant_source_ids")
        if chunks is None or sources is None:
            errors.append(f"{identifier}: chunk and source reviews must be explicit")
            chunks, sources = chunks or [], sources or []
        if row.get("answerable") is True and (not chunks or not sources):
            errors.append(f"{identifier}: answerable question needs chunk and source gold")
        if len(chunks) != len(set(chunks)) or len(sources) != len(set(sources)):
            errors.append(f"{identifier}: duplicate relevance ID")
        for chunk_id in chunks:
            if not lookup.has_chunk(chunk_id):
                errors.append(f"{identifier}: unknown chunk ID {chunk_id}")
            elif lookup.source_for_chunk(chunk_id) not in sources:
                errors.append(f"{identifier}: relevant chunk's source is absent from source gold")
        for source_id in sources:
            if not lookup.has_source(source_id):
                errors.append(f"{identifier}: unknown source ID {source_id}")
        if row.get("in_domain") is False and (chunks or sources):
            warnings.append(f"{identifier}: out-of-domain question has relevance labels")
        if tier == "deep":
            if row.get("answer_reviewed") is not True or row.get("citation_reviewed") is not True:
                errors.append(f"{identifier}: deep review flags are incomplete")
            try:
                datetime.fromisoformat(row["deep_reviewed_at"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{identifier}: invalid deep_reviewed_at")
            if not str(row.get("gold_answer") or "").strip():
                errors.append(f"{identifier}: missing reviewed gold answer")
            facts = row.get("required_facts")
            citations = row.get("gold_citation_source_ids")
            if not isinstance(facts, list) or any(not isinstance(x, str) or not x.strip() for x in facts):
                errors.append(f"{identifier}: invalid required facts")
            if not isinstance(citations, list) or any(not isinstance(x, str) or not x for x in citations):
                errors.append(f"{identifier}: invalid citation source IDs")
            if row.get("answerable") is True and (not facts or not citations):
                errors.append(f"{identifier}: answerable deep question needs facts and citations")
            for source_id in citations or []:
                if not lookup.has_source(source_id):
                    errors.append(f"{identifier}: unknown citation source ID {source_id}")
            indices = row.get("factual_paragraph_indices")
            if indices is not None and (not isinstance(indices, list) or
                                        any(type(value) is not int or value < 0 for value in indices)):
                errors.append(f"{identifier}: invalid factual paragraph indices")
            elif isinstance(indices, list) and row.get("gold_answer"):
                paragraph_count = len([part for part in row["gold_answer"].split("\n\n") if part.strip()])
                if any(value >= paragraph_count for value in indices):
                    warnings.append(f"{identifier}: factual paragraph index exceeds answer paragraphs")
    exact, near = duplicate_pairs(records)
    errors.extend(f"duplicate question: {a}, {b}" for a, b in exact)
    warnings.extend(f"near duplicate question: {a}, {b}" for a, b in near)
    return {"errors": errors, "warnings": warnings}


def workspace_records(workspace: Workspace, tier: str) -> list[dict[str, Any]]:
    records = []
    for candidate in workspace.all_candidates():
        if candidate["review_status"] != "accepted":
            continue
        if tier == "deep" and candidate["deep_status"] != "accepted":
            continue
        try:
            records.append(gold_record(candidate, tier))
        except Exception as exc:
            records.append({"id": candidate["candidate_id"], "question": candidate.get("question"),
                            "category": candidate.get("category"), "projection_error": type(exc).__name__})
    return records


def read_records(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"line {line_number}: invalid JSON")
                continue
            if not isinstance(value, dict):
                errors.append(f"line {line_number}: expected a question object")
                continue
            records.append(value)
    return records, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("evaluation/annotation/workspace"))
    parser.add_argument("--input", type=Path, help="Validate an exported JSONL instead of workspace")
    parser.add_argument("--tier", choices=("retrieval", "deep"), default="retrieval")
    parser.add_argument("--corpus", type=Path, default=Path("artifacts/corpus_v1/chunks.jsonl"))
    args = parser.parse_args(argv)
    lookup = CorpusLookup(args.corpus, args.workspace / "corpus_lookup.sqlite3")
    workspace = None
    try:
        if args.input:
            records, input_errors = read_records(args.input)
        else:
            workspace = Workspace(args.workspace)
            records = workspace_records(workspace, args.tier)
            input_errors = []
        result = validate_records(records, lookup, tier=args.tier)
        result["errors"] = input_errors + result["errors"]
        print(json.dumps({"count": len(records), **result}, ensure_ascii=False, indent=2))
        return 1 if result["errors"] else 0
    finally:
        if workspace:
            workspace.close()
        lookup.close()


if __name__ == "__main__":
    raise SystemExit(main())
