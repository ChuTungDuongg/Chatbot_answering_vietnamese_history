"""Export only human-accepted, validated records as Question schema v1 JSONL."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from evaluation.annotation.corpus import CorpusLookup
from evaluation.annotation.validate_dataset import validate_records, workspace_records
from evaluation.annotation.workspace import Workspace, dumps
from evaluation.annotation.silver_store import SilverStore
from evaluation.schema import Question


def export(workspace: Workspace, lookup: CorpusLookup, output: Path, *,
           tier: str, split: str | None = None) -> int:
    if output.exists():
        raise FileExistsError(output)
    records = workspace_records(workspace, tier)
    if split:
        records = [record for record in records if record.get("benchmark_split") == split]
    if not records:
        raise ValueError("No accepted records for this tier and split")
    result = validate_records(records, lookup, tier=tier)
    if result["errors"]:
        raise RuntimeError("Gold validation failed:\n" + "\n".join(result["errors"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    return len(records)


def silver_record(row: dict, tier: str) -> dict:
    if row["annotation_status"] != "auto_reviewed" or row["confidence"] not in {"high", "medium"}:
        raise ValueError("Default SILVER export requires auto-reviewed high/medium confidence")
    generated, labels = row["generated"], row["labels"]
    if not generated or not labels or labels.get("annotation_origin") != "automatic":
        raise ValueError("SILVER provenance or labels are missing")
    result = {"id": row["candidate_id"], "question": generated["question"],
              "category": generated["category"], "difficulty": generated["difficulty"],
              "question_type": generated.get("question_type") or generated["category"],
              "answerable": labels["answerable"], "in_domain": labels["in_domain"],
              "relevant_chunk_ids": labels["relevant_chunk_ids"],
              "relevant_source_ids": labels["relevant_source_ids"],
              "annotation_origin": "automatic", "annotation_status": "auto_reviewed",
              "confidence": row["confidence"], "notes": "SILVER: automatically adjudicated; not human GOLD"}
    if tier == "silver-deep":
        deep = row.get("deep")
        if row.get("deep_status") != "auto_reviewed" or not deep or not deep["all_supported"]:
            raise ValueError("Deep SILVER record lacks successful independent verification")
        result.update(gold_answer=deep["silver_answer"],
                      required_facts=[item["fact"] for item in deep["candidate_required_facts"]],
                      gold_citation_source_ids=deep["candidate_citation_source_ids"],
                      factual_paragraph_indices=deep.get("factual_paragraph_indices"))
    return Question.model_validate(result).model_dump(mode="json", exclude_none=True)


def export_silver(store: SilverStore, lookup: CorpusLookup, output: Path, *, tier: str) -> int:
    if tier not in {"silver-retrieval", "silver-deep"}:
        raise ValueError("Invalid SILVER tier")
    if output.exists():
        raise FileExistsError(output)
    from evaluation.annotation.silver_validate import validate
    validation = validate(store.rows(), lookup, tier=tier)
    if validation["errors"]:
        raise RuntimeError("SILVER validation failed:\n" + "\n".join(validation["errors"]))
    rows = [row for row in store.rows() if row["annotation_status"] == "auto_reviewed"
            and row["confidence"] in {"high", "medium"}]
    if tier == "silver-deep":
        rows = [row for row in rows if row["deep_status"] == "auto_reviewed"]
    records = [silver_record(row, tier) for row in rows]
    if not records:
        raise ValueError("No verified SILVER records for selected tier")
    seen = set()
    for record in records:
        if record["id"] in seen:
            raise ValueError("Duplicate SILVER question ID")
        seen.add(record["id"])
        chunks, sources = record["relevant_chunk_ids"], record["relevant_source_ids"]
        if record["answerable"] and (not chunks or not sources):
            raise ValueError("Answerable SILVER question lacks supporting evidence")
        if any(not lookup.has_chunk(item) or lookup.source_for_chunk(item) not in sources for item in chunks):
            raise ValueError("SILVER chunk/source evidence is not valid for corpus")
        if any(not lookup.has_source(item) for item in sources):
            raise ValueError("SILVER source is absent from corpus")
        if tier == "silver-deep":
            if record["answerable"] and (not record.get("required_facts") or
                                         not record.get("gold_citation_source_ids")):
                raise ValueError("Deep SILVER answer lacks supported facts/citations")
            if any(item not in sources for item in record.get("gold_citation_source_ids", [])):
                raise ValueError("SILVER citation source is not verified relevant")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    with temporary.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    return len(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--corpus", type=Path, default=Path("artifacts/corpus_v1/chunks.jsonl"))
    parser.add_argument("--tier", choices=("retrieval", "deep", "silver-retrieval", "silver-deep",
                                           "audit-gold"), required=True)
    parser.add_argument("--split", choices=("dev", "test"))
    parser.add_argument("--lookup-workspace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.workspace is None:
        args.workspace = (Path("evaluation/annotation/workspace_silver_v1")
                          if args.tier.startswith("silver-") else
                          Path("evaluation/annotation/workspace_silver_v1/audit_gold")
                          if args.tier == "audit-gold" else Path("evaluation/annotation/workspace"))
    if args.lookup_workspace is None and args.tier == "audit-gold":
        args.lookup_workspace = args.workspace.parent
    lookup = CorpusLookup(args.corpus, (args.lookup_workspace or args.workspace) / "corpus_lookup.sqlite3")
    workspace = (SilverStore(args.workspace) if args.tier.startswith("silver-")
                 else Workspace(args.workspace))
    try:
        if args.tier.startswith("silver-"):
            count = export_silver(workspace, lookup, args.output, tier=args.tier)
            print(f"exported {count} automatic {args.tier} questions to {args.output}")
        else:
            count = export(workspace, lookup, args.output,
                           tier="retrieval" if args.tier == "audit-gold" else args.tier, split=args.split)
            print(f"exported {count} human-reviewed {args.tier} questions to {args.output}")
        return 0
    finally:
        workspace.close()
        lookup.close()


if __name__ == "__main__":
    raise SystemExit(main())
