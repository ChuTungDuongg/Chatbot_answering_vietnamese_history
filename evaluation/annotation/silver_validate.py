"""Validate automatic SILVER labels and corpus references without silently repairing them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.annotation.corpus import CorpusLookup
from evaluation.annotation.policy import CATEGORIES, duplicate_pairs
from evaluation.annotation.silver_store import SilverStore


def validate(rows: list[dict[str, Any]], lookup: Any, *, tier: str = "silver-retrieval") -> dict[str, list[str]]:
    errors, warnings = [], []
    seen = set()
    for row in rows:
        identifier = row["candidate_id"]
        if identifier in seen:
            errors.append(f"{identifier}: duplicate candidate ID")
        seen.add(identifier)
        generated, labels = row.get("generated"), row.get("labels")
        if row["annotation_status"] != "auto_reviewed" or row["confidence"] not in {"high", "medium"}:
            continue
        if not generated or not labels or not str(generated.get("question") or "").strip():
            errors.append(f"{identifier}: missing question or automatic labels")
            continue
        if generated.get("category") not in CATEGORIES:
            errors.append(f"{identifier}: invalid category")
        if generated.get("difficulty") not in {"easy", "medium", "hard"}:
            errors.append(f"{identifier}: invalid difficulty")
        if labels.get("annotation_origin") != "automatic" or labels.get("annotation_status") != "auto_reviewed":
            errors.append(f"{identifier}: SILVER provenance or state mismatch")
        if type(labels.get("answerable")) is not bool or type(labels.get("in_domain")) is not bool:
            errors.append(f"{identifier}: answerability/domain label missing")
        if labels.get("answerable") and not labels.get("in_domain"):
            errors.append(f"{identifier}: out-of-domain question marked answerable")
        chunks = labels.get("relevant_chunk_ids") or []
        sources = labels.get("relevant_source_ids") or []
        if len(chunks) != len(set(chunks)) or len(sources) != len(set(sources)):
            errors.append(f"{identifier}: duplicate evidence ID")
        if labels.get("answerable") and (not chunks or not sources):
            errors.append(f"{identifier}: answerable question lacks direct evidence")
        for chunk in chunks:
            if not lookup.has_chunk(chunk):
                errors.append(f"{identifier}: unknown chunk {chunk}")
            elif lookup.source_for_chunk(chunk) not in sources:
                errors.append(f"{identifier}: chunk source absent from source labels")
        for source in sources:
            if not lookup.has_source(source):
                errors.append(f"{identifier}: unknown source {source}")
        if not labels.get("semantic_pass_agreement"):
            errors.append(f"{identifier}: semantic checks disagree")
        if labels.get("contradiction", {}).get("contradiction_detected"):
            errors.append(f"{identifier}: unresolved contradiction")
        if tier == "silver-deep" and row.get("deep_status") == "auto_reviewed":
            deep = row.get("deep") or {}
            if not deep.get("all_supported") or not deep.get("answer_evidence_coverage") or not deep.get("atomic_facts"):
                errors.append(f"{identifier}: deep answer evidence is insufficient")
            facts = deep.get("candidate_required_facts") or []
            citations = deep.get("candidate_citation_source_ids") or []
            if labels.get("answerable") and (not facts or not citations):
                errors.append(f"{identifier}: answerable deep item lacks facts/citations")
            for fact in facts:
                supporting = fact.get("supporting_chunk_ids") or []
                if not fact.get("fact") or not supporting or any(item not in chunks for item in supporting):
                    errors.append(f"{identifier}: required fact lacks verified chunk support")
                if set(fact.get("supporting_source_ids") or []) != {lookup.source_for_chunk(item) for item in supporting}:
                    errors.append(f"{identifier}: required fact source support disagrees")
            if any(source not in sources for source in citations):
                errors.append(f"{identifier}: citation source is not relevant")
            if not all(item.get("supported") for item in deep.get("fact_support") or []):
                errors.append(f"{identifier}: required fact verification failed")
            if not all(item.get("supported") and item.get("supporting_fact_indices")
                       for item in deep.get("citation_support") or []):
                errors.append(f"{identifier}: citation verification failed")
    selected = [row for row in rows if row["annotation_status"] == "auto_reviewed"
                and row["confidence"] in {"high", "medium"} and row.get("generated")]
    exact, near = duplicate_pairs([{"candidate_id": row["candidate_id"],
                                    "question": row["generated"]["question"]} for row in selected])
    errors += [f"duplicate question: {left} / {right}" for left, right in exact]
    warnings += [f"near duplicate question: {left} / {right}" for left, right in near]
    return {"errors": errors, "warnings": warnings}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("evaluation/annotation/workspace_silver_v1"))
    parser.add_argument("--corpus", type=Path, default=Path("artifacts/corpus_v1/chunks.jsonl"))
    parser.add_argument("--tier", choices=("silver-retrieval", "silver-deep"), default="silver-retrieval")
    args = parser.parse_args(argv)
    silver = SilverStore(args.workspace)
    lookup = CorpusLookup(args.corpus, args.workspace / "corpus_lookup.sqlite3")
    try:
        result = validate(silver.rows(), lookup, tier=args.tier)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result["errors"] else 0
    finally:
        silver.close()
        lookup.close()


if __name__ == "__main__":
    raise SystemExit(main())
