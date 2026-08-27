from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app.agents.prompts import EVIDENCE_AGENT_SYSTEM
from app.agents.schemas import EvidenceAgentRequest, EvidenceModelOutput
from app.agents.evidence_validation import (
    compressed_derived_from_own_claims,
    grounded_in_source,
    referenced_evidence_ids,
)
from training.common.datasets import split_rows
from training.common.jsonl import read_jsonl
from training.evidence_agent.coverage import (
    GENERIC_MISSING_PREFIX,
    accentfold,
    assess_answer_coverage,
)
from training.evidence_agent.conflicts import (
    MODEL_VISIBLE_RESERVED_MARKERS,
    conflict_metadata_is_question_relevant,
    conflict_values_incompatible,
)
from training.evidence_agent.prepare_dataset import GENERIC_SUMMARIES


REQUIRED_V2_BEHAVIORS = {
    "duplicate",
    "conflict",
    "partial",
    "insufficient",
    "irrelevant_disagreement",
}


def _normalized(text: str) -> str:
    return " ".join(str(text).casefold().split())


def validate_rows(rows: list[dict[str, Any]], *, require_v2_behaviors: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    row_ids: set[str] = set()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    status_distribution: Counter[str] = Counter()
    behavior_distribution: Counter[str] = Counter()
    source_distribution: Counter[str] = Counter()
    summaries: Counter[str] = Counter()
    original_lengths: list[int] = []
    compressed_lengths: list[int] = []
    empty_compressed = 0
    copy_through = 0
    selected_count = 0
    sufficient_rows = 0
    partial_rows = 0
    conflict_rows = 0
    question_relevant_conflicts = 0
    irrelevant_conflict_rows = 0
    conflict_without_two_valid_ids = 0
    conflict_without_incompatible_values = 0
    conflict_without_slot_value_details = 0
    model_visible_synthetic_markers = 0
    model_visible_synthetic_source_types = 0
    model_visible_reserved_titles = 0
    model_visible_serialized_metadata_fields = 0
    model_visible_conflict_id_suffixes = 0
    model_visible_duplicate_id_suffixes = 0
    cross_id_claim_attribution = 0
    cross_id_compressed_attribution = 0
    conflict_type_distribution: Counter[str] = Counter()
    hard_negative_compatible_pairs = 0
    hard_negative_complementary_pairs = 0
    hard_negative_irrelevant_disagreement = 0
    partial_with_full_gold_answer_coverage = 0
    sufficient_with_missing_information = 0
    partial_with_empty_missing_information = 0
    partial_without_useful_coverage = 0
    sufficient_with_suspicious_coverage = 0
    duplicate_rows_selecting_all_duplicates = 0
    generic_missing_information_count = 0
    relevance_by_class: dict[str, list[float]] = defaultdict(list)

    for index, row in enumerate(rows, 1):
        label = f"row {index}"
        row_id = str(row.get("id") or "")
        group_id = str(row.get("group_id") or "")
        behavior = str(row.get("behavior") or "")
        if not row_id:
            errors.append(f"{label}: missing id")
        elif row_id in row_ids:
            errors.append(f"{label}: duplicate id {row_id!r}")
        if row_id:
            row_ids.add(row_id)
        if not group_id:
            errors.append(f"{label}: missing group_id")
        else:
            groups[group_id].append(row)
        if not behavior:
            errors.append(f"{label}: missing behavior")

        messages = row.get("messages")
        if not isinstance(messages, list) or [item.get("role") for item in messages if isinstance(item, dict)] != ["system", "user", "assistant"]:
            errors.append(f"{label}: messages must be system/user/assistant")
            continue
        if messages[0].get("content") != EVIDENCE_AGENT_SYSTEM:
            errors.append(f"{label}: system prompt differs from EVIDENCE_AGENT_SYSTEM")
        try:
            visible_user_content = str(messages[1].get("content") or "")
            request = EvidenceAgentRequest.model_validate(json.loads(visible_user_content))
        except Exception as exc:
            errors.append(f"{label}: invalid EvidenceAgentRequest: {exc}")
            continue
        try:
            assistant_payload = json.loads(messages[2].get("content") or "")
            output = EvidenceModelOutput.model_validate(assistant_payload)
        except Exception as exc:
            errors.append(f"{label}: invalid runtime EvidenceModelOutput: {exc}")
            continue
        if row.get("input") != request.model_dump():
            errors.append(f"{label}: top-level input differs from training user message")
        if row.get("output") != output.model_dump():
            errors.append(f"{label}: top-level output differs from training assistant message")
        if row.get("question") != request.question:
            errors.append(f"{label}: question differs from canonical request")

        visible_folded = visible_user_content.casefold()
        visible_markers = [marker for marker in MODEL_VISIBLE_RESERVED_MARKERS if marker in visible_folded]
        if visible_markers:
            model_visible_synthetic_markers += 1
            errors.append(f"{label}: model-visible synthetic/class markers are forbidden: {visible_markers}")
        synthetic_sources = sum("synthetic" in item.source_type.casefold() for item in request.evidence)
        reserved_titles = sum(
            any(marker in str(item.title or "").casefold() for marker in MODEL_VISIBLE_RESERVED_MARKERS)
            for item in request.evidence
        )
        serialized_metadata = sum(
            marker in visible_folded
            for marker in ("augmentation_type", "parent_evidence_id", '"behavior"', '"synthetic"')
        )
        model_visible_synthetic_source_types += synthetic_sources
        model_visible_reserved_titles += reserved_titles
        model_visible_serialized_metadata_fields += serialized_metadata
        if synthetic_sources:
            errors.append(f"{label}: synthetic source_type is visible to the model")
        if reserved_titles:
            errors.append(f"{label}: reserved augmentation marker appears in an evidence title")
        if serialized_metadata:
            errors.append(f"{label}: augmentation metadata/class field was serialized into the prompt")
        conflict_suffixes = sum("__conflict" in item.evidence_id.casefold() for item in request.evidence)
        duplicate_suffixes = sum("__dup_" in item.evidence_id.casefold() for item in request.evidence)
        model_visible_conflict_id_suffixes += conflict_suffixes
        model_visible_duplicate_id_suffixes += duplicate_suffixes
        if conflict_suffixes:
            errors.append(f"{label}: model-visible conflict ID suffix is forbidden")
        if duplicate_suffixes:
            errors.append(f"{label}: model-visible duplicate ID suffix is forbidden")

        candidates = {item.evidence_id: item for item in request.evidence}
        selected_ids = [item.evidence_id for item in output.selected_evidence]
        invented = sorted(set(selected_ids) - set(candidates))
        if invented:
            errors.append(f"{label}: selected evidence IDs do not exist in input: {invented}")
        legacy_selected = row.get("output", {}).get("selected_ids")
        legacy_rejected = row.get("output", {}).get("rejected_ids")
        if legacy_selected is not None or legacy_rejected is not None:
            if set(legacy_selected or []) & set(legacy_rejected or []):
                errors.append(f"{label}: selected_ids and rejected_ids overlap")
            errors.append(f"{label}: derived selected/rejected ID fields must not be model targets")

        for item in output.selected_evidence:
            source = candidates.get(item.evidence_id)
            if source is None:
                continue
            selected_count += 1
            source_text = _normalized(source.text)
            compressed = _normalized(item.compressed_text)
            empty_compressed += int(not compressed)
            original_lengths.append(len(source.text))
            compressed_lengths.append(len(item.compressed_text))
            copy_through += int(compressed == source_text)
            for claim in item.claims:
                if not grounded_in_source(claim, source.text):
                    cross_id_claim_attribution += 1
                    errors.append(
                        f"{label}: claim for {item.evidence_id!r} is not grounded in that same evidence source"
                    )
            if not compressed_derived_from_own_claims(item, source.text):
                cross_id_compressed_attribution += 1
                errors.append(
                    f"{label}: compressed_text for {item.evidence_id!r} is not derivable from its own claims/source"
                )

        if output.summary in GENERIC_SUMMARIES:
            errors.append(f"{label}: generic template summary is forbidden")
        generic_missing = [
            item
            for item in output.missing_information
            if accentfold(item).startswith(GENERIC_MISSING_PREFIX)
        ]
        generic_missing_information_count += len(generic_missing)
        if generic_missing:
            errors.append(f"{label}: generic missing_information is forbidden")

        gold_answer = str((row.get("metadata") or {}).get("gold_answer") or "").strip()
        if gold_answer:
            input_coverage = assess_answer_coverage(
                request.question, gold_answer, [item.text for item in request.evidence]
            )
            selected_coverage = assess_answer_coverage(
                request.question,
                gold_answer,
                [candidates[evidence_id].text for evidence_id in selected_ids if evidence_id in candidates],
            )
            if output.status == "sufficient" and not selected_coverage.full:
                sufficient_with_suspicious_coverage += 1
                message = f"{label}: sufficient selected evidence does not cover the gold answer components"
                if selected_coverage.confident:
                    errors.append(message)
                else:
                    warnings.append(message + " (heuristic)")
            if behavior == "partial":
                if input_coverage.full:
                    partial_with_full_gold_answer_coverage += 1
                    errors.append(f"{label}: partial input still has full gold-answer coverage")
                if not selected_coverage.useful:
                    partial_without_useful_coverage += 1
                    errors.append(f"{label}: partial selection supports no required answer component")
        metadata = row.get("metadata") or {}
        if output.status == "conflicting":
            for conflict in output.conflicts:
                mentioned = referenced_evidence_ids(conflict, candidates)
                if len(mentioned) < 2:
                    conflict_without_two_valid_ids += 1
                    errors.append(f"{label}: conflict must name at least two existing evidence IDs")
        if behavior == "conflict" and output.status != "conflicting":
            errors.append(f"{label}: conflict behavior requires conflicting status")
        if behavior == "conflict" and output.status == "conflicting":
            conflict_type = str(metadata.get("conflict_type") or "other")
            conflict_type_distribution[conflict_type] += 1
            relevant = conflict_metadata_is_question_relevant(
                question=request.question,
                gold_answer=gold_answer,
                metadata=metadata,
            )
            question_relevant_conflicts += int(relevant)
            irrelevant_conflict_rows += int(not relevant)
            if not relevant:
                errors.append(f"{label}: conflict does not affect a question-relevant answer slot")
            parent_id = str(metadata.get("parent_evidence_id") or "")
            mutated_id = str(metadata.get("mutated_evidence_id") or "")
            original_value = str(metadata.get("original_value") or "")
            mutated_value = str(metadata.get("mutated_value") or "")
            valid_pair = parent_id in candidates and mutated_id in candidates and parent_id != mutated_id
            if not valid_pair:
                conflict_without_two_valid_ids += 1
                errors.append(f"{label}: conflict metadata does not identify two supplied evidence IDs")
            values_grounded = bool(
                valid_pair
                and grounded_in_source(original_value, candidates[parent_id].text)
                and grounded_in_source(mutated_value, candidates[mutated_id].text)
            )
            incompatible = values_grounded and conflict_values_incompatible(
                original_value, mutated_value, conflict_type
            )
            if not incompatible:
                conflict_without_incompatible_values += 1
                errors.append(f"{label}: conflict values are missing, ungrounded, or compatible")
            conflict_details = " ".join(output.conflicts)
            slot_label = str(metadata.get("slot_label") or "")
            detailed = all(
                grounded_in_source(value, conflict_details)
                for value in (parent_id, mutated_id, slot_label, original_value, mutated_value)
            )
            if not detailed:
                conflict_without_slot_value_details += 1
                errors.append(f"{label}: conflict description must name both IDs, the slot, and both values")
        if behavior == "duplicate":
            duplicate_ids = set(metadata.get("augmented_evidence_ids") or [])
            parent_id = str(metadata.get("parent_evidence_id") or "")
            if not duplicate_ids or not duplicate_ids <= set(candidates):
                errors.append(f"{label}: duplicate metadata must name generated evidence IDs")
            if duplicate_ids & set(selected_ids) or parent_id not in selected_ids:
                errors.append(f"{label}: duplicate behavior must reject generated duplicate IDs")
            if duplicate_ids and duplicate_ids <= set(selected_ids):
                duplicate_rows_selecting_all_duplicates += 1
            hard_negative_compatible_pairs += int(bool(duplicate_ids))
        if behavior == "irrelevant_disagreement":
            hard_negative_irrelevant_disagreement += 1
            if output.status == "conflicting":
                errors.append(f"{label}: irrelevant disagreement must not be labelled conflicting")
        if output.status == "sufficient" and len(output.selected_evidence) >= 2:
            hard_negative_complementary_pairs += 1
        if behavior == "partial" and not (
            output.status == "insufficient" and output.selected_evidence and output.missing_information
        ):
            errors.append(f"{label}: partial behavior requires useful selected evidence plus missing information")
        if output.status == "sufficient":
            sufficient_rows += 1
            sufficient_with_missing_information += int(bool(output.missing_information))
            if output.missing_information:
                errors.append(f"{label}: sufficient output must not contain missing_information")
        if behavior == "partial":
            partial_rows += 1
            partial_with_empty_missing_information += int(not output.missing_information)
        if output.status == "conflicting":
            conflict_rows += 1
        relevance_class = "partial" if behavior == "partial" else output.status
        relevance_by_class[relevance_class].extend(item.relevance for item in output.selected_evidence)
        if output.status == "insufficient" and output.selected_evidence and behavior != "partial":
            warnings.append(f"{label}: insufficient output retains evidence outside documented partial behavior")

        status_distribution[output.status] += 1
        behavior_distribution[behavior] += 1
        source_distribution[str(row.get("source_dataset") or "unknown")] += 1
        summaries[output.summary] += 1

    if not rows:
        errors.append("dataset is empty")
    missing_behaviors = REQUIRED_V2_BEHAVIORS - set(behavior_distribution)
    if require_v2_behaviors and missing_behaviors:
        errors.append(f"dataset is missing required v2 behaviors: {sorted(missing_behaviors)}")
    if require_v2_behaviors and len(status_distribution) < 3:
        errors.append("dataset must supervise sufficient, insufficient, and conflicting statuses")
    if rows and max(status_distribution.values(), default=0) / len(rows) > 0.90:
        errors.append("one status exceeds 90% of the dataset")

    simulated_overlap = {"train_eval": 0, "train_test": 0, "eval_test": 0}
    split_group_counts = {"train": 0, "eval": 0, "test": 0}
    split_behavior_distribution: dict[str, dict[str, int]] = {"train": {}, "eval": {}, "test": {}}
    split_missing_behaviors: dict[str, list[str]] = {"train": [], "eval": [], "test": []}
    if rows and all(row.get("group_id") for row in rows):
        splits = split_rows(rows, seed=42, group_key="group_id", stratify_key="behavior")
        split_groups = {
            name: {str(row["group_id"]) for row in getattr(splits, name)}
            for name in ("train", "eval", "test")
        }
        split_group_counts = {name: len(value) for name, value in split_groups.items()}
        simulated_overlap = {
            "train_eval": len(split_groups["train"] & split_groups["eval"]),
            "train_test": len(split_groups["train"] & split_groups["test"]),
            "eval_test": len(split_groups["eval"] & split_groups["test"]),
        }
        if any(simulated_overlap.values()):
            errors.append(f"group leakage across simulated splits: {simulated_overlap}")
        all_behaviors = set(behavior_distribution)
        for name in ("train", "eval", "test"):
            counts = Counter(str(row.get("behavior") or "unknown") for row in getattr(splits, name))
            split_behavior_distribution[name] = dict(counts)
            split_missing_behaviors[name] = sorted(all_behaviors - set(counts))

    mean_original = statistics.mean(original_lengths) if original_lengths else 0.0
    mean_compressed = statistics.mean(compressed_lengths) if compressed_lengths else 0.0
    relevance_status_shortcut_stats = {
        name: {
            "count": len(values),
            "unique_values": len(set(values)),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": statistics.mean(values) if values else None,
        }
        for name, values in sorted(relevance_by_class.items())
    }
    partial_relevance = relevance_by_class.get("partial", [])
    sufficient_relevance = relevance_by_class.get("sufficient", [])
    relevance_shortcut_detected = bool(
        len(partial_relevance) >= 5
        and len(sufficient_relevance) >= 5
        and len(set(partial_relevance)) == 1
        and len(set(sufficient_relevance)) == 1
        and partial_relevance[0] != sufficient_relevance[0]
    )
    if relevance_shortcut_detected:
        errors.append("relevance is a deterministic partial/sufficient class shortcut")
    return {
        "valid": not errors,
        "rows": len(rows),
        "unique_ids": len(row_ids),
        "unique_groups": len(groups),
        "status_distribution": dict(status_distribution),
        "behavior_distribution": dict(behavior_distribution),
        "source_distribution": dict(source_distribution),
        "unique_summaries": len(summaries),
        "generic_summary_frequency": sum(summaries[item] for item in GENERIC_SUMMARIES),
        "generic_summary_ratio": sum(summaries[item] for item in GENERIC_SUMMARIES) / max(len(rows), 1),
        "selected_evidence_items": selected_count,
        "empty_compressed_text": empty_compressed,
        "copy_through_compressed_text": copy_through,
        "mean_original_chars": mean_original,
        "mean_compressed_chars": mean_compressed,
        "compression_ratio": mean_compressed / max(mean_original, 1.0),
        "split_group_counts": split_group_counts,
        "split_group_overlap": simulated_overlap,
        "split_behavior_distribution": split_behavior_distribution,
        "split_missing_behaviors": split_missing_behaviors,
        "grounding_metric": "extractive claim containment (heuristic)",
        "sufficient_rows": sufficient_rows,
        "partial_rows": partial_rows,
        "conflict_rows": conflict_rows,
        "question_relevant_conflicts": question_relevant_conflicts,
        "irrelevant_conflict_rows": irrelevant_conflict_rows,
        "conflict_without_two_valid_ids": conflict_without_two_valid_ids,
        "conflict_without_incompatible_values": conflict_without_incompatible_values,
        "conflict_without_slot_value_details": conflict_without_slot_value_details,
        "model_visible_synthetic_markers": model_visible_synthetic_markers,
        "model_visible_synthetic_source_types": model_visible_synthetic_source_types,
        "model_visible_reserved_titles": model_visible_reserved_titles,
        "model_visible_serialized_metadata_fields": model_visible_serialized_metadata_fields,
        "model_visible_conflict_id_suffixes": model_visible_conflict_id_suffixes,
        "model_visible_duplicate_id_suffixes": model_visible_duplicate_id_suffixes,
        "cross_id_claim_attribution": cross_id_claim_attribution,
        "cross_id_compressed_attribution": cross_id_compressed_attribution,
        "conflict_type_distribution": dict(conflict_type_distribution),
        "hard_negative_compatible_pairs": hard_negative_compatible_pairs,
        "hard_negative_complementary_pairs": hard_negative_complementary_pairs,
        "hard_negative_irrelevant_disagreement": hard_negative_irrelevant_disagreement,
        "partial_with_full_gold_answer_coverage": partial_with_full_gold_answer_coverage,
        "sufficient_with_missing_information": sufficient_with_missing_information,
        "partial_with_empty_missing_information": partial_with_empty_missing_information,
        "partial_without_useful_coverage": partial_without_useful_coverage,
        "sufficient_with_suspicious_coverage": sufficient_with_suspicious_coverage,
        "duplicate_rows_selecting_all_duplicates": duplicate_rows_selecting_all_duplicates,
        "generic_missing_information_count": generic_missing_information_count,
        "relevance_status_shortcut_stats": relevance_status_shortcut_stats,
        "relevance_shortcut_detected": relevance_shortcut_detected,
        "errors": errors,
        "warnings": warnings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Evidence Agent v2.2 JSONL against the production runtime contract.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--allow-partial-fixture", action="store_true", help="Do not require every v2 behavior in tiny fixtures.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_rows(read_jsonl(args.dataset), require_v2_behaviors=not args.allow_partial_fixture)
    except (OSError, ValueError) as exc:
        report = {"valid": False, "errors": [str(exc)], "warnings": []}
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report.get("valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
