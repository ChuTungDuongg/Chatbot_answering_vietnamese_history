from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from typing import Any, Iterable

from app.agents.evidence.prompts import EVIDENCE_AGENT_SYSTEM
from app.agents.evidence.schemas import EvidenceAgentRequest, EvidenceModelOutput
from training.common.sft import (
    IGNORE_INDEX,
    AssistantOnlyTokenStats,
    build_assistant_only_example_with_stats,
)


# These are deterministic character ceilings tried before chat rendering. The
# final acceptance check always uses the real tokenizer and max_length.
EVIDENCE_TEXT_LIMITS = (2400, 1600, 1200, 900, 700, 500, 320, 200, 120, 60, 1)


@dataclass(frozen=True)
class EvidenceSFTStats:
    raw_prompt_tokens: int
    prompt_tokens: int
    assistant_tokens: int
    sequence_tokens: int
    supervised_tokens: int
    overlength: bool
    structured_truncation: bool
    evidence_chars_before: int
    evidence_chars_after: int


def _compact(value: str) -> str:
    return " ".join(value.split())


def _claim_preserving_excerpt(text: str, claims: Iterable[str], limit: int) -> str:
    """Cap one evidence body while retaining every grounded target claim."""
    compact = _compact(text)
    required = [_compact(claim) for claim in claims if _compact(claim)]
    if len(compact) <= limit and all(claim.casefold() in compact.casefold() for claim in required):
        return compact
    if not required:
        return compact[: max(limit, 1)].strip() or "…"

    # Claims are already extractively checked by the dataset validator. Build
    # deterministic source windows around them, then fall back to the grounded
    # claim string if whitespace normalization prevents an exact lookup.
    folded = compact.casefold()
    context = max((limit - sum(len(claim) for claim in required)) // max(2 * len(required), 1), 0)
    excerpts: list[str] = []
    for claim in required:
        position = folded.find(claim.casefold())
        if position < 0:
            excerpts.append(claim)
            continue
        start = max(position - context, 0)
        end = min(position + len(claim) + context, len(compact))
        excerpts.append(compact[start:end].strip())
    result = " … ".join(dict.fromkeys(item for item in excerpts if item))
    return result or "…"


def _capped_messages(row: dict[str, Any], *, text_limit: int) -> list[dict[str, str]]:
    request = EvidenceAgentRequest.model_validate(row.get("input"))
    output = EvidenceModelOutput.model_validate(row.get("output"))
    claims_by_id = {
        selected.evidence_id: selected.claims
        for selected in output.selected_evidence
    }
    evidence = []
    for candidate in request.evidence:
        payload = candidate.model_dump()
        claims = claims_by_id.get(candidate.evidence_id, [])
        payload["text"] = _claim_preserving_excerpt(candidate.text, claims, text_limit)
        evidence.append(payload)
    request_payload = {
        "question": request.question,
        "max_selected": request.max_selected,
        "evidence": evidence,
    }
    return [
        {"role": "system", "content": EVIDENCE_AGENT_SYSTEM},
        {"role": "user", "content": json.dumps(request_payload, ensure_ascii=False, sort_keys=True)},
        {"role": "assistant", "content": json.dumps(output.model_dump(), ensure_ascii=False, sort_keys=True)},
    ]


def build_evidence_assistant_only_example(
    tokenizer,
    row: dict[str, Any],
    *,
    max_length: int,
) -> tuple[dict[str, list[int]], EvidenceSFTStats, list[dict[str, str]]]:
    """Build a valid-JSON Evidence feature before generic token truncation."""
    raw_messages = row.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError(f"row {row.get('id', '<unknown>')} has no messages")
    raw_feature, raw_stats = build_assistant_only_example_with_stats(
        tokenizer, raw_messages, max_length=max_length
    )
    before_chars = sum(
        len(str(item.get("text") or ""))
        for item in (row.get("input") or {}).get("evidence", [])
    )

    if not raw_stats.prompt_truncated:
        supervised = sum(label != IGNORE_INDEX for label in raw_feature["labels"])
        return raw_feature, EvidenceSFTStats(
            raw_prompt_tokens=raw_stats.prompt_tokens,
            prompt_tokens=raw_stats.prompt_tokens,
            assistant_tokens=raw_stats.assistant_tokens,
            sequence_tokens=raw_stats.sequence_tokens,
            supervised_tokens=supervised,
            overlength=raw_stats.truncated,
            structured_truncation=False,
            evidence_chars_before=before_chars,
            evidence_chars_after=before_chars,
        ), raw_messages

    last_stats: AssistantOnlyTokenStats | None = None
    for text_limit in EVIDENCE_TEXT_LIMITS:
        messages = _capped_messages(row, text_limit=text_limit)
        feature, token_stats = build_assistant_only_example_with_stats(
            tokenizer, messages, max_length=max_length
        )
        last_stats = token_stats
        if token_stats.prompt_truncated:
            continue
        raw_target = [label for label in raw_feature["labels"] if label != IGNORE_INDEX]
        capped_target = [label for label in feature["labels"] if label != IGNORE_INDEX]
        if capped_target != raw_target:
            raise ValueError("structured evidence capping changed the assistant target token sequence")
        supervised = sum(label != IGNORE_INDEX for label in feature["labels"])
        if supervised == 0:
            raise ValueError("Evidence SFT invariant failed: sample has zero supervised tokens")
        after_request = json.loads(messages[1]["content"])
        after_chars = sum(len(item["text"]) for item in after_request["evidence"])
        return feature, EvidenceSFTStats(
            raw_prompt_tokens=raw_stats.prompt_tokens,
            prompt_tokens=token_stats.prompt_tokens,
            assistant_tokens=token_stats.assistant_tokens,
            sequence_tokens=token_stats.sequence_tokens,
            supervised_tokens=supervised,
            overlength=True,
            structured_truncation=True,
            evidence_chars_before=before_chars,
            evidence_chars_after=after_chars,
        ), messages

    final_prompt = last_stats.prompt_tokens if last_stats else "unknown"
    raise ValueError(
        f"row {row.get('id', '<unknown>')} cannot fit valid structured evidence and the complete assistant target "
        f"within max_length={max_length} (smallest prompt={final_prompt})"
    )


def _distribution(values: list[int]) -> dict[str, float | int | None]:
    return {
        "min": min(values) if values else None,
        "mean": statistics.mean(values) if values else None,
        "max": max(values) if values else None,
    }


def prepare_evidence_split(
    tokenizer,
    rows: list[dict[str, Any]],
    *,
    max_length: int,
    split_name: str,
) -> tuple[list[dict[str, list[int]]], dict[str, Any]]:
    """Tokenize one split and fail preflight on every unsafe sample."""
    if not rows:
        raise ValueError(f"tokenization preflight split {split_name!r} is empty")
    features: list[dict[str, list[int]]] = []
    stats: list[EvidenceSFTStats] = []
    errors: list[str] = []
    for row in rows:
        try:
            feature, item_stats, _ = build_evidence_assistant_only_example(
                tokenizer, row, max_length=max_length
            )
            if not feature["input_ids"]:
                raise ValueError("tokenization produced empty input_ids")
            if item_stats.supervised_tokens == 0:
                raise ValueError("tokenization produced zero supervised tokens")
            if item_stats.assistant_tokens == 0:
                raise ValueError("assistant target was completely truncated")
            features.append(feature)
            stats.append(item_stats)
        except (TypeError, ValueError, KeyError) as exc:
            errors.append(f"{row.get('id', '<unknown>')}: {exc}")
    if errors:
        raise ValueError(f"tokenization preflight failed for {split_name}: {errors[:5]}")

    report = {
        "rows": len(rows),
        "max_sequence_length": max(item.sequence_tokens for item in stats),
        "truncated_rows": sum(item.structured_truncation for item in stats),
        "overlength_rows": sum(item.overlength for item in stats),
        "zero_supervised_rows": sum(item.supervised_tokens == 0 for item in stats),
        "supervised_tokens": _distribution([item.supervised_tokens for item in stats]),
        "prompt_tokens": _distribution([item.prompt_tokens for item in stats]),
        "raw_prompt_tokens": _distribution([item.raw_prompt_tokens for item in stats]),
        "assistant_tokens": _distribution([item.assistant_tokens for item in stats]),
        "evidence_chars_before": _distribution([item.evidence_chars_before for item in stats]),
        "evidence_chars_after": _distribution([item.evidence_chars_after for item in stats]),
    }
    if report["zero_supervised_rows"]:
        raise ValueError(f"tokenization preflight failed for {split_name}: zero-supervised rows found")
    return features, report
