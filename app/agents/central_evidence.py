from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote, urlsplit

from app.agents.central_question import CentralQuestionAnalysis, _ascii_fold_vietnamese
from app.agents.config import CentralAgentConfig


def normalize_entity(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", _ascii_fold_vietnamese(value)))


def _score(row: dict[str, Any], key: str) -> float | None:
    try:
        value = float(row[key])
        return value if math.isfinite(value) else None
    except (KeyError, TypeError, ValueError):
        return None


def _title_matches(row: dict[str, Any], subject: str) -> bool:
    titles = [str(row.get(key) or "") for key in ("title", "page_title", "source_title")]
    if row.get("url"):
        titles.append(unquote(urlsplit(str(row["url"])).path.rsplit("/", 1)[-1]).replace("_", " "))
    return any(normalize_entity(re.sub(r"\s*\([^)]*\)\s*$", "", title)) == subject for title in titles)


def select_evidence(
    rows: list[dict[str, Any]],
    analysis: CentralQuestionAnalysis,
    config: CentralAgentConfig,
    *,
    compare_scores: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter within a retrieval batch; never compare scores from different queries/tools.

    Raw CrossEncoder scores have no guaranteed probability calibration. The default
    removes only a separated low tail (at least two stronger anchors). An absolute
    floor is opt-in for explicitly calibrated probability scores.
    """
    reasons: Counter[str] = Counter()
    excluded: set[int] = set()
    scores = {i: score for i, row in enumerate(rows) if (score := _score(row, "reranker_score")) is not None}
    ranked = sorted(scores, key=lambda i: scores[i], reverse=True)
    if compare_scores and ranked:
        if config.reranker_score_mode == "probability" and all(0 <= score <= 1 for score in scores.values()):
            floor = config.reranker_score_floor
            if floor is not None and scores[ranked[0]] >= config.reranker_strong_score:
                for i in ranked[1:]:
                    if scores[i] < floor:
                        excluded.add(i)
                        reasons["reranker_probability_floor"] += 1
        if len(ranked) >= 3:
            span = scores[ranked[0]] - scores[ranked[-1]]
            # Two stronger anchors avoid filtering a flat batch or a lone outlier.
            for cut in range(2, len(ranked)):
                gap = scores[ranked[cut - 1]] - scores[ranked[cut]]
                if span > 0 and gap / span >= config.reranker_tail_gap_ratio:
                    for i in ranked[cut:]:
                        if i not in excluded:
                            excluded.add(i)
                            reasons["reranker_separated_low_tail"] += 1
                    break

    subject = normalize_entity(analysis.subject or "") if analysis.question_type == "biography" else ""
    exact = {i for i, row in enumerate(rows) if subject and _title_matches(row, subject)}
    exact_kept = exact - excluded
    associated: set[int] = set(exact)
    if subject:
        page_ids = {
            (key, str(rows[i][key])) for i in exact_kept
            for key in ("url", "page_id", "source_page_id") if rows[i].get(key)
        }
        for i, row in enumerate(rows):
            text = normalize_entity(str(row.get("text") or row.get("content") or row.get("snippet") or ""))
            if f" {subject} " in f" {text} " or any(str(row.get(key) or "") == value for key, value in page_ids):
                associated.add(i)
        if len(exact_kept) >= config.biography_min_exact_hits:
            for i in range(len(rows)):
                if i not in associated and i not in excluded:
                    excluded.add(i)
                    reasons["biography_entity_collision"] += 1

    indices = [i for i in range(len(rows)) if i not in excluded]
    if subject and associated:
        # Stable tie-breaking and no invented probability scale for retrieval scores.
        indices.sort(key=lambda i: (
            i in exact, i in associated,
            _score(rows[i], "reranker_score") if compare_scores and i in scores else float("-inf"),
            (_score(rows[i], "final_retrieval_score") or 0) if compare_scores else 0,
        ), reverse=True)
    if subject and len(indices) > config.biography_max_sources:
        reasons["biography_context_limit"] += len(indices) - config.biography_max_sources
        indices = indices[:config.biography_max_sources]
    selected = [rows[i] for i in indices]
    return selected, {
        "retrieval_candidates_before_filter": len(rows),
        "retrieval_candidates_after_filter": len(selected),
        "retrieval_filtered_count": len(rows) - len(selected),
        "retrieval_filter_reasons": dict(reasons),
        "biography_entity": analysis.subject if subject else None,
        "biography_exact_title_hits": len(exact),
    }


@dataclass(frozen=True)
class SynthesisEvidence:
    alias: str
    real_source_id: str
    title: str
    source_kind: str
    text: str


def build_evidence_packet(sources: list[dict[str, Any]]) -> list[SynthesisEvidence]:
    packet: list[SynthesisEvidence] = []
    seen: set[str] = set()
    for row in sources:
        source_id = str(row.get("chunk_id") or "")
        text = str(row.get("text") or "").strip()
        if not source_id or source_id in seen or not text:
            continue
        seen.add(source_id)
        packet.append(SynthesisEvidence(
            alias=f"S{len(packet) + 1}", real_source_id=source_id,
            title=str(row.get("title") or ""), source_kind=str(row.get("source_kind") or "history"), text=text,
        ))
    return packet


def render_evidence_packet(packet: list[SynthesisEvidence]) -> str:
    # Real IDs stay on the host. No ranks, retrieval scores, or old tool observations.
    return "\n\n".join(
        f"[{item.alias}]\ntitle: {item.title}\nsource_kind: {item.source_kind}\ntext: {item.text}"
        for item in packet
    )
