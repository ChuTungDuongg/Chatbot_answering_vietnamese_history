from __future__ import annotations

import math
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote, urlsplit

from app.agents.central_question import CentralQuestionAnalysis
from app.agents.central_viewpoints import annotate_viewpoints
from app.agents.central_compaction import excerpt_evidence
from app.agents.central_relationships import select_relationship_evidence, mentions
from app.agents.central_entity_aliases import evidence_aliases
from app.agents.config import CentralAgentConfig
from app.agents.central_analytical import (
    annotate_evidence, coverage_select, evidence_targets, strong_evidence, entity_consistency, normalize_entity,
)


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
    target: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter within a retrieval batch; never compare scores from different queries/tools.

    Raw CrossEncoder scores have no guaranteed probability calibration. The default
    removes only a separated low tail (at least two stronger anchors). An absolute
    floor is opt-in for explicitly calibrated probability scores.
    """
    if analysis.relation_requested:
        annotated = [annotate_evidence(row, analysis) for row in rows]
        kept = [row for row in annotated if row["covered_entities"]]
        if compare_scores:
            kept.sort(key=lambda row: _score(row, "reranker_score") if _score(row, "reranker_score") is not None else float("-inf"), reverse=True)
            kept = [{**row, "retrieval_rank": rank} for rank, row in enumerate(kept, 1)]
        dropped = len(rows) - len(kept)
        return kept, {"retrieval_candidates_before_filter": len(rows), "retrieval_candidates_after_filter": len(kept),
            "retrieval_filtered_count": dropped, "retrieval_filter_reasons": {"unrequested_biography_entity": dropped} if dropped else {},
            "entity_disambiguation_filtered_count": dropped, "entity_disambiguation_filter_reasons": {},
            "biography_entity": analysis.subject, "biography_exact_title_hits": sum(bool(row["canonical_entities"]) for row in kept)}
    if analysis.question_type in {"cause", "comparison"} and (target or analysis.event or analysis.subject):
        annotated = [annotate_evidence(row, analysis, target) for row in rows]
        reasons = Counter(row["entity_filter_reason"] or "analytical_target_mismatch"
                          for row in annotated if not row["target_consistent"])
        kept = [row for row in annotated if row["target_consistent"]]
        # Ranks are only comparable within this batch. Never apply a global score floor
        # that could drop a whole target or a distinct analytical dimension.
        if compare_scores:
            kept.sort(key=lambda row: _score(row, "reranker_score") if _score(row, "reranker_score") is not None else float("-inf"), reverse=True)
            kept = [{**row, "retrieval_rank": rank} for rank, row in enumerate(kept, 1)]
        return kept, {
            "retrieval_candidates_before_filter": len(rows), "retrieval_candidates_after_filter": len(kept),
            "retrieval_filtered_count": len(rows) - len(kept), "retrieval_filter_reasons": dict(reasons),
            "entity_disambiguation_filtered_count": sum(reasons.values()),
            "entity_disambiguation_filter_reasons": dict(reasons),
            "chronology_downranked_count": sum(row["chronology_downranked"] for row in kept),
            "biography_entity": None, "biography_exact_title_hits": 0,
        }
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
        "entity_disambiguation_filtered_count": sum(1 for i in excluded if subject and i not in associated),
        "entity_disambiguation_filter_reasons": {"biography_entity_collision": sum(1 for i in excluded if subject and i not in associated)} if subject else {},
    }


def select_synthesis_evidence(rows: list[dict[str, Any]], analysis: CentralQuestionAnalysis, config: CentralAgentConfig) -> list[dict[str, Any]]:
    if analysis.relation_requested:
        previews = [{**row, "text": excerpt_evidence(str(row.get("text") or ""), analysis, config.evidence_excerpt_chars),
                     "evidence_chars_before_compaction": len(str(row.get("text") or ""))} for row in rows]
        selected = select_relationship_evidence(previews, analysis, config.biography_max_sources)
    elif analysis.comparison_targets:
        groups = []
        for target in analysis.comparison_targets:
            group, _ = select_evidence([row for row in rows if target in evidence_targets(row)], analysis, config, compare_scores=False, target=target)
            strong = [row for row in group if strong_evidence(row, min_chars=config.strong_evidence_min_chars)]
            groups.append(coverage_select(strong or group, analysis, config.analytical_max_sources))
        # Round-robin reservations before filling spare capacity; neither target can
        # exhaust the packet. Duplicate sources retain all verified target origins.
        combined: dict[str, dict[str, Any]] = {}
        for index in range(config.analytical_max_sources):
            for group in groups:
                if index >= len(group):
                    continue
                row = group[index]
                source_id = str(row["chunk_id"])
                if source_id in combined:
                    continue
                if len(combined) < config.analytical_max_sources:
                    combined[source_id] = row
        selected = list(combined.values())
        # Group order is stable, so S1/S2 belong to A and S3/S4 to B when quota=2.
        selected.sort(key=lambda row: next(i for i, target in enumerate(analysis.comparison_targets) if target in evidence_targets(row)))
    else:
        filtered, _ = select_evidence(rows, analysis, config, compare_scores=False)
        selected = [annotate_evidence(row, analysis) for row in filtered]
        if analysis.question_type == "cause":
            # Rank the excerpt that synthesis can actually receive, not the
            # dimensions aggregated over a larger source that will be discarded.
            selected = [annotate_evidence({**row,
                "text": excerpt_evidence(str(row.get("text") or ""), analysis, config.evidence_excerpt_chars),
                "evidence_chars_before_compaction": len(str(row.get("text") or "")),
            }, analysis) for row in selected]
            selected = [row for row in selected if row["text"]]
        if analysis.question_type in {"cause", "significance", "consequence", "evaluation"}:
            strong = [row for row in selected if strong_evidence(row, min_chars=config.strong_evidence_min_chars)]
            if len(strong) >= 3:
                selected = strong
            selected = coverage_select(selected, analysis, config.analytical_max_sources)
        else:
            selected = selected[:config.max_tool_results]
    # Source admission is independent of action-observation consumption. Share the
    # synthesis character budget fairly, then re-annotate exactly what the model sees.
    per_source = min(config.evidence_excerpt_chars, max(0, config.synthesis_char_budget // max(1, len(selected)) - 200))
    bounded = []
    for row in selected:
        original = str(row.get("text") or "")
        item = {**row, "text": excerpt_evidence(original, analysis, per_source), "evidence_chars_before_compaction": row.get("evidence_chars_before_compaction", len(original))}
        if not item["text"]:
            continue
        if analysis.comparison_targets:
            # Merging versions or truncating text must not confer another target's
            # support on a source that no longer contains that evidence.
            verified = [target for target in evidence_targets(item) if entity_consistency(item, target)[0]]
            if not verified:
                continue
            item.update(comparison_target=verified[0], comparison_targets=verified)
        bounded.append(annotate_evidence(item, analysis))
    return bounded


@dataclass(frozen=True)
class SynthesisEvidence:
    alias: str
    real_source_id: str
    title: str
    source_kind: str
    text: str
    comparison_target: str | None = None
    comparison_targets: tuple[str, ...] = ()
    viewpoint_sensitive: bool = False
    viewpoint_annotations: tuple[dict[str, Any], ...] = ()
    entity_aliases: tuple[dict[str, Any], ...] = ()

    def __post_init__(self):
        if self.comparison_target and not self.comparison_targets:
            object.__setattr__(self, "comparison_targets", (self.comparison_target,))
        # Derive from the exact visible text, not a legacy source-wide flag or stale
        # annotations that may refer to text discarded by the packet budget.
        annotations = tuple(annotate_viewpoints(self.text))
        object.__setattr__(self, "viewpoint_annotations", annotations)
        object.__setattr__(self, "viewpoint_sensitive", bool(annotations))
        # A truncated-away identity cannot confer aliases through stale metadata.
        aliases = [pair for pair in self.entity_aliases if mentions(f"{self.title} {self.text}", pair["name"])
                   and (pair["origin"] == "selected_entity_metadata" or mentions(self.text, pair["alias"]))]
        object.__setattr__(self, "entity_aliases", tuple(aliases or evidence_aliases(self.text, self.title)))


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
            comparison_target=row.get("comparison_target"), comparison_targets=tuple(evidence_targets(row)),
            viewpoint_sensitive=bool(row.get("viewpoint_sensitive")),
            entity_aliases=tuple(evidence_aliases(text, str(row.get("title") or ""), row.get("metadata"))),
        ))
    return packet


def render_evidence_packet(packet: list[SynthesisEvidence]) -> str:
    # Real IDs stay on the host. No ranks, retrieval scores, or old tool observations.
    def render(item):
        spans = [{"type": a["type"], "excerpt": a["text"][:140], "attribution_hint": a["attribution_hint"]}
                 for a in item.viewpoint_annotations if a["requires_attribution"]]
        note = "\nviewpoint_annotations: " + json.dumps(spans[:6], ensure_ascii=False) if spans else ""
        if item.entity_aliases:
            note += "\nentity_aliases: " + json.dumps(item.entity_aliases, ensure_ascii=False)
        return f"[{item.alias}]\ntitle: {item.title}\nsource_kind: {item.source_kind}{note}\ntext: {item.text}"
    targets = list(dict.fromkeys(target for item in packet for target in item.comparison_targets))
    if targets:
        return "\n\n".join(f"TARGET {chr(65 + i)} — {target}\n" + "\n\n".join(
            render(item) for item in packet if target in item.comparison_targets
        ) for i, target in enumerate(targets))
    return "\n\n".join(render(item) for item in packet)
