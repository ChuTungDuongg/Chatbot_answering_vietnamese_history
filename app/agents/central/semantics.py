"""Central semantic value types and lexical normalization; no higher-level dependencies."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

@dataclass(frozen=True)
class CentralQuestionAnalysis:
    question: str
    question_type: str | None
    analytical: bool
    comparison_targets: tuple[str, ...] = ()
    subject: str | None = None
    event: str | None = None
    event_type: str | None = None
    canonical_target: str | None = None
    actors: tuple[str, ...] = ()
    outcome: str | None = None
    facets: tuple[str, ...] = ()
    related_entities: tuple[str, ...] = ()
    relation_requested: bool = False
    relation_phrase: str | None = None
    comparison_targets_raw: tuple[str, ...] = ()
    comparison_canonical_targets: tuple[str, ...] = ()
    comparison_target_entity_types: tuple[str | None, ...] = ()
    target_resolution_events: tuple[dict, ...] = ()
    administrative_level: str | None = None
    time_scope: tuple[int, ...] = ()
    freshness_required: bool = False
    freshness_reason: str | None = None
    premise_requires_validation: bool = False
    raw_event_clause: str | None = None
    answer_depth: str = "simple_fact"
    viewpoint_requested: bool = False

    def telemetry(self) -> dict:
        return {
            "question_type": self.question_type, "answer_depth": self.answer_depth, "subject": self.subject,
            "viewpoint_requested": self.viewpoint_requested,
            "event": self.event, "actors": list(self.actors), "outcome": self.outcome,
            "event_type": self.event_type, "canonical_target": self.canonical_target,
            "comparison_targets": list(self.comparison_targets),
            "facet": self.facets[0] if self.facets else None, "facets": list(self.facets),
            "related_entities": list(self.related_entities), "relation_requested": self.relation_requested,
            "relation_phrase": self.relation_phrase,
            "comparison_targets_raw": list(self.comparison_targets_raw or self.comparison_targets),
            "comparison_targets_normalized": list(self.comparison_targets),
            "comparison_canonical_targets": list(self.comparison_canonical_targets or self.comparison_targets),
            "comparison_target_entity_types": list(self.comparison_target_entity_types),
            "target_resolution_events": list(self.target_resolution_events),
            "administrative_level": self.administrative_level,
            "requested_administrative_level": self.administrative_level,
            "time_scope": list(self.time_scope), "freshness_required": self.freshness_required,
            "freshness_reason": self.freshness_reason,
            "canonical_event": self.event, "raw_event_clause": self.raw_event_clause,
            "requested_facets": list(self.facets),
            "event_resolution_events": ([{"raw": self.raw_event_clause, "canonical": self.event,
                                          "reason": "analytical_clause_boundary"}] if self.raw_event_clause and self.raw_event_clause != self.event else []),
        }


def _ascii_fold_vietnamese(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
        if not unicodedata.combining(char)
    )


ACTOR_ALIASES = ((r"\b(?:my|hoa ky)\b", "Mỹ"), (r"\b(?:vnch|viet nam cong hoa)\b", "Việt Nam Cộng hòa"))


def normalized_actor_text(text):
    folded = _ascii_fold_vietnamese(text)
    for pattern, name in ACTOR_ALIASES:
        folded = re.sub(pattern, _ascii_fold_vietnamese(name), folded)
    return " ".join(re.findall(r"[a-z0-9]+", folded))


def normalize_entity(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", _ascii_fold_vietnamese(value)))


def target_key(value: str) -> str:
    person_tran = value.strip().casefold().startswith("trần ")
    value = normalize_entity(value)
    prefix = r"^(?:chien thang|chien dich)\s+" if person_tran else r"^(?:chien thang|chien dich|tran)\s+"
    value = re.sub(prefix, "", value)
    return re.sub(r"\bnam\s+(?=\d{3,4}\b)", "", value).strip()


def target_mentions(text: str, target: str) -> bool:
    key = target_key(target)
    return bool(key and f" {key} " in f" {target_key(text)} ")


BRACKET_RE = re.compile(r"\[([^\[\]\n]+)\]")
