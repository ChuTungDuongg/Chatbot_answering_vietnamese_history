"""Shared draft categories, duplicate checks, and review policy version."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher


REVIEW_POLICY_VERSION = 1
CATEGORIES = (
    "dynasties", "rulers", "historical_figures", "battles", "wars",
    "revolutions", "treaties", "colonial_period", "independence",
    "political_events", "cultural_history", "religious_history",
    "economic_history", "social_history", "historical_geography",
    "archaeology_heritage", "chronology", "cause_consequence",
    "comparison", "multi_hop", "ambiguous", "insufficient_evidence",
    "out_of_domain", "false_premise",
)
REVIEW_STATES = ("pending", "accepted", "rejected", "needs_review")
EVIDENCE_DECISIONS = ("relevant", "not_relevant", "uncertain")


def fold(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.casefold())
    normalized = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", normalized.replace("đ", "d"))).strip()


def question_key(value: str) -> str:
    return fold(value)


def similar(a: str, b: str, *, threshold: float = 0.88) -> bool:
    left, right = question_key(a), question_key(b)
    if left == right:
        return True
    if not left or not right:
        return False
    tokens_a, tokens_b = set(left.split()), set(right.split())
    overlap = len(tokens_a & tokens_b) / max(1, len(tokens_a | tokens_b))
    return overlap >= 0.75 and SequenceMatcher(None, left, right).ratio() >= threshold


def duplicate_pairs(rows: list[dict]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    exact, near = [], []
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            left_id = str(left.get("id") or left.get("candidate_id"))
            right_id = str(right.get("id") or right.get("candidate_id"))
            a, b = str(left.get("question") or ""), str(right.get("question") or "")
            if question_key(a) == question_key(b):
                exact.append((left_id, right_id))
            elif similar(a, b):
                near.append((left_id, right_id))
    return exact, near
