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
    prepared = []
    for row in rows:
        key = question_key(str(row.get("question") or ""))
        prepared.append((str(row.get("id") or row.get("candidate_id")), key, set(key.split())))
    for index, (left_id, left_key, left_tokens) in enumerate(prepared):
        for right_id, right_key, right_tokens in prepared[index + 1:]:
            if left_key == right_key:
                exact.append((left_id, right_id))
            elif (left_key and right_key
                  and len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens)) >= 0.75
                  and SequenceMatcher(None, left_key, right_key).ratio() >= 0.88):
                near.append((left_id, right_id))
    return exact, near
