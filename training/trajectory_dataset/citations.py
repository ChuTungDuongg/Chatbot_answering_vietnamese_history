from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


_BRACKETED_TOKEN = re.compile(r"\[([^\[\]]+)\]")
_STRUCTURED_ID = re.compile(r"^[\w.-]+$", flags=re.UNICODE)
_COMPACT_ID = re.compile(r"^([A-Za-z]+)\d+$")


@dataclass(frozen=True)
class EvidenceCitations:
    """Allowed evidence citations and canonical-looking unknown references."""

    citations: tuple[str, ...]
    unknown_ids: tuple[str, ...]


def format_evidence_citation(evidence_id: str) -> str:
    """Write the canonical trajectory citation form: ``[evidence_id]``."""
    value = str(evidence_id).strip()
    if not value or "[" in value or "]" in value or any(character.isspace() for character in value):
        raise ValueError("evidence IDs must be non-empty bracket-free tokens")
    return f"[{value}]"


def _looks_like_unknown_evidence_id(value: str, allowed_ids: set[str]) -> bool:
    """Distinguish structured project IDs from ordinary bracketed prose.

    Known IDs are always recognized exactly, regardless of their shape. An
    unknown token is evidence-like only when it is a compact structured token
    containing a letter and an ID separator, or when it follows the same short
    alpha+number family as an allowed ID (for fixtures such as c1/c2).
    """
    if not value or len(value) > 256 or not _STRUCTURED_ID.fullmatch(value):
        return False
    if not any(character.isalpha() for character in value):
        return False
    if any(separator in value for separator in ("_", "-", ".")):
        return True
    candidate = _COMPACT_ID.fullmatch(value)
    if candidate is None:
        return False
    candidate_family = candidate.group(1)
    return any(
        (allowed_match := _COMPACT_ID.fullmatch(allowed)) is not None
        and allowed_match.group(1) == candidate_family
        for allowed in allowed_ids
    )


def extract_evidence_citations(answer: str, allowed_ids: Iterable[str]) -> EvidenceCitations:
    """Parse citations relative to observed IDs without treating all brackets as IDs.

    Numeric references and ordinary bracketed prose are ignored. Exact allowed
    IDs are returned as citations. Canonical-looking but unobserved IDs are
    returned separately so grounding checks can still reject them.
    """
    allowed = {str(value).strip() for value in allowed_ids if str(value).strip()}
    citations: list[str] = []
    unknown: list[str] = []
    for raw_value in _BRACKETED_TOKEN.findall(str(answer or "")):
        value = raw_value.strip()
        if value in allowed:
            if value not in citations:
                citations.append(value)
        elif _looks_like_unknown_evidence_id(value, allowed) and value not in unknown:
            unknown.append(value)
    return EvidenceCitations(tuple(citations), tuple(unknown))
