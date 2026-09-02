from __future__ import annotations

import unicodedata
from typing import Iterable

from app.agents.evidence.schemas import SelectedEvidence


def normalize_grounding(value: str) -> str:
    normalized = unicodedata.normalize("NFD", str(value).casefold().replace("đ", "d"))
    folded = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    return " ".join(folded.split())


def grounded_in_source(value: str, source_text: str) -> bool:
    normalized = normalize_grounding(value)
    return bool(normalized and normalized in normalize_grounding(source_text))


def compressed_derived_from_own_claims(item: SelectedEvidence, source_text: str) -> bool:
    if not item.claims or any(not grounded_in_source(claim, source_text) for claim in item.claims):
        return False
    compressed = normalize_grounding(item.compressed_text)
    return compressed in {
        normalize_grounding(source_text),
        normalize_grounding(" ".join(item.claims)),
    } or grounded_in_source(item.compressed_text, source_text)


def referenced_evidence_ids(description: str, evidence_ids: Iterable[str]) -> list[str]:
    return [evidence_id for evidence_id in evidence_ids if evidence_id and evidence_id in description]

