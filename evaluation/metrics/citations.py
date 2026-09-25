"""Citation ID checks with explicit retrieved and optional corpus/gold sets."""
from __future__ import annotations

import re
from typing import Any


BRACKET = re.compile(r"\[([^\[\]\n]{1,128})\]")


def aliases_for_sources(sources: list[dict[str, Any]]) -> dict[str, str]:
    """Map visible aliases to canonical source IDs, retaining first occurrence."""
    aliases: dict[str, str] = {}
    for index, source in enumerate(sources, 1):
        canonical = str(source.get("source_id") or source.get("chunk_id") or "")
        if not canonical:
            continue
        display_index = source.get("display_index") or index
        for alias in (source.get("source_id"), source.get("chunk_id"), source.get("citation_id"),
                      f"S{display_index}", str(display_index)):
            if alias is not None:
                aliases.setdefault(str(alias), canonical)
    return aliases


def parse_citations(answer: str, known_aliases: set[str]) -> list[str]:
    found: list[str] = []
    for match in BRACKET.finditer(answer):
        for raw in re.split(r"[,;]", match.group(1)):
            value = raw.strip()
            if not value:
                continue
            # A bracketed historical year is not a citation unless an explicit
            # retrieved alias has that exact spelling.
            if re.fullmatch(r"\d{3,4}", value) and value not in known_aliases:
                continue
            if value in known_aliases or re.fullmatch(r"[\w:.-]{1,128}", value, re.UNICODE):
                found.append(value)
    return found


def score_citations(answer: str, sources: list[dict[str, Any]], *,
                    gold_source_ids: list[str] | None = None,
                    known_source_ids: set[str] | None = None,
                    factual_paragraph_indices: list[int] | None = None) -> dict[str, Any]:
    aliases = aliases_for_sources(sources)
    cited_aliases = parse_citations(answer, set(aliases))
    resolved = [aliases.get(alias) for alias in cited_aliases]
    total = len(cited_aliases)
    valid = sum(item is not None for item in resolved)
    cited_sources = {item for item in resolved if item is not None}
    cited_targets = {item if item is not None else alias
                     for alias, item in zip(cited_aliases, resolved)}
    result: dict[str, Any] = {
        "cited_aliases": cited_aliases,
        "resolved_source_ids": sorted(cited_sources),
        "invalid_citation_ids": [alias for alias, item in zip(cited_aliases, resolved) if item is None],
        "citation_validity_rate": valid / total if total else None,
        "cited_retrieved_rate": valid / total if total else None,
        "citation_precision": None,
        "citation_recall": None,
        "source_id_existence_rate": None,
        "answer_citation_coverage": None,
    }
    if gold_source_ids is not None:
        gold = set(gold_source_ids)
        result["citation_precision"] = len(cited_sources & gold) / len(cited_targets) if cited_targets else None
        result["citation_recall"] = len(cited_sources & gold) / len(gold) if gold else None
    if known_source_ids is not None and total:
        result["source_id_existence_rate"] = sum(
            item in known_source_ids if item is not None else alias in known_source_ids
            for alias, item in zip(cited_aliases, resolved)
        ) / total
    if factual_paragraph_indices is not None:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", answer) if part.strip()]
        indices = [index for index in factual_paragraph_indices if 0 <= index < len(paragraphs)]
        if indices:
            result["answer_citation_coverage"] = sum(
                any(alias in aliases for alias in parse_citations(paragraphs[index], set(aliases)))
                for index in indices
            ) / len(indices)
    return result
