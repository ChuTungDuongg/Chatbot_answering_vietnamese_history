from __future__ import annotations

import re
from dataclasses import dataclass

from app.agents.central_evidence import SynthesisEvidence
from app.agents.central_question import _ascii_fold_vietnamese


BRACKET_RE = re.compile(r"\[([^\[\]\n]+)\]")
_ALIAS_FORM = re.compile(r"s\s*([1-9]\d*)", re.I)


@dataclass(frozen=True)
class CitationCheck:
    answer: str
    source_ids: list[str]
    invalid: list[str]
    normalized: bool
    uncited_paragraphs: int


def check_citations(answer: str, packet: list[SynthesisEvidence]) -> CitationCheck:
    aliases = {item.alias: item.real_source_id for item in packet}
    real_to_alias = {item.real_source_id: item.alias for item in packet}
    source_ids: list[str] = []
    invalid: list[str] = []

    def replace(match: re.Match[str]) -> str:
        value = match.group(1).strip()
        # Backward compatibility: an exact selected real ID (including Unicode)
        # is accepted and canonicalized to its request-local alias.
        if value in real_to_alias:
            found = [real_to_alias[value]]
        else:
            parts = re.split(r"\s*[,;]\s*", value)
            forms = [_ALIAS_FORM.fullmatch(part) for part in parts]
            found = [f"S{form.group(1)}" for form in forms if form]
            if len(found) != len(parts) or any(alias not in aliases for alias in found):
                # A bare year is prose; [1] and placeholders are invalid citations.
                if not re.fullmatch(r"[1-9]\d{2,3}", value):
                    invalid.append(value)
                return match.group(0)
        for alias in found:
            source_id = aliases[alias]
            if source_id not in source_ids:
                source_ids.append(source_id)
        return " ".join(f"[{alias}]" for alias in dict.fromkeys(found))

    # A doubled pair around a supplied alias has only one possible source.
    unwrapped = re.sub(
        r"\[\[\s*(s\s*[1-9]\d*)\s*\]\]",
        lambda match: "[" + match.group(1) + "]" if re.sub(r"\s+", "", match.group(1)).upper() in aliases else match.group(0),
        answer, flags=re.I,
    )
    normalized_answer = BRACKET_RE.sub(replace, unwrapped)
    uncited = 0
    for paragraph in re.split(r"\n\s*\n|\n(?=\s*(?:[-*]|\d+\.)\s)", normalized_answer):
        paragraph = paragraph.strip()
        if not paragraph or paragraph.startswith("#") and "\n" not in paragraph:
            continue
        plain = BRACKET_RE.sub("", paragraph).strip(" *:-\n")
        folded = _ascii_fold_vietnamese(plain)
        if not plain or (paragraph.endswith(":") and len(plain.split()) <= 10):
            continue
        if re.fullmatch(r"\*\*[^*]+\*\*:?", paragraph):
            continue
        if folded.startswith(("bang chung hien co chua", "bang chung duoc cung cap chua", "chua du bang chung")):
            continue
        if not any(value in aliases for value in BRACKET_RE.findall(paragraph)):
            uncited += 1
    return CitationCheck(
        normalized_answer, source_ids, list(dict.fromkeys(invalid)),
        normalized_answer != answer, uncited,
    )


def expand_citations(answer: str, packet: list[SynthesisEvidence]) -> str:
    aliases = {item.alias: item.real_source_id for item in packet}
    # One substitution pass: newly inserted IDs can never be expanded again.
    return BRACKET_RE.sub(
        lambda match: f"[{aliases[match.group(1)]}]" if match.group(1) in aliases else match.group(0), answer,
    )
