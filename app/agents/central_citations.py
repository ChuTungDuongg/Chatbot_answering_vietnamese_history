from __future__ import annotations

import re
from dataclasses import dataclass

from app.agents.central_evidence import SynthesisEvidence
from app.agents.central_question import _ascii_fold_vietnamese
from app.agents.central_analytical import target_mentions, viewpoint_sensitive


BRACKET_RE = re.compile(r"\[([^\[\]\n]+)\]")
_ALIAS_FORM = re.compile(r"s\s*([1-9]\d*)", re.I)


@dataclass(frozen=True)
class CitationCheck:
    answer: str
    source_ids: list[str]
    invalid: list[str]
    normalized: bool
    uncited_paragraphs: int
    target_mismatches: list[str]
    unattributed_viewpoints: int


def check_citations(answer: str, packet: list[SynthesisEvidence]) -> CitationCheck:
    aliases = {item.alias: item.real_source_id for item in packet}
    real_to_alias = {item.real_source_id: item.alias for item in packet}
    source_ids: list[str] = []
    invalid: list[str] = []

    def replace(match: re.Match[str]) -> str:
        value = match.group(1).strip()
        # Numeric prose is never interpreted as a source, even for a numeric real ID.
        if value.isdigit():
            return match.group(0)
        # Backward compatibility: an exact selected real ID (including Unicode)
        # is accepted and canonicalized to its request-local alias.
        if value in real_to_alias:
            found = [real_to_alias[value]]
        else:
            parts = re.split(r"\s*[,;]\s*", value)
            forms = [_ALIAS_FORM.fullmatch(part) for part in parts]
            found = [f"S{form.group(1)}" for form in forms if form]
            if len(found) != len(parts) or any(alias not in aliases for alias in found):
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
    mismatches: list[str] = []
    unattributed = 0
    targets = list(dict.fromkeys(target for item in packet for target in item.comparison_targets))
    alias_targets = {item.alias: set(item.comparison_targets) for item in packet}
    section_targets: set[str] = set()
    table_targets: list[set[str]] = []
    paragraphs = re.split(r"\n\s*\n|\n(?=\s*(?:(?:[-*]|\d+\.)\s|#{1,6}\s|\|))", normalized_answer)
    for paragraph_index, paragraph in enumerate(paragraphs):
        paragraph = paragraph.strip()
        first_line = paragraph.split("\n")[0]
        heading = first_line.startswith("#") or bool(re.fullmatch(r"\*\*[^*]+\*\*:?", first_line)) or first_line.endswith(":")
        if heading:
            section_targets = {target for target in targets if target_mentions(first_line, target)}
        if not paragraph or heading and "\n" not in paragraph:
            continue
        plain = BRACKET_RE.sub("", paragraph).strip(" *:-\n")
        folded = _ascii_fold_vietnamese(plain)
        if not plain or (paragraph.endswith(":") and len(plain.split()) <= 10):
            continue
        if re.fullmatch(r"\*\*[^*]+\*\*:?", paragraph):
            continue
        if folded.startswith(("bang chung hien co chua", "bang chung duoc cung cap chua", "chua du bang chung")):
            continue
        if paragraph.startswith("|"):
            cells = paragraph.strip("|").split("|")
            cell_targets = [{target for target in targets if target_mentions(cell, target)} for cell in cells]
            next_paragraph = paragraphs[paragraph_index + 1].strip() if paragraph_index + 1 < len(paragraphs) else ""
            if any(cell_targets) and re.fullmatch(r"[| :\-]+", next_paragraph):
                table_targets = cell_targets
                continue
            if table_targets and not re.fullmatch(r"[| :\-]+", plain):
                for expected, cell in zip(table_targets, cells):
                    supported = set().union(*(alias_targets.get(alias, set()) for alias in BRACKET_RE.findall(cell)))
                    # Explicitly unavailable dimensions do not assert a historical fact.
                    if "chua du bang chung" not in _ascii_fold_vietnamese(cell):
                        mismatches.extend(sorted(expected - supported))
        else:
            table_targets = []
        if re.fullmatch(r"[| :\-]+", plain) or paragraph.startswith("|") and not re.search(r"[.!]|\[S\d+\]", paragraph) and any(cue in folded for cue in ("tieu chi", "phuong dien")):
            continue
        cited_aliases = [value for value in BRACKET_RE.findall(paragraph) if value in aliases]
        if not cited_aliases:
            uncited += 1
        discussed = {target for target in targets if target_mentions(plain, target)} or section_targets
        supported_targets = set().union(*(alias_targets[alias] for alias in cited_aliases)) if cited_aliases else set()
        mismatches.extend(sorted(discussed - supported_targets))
        if viewpoint_sensitive(plain) and not any(cue in folded for cue in ("theo ", "nguon ", "tac gia", "tuyen bo", "nhan dinh", "quan diem", "khau hieu", "loi ke")):
            unattributed += 1
    return CitationCheck(
        normalized_answer, source_ids, list(dict.fromkeys(invalid)),
        normalized_answer != answer, uncited, list(dict.fromkeys(mismatches)), unattributed,
    )


def expand_citations(answer: str, packet: list[SynthesisEvidence]) -> str:
    aliases = {alias: item["display_index"] for alias, item in citation_display_map(packet).items()}
    # Only already validated aliases reach here. Never parse numeric brackets.
    return BRACKET_RE.sub(
        lambda match: f"[{aliases[match.group(1)]}]" if match.group(1) in aliases else match.group(0), answer,
    )


def citation_display_map(packet: list[SynthesisEvidence]) -> dict[str, dict]:
    return {item.alias: {
        "display_index": index, "source_id": item.real_source_id, "title": item.title,
        "source_kind": item.source_kind, "comparison_target": item.comparison_target,
        "comparison_targets": list(item.comparison_targets),
    } for index, item in enumerate(packet, 1)}
