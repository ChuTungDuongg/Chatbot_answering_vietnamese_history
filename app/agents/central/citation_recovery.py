"""Conservative citation-only recovery. No retrieval, new models or factual rewriting."""
from __future__ import annotations

import json
import re

from app.agents.central.citation_support import Paragraph, supported_aliases
from app.agents.central.semantics import target_mentions
from app.agents.central.citations import check_citations


PURE_CITATION_ISSUES = {"missing_valid_citations", "uncited_factual_paragraphs"}
_BREAK = r"\n\s*\n|\n(?=\s*(?:(?:[-*]|\d+\.)\s|#{1,6}\s|\|))"


def uncited_paragraphs(answer, packet, analysis=None):
    parts = re.split(f"({_BREAK})", answer)
    new_facts = {item["paragraph"] for item in check_citations(answer, packet, analysis).paragraph_classifications
                 if item["kind"] == "new_factual_claim"}
    all_targets = {target for source in packet for target in source.comparison_targets}
    section_targets = set()
    paragraphs = []
    for position in range(0, len(parts), 2):
        text = parts[position].strip()
        if text.startswith("#") or re.fullmatch(r"\*\*[^*]+\*\*:?", text) or text.endswith(":"):
            section_targets = {target for target in all_targets if target_mentions(text, target)}
        if text and position // 2 + 1 in new_facts:
            targets = {target for target in all_targets if target_mentions(text, target)} or section_targets
            paragraphs.append(Paragraph(f"P{position // 2 + 1}", position, text, frozenset(targets)))
    return parts, paragraphs


def insert_mapping(parts, paragraphs, mapping):
    parts = list(parts)
    for paragraph in paragraphs:
        if aliases := mapping.get(paragraph.key):
            parts[paragraph.position] = parts[paragraph.position].rstrip() + " " + " ".join(f"[{alias}]" for alias in aliases)
    return "".join(parts)


def align_citations(answer, packet, config, analysis=None):
    parts, paragraphs = uncited_paragraphs(answer, packet, analysis)
    mapping, confidence = {}, {}
    for paragraph in paragraphs:
        aliases, score = supported_aliases(paragraph, packet, threshold=config.citation_alignment_threshold, margin=config.citation_alignment_margin, analysis=analysis)
        confidence[paragraph.key] = score
        if aliases:
            mapping[paragraph.key] = aliases
    return insert_mapping(parts, paragraphs, mapping), confidence


def citation_repair_messages(answer, packet, analysis=None):
    _, paragraphs = uncited_paragraphs(answer, packet, analysis)
    return [
        {"role": "system", "content": 'Chỉ gán trích dẫn, không viết lại nội dung. Trả duy nhất JSON: {"P1":["S1"],"P2":["S2"]}. Chỉ dùng mã đoạn và bí danh được cấp; đoạn không được bằng chứng hỗ trợ thì dùng [].'},
        {"role": "user", "content": json.dumps({
            "paragraphs": {p.key: p.text for p in paragraphs},
            "evidence": [{"alias": s.alias, "title": s.title, "text": s.text, "comparison_targets": s.comparison_targets} for s in packet],
        }, ensure_ascii=False)},
    ]


def apply_citation_mapping(answer, content, packet, config, analysis=None):
    # Parse strict bounded JSON. A model mapping is a proposal, never proof.
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate paragraph key")
            result[key] = value
        return result
    try:
        content = content.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*\n([\s\S]*?)\n```", content)
        value = json.loads(fenced.group(1) if fenced else content, object_pairs_hook=unique)
    except (ValueError, TypeError):
        return answer
    parts, paragraphs = uncited_paragraphs(answer, packet, analysis)
    if not isinstance(value, dict) or set(value) - {p.key for p in paragraphs}:
        return answer
    by_alias = {s.alias: s for s in packet}
    accepted = {}
    for paragraph in paragraphs:
        proposed = value.get(paragraph.key, [])
        if not isinstance(proposed, list) or not all(isinstance(a, str) and a in by_alias for a in proposed) or len(proposed) > len(packet):
            return answer
        if not proposed:
            continue
        supported, _ = supported_aliases(paragraph, [by_alias[a] for a in proposed], threshold=config.citation_alignment_threshold, margin=0, analysis=analysis)
        if supported and set(supported) == set(proposed):
            accepted[paragraph.key] = proposed
    return insert_mapping(parts, paragraphs, accepted)
