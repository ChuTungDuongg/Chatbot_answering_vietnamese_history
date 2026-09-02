"""Conservative citation-only recovery. No retrieval, new models or factual rewriting."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.agents.central_analytical import normalize_entity, target_mentions
from app.agents.central_citations import BRACKET_RE, check_citations
from app.agents.central_compaction import sentence_units
from app.agents.central_grounding import grounding_risks


PURE_CITATION_ISSUES = {"missing_valid_citations", "uncited_factual_paragraphs"}
_BREAK = r"\n\s*\n|\n(?=\s*(?:(?:[-*]|\d+\.)\s|#{1,6}\s|\|))"
_STOP = set("va la cua cac nhung mot voi trong cho o tu da duoc nhieu nay do su ve co de khi thi ma den vao nhu cung theo".split())


@dataclass(frozen=True)
class Paragraph:
    key: str
    position: int
    text: str
    targets: frozenset[str]


def uncited_paragraphs(answer, packet):
    parts = re.split(f"({_BREAK})", answer)
    all_targets = {target for source in packet for target in source.comparison_targets}
    section_targets = set()
    paragraphs = []
    for position in range(0, len(parts), 2):
        text = parts[position].strip()
        if text.startswith("#") or re.fullmatch(r"\*\*[^*]+\*\*:?", text) or text.endswith(":"):
            section_targets = {target for target in all_targets if target_mentions(text, target)}
        if text and check_citations(text, packet).uncited_paragraphs:
            targets = {target for target in all_targets if target_mentions(text, target)} or section_targets
            paragraphs.append(Paragraph(f"P{position // 2 + 1}", position, text, frozenset(targets)))
    return parts, paragraphs


def _words(text):
    return normalize_entity(re.sub(r"\bVNCH\b", "Việt Nam Cộng hòa", text, flags=re.I)).split()


def sentence_support(claim, source) -> float:
    """Measured lexical/phrase support, not an entailment guarantee.

    Every sentence needs a distinctive match; new entities, numbers and changed
    negation cannot be rescued by common topic words or the source's rank.
    """
    risks = grounding_risks(claim, "", [source])
    if any(risks.values()):
        return 0.0
    words = _words(claim)
    significant = set(words) - _STOP
    if len(significant) < 5:
        return 0.0
    numbers = set(re.findall(r"\b\d+\b", claim))
    phrases = set(zip(words, words[1:], words[2:]))
    units = sentence_units(source.text)
    best = 0.0
    for index in range(len(units)):
        for size in (1, 2, 3):
            evidence = " ".join(units[index:index + size])
            evidence_words = _words(evidence)
            evidence_set = set(evidence_words)
            if not numbers <= set(re.findall(r"\b\d+\b", evidence)):
                continue
            if (set(words) & {"khong", "chua"}) != (evidence_set & {"khong", "chua"}):
                continue
            common = significant & evidence_set
            shared_phrases = phrases & set(zip(evidence_words, evidence_words[1:], evidence_words[2:]))
            if len(common) < 5 or len(shared_phrases) < 2:
                continue
            recall = len(common) / len(significant)
            phrase_recall = len(shared_phrases) / max(1, len(phrases))
            best = max(best, 0.8 * recall + 0.2 * phrase_recall)
    return round(best, 4)


def supported_aliases(paragraph, packet, *, threshold, margin):
    # Table citations belong in individual cells; don't append a row-wide guess.
    if paragraph.text.startswith("|"):
        return [], 0.0
    candidates = [source for source in packet if not paragraph.targets or set(source.comparison_targets) & paragraph.targets]
    prose = BRACKET_RE.sub(lambda match: match.group(1) if match.group(1).isdigit() else "", paragraph.text)
    prose = re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", prose)
    clauses = [part for sentence in sentence_units(prose) for part in re.split(r";\s*", sentence) if part.strip()]
    aliases, confidences = [], []
    for clause in clauses:
        scores = sorted(((sentence_support(clause, source), source.alias) for source in candidates), reverse=True)
        if not scores or scores[0][0] < threshold:
            return [], scores[0][0] if scores else 0.0
        best, alias = scores[0]
        # A close runner-up is fine only if it independently meets the same strict
        # support threshold; otherwise the match is ambiguous and requires review.
        if len(scores) > 1 and best - scores[1][0] < margin and scores[1][0] < threshold:
            return [], best
        aliases.append(alias)
        confidences.append(best)
    aliases = list(dict.fromkeys(aliases))
    supported_targets = {target for source in candidates if source.alias in aliases for target in source.comparison_targets}
    if not paragraph.targets <= supported_targets:
        return [], min(confidences, default=0.0)
    return aliases, min(confidences, default=0.0)


def insert_mapping(parts, paragraphs, mapping):
    parts = list(parts)
    for paragraph in paragraphs:
        if aliases := mapping.get(paragraph.key):
            parts[paragraph.position] = parts[paragraph.position].rstrip() + " " + " ".join(f"[{alias}]" for alias in aliases)
    return "".join(parts)


def align_citations(answer, packet, config):
    parts, paragraphs = uncited_paragraphs(answer, packet)
    mapping, confidence = {}, {}
    for paragraph in paragraphs:
        aliases, score = supported_aliases(paragraph, packet, threshold=config.citation_alignment_threshold, margin=config.citation_alignment_margin)
        confidence[paragraph.key] = score
        if aliases:
            mapping[paragraph.key] = aliases
    return insert_mapping(parts, paragraphs, mapping), confidence


def citation_repair_messages(answer, packet):
    _, paragraphs = uncited_paragraphs(answer, packet)
    return [
        {"role": "system", "content": 'Chỉ gán trích dẫn, không viết lại nội dung. Trả duy nhất JSON: {"P1":["S1"],"P2":["S2"]}. Chỉ dùng mã đoạn và bí danh được cấp; đoạn không được bằng chứng hỗ trợ thì dùng [].'},
        {"role": "user", "content": json.dumps({
            "paragraphs": {p.key: p.text for p in paragraphs},
            "evidence": [{"alias": s.alias, "title": s.title, "text": s.text, "comparison_targets": s.comparison_targets} for s in packet],
        }, ensure_ascii=False)},
    ]


def apply_citation_mapping(answer, content, packet, config):
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
    parts, paragraphs = uncited_paragraphs(answer, packet)
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
        supported, _ = supported_aliases(paragraph, [by_alias[a] for a in proposed], threshold=config.citation_alignment_threshold, margin=0)
        if supported and set(supported) == set(proposed):
            accepted[paragraph.key] = proposed
    return insert_mapping(parts, paragraphs, accepted)
