"""Deterministic two-entity coverage and relationship claims within Central V2."""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from app.agents.central_question import _ascii_fold_vietnamese


RELATION_CAVEAT = "Bằng chứng hiện có chưa xác lập một mối quan hệ trực tiếp cụ thể giữa hai người."
_INTERACTION = re.compile(r"\b(?:gap(?: go)?|hoi dam|tro chuyen|lam viec (?:cung|voi)|cong su|cap tren|chi huy truc tiep|truc tiep chi huy|doi dau voi|doi thu ca nhan|thu dich ca nhan|trao doi thu|lien lac truc tiep)\b")
_OPPOSING_CONTEXT = re.compile(r"\b(?:doi lap|doi nghich|chien tuyen doi|hai phia doi)\b")


def normalized(text):
    return " ".join(re.findall(r"[a-z0-9]+", _ascii_fold_vietnamese(text)))


def mentions(text, entity):
    return bool(entity and f" {normalized(entity)} " in f" {normalized(text)} ")


def relation_spans(text, analysis):
    first, second = (re.escape(normalized(entity)) for entity in [analysis.subject, analysis.related_entities[0]])
    modifiers = r"(?:(?:da|tung|co|la|khong|chua|mot|nguoi) )*"
    predicate = r"(?:gap(?: go)?|hoi dam|tro chuyen|lam viec|cong su|cap tren|chi huy truc tiep|truc tiep chi huy|doi dau|doi thu ca nhan|thu dich ca nhan|trao doi thu|lien lac truc tiep)"
    def connects(a, b):
        return rf"\b{a} {modifiers}{predicate} (?:(?:voi|cua|cung) )?{b}\b|\b{a} va {b} {modifiers}{predicate} (?:voi )?nhau\b"
    connected = re.compile(connects(first, second) + "|" + connects(second, first))
    return [sentence.strip() for sentence in re.split(r"[.!?;\n]", text)
            if connected.search(normalized(sentence)) and not has_relation_caveat(sentence)]


def annotate_relationship(row, analysis):
    row = dict(row)
    text, title = str(row.get("text") or row.get("snippet") or ""), str(row.get("title") or "")
    entities = [analysis.subject, *analysis.related_entities]
    covered = [entity for entity in entities if mentions(text, entity) or normalized(title) == normalized(entity)]
    canonical = [entity for entity in entities if normalized(re.sub(r"\([^)]*\)", "", title)) == normalized(entity)]
    spans = relation_spans(text, analysis)
    roles = (["primary_subject"] if analysis.subject in covered else []) + (["related_entity"] if any(entity in covered for entity in analysis.related_entities) else [])
    if spans:
        roles.append("relationship")
    return {**row, "covered_entities": covered, "canonical_entities": canonical,
            "entity_roles": roles, "entity_role": roles[0] if roles else None,
            "direct_relation_spans": spans, "target_consistent": bool(covered)}


def usable(row):
    text = str(row.get("text") or "")
    return row.get("citable", True) and len(text) >= 60 and len(text.split()) >= 10


def select_relationship_evidence(rows, analysis, limit):
    rows = [annotate_relationship(row, analysis) for row in rows]
    selected = []
    def reserve(candidates):
        candidates = [row for row in candidates if row not in selected and usable(row)]
        if candidates and len(selected) < limit:
            selected.append(min(candidates, key=lambda row: (not bool(row["canonical_entities"]), row.get("retrieval_rank", 999))))
    for entity in [analysis.subject, *analysis.related_entities]:
        reserve([row for row in rows if entity in row["canonical_entities"]] or [row for row in rows if entity in row["covered_entities"]])
    reserve([row for row in rows if row["direct_relation_spans"]])
    # Fill with canonical identities/context before peripheral anecdotes.
    for row in sorted(rows, key=lambda row: (not bool(row["canonical_entities"]), row.get("retrieval_rank", 999))):
        if row not in selected and usable(row) and row["covered_entities"] and len(selected) < limit:
            selected.append(row)
    return selected


def relationship_coverage(rows, analysis):
    rows = [annotate_relationship(row, analysis) for row in rows if usable(row)]
    counts = {entity: len({normalized(row["text"]) for row in rows if entity in row["covered_entities"]})
              for entity in [analysis.subject, *analysis.related_entities]}
    direct = len({normalized(span) for row in rows for span in row["direct_relation_spans"]})
    enough = all(counts.values())
    return enough, {
        "relationship_balance": {**{entity: {"strong_evidence_count": count} for entity, count in counts.items()},
                                 "direct_relation": {"strong_evidence_count": direct}},
        "primary_subject_evidence_count": counts[analysis.subject],
        "related_entity_evidence_count": sum(counts[e] for e in analysis.related_entities),
        "direct_relation_evidence_count": direct,
        "partial_answer": bool(enough and not direct),
        "unresolved_facets": ["relationship"] if not direct else [],
        "evidence_sufficient": bool(enough),
        "evidence_sufficiency_reason": "relationship_entities_and_direct_evidence" if enough and direct else "relationship_entities_with_explicit_limit" if enough else "relationship_entity_coverage_insufficient",
    }


def has_relation_caveat(answer):
    folded = normalized(answer)
    return bool(re.search(r"\b(?:chua (?:co )?(?:du |duoc )?(?:bang chung|xac lap|xac dinh)|khong du bang chung)\b", folded)
                and re.search(r"\b(?:quan he|lien he|tuong tac|gap|lam viec)\b", folded))


def relationship_answer_issues(answer, analysis, packet):
    if not analysis.relation_requested:
        return []
    issues = []
    if not mentions(answer, analysis.subject):
        issues.append("relationship_primary_identity_missing")
    source_by_alias = {source.alias: source for source in packet}
    for paragraph in re.split(r"\n\s*\n", answer):
        aliases = re.findall(r"\[(S\d+)\]", paragraph)
        evidence = [source_by_alias[alias] for alias in aliases if alias in source_by_alias] if aliases else packet
        spans = [span for source in evidence for span in relation_spans(source.text, analysis)]
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            # Evidence limits are not assertions that an interaction never happened.
            if has_relation_caveat(sentence):
                continue
            if _OPPOSING_CONTEXT.search(normalized(sentence)) and not any(_OPPOSING_CONTEXT.search(normalized(source.text)) for source in evidence):
                issues.append("unsupported_relationship_claim")
            if not _INTERACTION.search(normalized(sentence)):
                continue
            claim = re.sub(r"\[S\d+\]", "", sentence)
            words = normalized(claim).split()
            # Require a documented relationship proposition, not co-occurrence or
            # two unrelated biographies. Citation/grounding checks remain separate.
            supported = False
            for span in spans:
                source_words = normalized(span).split()
                if ("khong" in words) != ("khong" in source_words):
                    continue
                if not set(re.findall(r"\d+", claim)) <= set(re.findall(r"\d+", span)):
                    continue
                if _INTERACTION.findall(normalized(claim)) != _INTERACTION.findall(normalized(span)):
                    continue
                if SequenceMatcher(None, words, source_words, autojunk=False).ratio() >= .7:
                    supported = True
                    break
            if not supported:
                issues.append("unsupported_relationship_claim")
    return list(dict.fromkeys(issues))


def relationship_contract(analysis, debug):
    return (
        f"Câu hỏi gồm danh tính {analysis.subject} và mối liên hệ với {', '.join(analysis.related_entities)}. "
        "Trình bày ngắn gọn danh tính được nguồn hỗ trợ và xử lý riêng phần quan hệ. "
        "Nguồn riêng về hai người không chứng minh họ từng gặp, trò chuyện, làm việc chung, chỉ huy nhau hay có thù địch cá nhân. "
        "Bối cảnh cùng thời/khác phía chỉ được nêu khi nguồn xác lập bối cảnh đó. "
        + ("Chưa tìm thấy bằng chứng về tương tác trực tiếp. Giữ phần được hỗ trợ, không suy ra họ chưa từng gặp; kết thúc phần quan hệ bằng: " + RELATION_CAVEAT
           if not debug.get("direct_relation_evidence_count") else "Chỉ nêu tương tác trực tiếp đúng như đoạn bằng chứng liên hệ đã ghi nhận.")
    )
