"""Deterministic, model-free prompt budgeting with intact sentence/quote boundaries."""
from __future__ import annotations

import re

from app.agents.central_analytical import DIMENSION_CUES, cause_sentence_relevance, normalize_entity
from app.agents.central_question import analyze_central_question
from app.agents.central_facets import neutral_preference, viewpoint_cost, evidence_facets, multi_facet


def sentence_units(text: str) -> list[str]:
    # Keep complete quotations (including multi-sentence speech) in one unit.
    protected = [(m.start(), m.end()) for m in re.finditer(r'“[^”]*”|"[^"\n]*"|‘[^’]*’', text)]
    boundaries = [0]
    for match in re.finditer(r'(?<=[.!?])\s+|\n+', text):
        if not any(start <= match.start() < end for start, end in protected):
            boundaries.append(match.end())
    boundaries.append(len(text))
    units = [text[a:b].strip() for a, b in zip(boundaries, boundaries[1:]) if text[a:b].strip()]
    merged = []
    for unit in units:
        if merged and unit.startswith(('“', '"', '‘')) and re.search(r"\b(?:noi|cho rang|nhan dinh|tuyen bo|said|argued)\b", normalize_entity(merged[-1])):
            merged[-1] += "\n" + unit
        else:
            merged.append(unit)
    return merged


def excerpt_evidence(text: str, analysis, limit: int) -> str:
    if len(text) <= limit:
        return text
    units = sentence_units(text)
    # Equivalent resolved cause questions score excerpts from the same semantic query.
    question = " ".join(filter(None, [analysis.event, analysis.subject, *analysis.actors, analysis.outcome])) if analysis.answer_depth == "broad_analysis" else analysis.question
    query = set(normalize_entity(" ".join(filter(None, [question, analysis.event, analysis.subject, *analysis.facets]))).split())
    requested = set(analysis.facets)
    def score(value):
        folded = normalize_entity(value)
        words = set(folded.split())
        dims = {dim for dim, cues in DIMENSION_CUES.items() if any(cue in folded for cue in cues)}
        cause = 8 * cause_sentence_relevance(value) if analysis.question_type == "cause" and not requested & {"result", "significance", "consequence"} else 0
        administrative = 0
        if analysis.administrative_level:
            from app.agents.central_administration import annotate_administration
            relevance = annotate_administration({"text": value}, analysis)
            administrative = 30 * relevance["administrative_level_consistent"] + 15 * bool(relevance["policy_cause_spans"])
        neutral = -24 * viewpoint_cost(value) if neutral_preference(analysis) else 0
        facet_gain = 6 * len(set(evidence_facets({"text": value, "cause_facet_score": cause_sentence_relevance(value)})) & requested) if multi_facet(analysis) else 0
        return len(words & query) + 3 * len(dims & requested) + len(dims) + cause + administrative + neutral + facet_gain
    ranked = sorted(range(len(units)), key=lambda i: (-score(units[i]), i))
    chosen: set[int] = set()
    window_count = 0
    # At most two query-relevant windows. Include preceding speaker/context and
    # following explanation where the complete units fit; never cut a sentence.
    for center in ranked:
        if center in chosen or len(units[center]) > limit:
            continue
        window = {center}
        for neighbor in (center - 1, center + 1):
            if 0 <= neighbor < len(units):
                if neutral_preference(analysis) and not viewpoint_cost(units[center]) and viewpoint_cost(units[neighbor]) > .25:
                    continue
                proposal = chosen | window | {neighbor}
                if len("\n\n[…]\n\n".join(units[i] for i in sorted(proposal))) <= limit:
                    window.add(neighbor)
        proposal = chosen | window
        if len("\n\n[…]\n\n".join(units[i] for i in sorted(proposal))) <= limit:
            chosen = proposal
            window_count += 1
        if window_count == 2:
            break
    # Keep at most two contiguous windows, including their original sentence order.
    groups: list[list[int]] = []
    for i in sorted(chosen):
        if not groups or i != groups[-1][-1] + 1:
            groups.append([])
        groups[-1].append(i)
    return "\n\n[…]\n\n".join(" ".join(units[i] for i in group) for group in groups[:2]) if groups else ""


def compact_history(question: str, history, *, max_messages: int, char_budget: int, debug: dict | None = None, analysis=None) -> list[dict[str, str]]:
    analysis = analysis or analyze_central_question(question)
    unresolved = re.search(r"\b(?:ong ay|ba ay|nguoi nay|nguoi do|su kien do|cuoc chien do|chuc vu do|con cai kia|vay thi sao|con|vay)\b", normalize_entity(question))
    standalone = not unresolved and bool(analysis.subject or analysis.event or analysis.comparison_targets)
    if debug is not None:
        debug.update(history_relevance_mode="self_contained" if standalone else "bounded_context",
                     history_turns_considered=len(history or []), history_turns_selected=0)
    if standalone:
        return []
    if not max_messages or not char_budget:
        return []
    # Standalone repeat attempts need no previous generated answer. Follow-ups
    # retain recent dialogue but never source metadata or trace payloads.
    clean = []
    duplicate = False
    for item in history or []:
        role, content = item.get("role"), str(item.get("content") or "").strip()
        if role == "user":
            duplicate = normalize_entity(content) == normalize_entity(question)
        if duplicate or role not in {"user", "assistant"} or not content:
            continue
        if role == "assistant" and (item.get("status") not in {None, "ok", "done"} or content.startswith(("Mình chưa tìm thấy đủ bằng chứng", "Đã tìm thấy tư liệu phù hợp"))):
            continue
        clean.append({"role": role, "content": content})
    selected = []
    remaining = char_budget
    for item in reversed(clean[-max_messages:]):
        content = item["content"]
        if len(content) > remaining:
            content = " ".join(sentence_units(content)[:1])
            if len(content) > remaining:
                continue
        selected.append({**item, "content": content})
        remaining -= len(content)
    if debug is not None:
        debug["history_turns_selected"] = len(selected)
    return list(reversed(selected))
