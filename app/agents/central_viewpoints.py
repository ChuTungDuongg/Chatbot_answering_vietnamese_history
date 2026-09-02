"""Span-level quotation/opinion checks, independent of source-level sensitivity flags."""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

from app.agents.central_question import _ascii_fold_vietnamese


_QUOTES = re.compile(r'''“([^”]{2,600})”|"([^"]{2,600})"|‘([^’]{2,600})’|(?<!\w)'([^'\n]{2,600})'(?!\w)''')
_LOADED = re.compile(r"\b(?:lu tay sai|bon de quoc|chien tranh phi nghia|bon ban nuoc|quy du khat mau)\b")
_FIRST_PERSON = re.compile(r"\b(?:chung ta|chung toi|toi|ta nhat dinh)\b")
_SPEECH = re.compile(r"\b(?:noi|tuyen bo|cho rang|nhan dinh|khang dinh|viet rang|said|stated|argued)\b")
_NAME = r"[A-ZÀ-ỸĐ][^\W\d_]*(?:\s+[A-ZÀ-ỸĐ][^\W\d_]*){0,4}"
_SPEAKER = re.compile(rf"(?<!\w)(?P<speaker>{_NAME})\s+(?:(?:đã|từng|cũng)\s+)?(?:nói|tuyên bố|cho rằng|nhận định|khẳng định|viết rằng|said|stated|argued)\b")
_SPEAKER_LABEL = re.compile(rf"^\s*(?P<speaker>{_NAME})\s*:\s*$")
_FUNCTION_WORDS = set("va la cua cac nhung mot voi trong cho o tu da duoc nay do su ve co de khi thi ma den vao nhu cung theo".split())
_NEGATION = {"khong", "chua", "chang"}


def _fold(text: str) -> str:
    return _ascii_fold_vietnamese(unicodedata.normalize("NFC", text))


def _words(text: str) -> list[str]:
    folded = re.sub(r"\bvnch\b", "viet nam cong hoa", _fold(text))
    return re.findall(r"[a-z0-9]+", folded)


def _speaker_hint(context: str) -> str | None:
    # The Unicode range À-Ỹ also contains lowercase letters. Check actual case
    # and whole words so "văn tuyên bố" cannot produce the supposed speaker "ăn".
    matches = [m for m in _SPEAKER.finditer(context) if all(word[0].isupper() for word in m.group("speaker").split())]
    label = _SPEAKER_LABEL.fullmatch(context)
    if label and not all(word[0].isupper() for word in label.group("speaker").split()):
        label = None
    return matches[-1].group("speaker") if matches else label.group("speaker") if label else None


def annotate_viewpoints(text: str) -> list[dict[str, Any]]:
    """Offsets refer to the supplied text. A quotation can be a term, not a viewpoint.

    First-person language is annotated only within a quote or explicit speech
    context. Generic historical vocabulary is deliberately absent from _LOADED.
    """
    annotations: list[dict[str, Any]] = []

    def add(kind: str, start: int, end: int, reason: str, hint: str | None = None, required: bool = True):
        annotations.append({"type": kind, "start": start, "end": end,
                            "text": text[start:end], "reason": reason,
                            "attribution_hint": hint, "requires_attribution": required})

    for quote in _QUOTES.finditer(text):
        group = next(i for i in range(1, 5) if quote.group(i) is not None)
        start, end = quote.span(group)
        preceding = re.split(r"[.!?]\s+|\n", text[max(0, quote.start() - 180):quote.start()])[-1]
        hint = _speaker_hint(preceding)
        quoted = _fold(text[start:end])
        speech = bool(hint or _SPEECH.search(_fold(preceding)))
        required = speech or bool(_LOADED.search(quoted) or _FIRST_PERSON.search(quoted))
        add("direct_quote", start, end, "attributed_speech" if speech else "quoted_text", hint, required)
        if _FIRST_PERSON.search(quoted):
            add("first_person", start, end, "first_person_in_quote", hint)

    # Folding an NFC string is length-preserving. Map matches back to the original
    # text by sentence, avoiding offset assumptions for decomposed Unicode input.
    for sentence in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", text):
        original = sentence.group()
        folded = _fold(original)
        if _LOADED.search(folded):
            add("evaluative_language", sentence.start(), sentence.end(), "high_confidence_loaded_language", _speaker_hint(original))
        speaker = _speaker_hint(original)
        if speaker and not any(sentence.start() <= a["start"] < sentence.end() for a in annotations if a["type"] == "direct_quote"):
            speech = _SPEECH.search(folded)
            if speech:
                # Capture the proposition after the reporting verb, not surrounding
                # neutral facts or the speaker's name.
                nfc = unicodedata.normalize("NFC", original)
                proposition = nfc[speech.end():].lstrip(" :,")
                start = text.find(proposition, sentence.start(), sentence.end())
                if start >= 0 and len(_words(proposition)) >= 5:
                    add("attributed_opinion", start, sentence.end(), "explicit_reporting_context", speaker)
                if _FIRST_PERSON.search(folded):
                    add("first_person", sentence.start(), sentence.end(), "first_person_in_speech", speaker)
    return sorted(annotations, key=lambda a: (a["start"], a["end"], a["type"]))


def _direct_quote_match(claim: str, excerpt: str) -> tuple[float, str] | None:
    """Require quoted overlap or a distinctive contiguous copy, never a topic match."""
    # Acronym expansion is useful for opinion paraphrases, but must not inflate
    # contiguous-copy length (VNCH is one copied word, not four).
    literal_words = lambda value: re.findall(r"[a-z0-9]+", _fold(value))
    claim_words, quote_words = literal_words(claim), literal_words(excerpt)
    if len(quote_words) < 5 or (set(claim_words) & _NEGATION) != (set(quote_words) & _NEGATION):
        return None
    for quote in _QUOTES.finditer(claim):
        quoted = literal_words(next(value for value in quote.groups() if value is not None))
        match = SequenceMatcher(None, quoted, quote_words, autojunk=False).find_longest_match()
        if match.size >= 5 and match.size / max(1, len(quoted)) >= .9 and len(set(quoted[match.a:match.a + match.size]) - _FUNCTION_WORDS) >= 3:
            return match.size / len(quoted), "answer_quotation_matches_sensitive_span"
    match = SequenceMatcher(None, claim_words, quote_words, autojunk=False).find_longest_match()
    meaningful = set(claim_words[match.a:match.a + match.size]) - _FUNCTION_WORDS
    if match.size >= 8 and len(meaningful) >= 4:
        return match.size / min(len(claim_words), len(quote_words)), "contiguous_sensitive_span_copy"
    # A complete short first-person utterance is distinctive even without marks:
    # e.g. "chúng ta nhất định thắng lợi". This exception cannot admit topic labels.
    if 5 <= len(quote_words) < 8 and match.size == len(quote_words) and _FIRST_PERSON.search(_fold(excerpt)) and len(meaningful) >= 3:
        return 1.0, "complete_short_first_person_quote"
    return None


def _opinion_match(claim: str, excerpt: str) -> float:
    claim_words, quote_words = _words(claim), _words(excerpt)
    if min(len(claim_words), len(quote_words)) < 6:
        return 0.0
    # Compare bounded windows so a copied opinion cannot hide in a long paragraph.
    for size in range(max(5, len(quote_words) - 4), min(len(claim_words), len(quote_words) + 4) + 1):
        for start in range(len(claim_words) - size + 1):
            window = claim_words[start:start + size]
            # Preserve negation: a neutral or opposite proposition is not a copied opinion.
            if (set(window) & _NEGATION) != (set(quote_words) & _NEGATION):
                continue
            matcher = SequenceMatcher(None, window, quote_words, autojunk=False)
            if matcher.ratio() >= .86 and matcher.find_longest_match().size >= 3 and len((set(window) & set(quote_words)) - _FUNCTION_WORDS) >= 4:
                return matcher.ratio()
    return 0.0


def _concrete_span(source: Any, annotation: dict[str, Any]) -> str | None:
    span = annotation.get("text")
    if not isinstance(span, str) or not span.strip() or _words(span) == _words(source.title):
        return None
    start, end = annotation.get("start"), annotation.get("end")
    if isinstance(start, int) and isinstance(end, int):
        return span if 0 <= start < end <= len(source.text) and source.text[start:end] == span else None
    return span if span in source.text else None


def _attributed(sentence: str, hint: str | None) -> bool:
    folded = _fold(sentence)
    if re.search(r"\btheo (?:nguon|tac gia|nhan dinh|quan diem|phat bieu|loi|mot so nha|cac nha)\b", folded):
        return True
    if hint and f" {' '.join(_words(hint))} " in f" {' '.join(_words(sentence))} " and re.search(r"\b(?:theo|noi|tuyen bo|cho rang|nhan dinh|quan diem|loi)\b", folded):
        return True
    return bool(re.search(r"\b(?:nguon|tac gia|nha phe binh|nha su hoc).{0,90}\b(?:cho rang|tuyen bo|nhan dinh|goi|viet|noi)\b", folded)
                or re.search(r"\b(?:la|trich) (?:loi tuyen bo|loi ke|khau hieu|nhan dinh|quan diem)\b", folded))


def viewpoint_attribution_issues(paragraph: str, cited_sources: list[Any]) -> list[dict[str, Any]]:
    """Only actual answer claims matter; never inspect viewpoint_sensitive booleans."""
    issues = []
    source_annotations = [(source.alias, annotation, span) for source in cited_sources
                          for annotation in source.viewpoint_annotations if annotation.get("requires_attribution")
                          if (span := _concrete_span(source, annotation))]
    for sentence in re.split(r"(?<=[.!?])\s+|\n", paragraph):
        if not sentence.strip():
            continue
        matched = []
        for alias, annotation, span in source_annotations:
            direct = _direct_quote_match(sentence, span) if annotation["type"] == "direct_quote" else None
            if direct:
                matched.append((alias, annotation, "direct_quote", direct[0], direct[1]))
            elif (annotation.get("attribution_hint") or annotation.get("reason") == "attributed_speech" or annotation["type"] in {"attributed_opinion", "evaluative_language"}) and (score := _opinion_match(sentence, span)):
                matched.append((alias, annotation, "viewpoint_paraphrase", score, "close_match_to_specific_attributed_proposition"))
        # High-confidence loaded language or first-person speech is also unsafe
        # when invented by the answer rather than copied from a supplied source.
        # A newly invented quotation is not evidence of copying a source quote.
        # Retain the separate loaded-language / first-person speech guard.
        own = [a for a in annotate_viewpoints(sentence) if a["requires_attribution"] and a["type"] in {"evaluative_language", "first_person"}]
        matched.sort(key=lambda item: item[2] != "direct_quote")
        for alias, annotation, kind, score, reason in matched + [(None, a, a["type"], None, "answer_contains_sensitive_language") for a in own]:
            if _attributed(sentence, annotation.get("attribution_hint")):
                continue
            # An answer's quotation may inherit a known speaker from the matching source.
            if alias is None and any(_attributed(sentence, a.get("attribution_hint")) for _, a, _, _, _ in matched):
                continue
            issues.append({"source_alias": alias, "type": kind,
                           "answer_claim": sentence.strip(), "matched_sensitive_span": annotation["text"] if alias else None,
                           "overlap_score": round(score, 4) if score is not None else None,
                           "attribution_hint": annotation.get("attribution_hint"), "reason": reason,
                           # Retain existing debug/repair consumers while providing explicit fields.
                           "claim": sentence.strip(), "source_excerpt": annotation["text"] if alias else None})
    # Multiple annotations can describe the same claim; report it once.
    by_claim = {}
    for issue in issues:
        by_claim.setdefault(issue["answer_claim"], issue)
    return list(by_claim.values())
