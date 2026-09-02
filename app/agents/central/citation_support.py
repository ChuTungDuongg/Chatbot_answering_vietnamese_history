"""Conservative paragraph support and summary classification, below citation validation."""
from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import groupby

from app.agents.central.semantics import BRACKET_RE, _ascii_fold_vietnamese, normalized_actor_text
from app.agents.central.compaction import sentence_units
from app.agents.central.grounding import grounding_risks

_STOP = set("va la cua cac nhung mot voi trong cho o tu da duoc nhieu nay do su ve co de khi thi ma den vao nhu cung theo".split())


@dataclass(frozen=True)
class Paragraph:
    key: str
    position: int
    text: str
    targets: frozenset[str]


def _words(text):
    return normalized_actor_text(text).split()


def sentence_support(claim, source) -> float:
    """Measured lexical/phrase support, not an entailment guarantee.

    Every sentence needs a distinctive match; new entities, numbers and changed
    negation cannot be rescued by common topic words or the source's rank.
    """
    risks = grounding_risks(claim, "", [source])
    if any(risks.values()):
        return 0.0
    words = _words(claim)
    supplied_words = set(_words(source.title + " " + source.text))
    # Existing grounding checks recognize common historical names. Arbitrary
    # capitalized entities must also be present before lexical alignment is safe.
    name_runs = (" ".join(tokens) for capitalized, tokens in groupby(
        re.findall(r"[^\W\d_]+", claim), key=lambda token: token[0].isupper()) if capitalized)
    names = {word for name in name_runs for word in _words(name) if word not in _STOP}
    if not names <= supplied_words:
        return 0.0
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


def supported_aliases(paragraph, packet, *, threshold, margin, analysis=None):
    # Table citations belong in individual cells; don't append a row-wide guess.
    if paragraph.text.startswith("|"):
        return [], 0.0
    candidates = [source for source in packet if not paragraph.targets or set(source.comparison_targets) & paragraph.targets]
    prose = BRACKET_RE.sub(lambda match: match.group(1) if match.group(1).isdigit() else "", paragraph.text)
    prose = re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", prose)
    clauses = [part for sentence in sentence_units(prose) for part in re.split(r";\s*", sentence) if part.strip()]
    aliases, confidences = [], []
    for clause in clauses:
        scoped = candidates
        if analysis and analysis.event_type:
            from app.agents.central.events import relational_target_features, event_matches
            from app.agents.central.depth import actor_scope
            # A joint event claim cannot borrow a similar sentence from A–C.
            joint_claim = event_matches(clause, analysis.event) or len(actor_scope(clause, analysis.actors)) > 1
            if joint_claim:
                scoped = [source for source in candidates if relational_target_features(
                    {"title": source.title, "text": source.text}, analysis)["direct_target_coverage"]]
            else:
                direct = [source for source in candidates if relational_target_features(
                    {"title": source.title, "text": source.text}, analysis)["direct_target_coverage"]
                          and sentence_support(clause, source) >= threshold]
                if direct:
                    scoped = direct
        scores = sorted(((sentence_support(clause, source), source.alias) for source in scoped), reverse=True)
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


def supported_synthesis_summary(paragraph, preceding, packet, analysis=None):
    """Only short restatements of immediately preceding, independently supported claims.

    Ordered clause matches prevent a bag of familiar entities from licensing a
    new relation, year, negation or causal predicate. Uncertain paraphrases remain
    new claims and can take the normal citation-alignment path.
    """
    cue = re.match(r"^(?:tom lai|nhu vay|vi vay|do do|tu do|nhin chung)\b\s*[, :]?\s*", _ascii_fold_vietnamese(paragraph))
    if not cue or len(paragraph.split()) > 90 or not preceding:
        return False
    by_alias = {source.alias: source for source in packet}
    supported_claims = []
    for prior in reversed(preceding):
        aliases = [alias for alias in BRACKET_RE.findall(prior) if alias in by_alias]
        if not aliases or prior.lstrip().startswith(("#", "|")):
            break
        sources = [by_alias[alias] for alias in aliases]
        p = Paragraph("prior", 0, prior, frozenset())
        support, _ = supported_aliases(p, sources, threshold=.88, margin=0, analysis=analysis)
        if not support:
            break
        supported_claims.extend(sentence_units(BRACKET_RE.sub("", prior)))
    if not supported_claims:
        return False
    # Normalization removes punctuation, so locate the prefix separately in the
    # original prose to preserve clause/sentence boundaries.
    prose = paragraph[cue.end():]
    clauses = [c for c in re.split(r"[.!?;]|\s+và\s+", prose, flags=re.I) if c.strip()]
    for clause in clauses:
        words = [w for w in _words(clause) if w not in _STOP]
        if len(words) < 3:
            return False
        def ordered_match(prior):
            available = iter(w for w in _words(prior) if w not in _STOP)
            return all(any(candidate == word for candidate in available) for word in words)
        if not any(ordered_match(prior) for prior in supported_claims):
            return False
    return bool(clauses)
