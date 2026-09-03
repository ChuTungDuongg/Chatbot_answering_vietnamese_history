"""Corpus-backed comparison names and deterministic entity-type constraints."""
from __future__ import annotations

import re
from dataclasses import replace

from app.agents.central.semantics import _ascii_fold_vietnamese


# Entity grammar, not a list of historical people/events.
HEADS = (
    ("vương triều", "dynasty"), ("chính quyền", "government"),
    ("hiệp định", "agreement"), ("hội nghị", "conference"),
    ("chiến dịch", "campaign"), ("chiến thắng", "battle"),
    ("chiến tranh", "war"), ("cách mạng", "revolution"),
    ("khởi nghĩa", "uprising"), ("phong trào", "movement"),
    ("triều đại", "dynasty"), ("trận", "battle"),
    ("nhà", "dynasty"), ("triều", "dynasty"),
)


def name_key(text):
    folded = _ascii_fold_vietnamese(text)
    return re.sub(r"\bnam\s+(?=\d{4}\b)", "", " ".join(re.findall(r"[a-z0-9]+", folded)))


def entity_head(text):
    for head, kind in HEADS:
        # Preserve the distinction between the surname Trần and trận.
        if head == "trận" and text.lower().startswith("trần "):
            continue
        if name_key(text).startswith(name_key(head) + " "):
            return head, kind
    return None, None


class EntityTitleIndex:
    """Built once from loaded corpus titles; exact names plus optional date suffix."""
    def __init__(self, titles=()):
        self.names = {}
        for title in sorted(set(titles)):
            key = name_key(title)
            self.names.setdefault(key, set()).add(title)
            self.names.setdefault(re.sub(r"\s+\d{4}$", "", key), set()).add(title)

    def resolve(self, target, expected_type=None):
        choices = [title for title in self.names.get(name_key(target), ())
                   if not expected_type or entity_head(title)[1] == expected_type]
        # Prefer a unique dated identity; never invent a date or pick among homonyms.
        dated = [title for title in choices if re.search(r"\b\d{4}$", title)]
        choices = dated or choices
        return choices[0] if len(choices) == 1 else None


def resolve_comparison_targets(analysis, resolve=None):
    raw = analysis.comparison_targets_raw or analysis.comparison_targets
    if not raw:
        return analysis
    first_head, _ = entity_head(raw[0])
    normalized, canonical, types, events = [], [], [], []
    for position, display in enumerate(raw):
        head, kind = entity_head(display)
        inherited = None
        target = display
        confirmed = resolve(display, kind) if resolve else None
        if position and first_head and not head:
            candidate = first_head[:1].upper() + first_head[1:] + " " + display
            candidate_kind = entity_head(candidate)[1]
            resolved = resolve(candidate, candidate_kind) if resolve else None
            # Ambiguous named-person/event coordination needs corpus confirmation.
            # Short ellipsis is a scoped search hypothesis until evidence confirms it;
            # its bare geographic name can never enter retrieval or sufficiency.
            if resolved or (len(display.split()) <= 2 and candidate_kind in {"agreement", "dynasty"}):
                target, kind, inherited, confirmed = candidate, candidate_kind, first_head, resolved
        normalized.append(target)
        canonical.append(confirmed or target)
        types.append(kind)
        events.append({"raw": display, "display_target": display, "inherited_head": inherited,
                       "normalized": target, "canonical": confirmed or target,
                       "expected_entity_type": kind, "confidence": 1.0 if confirmed else 0.0,
                       "status": "corpus_confirmed" if confirmed else "evidence_confirmation_required"})
    return replace(analysis, comparison_targets_raw=tuple(raw), comparison_targets=tuple(normalized),
                   comparison_canonical_targets=tuple(canonical), comparison_target_entity_types=tuple(types),
                   target_resolution_events=tuple(events))


def canonical_for(analysis, target):
    if target in analysis.comparison_targets and analysis.comparison_canonical_targets:
        return analysis.comparison_canonical_targets[analysis.comparison_targets.index(target)]
    return target


def entity_type_consistent(row, target):
    """A title/lead of the wrong type cannot be rescued by incidental body mentions."""
    _, expected = entity_head(target)
    if not expected:
        return True
    title = str(row.get("title") or row.get("page_title") or "")
    text = str(row.get("text") or row.get("content") or row.get("snippet") or "")
    metadata = row.get("metadata") or {}
    actual = metadata.get("entity_type") or metadata.get("page_type") or entity_head(title)[1]
    compatible = {expected} | ({"battle", "campaign"} if expected in {"battle", "campaign"} else set())
    head, _ = entity_head(target)
    bare_name = re.sub(r"\s+\d{4}$", "", name_key(target)[len(name_key(head)):].strip())
    if expected == "agreement" and name_key(title) == bare_name:
        return False
    if actual in {"city", "town", "stadium", "monument", "song", "football club", "television show"}:
        return False
    lead = name_key(text[:250])
    if re.search(r"\bla (?:mot )?(?:thu do|thanh pho|cau lac bo|chuong trinh|san van dong)\b", lead):
        return False
    if actual in {kind for _, kind in HEADS}:
        # A historical overview may contain separate, substantive paragraphs for
        # both comparison targets. The page title does not erase the second one.
        return actual in compatible or any(name_key(head) + " " in name_key(text)
                                            for head, kind in HEADS if kind in compatible)
    # Untyped excerpts must actually name the typed event, not just its place.
    return any(name_key(head) + " " in name_key(title + " " + text) for head, kind in HEADS if kind in compatible)
