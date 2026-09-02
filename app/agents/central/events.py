"""Bounded relational-event grammar and excerpt-based target matching.

Names are grammatical captures, never keys in a historical answer table.
"""
from __future__ import annotations

import re
import unicodedata

from app.agents.central.semantics import ACTOR_ALIASES, _ascii_fold_vietnamese, normalized_actor_text


# Specific actions precede their optional controller (agree/decide to act).
EVENTS = (
    (r"binh thuong hoa quan he(?: ngoai giao)?", "bình thường hóa quan hệ", "bilateral_relation_change"),
    (r"thiet lap quan he(?: ngoai giao)?", "thiết lập quan hệ ngoại giao", "bilateral_relation_change"),
    (r"cat dut quan he(?: ngoai giao)?", "cắt đứt quan hệ", "bilateral_relation_change"),
    (r"tuyen bo doc lap", "tuyên bố độc lập", "political_state_change"),
    (r"ky (?:ket )?hiep dinh", "ký hiệp định", "agreement"),
    (r"ky (?:ket )?hiep uoc", "ký hiệp ước", "agreement"),
    (r"ky (?:ket )?thoa thuan", "ký thỏa thuận", "agreement"),
    (r"dam phan", "đàm phán", "negotiation"),
    (r"rut quan", "rút quân", "military_state_change"),
    (r"tham chien", "tham chiến", "military_state_change"),
    (r"lien minh", "liên minh", "bilateral_relation_change"),
    (r"cai cach", "cải cách", "political_state_change"),
    (r"thong nhat", "thống nhất", "political_state_change"),
    (r"chia cat", "chia cắt", "political_state_change"),
    (r"cong nhan", "công nhận", "bilateral_relation_change"),
    (r"cham dut", "chấm dứt", "state_change"),
    (r"rut khoi", "rút khỏi", "state_change"),
    (r"tham gia", "tham gia", "participation"),
    (r"doi dau", "đối đầu", "bilateral_relation_change"),
    (r"tuyen bo", "tuyên bố", "declaration"),
    (r"dong y", "đồng ý", "agreement"),
    (r"thiet lap", "thiết lập", "state_change"),
    (r"ky", "ký", "agreement"),
)
_PREFIX = re.compile(r"^(?:(?:vi sao|tai sao|nguyen nhan(?: nao)?(?: khien| cua| dan den)?|ly do(?: nao)?(?: khien| cua)?|su|viec|giua|ca)\s+)+")
_CONTROL = re.compile(r"\s+(?:(?:da|lai|cung|deu|dong y|quyet dinh|tien hanh|bat dau|tien toi)\s*)+$")
_ORG = re.compile(r"^(?:chinh phu|dang|nha nuoc|vuong quoc|de quoc|cong hoa|lien bang|trieu dai|nha|mat tran|lien minh)\s+")
_CONNECTORS = {"de", "von", "van", "of", "the"}
_ORG_WORDS = set("dân chủ cộng sản lao động quốc gia nhân dân cách mạng giải phóng thống nhất tự do hòa bình".split())


def named_actor(value):
    """Require a proper-name core, allowing generic organization prefixes."""
    value = value.strip(" ,;:.?!–—-")
    if any(re.fullmatch(pattern, _ascii_fold_vietnamese(value)) for pattern, _ in ACTOR_ALIASES):
        return value
    prefix = _ORG.match(_ascii_fold_vietnamese(value))
    core = value[prefix.end():] if prefix else value
    words = core.split()
    if not 1 <= len(value.split()) <= 9 or not words:
        return None
    if not all((word[0].isupper() and all(c.isalpha() or c in "-'’" for c in word))
               or word.casefold() in _CONNECTORS or prefix and word.casefold() in _ORG_WORDS for word in words):
        return None
    if not any(word[0].isupper() for word in words):
        return None
    return value


def coordinated_actors(value):
    value = value.strip()
    prefix = _PREFIX.match(_ascii_fold_vietnamese(value))
    if prefix:
        value = value[prefix.end():]
    control = _CONTROL.search(_ascii_fold_vietnamese(value))
    if control:
        value = value[:control.start()]
    # Split by folded offsets so accent variants retain their original names.
    boundaries = list(re.finditer(r"\s+(?:va|voi|lan|cung)\s+|\s*,\s*", _ascii_fold_vietnamese(value)))
    pieces, start = [], 0
    for boundary in boundaries:
        pieces.append(value[start:boundary.start()])
        start = boundary.end()
    pieces.append(value[start:])
    names = [named_actor(piece) for piece in pieces]
    # All conjuncts must be entities. Do not rescue half of an ordinary noun list.
    if not names or any(name is None for name in names):
        return ()
    unique = {}
    for name in names:
        unique.setdefault(normalized_actor_text(name), name)
    return tuple(unique.values())


def extract_relational_event(question):
    text = unicodedata.normalize("NFC", " ".join(question.split())).strip(" .?!")
    folded = _ascii_fold_vietnamese(text)
    nominal = {}
    candidates = ((pattern, canonical, kind, match) for pattern, canonical, kind in EVENTS
                  for match in re.finditer(rf"\b(?:{pattern})\b", folded))
    for pattern, canonical, kind, match in candidates:
        if pattern == "ky" and text[match.start():match.end()].casefold() not in {"ký", "ky"}:
            continue
        before, after = text[:match.start()].strip(), text[match.end():].strip()
        actors = coordinated_actors(before)
        # Nominal form: event giữa A và B; or subject performs event với B.
        scope = re.search(r"\b(?:giua|voi|cua)\s+(?P<actors>.+?)(?:\s+(?:nham|de|vao|nam|vi|do|ve|nhu the nao|la gi)\b|$)", _ascii_fold_vietnamese(after))
        if scope:
            other = coordinated_actors(after[slice(*scope.span("actors"))])
            actors = tuple(dict.fromkeys((*actors, *other)))
        # Recognition of a predicate alone must not promote ordinary conjuncts.
        if not actors and not re.search(r"\b(?:su|viec|nguyen nhan|vi sao|tai sao|ly do)\b", folded[:match.start()]):
            continue
        event = canonical
        # Retain the action's object (treaty name, policy, institution), not dates,
        # participant clauses or an explanation of why it happened.
        obj = re.split(r"\s+(?:giữa|với|của|nhằm|để|vào|năm|vì|do|về|như thế nào|là gì)\b", " " + after, maxsplit=1, flags=re.I)[0].strip()
        if obj and len(obj.split()) <= 10 and canonical not in {"bình thường hóa quan hệ", "thiết lập quan hệ ngoại giao", "cắt đứt quan hệ"}:
            event += " " + obj
        result = {"event": event, "event_type": kind, "actors": actors,
                  "canonical_target": " ".join((event, *actors)).strip()}
        if actors:
            return result
        # Prefer a predicate with a parsed subject over a possible noun in that
        # subject. Keep a nominal event only when no such predicate is found.
        if not nominal:
            nominal = result
    return nominal


def event_matches(text, event, *, nominal=False):
    """Match the same action/object despite inflection, accents and punctuation."""
    folded = normalized_actor_text(text)
    target = normalized_actor_text(event)
    # Optional diplomatic modifier carries no change of action identity.
    folded = re.sub(r"\bquan he ngoai giao\b", "quan he", folded)
    target = re.sub(r"\bquan he ngoai giao\b", "quan he", target)
    folded = re.sub(r"\bky ket\b", "ky", folded)
    if nominal and re.match(r"^ky (?:hiep dinh|hiep uoc|thoa thuan)\b", target):
        target = target.removeprefix("ky ")
    return bool(target and f" {target} " in f" {folded} ")


def relational_target_features(row, analysis):
    from app.agents.central.depth import actor_scope
    title = str(row.get("title") or row.get("page_title") or row.get("source_title") or "")
    metadata = row.get("metadata") or {}
    scope_title = " ".join([title, *(str(metadata.get(key) or "") for key in ("event", "canonical_event", "canonical_title"))])
    text = str(row.get("text") or row.get("content") or row.get("snippet") or "")
    required = set(analysis.actors)
    title_actors = set(actor_scope(scope_title, analysis.actors))
    text_actors = set(actor_scope(text, analysis.actors))
    title_event = event_matches(scope_title, analysis.event, nominal=True)
    title_joint = title_event and required <= title_actors
    # A page-wide bag of entities is not evidence of a joint relation.
    units = [s for s in re.split(r"[.!?;\n]+", text) if s.strip()]
    joint = any(event_matches(s, analysis.event) and required <= set(actor_scope(s, analysis.actors)) for s in units)
    body_event = any(event_matches(s, analysis.event) for s in units)
    explicit_other = False
    if title_joint and not joint:
        for sentence in units:
            relation = extract_relational_event(sentence)
            named = {normalized_actor_text(actor) for actor in relation.get("actors", ())}
            if len(named) >= 2 and not {normalized_actor_text(actor) for actor in required} <= named:
                explicit_other = True
                break
    title_joint = title_joint and not explicit_other
    direct = bool(title_joint or joint) if required else title_event or body_event
    covered = title_actors | text_actors if title_joint else text_actors
    event_present = title_event or body_event
    # Primary/direct = 3, same action with only some actors = 1, background = 0.
    tier = 3 if direct else 1 if event_present and (covered or not required) else 0
    return {"direct_target_coverage": direct, "target_consistency_score": tier,
            "relational_source_title": " ".join(scope_title.split()),
            "event_target_match": event_present, "actor_scope": sorted(covered, key=analysis.actors.index),
            "overview_anchor": bool(title_joint), "target_consistent": bool(tier),
            "canonical_target_consistent": direct if len(required) > 1 else bool(tier),
            "entity_filter_reason": None, "target_match_uncertain": not bool(tier)}
