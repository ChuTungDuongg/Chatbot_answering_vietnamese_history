"""Only aliases explicitly stated in model-visible evidence or its entity metadata."""
import re

from app.agents.central_relationships import mentions


_NAME = r"[A-ZÀ-ỸĐ][^\W\d_]*(?:\s+[A-ZÀ-ỸĐ][^\W\d_]*){1,5}"
_PAIR = re.compile(rf"(?<!\w)(?P<name>{_NAME})(?:,\s*|\s+)(?:sau này (?:được biết đến (?:với tên|là)|mang tên)|còn (?:gọi là|có tên là)|later known as)\s+(?P<alias>{_NAME})|(?<!\w)(?P<birth>{_NAME})\s*\((?P<other>{_NAME})\)")


def evidence_aliases(text, title="", metadata=None):
    pairs = []
    for match in _PAIR.finditer(text):
        name, alias = (match.group("name"), match.group("alias")) if match.group("name") else (match.group("birth"), match.group("other"))
        if all(word[0].isupper() for word in (name + " " + alias).split()):
            pairs.append({"name": name, "alias": alias, "origin": "selected_text"})
    mapping = (metadata or {}).get("entity_aliases")
    if isinstance(mapping, dict):
        for name, aliases in mapping.items():
            if isinstance(name, str) and mentions(f"{title} {text}", name) and isinstance(aliases, list):
                pairs.extend({"name": name, "alias": alias, "origin": "selected_entity_metadata"}
                             for alias in aliases if isinstance(alias, str) and alias.strip())
    return pairs
