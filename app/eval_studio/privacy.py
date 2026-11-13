"""Reject obvious credentials in metadata and backup artifacts."""

import re

SECRET_PATTERN = re.compile(
    r"(?:sk-(?:proj-)?[A-Za-z0-9_-]{16,}|Bearer\s+[A-Za-z0-9._~-]{16,}|(?:api[_-]?key|session[_-]?secret|password)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9._~-]{12,}|https?://[^/\s:@]+:[^/\s@]+@)",
    re.IGNORECASE,
)


def reject_secrets(value):
    if isinstance(value, str) and SECRET_PATTERN.search(value):
        raise ValueError("Potential credential in evaluation data; remove it")
    if isinstance(value, dict):
        for child in value.values():
            reject_secrets(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            reject_secrets(child)
