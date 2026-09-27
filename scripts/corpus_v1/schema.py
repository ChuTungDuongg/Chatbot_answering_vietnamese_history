"""Stable identities and serializable V1 row contracts."""

import hashlib
import json


def stable_id(prefix: str, *parts: object) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return prefix + "_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
