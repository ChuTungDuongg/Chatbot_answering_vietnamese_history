"""Bounded external evidence; protocol output can never become instructions."""

import hashlib
import json
import re
from itertools import islice
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

from app.mcp.schemas import MCPToolError

SECRET_KEY = re.compile(r"api.?key|authorization|cookie|password|secret|credential|token|headers|environment", re.I)


def redact(text, secrets=()):
    text = str(text)
    for value in sorted(secrets, key=len, reverse=True): text = text.replace(value, "[redacted]")
    return re.sub(r"(?i)(api[_ -]?key|bearer|token|password|secret)\s*[:= ]\s*\S+", r"\1=[redacted]", text)


def clipped_redact(value, limit, secrets=()):
    # Keep enough lookahead to redact a secret that crosses the clipping boundary.
    overlap = max((len(secret) for secret in secrets), default=0)
    return redact(str(value)[:limit + overlap], secrets)[:limit]


def bounded(value, budget, secrets, depth=0, remaining=None):
    remaining = remaining if remaining is not None else [budget, 128]
    if depth > 4 or remaining[0] <= 0 or remaining[1] <= 0: return "[bounded]"
    remaining[1] -= 1
    if isinstance(value, dict):
        result = {}
        for key, item in islice(value.items(), 24):
            if remaining[0] <= 0 or remaining[1] <= 0: break
            if SECRET_KEY.search(str(key)): continue
            safe_key = clipped_redact(key, 80, secrets)
            remaining[0] -= len(safe_key)
            result[safe_key] = bounded(item, min(budget, 1000), secrets, depth+1, remaining)
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value[:16]:
            if remaining[0] <= 0 or remaining[1] <= 0: break
            result.append(bounded(item, min(budget, 1000), secrets, depth+1, remaining))
        return result
    if isinstance(value, str):
        text = clipped_redact(value, min(budget, remaining[0]), secrets)
        remaining[0] -= len(text)
        return text
    if value is None or isinstance(value, (int, float, bool)): return value
    return "[unsupported content]"


def public_url(value, secrets):
    if not isinstance(value, str) or len(value) > 2000: return None
    if any(secret in value for secret in secrets): return None
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password: return None
    query = [(k,v) for k,v in parse_qsl(parsed.query) if not SECRET_KEY.search(k)]
    return urlunparse(parsed._replace(query=urlencode(query)))


def normalize(result, *, server, tool, label, limit, secrets=()):
    if getattr(result, "is_error", False): raise MCPToolError("MCP tool không hoàn tất thành công.")
    structured = getattr(result, "structured_content", None)
    parts = []
    truncated = False
    link = None
    for block in (getattr(result, "content", None) or [])[:16]:
        kind = getattr(block, "type", None)
        if kind == "text":
            value = str(block.text)
            truncated |= len(value) > limit
            parts.append(clipped_redact(value, limit, secrets))
        elif kind == "resource_link":
            link = public_url(getattr(block, "uri", None), secrets)
            parts.append(redact(str(getattr(block, "description", None) or getattr(block, "name", "External source"))[:limit], secrets))
        elif kind == "resource":
            resource = getattr(block, "resource", None)
            value = getattr(resource, "text", None)
            if isinstance(value, str):
                truncated |= len(value) > limit
                parts.append(clipped_redact(value, limit, secrets))
                link = public_url(getattr(resource, "uri", None), secrets) or link
    if structured is not None:
        remaining = [limit, 128]
        parts.append(json.dumps(bounded(structured, limit, secrets, remaining=remaining), ensure_ascii=False))
        truncated |= remaining[0] <= 0 or remaining[1] <= 0
    if not parts: raise MCPToolError("MCP tool trả nội dung không hỗ trợ hoặc không có dữ liệu.")
    text = "\n".join(parts)
    truncated |= len(text) > limit
    text = text[:limit]
    if truncated: text = text[:limit-15] + "\n[truncated]"
    source = structured if isinstance(structured, dict) else {}
    url = public_url(source.get("url"), secrets) or link
    identifier = clipped_redact(source.get("source_id", ""), 160, secrets) or None
    digest = hashlib.sha256(f"{server}\0{tool}\0{text}".encode()).hexdigest()[:16]
    return [{"chunk_id": f"mcp:{server}:{digest}", "source_id": f"mcp:{server}:{identifier}" if identifier else None,
             "source_kind": "mcp", "title": f"{label} · {tool[:64]}", "text": text, "url": url,
             "metadata": {"provider": "mcp", "server": server, "tool": tool,
                          "remote_source_id": identifier, "truncated": truncated, "untrusted": True}}]
