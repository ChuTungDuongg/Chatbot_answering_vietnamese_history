from __future__ import annotations

from enum import Enum
from typing import Any


class ChatMode(str, Enum):
    """Canonical user-facing application modes."""

    HYBRID = "hybrid"
    THREE_LLM = "three_llm"
    CENTRAL = "central"


LEGACY_CHAT_MODE_ALIASES: dict[str, ChatMode] = {
    "fast": ChatMode.HYBRID,
    "hybrid_rag": ChatMode.HYBRID,
    "agentic_rag": ChatMode.THREE_LLM,
    "agent": ChatMode.CENTRAL,
}


def normalize_chat_mode(
    value: Any,
    *,
    default: ChatMode | None = None,
) -> ChatMode:
    """Resolve canonical modes while retaining old API-client compatibility."""

    if value is None or str(value).strip() == "":
        if default is None:
            raise ValueError("Chat mode is required.")
        return default
    if isinstance(value, ChatMode):
        return value
    normalized = str(value).strip().lower()
    if normalized in LEGACY_CHAT_MODE_ALIASES:
        return LEGACY_CHAT_MODE_ALIASES[normalized]
    try:
        return ChatMode(normalized)
    except ValueError as exc:
        choices = ", ".join(mode.value for mode in ChatMode)
        raise ValueError(f"Unsupported chat mode {value!r}. Expected one of: {choices}.") from exc
