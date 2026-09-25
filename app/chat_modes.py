"""The two supported inference modes."""

from enum import Enum
from typing import Any


class ChatMode(str, Enum):
    HYBRID = "hybrid"
    CENTRAL = "central"


def normalize_chat_mode(value: Any, *, default: ChatMode | None = None) -> ChatMode:
    if value is None or str(value).strip() == "":
        if default is None:
            raise ValueError("Chat mode is required.")
        return default
    if isinstance(value, ChatMode):
        return value
    try:
        return ChatMode(str(value).strip().lower())
    except ValueError as exc:
        raise ValueError("Unsupported chat mode. Expected 'hybrid' or 'central'.") from exc
