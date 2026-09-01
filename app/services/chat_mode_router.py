from __future__ import annotations

from typing import Any

from app.chat_modes import ChatMode, normalize_chat_mode


class ChatModeRouter:
    """Single app-level dispatch point for all user-facing chat modes."""

    def __init__(self, *, hybrid: Any, three_llm: Any, central: Any):
        self._runtimes = {
            ChatMode.HYBRID: hybrid,
            ChatMode.THREE_LLM: three_llm,
            ChatMode.CENTRAL: central,
        }

    def runtime_for(self, mode: ChatMode | str) -> Any:
        canonical = normalize_chat_mode(mode)
        runtime = self._runtimes[canonical]
        if runtime is None:
            raise RuntimeError(f"Chat mode {canonical.value!r} is not enabled in this deployment.")
        return runtime

    def chat(self, mode: ChatMode | str, **kwargs: Any) -> dict[str, Any]:
        return self.runtime_for(mode).chat(**kwargs)
