from __future__ import annotations

from typing import Any

from app.chat_modes import ChatMode, normalize_chat_mode


class ChatModeRouter:
    """Single app-level dispatch point for all user-facing chat modes."""

    def __init__(self, *, fast: Any, hybrid: Any, agent: Any):
        self._runtimes = {
            ChatMode.FAST: fast,
            ChatMode.HYBRID: hybrid,
            ChatMode.AGENT: agent,
        }
        missing = [mode.value for mode, runtime in self._runtimes.items() if runtime is None]
        if missing:
            raise ValueError(f"Chat mode runtimes are missing: {missing}")

    def runtime_for(self, mode: ChatMode | str) -> Any:
        return self._runtimes[normalize_chat_mode(mode)]

    def chat(self, mode: ChatMode | str, **kwargs: Any) -> dict[str, Any]:
        return self.runtime_for(mode).chat(**kwargs)
