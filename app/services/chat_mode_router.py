"""Dispatch between the two explicitly configured inference runtimes."""

from typing import Any

from app.chat_modes import ChatMode, normalize_chat_mode


class ChatModeRouter:
    def __init__(self, *, hybrid: Any, central: Any):
        self._runtimes = {ChatMode.HYBRID: hybrid, ChatMode.CENTRAL: central}

    def runtime_for(self, mode: ChatMode | str) -> Any:
        canonical = normalize_chat_mode(mode)
        runtime = self._runtimes[canonical]
        if runtime is None:
            raise RuntimeError(f"Chat mode {canonical.value!r} is not enabled in this deployment.")
        return runtime
