from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any


class LazyRuntime:
    """Thread-safe on-demand model runtime proxy used to avoid cross-mode GPU startup cost."""

    def __init__(self, factory: Callable[[], Any], *, name: str):
        self._factory = factory
        self._name = name
        self._instance: Any | None = None
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._instance is not None

    def get(self) -> Any:
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = self._factory()
        return self._instance

    def __getattr__(self, name: str) -> Any:
        return getattr(self.get(), name)

    def __repr__(self) -> str:
        return f"LazyRuntime(name={self._name!r}, loaded={self.loaded})"

