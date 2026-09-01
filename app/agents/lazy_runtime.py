from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any


class LazyRuntime:
    """Thread-safe on-demand model runtime proxy used to avoid cross-mode GPU startup cost."""

    def __init__(self, factory: Callable[[], Any], *, name: str):
        self._factory = factory
        self._name = name
        self._instance: Any | None = None
        self._lock = threading.RLock()
        self._load_elapsed_ms: float = 0.0
        self._load_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._instance is not None

    @property
    def is_ready(self) -> bool:
        return self.loaded and self._load_error is None

    @property
    def load_elapsed_ms(self) -> float:
        return self._load_elapsed_ms

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def get(self) -> Any:
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    started = time.perf_counter()
                    try:
                        self._instance = self._factory()
                        self._load_error = None
                    except Exception as exc:
                        self._load_error = f"{type(exc).__name__}: {exc}"
                        raise
                    finally:
                        self._load_elapsed_ms = (time.perf_counter() - started) * 1000
        return self._instance

    def ensure_ready(self) -> Any:
        return self.get()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.get(), name)

    def __repr__(self) -> str:
        return f"LazyRuntime(name={self._name!r}, loaded={self.loaded})"
