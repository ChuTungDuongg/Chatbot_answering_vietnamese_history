from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {source} at line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object in {source} at line {line_number}")
            yield value


def read_jsonl(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        rows.append(row)
        if limit is not None and len(rows) >= max(limit, 0):
            break
    return rows


def _temp_path(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(descriptor)
    return Path(name)


def atomic_write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    target = Path(path)
    temporary = _temp_path(target)
    count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def atomic_write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    temporary = _temp_path(target)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


class IncrementalJsonlWriter:
    """Append resumable records while making completed IDs observable."""

    def __init__(self, path: str | Path, *, resume: bool, checkpoint_every: int = 100):
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every must be at least 1")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.completed_ids: set[str] = set()
        if resume and self.path.exists():
            for row in iter_jsonl(self.path):
                row_id = str(row.get("id") or "")
                if row_id:
                    self.completed_ids.add(row_id)
        self._handle = self.path.open("a" if resume else "w", encoding="utf-8", newline="\n")
        self._checkpoint_every = checkpoint_every
        self._since_checkpoint = 0
        self.written = 0
        self.skipped = 0

    def write(self, row: dict[str, Any]) -> bool:
        row_id = str(row.get("id") or "")
        if not row_id:
            raise ValueError("Incremental rows require a non-empty id")
        if row_id in self.completed_ids:
            self.skipped += 1
            return False
        self._handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self.completed_ids.add(row_id)
        self.written += 1
        self._since_checkpoint += 1
        if self._since_checkpoint >= self._checkpoint_every:
            self.checkpoint()
        return True

    def checkpoint(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._since_checkpoint = 0

    def close(self) -> None:
        if not self._handle.closed:
            self.checkpoint()
            self._handle.close()

    def __enter__(self) -> "IncrementalJsonlWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
