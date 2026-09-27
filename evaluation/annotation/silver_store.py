"""Transactional SILVER checkpoints, isolated from the human GOLD workspace."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from evaluation.annotation.workspace import dumps, now


def digest(value: Any) -> str:
    return hashlib.sha256(dumps(value).encode("utf-8")).hexdigest()


@contextmanager
def exclusive_run(directory: Path) -> Iterator[None]:
    """OS advisory lock is released after process death; SQLite stays committed."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "silver.run.lock").open("a+b") as handle:
        handle.seek(0)
        handle.write(b"0")
        handle.flush()
        handle.seek(0)
        if __import__("os").name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another SILVER run holds this workspace") from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("Another SILVER run holds this workspace") from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


class SilverStore:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "silver.sqlite3"
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS candidates(
                seq INTEGER PRIMARY KEY, candidate_id TEXT UNIQUE NOT NULL,
                seed_json TEXT NOT NULL, generated_json TEXT,
                evidence_json TEXT, labels_json TEXT, deep_json TEXT,
                annotation_status TEXT NOT NULL DEFAULT 'draft',
                confidence TEXT, deep_status TEXT NOT NULL DEFAULT 'not_selected',
                rejection_reason TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS calls(
                candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id),
                stage TEXT NOT NULL, input_sha256 TEXT NOT NULL,
                response_json TEXT NOT NULL, completed_at TEXT NOT NULL,
                PRIMARY KEY(candidate_id,stage));
            CREATE INDEX IF NOT EXISTS silver_status ON candidates(annotation_status,deep_status);
        """)

    def close(self) -> None:
        self.db.close()

    def metadata(self, key: str) -> Any | None:
        row = self.db.execute("SELECT value_json FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set_metadata(self, key: str, value: Any) -> None:
        encoded = dumps(value)
        with self.lock, self.db:
            existing = self.db.execute("SELECT value_json FROM metadata WHERE key=?", (key,)).fetchone()
            if existing and existing[0] != encoded:
                raise RuntimeError(f"SILVER resume provenance mismatch: {key}")
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES (?,?)", (key, encoded))

    def count(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])

    def insert_seed(self, seq: int, seed: dict[str, Any]) -> dict[str, Any]:
        candidate_id = f"silver_{seq:06d}"
        with self.lock, self.db:
            self.db.execute("""INSERT OR IGNORE INTO candidates
                (seq,candidate_id,seed_json,created_at,updated_at) VALUES (?,?,?,?,?)""",
                (seq, candidate_id, dumps(seed), now(), now()))
        row = self.get(candidate_id)
        if row["seed"] != seed:
            raise RuntimeError("SILVER candidate seed changed on resume")
        return row

    def get(self, candidate_id: str) -> dict[str, Any]:
        with self.lock:
            row = self.db.execute("SELECT * FROM candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
        if row is None:
            raise KeyError(candidate_id)
        result = dict(row)
        for key in ("seed", "generated", "evidence", "labels", "deep"):
            raw = result.pop(f"{key}_json", None)
            result[key] = json.loads(raw) if raw is not None else None
        return result

    def rows(self) -> list[dict[str, Any]]:
        ids = [row[0] for row in self.db.execute("SELECT candidate_id FROM candidates ORDER BY seq")]
        return [self.get(candidate_id) for candidate_id in ids]

    def save(self, candidate_id: str, **fields: Any) -> None:
        allowed = {"generated", "evidence", "labels", "deep", "annotation_status",
                   "confidence", "deep_status", "rejection_reason"}
        if not fields or set(fields) - allowed:
            raise ValueError("Unsupported SILVER checkpoint fields")
        updates = {}
        for key, value in fields.items():
            updates[f"{key}_json" if key in {"generated", "evidence", "labels", "deep"} else key] = (
                dumps(value) if key in {"generated", "evidence", "labels", "deep"} and value is not None else value)
        updates["updated_at"] = now()
        assignments = ",".join(f"{key}=?" for key in updates)
        with self.lock, self.db:
            cursor = self.db.execute(f"UPDATE candidates SET {assignments} WHERE candidate_id=?",
                                     (*updates.values(), candidate_id))
            if cursor.rowcount != 1:
                raise KeyError(candidate_id)

    def cached_call(self, candidate_id: str, stage: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        with self.lock:
            row = self.db.execute("SELECT input_sha256,response_json FROM calls WHERE candidate_id=? AND stage=?",
                                  (candidate_id, stage)).fetchone()
        if row is None:
            return None
        if row[0] != digest(payload):
            raise RuntimeError(f"SILVER resume input changed: {stage}")
        return json.loads(row[1])

    def save_call(self, candidate_id: str, stage: str, payload: dict[str, Any], result: dict[str, Any]) -> None:
        with self.lock, self.db:
            self.db.execute("""INSERT INTO calls VALUES (?,?,?,?,?)""",
                            (candidate_id, stage, digest(payload), dumps(result), now()))

    def call_count(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM calls").fetchone()[0])
