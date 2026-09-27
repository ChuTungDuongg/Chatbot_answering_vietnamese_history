"""Small ID/byte-offset sidecar for safe random access to the immutable corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any


class CorpusLookup:
    def __init__(self, corpus: Path, database: Path):
        self.corpus = corpus.resolve()
        self.database = database.resolve()
        if not self.database.is_file():
            raise FileNotFoundError(f"Corpus lookup missing: {self.database}; run corpus index first")
        self.connection = sqlite3.connect(self.database, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        metadata = dict(self.connection.execute("SELECT key,value FROM metadata"))
        stat = self.corpus.stat()
        if (metadata.get("corpus_bytes") != str(stat.st_size) or
                metadata.get("corpus_mtime_ns") != str(stat.st_mtime_ns)):
            self.connection.close()
            raise RuntimeError("Corpus changed since lookup build; rebuild lookup explicitly")
        self.corpus_sha256 = metadata["corpus_sha256"]
        self.count = int(metadata["count"])

    @classmethod
    def build(cls, corpus: Path, database: Path, *, expected_sha: str | None = None,
              expected_count: int | None = None) -> "CorpusLookup":
        corpus, database = corpus.resolve(), database.resolve()
        database.parent.mkdir(parents=True, exist_ok=True)
        if database.exists() or database.with_name(database.name + ".partial").exists():
            raise FileExistsError("Corpus lookup or partial lookup already exists; inspect before rebuilding")
        temporary = database.with_name(database.name + ".partial")
        connection = sqlite3.connect(temporary)
        try:
            connection.executescript("""
                CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
                    document_id TEXT NOT NULL, chunk_index INTEGER NOT NULL,
                    row_index INTEGER NOT NULL, byte_offset INTEGER NOT NULL, byte_length INTEGER NOT NULL);
                CREATE INDEX chunks_source ON chunks(source_id);
                CREATE INDEX chunks_document ON chunks(document_id, chunk_index);
            """)
            digest = hashlib.sha256()
            count = 0
            with corpus.open("rb") as handle:
                while raw := handle.readline():
                    offset = handle.tell() - len(raw)
                    digest.update(raw)
                    if not raw.strip():
                        continue
                    row = json.loads(raw)
                    connection.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
                                       (row["chunk_id"], row["source_id"], row["document_id"],
                                        int(row["chunk_index"]), count, offset, len(raw)))
                    count += 1
                    if count % 10000 == 0:
                        connection.commit()
            sha = digest.hexdigest()
            if expected_sha is not None and sha != expected_sha:
                raise RuntimeError("Corpus SHA differs from expected V1 identity")
            if expected_count is not None and count != expected_count:
                raise RuntimeError("Corpus count differs from expected V1 identity")
            stat = corpus.stat()
            connection.executemany("INSERT INTO metadata VALUES (?,?)", (
                ("corpus_sha256", sha), ("corpus_bytes", str(stat.st_size)),
                ("corpus_mtime_ns", str(stat.st_mtime_ns)), ("count", str(count))))
            connection.commit()
        finally:
            connection.close()
        os.replace(temporary, database)
        return cls(corpus, database)

    def close(self) -> None:
        self.connection.close()

    def has_chunk(self, chunk_id: str) -> bool:
        return self.connection.execute("SELECT 1 FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone() is not None

    def has_source(self, source_id: str) -> bool:
        return self.connection.execute("SELECT 1 FROM chunks WHERE source_id=? LIMIT 1", (source_id,)).fetchone() is not None

    def source_for_chunk(self, chunk_id: str) -> str | None:
        row = self.connection.execute("SELECT source_id FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        return row[0] if row else None

    def _read(self, row: sqlite3.Row) -> dict[str, Any]:
        with self.corpus.open("rb") as handle:
            handle.seek(row["byte_offset"])
            raw = handle.read(row["byte_length"])
        value = json.loads(raw)
        if value.get("chunk_id") != row["chunk_id"]:
            raise RuntimeError("Corpus lookup offset mismatch")
        return value

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        return self._read(row) if row else None

    def nearby(self, chunk_id: str, radius: int = 2) -> list[dict[str, Any]]:
        row = self.connection.execute("SELECT document_id,chunk_index FROM chunks WHERE chunk_id=?",
                                      (chunk_id,)).fetchone()
        if not row:
            return []
        matches = self.connection.execute("""SELECT * FROM chunks WHERE document_id=?
            AND chunk_index BETWEEN ? AND ? ORDER BY chunk_index""",
            (row["document_id"], row["chunk_index"] - radius, row["chunk_index"] + radius))
        return [self._read(item) for item in matches]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("artifacts/corpus_v1/chunks.jsonl"))
    parser.add_argument("--workspace", type=Path, default=Path("evaluation/annotation/workspace"))
    args = parser.parse_args(argv)
    runtime = args.corpus.parent / "runtime/manifest.json"
    identity = json.loads(runtime.read_text(encoding="utf-8"))["corpus"] if runtime.is_file() else {}
    lookup = CorpusLookup.build(args.corpus, args.workspace / "corpus_lookup.sqlite3",
                                expected_sha=identity.get("corpus_sha256"),
                                expected_count=identity.get("count"))
    try:
        print(f"indexed {lookup.count} corpus chunk IDs and offsets")
    finally:
        lookup.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
