"""Crash-safe, manually adjudicated Codex SILVER batches and deterministic master merge.

No model client is imported here. Every add is one SQLite transaction; completed
JSONL batches are immutable. Corpus evidence is read by byte offset.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.annotation.policy import CATEGORIES, duplicate_pairs, similar
from evaluation.annotation.workspace import dumps, now
from evaluation.schema import Question, load_questions


WORKSPACE = Path("evaluation/annotation/workspace_codex_500")
MASTER = Path("evaluation/datasets/v1_silver/questions_5000.jsonl")
CORPUS = Path("artifacts/corpus_v1/chunks.jsonl")
LOOKUP = Path("evaluation/annotation/workspace/corpus_lookup.sqlite3")
BATCH_SIZE = 50
TARGET_QUESTIONS = 5000
MAX_BATCHES = TARGET_QUESTIONS // BATCH_SIZE
DIFFICULTY_QUOTA = {"easy": 18, "medium": 22, "hard": 10}
ID_PREFIX = "vn_hist_silver_"
REQUIRED = {"id", "question", "category", "difficulty", "gold_answer", "relevant_chunk_ids",
            "relevant_source_ids", "gold_citation_source_ids", "required_facts",
            "factual_paragraph_indices", "answerable", "in_domain", "notes",
            "annotation_origin", "annotation_status", "confidence"}


class ReadOnlyCorpus:
    def __init__(self, corpus: Path = CORPUS, lookup: Path = LOOKUP):
        self.corpus = corpus.resolve()
        self.db = sqlite3.connect(f"file:{lookup.resolve().as_posix()}?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row
        metadata = dict(self.db.execute("SELECT key,value FROM metadata"))
        stat = self.corpus.stat()
        if metadata["corpus_bytes"] != str(stat.st_size) or metadata["corpus_mtime_ns"] != str(stat.st_mtime_ns):
            raise RuntimeError("Corpus differs from the read-only lookup")
        self.sha256 = metadata["corpus_sha256"]
        self.count = int(metadata["count"])

    def close(self) -> None:
        self.db.close()

    def chunk(self, chunk_id: str) -> dict[str, Any] | None:
        entry = self.db.execute("SELECT byte_offset,byte_length FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        if entry is None:
            return None
        with self.corpus.open("rb") as handle:
            handle.seek(entry["byte_offset"])
            return json.loads(handle.read(entry["byte_length"]))

    def has_source(self, source_id: str) -> bool:
        return self.db.execute("SELECT 1 FROM chunks WHERE source_id=? LIMIT 1", (source_id,)).fetchone() is not None


def validate_record(record: dict[str, Any], audit: dict[str, Any], corpus: ReadOnlyCorpus) -> None:
    missing = REQUIRED - record.keys()
    if missing:
        raise ValueError(f"Missing required SILVER fields: {sorted(missing)}")
    Question.model_validate(record)
    if record["category"] not in CATEGORIES or record["difficulty"] not in {"easy", "medium", "hard"}:
        raise ValueError("Invalid category or difficulty")
    if record["annotation_origin"] != "automatic" or record["annotation_status"] != "auto_reviewed":
        raise ValueError("Codex labels must remain automatic SILVER")
    if record["confidence"] not in {"high", "medium"} or "SILVER" not in record["notes"]:
        raise ValueError("Accepted batch record needs SILVER provenance and usable confidence")
    if type(record["answerable"]) is not bool or type(record["in_domain"]) is not bool:
        raise ValueError("Answerability and domain need explicit booleans")
    if record["answerable"] and not record["in_domain"]:
        raise ValueError("Out-of-domain question cannot be answerable")
    if not str(record["gold_answer"] or "").strip():
        raise ValueError("Every record needs an evidence-grounded SILVER answer")
    for field in ("relevant_chunk_ids", "relevant_source_ids", "gold_citation_source_ids",
                  "required_facts", "factual_paragraph_indices"):
        if not isinstance(record[field], list) or len(record[field]) != len(set(record[field])):
            raise ValueError(f"Invalid or duplicate {field}")
    if record["answerable"] and not all(record[field] for field in
                                         ("relevant_chunk_ids", "relevant_source_ids",
                                          "gold_citation_source_ids", "required_facts")):
        raise ValueError("Answerable item lacks supporting labels")
    if any(type(index) is not int or index < 0 for index in record["factual_paragraph_indices"]):
        raise ValueError("Factual paragraph indices must be nonnegative")
    if any(not isinstance(fact, str) or not fact.strip() for fact in record["required_facts"]):
        raise ValueError("Required facts must be nonempty strings")
    chunks = {}
    for chunk_id in record["relevant_chunk_ids"]:
        row = corpus.chunk(chunk_id)
        if row is None or row.get("source_id") not in record["relevant_source_ids"]:
            raise ValueError("Relevant chunk absent or source mapping disagrees")
        chunks[chunk_id] = row
    if any(not corpus.has_source(source) for source in record["relevant_source_ids"]):
        raise ValueError("Unknown relevant source")
    if not set(record["gold_citation_source_ids"]) <= set(record["relevant_source_ids"]):
        raise ValueError("Citation source is not among relevant sources")
    if not isinstance(audit.get("evidence"), list) or not audit["evidence"]:
        raise ValueError("Each Codex annotation needs inspected corpus evidence")
    if not str(audit.get("pass_b") or "").strip():
        raise ValueError("Independent Codex verification note is missing")
    covered_facts: set[int] = set()
    covered_citations: set[str] = set()
    relevant_audit: set[str] = set()
    for item in audit["evidence"]:
        chunk_id = item.get("chunk_id")
        row = corpus.chunk(chunk_id) if isinstance(chunk_id, str) else None
        quote = item.get("quote")
        decision = item.get("decision")
        if row is None or not isinstance(quote, str) or not quote.strip() or quote not in row.get("text", ""):
            raise ValueError("Evidence quote must occur in the inspected corpus chunk")
        if decision not in {"relevant", "not_relevant", "uncertain"}:
            raise ValueError("Evidence decision is invalid")
        indices = item.get("supports_fact_indices", [])
        if not isinstance(indices, list) or any(type(i) is not int or i < 0 or
                                                i >= len(record["required_facts"]) for i in indices):
            raise ValueError("Evidence fact mapping is invalid")
        if decision == "relevant":
            if chunk_id not in chunks:
                raise ValueError("Relevant audit chunk absent from record labels")
            relevant_audit.add(chunk_id)
            covered_facts.update(indices)
            if indices:
                covered_citations.add(row["source_id"])
        elif indices:
            raise ValueError("Non-relevant evidence cannot support a fact")
    if relevant_audit != set(record["relevant_chunk_ids"]):
        raise ValueError("Every selected relevant chunk needs direct audit evidence")
    if covered_facts != set(range(len(record["required_facts"]))):
        raise ValueError("Every required fact needs inspected chunk support")
    if not set(record["gold_citation_source_ids"]) <= covered_citations:
        raise ValueError("Each citation source needs support for a factual claim")


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]], *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A killed writer may leave its temporary file behind. A fresh process can
    # still retry without removing a file whose owner might still be running.
    temp = path.with_name(path.name + f".partial-{os.getpid()}")
    if temp.exists():
        raise FileExistsError(temp)
    with temp.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists() and not replace:
        temp.unlink()
        raise FileExistsError(path)
    os.replace(temp, path)


class CodexBatchStore:
    def __init__(self, workspace: Path = WORKSPACE, corpus: ReadOnlyCorpus | None = None):
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.corpus = corpus or ReadOnlyCorpus()
        self.owns_corpus = corpus is None
        self.db = sqlite3.connect(self.workspace / "codex_silver.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS records(
                id TEXT PRIMARY KEY, batch INTEGER NOT NULL, ordinal INTEGER NOT NULL,
                record_json TEXT NOT NULL, audit_json TEXT NOT NULL, created_at TEXT NOT NULL,
                UNIQUE(batch, ordinal));
            CREATE TABLE IF NOT EXISTS rejected(
                id INTEGER PRIMARY KEY AUTOINCREMENT, batch INTEGER NOT NULL,
                reason TEXT NOT NULL, created_at TEXT NOT NULL);
        """)
        with self.db:
            for key, value in (("corpus_sha256", self.corpus.sha256), ("corpus_count", str(self.corpus.count))):
                previous = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
                if previous and previous[0] != value:
                    raise RuntimeError("Codex batch corpus identity changed")
                self.db.execute("INSERT OR IGNORE INTO metadata VALUES (?,?)", (key, value))

    def close(self) -> None:
        self.db.close()
        if self.owns_corpus:
            self.corpus.close()

    def _batch_path(self, batch: int) -> Path:
        return self.workspace / "batches" / f"batch_{batch:02d}.jsonl"

    def _audit_path(self, batch: int) -> Path:
        return self.workspace / "batches" / f"batch_{batch:02d}.audit.jsonl"

    def completed(self) -> int:
        if self._batch_path(MAX_BATCHES + 1).exists():
            raise RuntimeError(f"Batch {MAX_BATCHES + 1:02d} exceeds the benchmark target")
        count = 0
        for batch in range(1, MAX_BATCHES + 1):
            if not self._batch_path(batch).is_file():
                break
            count += 1
        if any(self._batch_path(batch).exists() for batch in range(count + 2, MAX_BATCHES + 1)):
            raise RuntimeError("Completed batch files are not contiguous")
        return count

    def next_batch(self) -> int:
        complete = self.completed()
        if complete >= MAX_BATCHES:
            raise RuntimeError(f"All {MAX_BATCHES} batches are complete")
        return complete + 1

    def draft_rows(self, batch: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        return [(json.loads(row[0]), json.loads(row[1])) for row in self.db.execute(
            "SELECT record_json,audit_json FROM records WHERE batch=? ORDER BY ordinal", (batch,))]

    def all_rows(self) -> list[dict[str, Any]]:
        return [json.loads(row[0]) for row in self.db.execute("SELECT record_json FROM records ORDER BY ordinal")]

    def add(self, record: dict[str, Any], audit: dict[str, Any]) -> None:
        batch = self.next_batch()
        existing = self.draft_rows(batch)
        if len(existing) >= BATCH_SIZE:
            raise RuntimeError("Batch already has 50 records; complete it")
        ordinal = (batch - 1) * BATCH_SIZE + len(existing) + 1
        expected_id = f"{ID_PREFIX}{ordinal:04d}"
        if record.get("id") != expected_id:
            raise ValueError(f"Next stable ID must be {expected_id}")
        validate_record(record, audit, self.corpus)
        previous = self.all_rows()
        if any(similar(record["question"], item["question"]) for item in previous):
            raise ValueError("Duplicate or near-duplicate question across Codex batches")
        with self.db:
            self.db.execute("INSERT INTO records VALUES (?,?,?,?,?,?)",
                            (record["id"], batch, ordinal, dumps(record), dumps(audit), now()))

    def revise(self, record: dict[str, Any], audit: dict[str, Any]) -> None:
        batch = self.next_batch()
        row = self.db.execute("SELECT batch FROM records WHERE id=?", (record.get("id"),)).fetchone()
        if row is None or row[0] != batch:
            raise ValueError("Only a saved record in the incomplete current batch can be revised")
        validate_record(record, audit, self.corpus)
        if any(similar(record["question"], other["question"]) for other in self.all_rows()
               if other["id"] != record["id"]):
            raise ValueError("Revised question duplicates another Codex record")
        with self.db:
            self.db.execute("UPDATE records SET record_json=?,audit_json=? WHERE id=?",
                            (dumps(record), dumps(audit), record["id"]))

    def reject(self, reason: str) -> None:
        if not reason.strip():
            raise ValueError("Rejection reason is required")
        with self.db:
            self.db.execute("INSERT INTO rejected(batch,reason,created_at) VALUES (?,?,?)",
                            (self.next_batch(), reason, now()))

    def _validate_batch(self, batch: int, pairs: list[tuple[dict, dict]]) -> list[dict[str, Any]]:
        if len(pairs) != BATCH_SIZE:
            raise ValueError(f"Batch {batch:02d} has {len(pairs)} records, expected 50")
        result = []
        for index, (record, audit) in enumerate(pairs, 1):
            expected = f"{ID_PREFIX}{((batch - 1) * BATCH_SIZE + index):04d}"
            if record.get("id") != expected:
                raise ValueError(f"Batch {batch:02d} ID order mismatch: {expected}")
            validate_record(record, audit, self.corpus)
            result.append(record)
        exact, near = duplicate_pairs(result)
        if exact or near:
            raise ValueError(f"Batch {batch:02d} contains duplicate questions")
        difficulty_counts = Counter(row["difficulty"] for row in result)
        if difficulty_counts != DIFFICULTY_QUOTA:
            raise ValueError(f"Batch {batch:02d} difficulty counts {dict(difficulty_counts)}; "
                             f"expected {DIFFICULTY_QUOTA}")
        return result

    def merge(self, master: Path = MASTER) -> int:
        count = self.completed()
        all_records: list[dict[str, Any]] = []
        for batch in range(1, count + 1):
            with self._batch_path(batch).open(encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
            with self._audit_path(batch).open(encoding="utf-8") as handle:
                audits = [json.loads(line) for line in handle if line.strip()]
            self._validate_batch(batch, list(zip(records, audits, strict=True)))
            all_records.extend(records)
        exact, near = duplicate_pairs(all_records)
        if exact or near or len(all_records) != count * BATCH_SIZE:
            raise ValueError("Cross-batch duplicate or cumulative count mismatch")
        _atomic_jsonl(master, all_records, replace=True)
        if len(load_questions(master)) != count * BATCH_SIZE:
            raise RuntimeError("Master schema/count validation failed")
        return len(all_records)

    def complete(self, master: Path = MASTER) -> int:
        if self.completed() == MAX_BATCHES:
            return self.merge(master)
        batch = self.next_batch()
        pairs = self.draft_rows(batch)
        if not pairs and batch > 1:
            # The preceding run may have published the immutable batch but
            # crashed before rebuilding the cumulative master.
            return self.merge(master)
        records = self._validate_batch(batch, pairs)
        previous = self.all_rows()[:(batch - 1) * BATCH_SIZE]
        exact, near = duplicate_pairs(previous + records)
        if exact or near:
            raise ValueError("Cross-batch duplicate question")
        # Audit file is written first; an interrupted completion can be resumed.
        audit_path = self._audit_path(batch)
        if not audit_path.is_file():
            _atomic_jsonl(audit_path, [audit for _, audit in pairs], replace=False)
        else:
            with audit_path.open(encoding="utf-8") as handle:
                saved = [json.loads(line) for line in handle if line.strip()]
            if saved != [audit for _, audit in pairs]:
                raise RuntimeError("Existing batch audit differs from SQLite records")
        batch_path = self._batch_path(batch)
        _atomic_jsonl(batch_path, records, replace=False)
        return self.merge(master)

    def status(self) -> dict[str, Any]:
        complete = self.completed()
        current = complete + 1 if complete < MAX_BATCHES else None
        rows = self.draft_rows(current) if current else []
        all_records = self.all_rows()
        source_counts = Counter(source for item in all_records for source in item["relevant_source_ids"])
        rejected = self.db.execute("SELECT count(*) FROM rejected").fetchone()[0]
        return {"completed_batches": complete, "previous_cumulative": complete * BATCH_SIZE,
                "next_batch": current, "current_saved": len(rows),
                "current_next_id": f"{ID_PREFIX}{(complete * BATCH_SIZE + len(rows) + 1):04d}" if current else None,
                "all_saved": len(all_records), "rejected_drafts": rejected,
                "top_sources": source_counts.most_common(10)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "add", "revise", "reject", "complete", "merge"))
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--master", type=Path, default=MASTER)
    parser.add_argument("--input", type=Path, help="JSON object containing record and audit")
    parser.add_argument("--reason")
    args = parser.parse_args(argv)
    store = CodexBatchStore(args.workspace)
    try:
        if args.action == "status":
            print(json.dumps(store.status(), ensure_ascii=False, indent=2))
        elif args.action in {"add", "revise"}:
            if args.input is None:
                parser.error(f"{args.action} requires --input")
            item = json.loads(args.input.read_text(encoding="utf-8"))
            if args.action == "add":
                store.add(item["record"], item["audit"])
            else:
                store.revise(item["record"], item["audit"])
            print(json.dumps(store.status(), ensure_ascii=False))
        elif args.action == "reject":
            store.reject(args.reason or "")
        elif args.action == "complete":
            print(f"master_records={store.complete(args.master)}")
        else:
            print(f"master_records={store.merge(args.master)}")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
