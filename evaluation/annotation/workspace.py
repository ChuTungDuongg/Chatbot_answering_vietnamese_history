"""Transactional local annotation workspace; draft and reviewer fields stay separate."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.annotation.policy import CATEGORIES, EVIDENCE_DECISIONS, REVIEW_POLICY_VERSION, REVIEW_STATES


EDITABLE = {"question", "category", "difficulty", "question_type", "answerable", "in_domain",
            "notes", "draft_answer", "gold_answer", "required_facts", "gold_citation_source_ids",
            "factual_paragraph_indices", "benchmark_split", "reviewer_id",
            "chunk_decisions", "source_decisions"}
STAGES = ("dense", "bm25", "rrf", "reranked")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def initial_review(candidate: dict[str, Any]) -> dict[str, Any]:
    return {"question": candidate["question"], "category": candidate["category"],
            "difficulty": candidate.get("difficulty"), "question_type": candidate.get("question_type"),
            "answerable": None, "in_domain": None, "notes": "",
            "draft_answer": candidate.get("draft_answer"), "gold_answer": None,
            "required_facts": None, "gold_citation_source_ids": None,
            "factual_paragraph_indices": None, "benchmark_split": None, "reviewer_id": None,
            "review_policy_version": None,
            "chunk_decisions": {}, "source_decisions": {}}


class Workspace:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database = self.directory / "annotation.sqlite3"
        self.connection = sqlite3.connect(self.database, timeout=30, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS candidates(
                candidate_id TEXT PRIMARY KEY, draft_json TEXT NOT NULL, review_json TEXT NOT NULL,
                review_status TEXT NOT NULL DEFAULT 'pending', deep_status TEXT NOT NULL DEFAULT 'not_selected',
                evidence_complete INTEGER NOT NULL DEFAULT 0,
                retrieval_reviewed INTEGER NOT NULL DEFAULT 0,
                answer_reviewed INTEGER NOT NULL DEFAULT 0,
                citation_reviewed INTEGER NOT NULL DEFAULT 0,
                reviewed_at TEXT, deep_reviewed_at TEXT, review_version INTEGER NOT NULL DEFAULT 0,
                revision INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS evidence(
                candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id),
                stage TEXT NOT NULL, rank INTEGER NOT NULL, chunk_id TEXT NOT NULL,
                source_id TEXT, document_id TEXT, score REAL, details_json TEXT NOT NULL,
                PRIMARY KEY(candidate_id,stage,rank));
            CREATE TABLE IF NOT EXISTS events(
                id INTEGER PRIMARY KEY AUTOINCREMENT, candidate_id TEXT NOT NULL,
                action TEXT NOT NULL, at TEXT NOT NULL, revision INTEGER NOT NULL);
            CREATE INDEX IF NOT EXISTS candidates_status ON candidates(review_status,deep_status);
        """)

    def close(self) -> None:
        self.connection.close()

    def set_metadata(self, key: str, value: Any) -> None:
        encoded = dumps(value)
        existing = self.connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        if existing and existing[0] != encoded:
            raise RuntimeError(f"Workspace metadata mismatch: {key}")
        with self.connection:
            self.connection.execute("INSERT OR IGNORE INTO metadata VALUES (?,?)", (key, encoded))

    def get_metadata(self, key: str) -> Any | None:
        row = self.connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def insert_candidates(self, candidates: list[dict[str, Any]]) -> int:
        inserted = 0
        with self.connection:
            for candidate in candidates:
                if candidate.get("review_status") != "pending":
                    raise ValueError("Generated candidates must begin pending")
                if not candidate.get("candidate_id") or not candidate.get("question"):
                    raise ValueError("Candidate ID and question are required")
                if candidate.get("category") not in CATEGORIES:
                    raise ValueError("Unknown candidate category")
                cursor = self.connection.execute("""INSERT OR IGNORE INTO candidates
                    (candidate_id,draft_json,review_json,updated_at) VALUES (?,?,?,?)""",
                    (candidate["candidate_id"], dumps(candidate), dumps(initial_review(candidate)), now()))
                inserted += cursor.rowcount
        return inserted

    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])

    def list_candidates(self, *, status: str | None = None, limit: int = 100,
                        offset: int = 0) -> list[dict[str, Any]]:
        where, args = ("WHERE review_status=?", (status,)) if status else ("", ())
        rows = self.connection.execute(f"SELECT * FROM candidates {where} ORDER BY candidate_id LIMIT ? OFFSET ?",
                                       (*args, limit, offset))
        return [self._decode(row, include_evidence=False) for row in rows]

    def all_candidates(self) -> list[dict[str, Any]]:
        return [self._decode(row, include_evidence=False)
                for row in self.connection.execute("SELECT * FROM candidates ORDER BY candidate_id")]

    def _decode(self, row: sqlite3.Row, *, include_evidence: bool) -> dict[str, Any]:
        draft, review = json.loads(row["draft_json"]), json.loads(row["review_json"])
        value = {**draft, **review, "candidate_id": row["candidate_id"],
                 "original_draft_answer": draft.get("draft_answer"),
                 "review_status": row["review_status"], "deep_status": row["deep_status"],
                 "evidence_complete": bool(row["evidence_complete"]),
                 "retrieval_reviewed": bool(row["retrieval_reviewed"]),
                 "answer_reviewed": bool(row["answer_reviewed"]),
                 "citation_reviewed": bool(row["citation_reviewed"]),
                 "reviewed_at": row["reviewed_at"], "deep_reviewed_at": row["deep_reviewed_at"],
                 "review_version": row["review_version"], "revision": row["revision"]}
        if include_evidence:
            value["evidence"] = self.evidence(row["candidate_id"])
        return value

    def get(self, candidate_id: str, *, include_evidence: bool = True) -> dict[str, Any]:
        row = self.connection.execute("SELECT * FROM candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
        if not row:
            raise KeyError(candidate_id)
        return self._decode(row, include_evidence=include_evidence)

    def evidence(self, candidate_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT * FROM evidence WHERE candidate_id=? ORDER BY stage,rank",
                                       (candidate_id,))
        return [{"stage": row["stage"], "rank": row["rank"], "chunk_id": row["chunk_id"],
                 "source_id": row["source_id"], "document_id": row["document_id"],
                 "score": row["score"], **json.loads(row["details_json"])} for row in rows]

    def set_evidence(self, candidate_id: str, items: list[dict[str, Any]]) -> None:
        candidate = self.get(candidate_id, include_evidence=False)
        if candidate["evidence_complete"]:
            return
        with self.connection:
            for item in items:
                if item["stage"] not in STAGES or int(item["rank"]) < 1:
                    raise ValueError("Invalid evidence stage or rank")
                self.connection.execute("""INSERT INTO evidence VALUES (?,?,?,?,?,?,?,?)""",
                    (candidate_id, item["stage"], int(item["rank"]), item["chunk_id"],
                     item.get("source_id"), item.get("document_id"), item.get("score"),
                     dumps(item.get("details") or {})))
            self.connection.execute("UPDATE candidates SET evidence_complete=1,updated_at=? WHERE candidate_id=?",
                                    (now(), candidate_id))

    def _check_decisions(self, review: dict[str, Any], lookup) -> tuple[list[str], list[str]]:
        for kind in ("chunk", "source"):
            decisions = review.get(f"{kind}_decisions")
            if not isinstance(decisions, dict):
                raise ValueError(f"{kind} decisions must be an object")
            for item, decision in decisions.items():
                if decision not in EVIDENCE_DECISIONS:
                    raise ValueError(f"Invalid {kind} decision")
                known = lookup.has_chunk(item) if kind == "chunk" else lookup.has_source(item)
                if not known:
                    raise ValueError(f"Unknown {kind} ID: {item}")
        chunks = sorted(k for k, v in review["chunk_decisions"].items() if v == "relevant")
        sources = sorted(k for k, v in review["source_decisions"].items() if v == "relevant")
        for chunk_id in chunks:
            if lookup.source_for_chunk(chunk_id) not in sources:
                raise ValueError("Every relevant chunk's source must also be marked relevant")
        return chunks, sources

    def apply_review(self, candidate_id: str, patch: dict[str, Any], action: str, lookup,
                     *, expected_revision: int | None = None) -> dict[str, Any]:
        if action not in ("save", "accept", "reject", "skip", "needs_review",
                          "accept_deep", "mark_deep_needs_review"):
            raise ValueError("Unknown review action")
        unknown = set(patch) - EDITABLE
        if unknown:
            raise ValueError(f"Fields cannot be edited through review: {sorted(unknown)}")
        current = self.get(candidate_id, include_evidence=False)
        if expected_revision is not None and expected_revision != current["revision"]:
            raise RuntimeError("Review changed in another session; reload before saving")
        review = json.loads(self.connection.execute("SELECT review_json FROM candidates WHERE candidate_id=?",
                                              (candidate_id,)).fetchone()[0])
        changed = {key for key, value in patch.items() if review.get(key) != value}
        review.update(patch)
        question_changed = "question" in changed
        retrieval_fields = {"question", "category", "difficulty", "question_type", "answerable",
                            "in_domain", "notes", "benchmark_split", "chunk_decisions", "source_decisions"}
        deep_fields = {"gold_answer", "required_facts", "gold_citation_source_ids",
                       "factual_paragraph_indices"}
        if question_changed and action in ("accept", "accept_deep"):
            raise ValueError("Save the edited question, rerun candidate evidence with --resume, then accept")
        if action == "accept_deep" and changed & retrieval_fields:
            raise ValueError("Save and accept retrieval edits before deep acceptance")
        if action == "mark_deep_needs_review" and changed & retrieval_fields:
            raise ValueError("Save retrieval edits separately before reopening deep review")
        if question_changed:
            review["chunk_decisions"] = {}
            review["source_decisions"] = {}
        if not str(review.get("question") or "").strip():
            raise ValueError("Question must not be empty")
        if review.get("category") not in CATEGORIES:
            raise ValueError("Category is not in annotation policy")
        if review.get("difficulty") not in (None, "easy", "medium", "hard"):
            raise ValueError("Invalid difficulty")
        if review.get("benchmark_split") not in (None, "dev", "test"):
            raise ValueError("Benchmark split must be dev or test")
        for field in ("answerable", "in_domain"):
            if review.get(field) is not None and type(review[field]) is not bool:
                raise ValueError(f"{field} must be true, false, or null")
        chunks, sources = self._check_decisions(review, lookup)
        status = current["review_status"]
        deep_status = current["deep_status"]
        retrieval_reviewed = current["retrieval_reviewed"]
        answer_reviewed = current["answer_reviewed"]
        citation_reviewed = current["citation_reviewed"]
        reviewed_at = current["reviewed_at"]
        deep_reviewed_at = current["deep_reviewed_at"]
        review_version = current["review_version"]
        if action == "accept":
            if not current["evidence_complete"]:
                raise ValueError("Retrieve candidate evidence before acceptance")
            if review.get("answerable") is None or review.get("in_domain") is None:
                raise ValueError("Reviewer must set answerable and in_domain")
            if review["answerable"] and not review["in_domain"]:
                raise ValueError("Out-of-domain question cannot be marked answerable")
            if review["answerable"] and not chunks:
                raise ValueError("Answerable question needs an explicitly relevant chunk")
            if review["answerable"] and not sources:
                raise ValueError("Answerable question needs an explicitly relevant source")
            status, retrieval_reviewed, reviewed_at = "accepted", True, now()
            review["review_policy_version"] = REVIEW_POLICY_VERSION
            review_version += 1
            if deep_status == "accepted":
                deep_status = "needs_review"
                answer_reviewed = citation_reviewed = False
        elif action in ("reject", "needs_review"):
            status, retrieval_reviewed = ("rejected" if action == "reject" else "needs_review"), False
            if deep_status == "accepted":
                deep_status = "needs_review"
                answer_reviewed = citation_reviewed = False
        elif action == "accept_deep":
            if status != "accepted" or deep_status not in ("pending", "needs_review"):
                raise ValueError("Deep review requires selected, accepted retrieval review")
            if not str(review.get("gold_answer") or "").strip():
                raise ValueError("Deep review requires a reviewer-written gold answer")
            if review.get("required_facts") is None or review.get("gold_citation_source_ids") is None:
                raise ValueError("Deep review requires explicit fact and citation labels")
            if (not isinstance(review["required_facts"], list) or
                    any(not isinstance(item, str) or not item.strip() for item in review["required_facts"]) or
                    not isinstance(review["gold_citation_source_ids"], list) or
                    any(not isinstance(item, str) or not item for item in review["gold_citation_source_ids"])):
                raise ValueError("Deep fact and citation labels must be nonempty strings")
            indices = review.get("factual_paragraph_indices")
            if indices is not None and (not isinstance(indices, list) or
                                        any(type(item) is not int or item < 0 for item in indices)):
                raise ValueError("Factual paragraph indices must be nonnegative integers")
            if review["answerable"] and (not review["required_facts"] or not review["gold_citation_source_ids"]):
                raise ValueError("Answerable deep review requires facts and citation sources")
            if any(not lookup.has_source(source) for source in review["gold_citation_source_ids"]):
                raise ValueError("Citation gold contains an unknown source")
            deep_status, answer_reviewed, citation_reviewed = "accepted", True, True
            deep_reviewed_at = now()
            review_version += 1
        elif action == "mark_deep_needs_review":
            deep_status, answer_reviewed, citation_reviewed = "needs_review", False, False
        elif changed and status == "accepted":
            if changed & retrieval_fields:
                status, retrieval_reviewed = "needs_review", False
            if deep_status == "accepted" and changed & (retrieval_fields | deep_fields):
                deep_status, answer_reviewed, citation_reviewed = "needs_review", False, False
        if action in ("accept", "accept_deep") and not str(review.get("reviewer_id") or "").strip():
            review["reviewer_id"] = None
        updated = now()
        with self.connection:
            if question_changed:
                self.connection.execute("DELETE FROM evidence WHERE candidate_id=?", (candidate_id,))
            cursor = self.connection.execute("""UPDATE candidates SET review_json=?,review_status=?,
                deep_status=?,retrieval_reviewed=?,answer_reviewed=?,citation_reviewed=?,
                reviewed_at=?,deep_reviewed_at=?,review_version=?,evidence_complete=?,
                revision=revision+1,updated_at=?
                WHERE candidate_id=? AND revision=?""",
                (dumps(review), status, deep_status, int(retrieval_reviewed), int(answer_reviewed),
                 int(citation_reviewed), reviewed_at, deep_reviewed_at, review_version,
                 int(current["evidence_complete"] and not question_changed),
                 updated, candidate_id, current["revision"]))
            if cursor.rowcount != 1:
                raise RuntimeError("Concurrent review update; reload before saving")
            self.connection.execute("INSERT INTO events(candidate_id,action,at,revision) VALUES (?,?,?,?)",
                                    (candidate_id, action, updated, current["revision"] + 1))
        return self.get(candidate_id)

    def select_deep(self, ids: list[str]) -> None:
        with self.connection:
            for candidate_id in ids:
                row = self.get(candidate_id, include_evidence=False)
                if row["review_status"] != "accepted":
                    raise ValueError("Deep queue may contain only accepted retrieval records")
                if row["deep_status"] == "not_selected":
                    self.connection.execute("UPDATE candidates SET deep_status='pending' WHERE candidate_id=?",
                                            (candidate_id,))

    def backup(self, output: Path) -> Path:
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".partial")
        if temporary.exists():
            raise FileExistsError(temporary)
        with temporary.open("x", encoding="utf-8") as handle:
            for key, value in self.connection.execute("SELECT key,value FROM metadata"):
                handle.write(dumps({"type": "metadata", "key": key, "value": json.loads(value)}) + "\n")
            for row in self.connection.execute("SELECT * FROM candidates ORDER BY candidate_id"):
                item = {key: row[key] for key in row.keys()}
                handle.write(dumps({"type": "candidate", "row": item}) + "\n")
            for row in self.connection.execute("SELECT * FROM evidence ORDER BY candidate_id,stage,rank"):
                handle.write(dumps({"type": "evidence", "row": dict(row)}) + "\n")
            for row in self.connection.execute("SELECT * FROM events ORDER BY id"):
                handle.write(dumps({"type": "event", "row": dict(row)}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        return output

    def restore(self, path: Path) -> None:
        if self.count():
            raise RuntimeError("Restore requires an empty workspace")
        with self.connection, path.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                if item["type"] == "metadata":
                    self.connection.execute("INSERT INTO metadata VALUES (?,?)",
                                            (item["key"], dumps(item["value"])))
                elif item["type"] in ("candidate", "evidence", "event"):
                    table = {"candidate": "candidates", "evidence": "evidence",
                             "event": "events"}[item["type"]]
                    row = item["row"]
                    names = list(row)
                    placeholders = ",".join("?" for _ in names)
                    self.connection.execute(f"INSERT INTO {table} ({','.join(names)}) VALUES ({placeholders})",
                                            tuple(row[name] for name in names))
                else:
                    raise ValueError("Unknown workspace backup row")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("backup", "restore"))
    parser.add_argument("--workspace", type=Path, default=Path("evaluation/annotation/workspace"))
    parser.add_argument("--file", type=Path, required=True)
    args = parser.parse_args(argv)
    workspace = Workspace(args.workspace)
    try:
        if args.action == "backup":
            print(workspace.backup(args.file))
        else:
            workspace.restore(args.file)
            print(f"restored {workspace.count()} candidates")
    finally:
        workspace.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
