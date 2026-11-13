import hashlib
import json
import re
import sqlite3
from collections import Counter
from contextlib import contextmanager
from difflib import SequenceMatcher
from pathlib import Path
from uuid import uuid4
from .models import DATASET_ID, QUOTAS, Question, ResponseInput, Review, now


class ConflictError(Exception):
    pass


# Imported IDs outside this range remain external identifiers, not allocator state.
MAX_ALLOCATED_ID = 999_999_999


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def advance_question_counter(db, question_id):
    match = re.fullmatch(r"HIST-EVAL-(\d+)", question_id)
    if match and int(match.group(1)) <= MAX_ALLOCATED_ID:
        db.execute(
            "INSERT INTO evaluation_id_counters VALUES ('question', ?) ON CONFLICT(name) DO UPDATE SET value=MAX(value, excluded.value)",
            (int(match.group(1)),),
        )


class EvalStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS evaluation_datasets (id TEXT PRIMARY KEY, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS evaluation_questions (id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, version INTEGER NOT NULL, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS evaluation_responses (id TEXT PRIMARY KEY, question_id TEXT NOT NULL, data TEXT NOT NULL, FOREIGN KEY(question_id) REFERENCES evaluation_questions(id) ON DELETE RESTRICT);
                CREATE TABLE IF NOT EXISTS human_reviews (id TEXT PRIMARY KEY, response_id TEXT NOT NULL, data TEXT NOT NULL, FOREIGN KEY(response_id) REFERENCES evaluation_responses(id) ON DELETE RESTRICT);
                CREATE TABLE IF NOT EXISTS dataset_versions (version TEXT PRIMARY KEY, data TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS evaluation_id_counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
                CREATE INDEX IF NOT EXISTS evaluation_responses_question_idx ON evaluation_responses(question_id);
                CREATE INDEX IF NOT EXISTS human_reviews_response_idx ON human_reviews(response_id);
            """)
            # Repair an allocator poisoned by an older unbounded numeric import.
            db.execute(
                "DELETE FROM evaluation_id_counters WHERE name='question' AND value>?", (MAX_ALLOCATED_ID,)
            )
            inserted = db.execute(
                "INSERT OR IGNORE INTO evaluation_datasets VALUES (?, ?)", (DATASET_ID, now())
            ).rowcount
            if inserted:
                n = 0
                for category, count in QUOTAS.items():
                    for _ in range(count):
                        n += 1
                        q = Question(id=f"HIST-EVAL-{n:03}", category=category, smoke=n % 2 == 1)
                        db.execute(
                            "INSERT INTO evaluation_questions VALUES (?, ?, ?, ?)",
                            (q.id, DATASET_ID, q.version, canonical(q.model_dump())),
                        )
            advance_question_counter(db, "HIST-EVAL-040")
            for (question_id,) in db.execute("SELECT id FROM evaluation_questions"):
                advance_question_counter(db, question_id)
            for (snapshot_data,) in db.execute("SELECT data FROM dataset_versions"):
                for case in json.loads(snapshot_data).get("cases", []):
                    advance_question_counter(db, case.get("id", ""))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA busy_timeout=15000")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def questions(self):
        with self.connect() as db:
            return [
                json.loads(row[0]) for row in db.execute("SELECT data FROM evaluation_questions ORDER BY id")
            ]

    def get_question(self, id):
        with self.connect() as db:
            row = db.execute("SELECT data FROM evaluation_questions WHERE id=?", (id,)).fetchone()
        if not row:
            raise KeyError(id)
        return json.loads(row[0])

    def save_question(self, data):
        q = Question.model_validate(data)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT data, version FROM evaluation_questions WHERE id=?", (q.id,)).fetchone()
            if not row:
                raise KeyError(q.id)
            if row[1] != q.version:
                raise ConflictError("Question was changed elsewhere; reload before saving")
            original = json.loads(row[0])
            q.created_at, q.updated_at, q.version = original["created_at"], now(), q.version + 1
            db.execute(
                "UPDATE evaluation_questions SET version=?, data=? WHERE id=?",
                (q.version, canonical(q.model_dump()), q.id),
            )
        return q.model_dump()

    def create_question(self, source=None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            counter = db.execute("SELECT value FROM evaluation_id_counters WHERE name='question'").fetchone()[
                0
            ]
            if counter >= MAX_ALLOCATED_ID:
                raise ConflictError("Question ID allocation range exhausted")
            id = f"HIST-EVAL-{counter+1:03}"
            advance_question_counter(db, id)
            data = dict(source or {})
            data.update(
                id=id,
                dataset_id=DATASET_ID,
                version=1,
                created_at=now(),
                updated_at=now(),
                status="draft",
                smoke=False,
            )
            q = Question.model_validate(data)
            db.execute(
                "INSERT INTO evaluation_questions VALUES (?, ?, ?, ?)",
                (id, DATASET_ID, 1, canonical(q.model_dump())),
            )
        return q.model_dump()

    def delete_question(self, id):
        with self.connect() as db:
            if not db.execute("DELETE FROM evaluation_questions WHERE id=?", (id,)).rowcount:
                raise KeyError(id)

    def create_response(self, data):
        item = ResponseInput.model_validate(data).model_dump()
        item.update(id=str(uuid4()), created_at=now())
        with self.connect() as db:
            if not db.execute(
                "SELECT 1 FROM evaluation_questions WHERE id=?", (item["question_id"],)
            ).fetchone():
                raise KeyError(item["question_id"])
            db.execute(
                "INSERT INTO evaluation_responses VALUES (?, ?, ?)",
                (item["id"], item["question_id"], canonical(item)),
            )
        return item

    def responses(self, question_id=None):
        with self.connect() as db:
            rows = (
                db.execute(
                    "SELECT data FROM evaluation_responses WHERE question_id=? ORDER BY rowid DESC",
                    (question_id,),
                )
                if question_id
                else db.execute("SELECT data FROM evaluation_responses ORDER BY rowid DESC")
            )
            return [json.loads(r[0]) for r in rows]

    def create_review(self, data):
        review = Review.model_validate(data)
        item = review.model_dump()
        item.update(id=str(uuid4()), created_at=now(), weighted_score=review.weighted_score())
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute(
                "SELECT 1 FROM evaluation_responses WHERE id=?", (item["response_id"],)
            ).fetchone():
                raise KeyError(item["response_id"])
            # Network retries must not append a second identical review. Changed
            # conclusions remain new append-only records, preserving audit history.
            saved = db.execute(
                "SELECT data FROM human_reviews WHERE response_id=? ORDER BY rowid DESC LIMIT 1",
                (item["response_id"],),
            ).fetchone()
            if saved:
                previous = json.loads(saved[0])
                if {k: previous.get(k) for k in review.model_dump()} == review.model_dump():
                    return previous
            db.execute(
                "INSERT INTO human_reviews VALUES (?, ?, ?)",
                (item["id"], item["response_id"], canonical(item)),
            )
        return item

    def reviews(self):
        with self.connect() as db:
            return [
                json.loads(r[0]) for r in db.execute("SELECT data FROM human_reviews ORDER BY rowid DESC")
            ]

    def validation(self):
        cases = self.questions()
        counts = Counter(q["category"] for q in cases)
        warnings = {}
        for q in cases:
            missing = [
                name
                for name in (
                    "question",
                    "reference_answer",
                    "expected_facts",
                    "must_cover_points",
                    "reference_sources",
                )
                if not q[name]
                or (isinstance(q[name], str) and not q[name].strip())
                or (isinstance(q[name], list) and not any(x.strip() for x in q[name]))
            ]
            if q["category"] == "Ranking / Historical Judgment" and not any(
                x.strip() for x in q["ranking_criteria"]
            ):
                missing.append("ranking_criteria")
            if q["category"] == "Ambiguous / Controversial" and not q["uncertainty_notes"].strip():
                missing.append("uncertainty_notes")
            if missing:
                warnings[q["id"]] = missing
        duplicates = []
        for i, q in enumerate(cases):
            a = " ".join(q["question"].casefold().split())
            if len(a) < 8:
                continue
            for other in cases[i + 1 :]:
                b = " ".join(other["question"].casefold().split())
                if b and SequenceMatcher(None, a, b).ratio() >= 0.88:
                    duplicates.append([q["id"], other["id"]])
        responses = self.responses()
        return {
            "category_counts": {key: counts[key] for key in QUOTAS},
            "quota": QUOTAS,
            "smoke_count": sum(q["smoke"] for q in cases),
            "smoke_by_category": {
                key: sum(q["smoke"] and q["category"] == key for q in cases) for key in QUOTAS
            },
            "warnings": warnings,
            "duplicates": duplicates,
            "review_metadata_missing": [
                r["id"]
                for r in responses
                if not all(r.get(k, "").strip() for k in ("provider", "model", "chat_mode"))
            ],
            "reviews_missing_reviewer": [r["id"] for r in self.reviews() if not r["reviewer"].strip()],
        }

    def export(self, smoke=False, approved=False, base_only=False):
        base_ids = {f"HIST-EVAL-{i:03}" for i in range(1, 41)}
        cases = [
            q
            for q in self.questions()
            if (not smoke or q["smoke"])
            and (not approved or q["status"] == "approved")
            and (not base_only or q["id"] in base_ids)
        ]
        with self.connect() as db:
            created = db.execute(
                "SELECT created_at FROM evaluation_datasets WHERE id=?", (DATASET_ID,)
            ).fetchone()[0]
        manifest = {
            "dataset_id": DATASET_ID,
            "dataset_version": "draft",
            "schema_version": "1.0",
            "created_at": created,
            "case_count": len(cases),
            "category_distribution": {k: sum(q["category"] == k for q in cases) for k in QUOTAS},
            "content_checksum": "sha256:" + hashlib.sha256(canonical(cases).encode("utf-8")).hexdigest(),
            "export_timestamp": now(),
        }
        return {"manifest": manifest, "cases": cases}

    def snapshot(self, version):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", version):
            raise ValueError("Invalid version")
        data = self.export()
        data["manifest"]["dataset_version"] = version
        with self.connect() as db:
            try:
                db.execute("INSERT INTO dataset_versions VALUES (?, ?, ?)", (version, canonical(data), now()))
            except sqlite3.IntegrityError as exc:
                raise ConflictError("Version already exists") from exc
        return data

    def list_snapshots(self):
        with self.connect() as db:
            return [
                {"version": row[0], "created_at": row[1]}
                for row in db.execute(
                    "SELECT version, created_at FROM dataset_versions ORDER BY created_at DESC"
                )
            ]

    def get_snapshot(self, version):
        with self.connect() as db:
            row = db.execute("SELECT data FROM dataset_versions WHERE version=?", (version,)).fetchone()
        if not row:
            raise KeyError(version)
        return json.loads(row[0])

    def import_questions(self, cases, overwrite=False):
        questions = [Question.model_validate(q) for q in cases]
        if len({q.id for q in questions}) != len(questions):
            raise ValueError("Duplicate IDs in import")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for q in questions:
                existing = db.execute(
                    "SELECT data, version FROM evaluation_questions WHERE id=?", (q.id,)
                ).fetchone()
                if existing and not overwrite:
                    raise ConflictError("Import contains existing IDs; confirmation required")
                if existing:
                    # Exact exported content can round-trip while other imports advance the revision.
                    previous = json.loads(existing[0])
                    if previous != q.model_dump() and q.version != existing[1]:
                        raise ConflictError("Question revision differs; refresh before overwrite")
                    if previous != q.model_dump():
                        q.version, q.updated_at = existing[1] + 1, now()
                db.execute(
                    "INSERT INTO evaluation_questions VALUES (?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET version=excluded.version, data=excluded.data",
                    (q.id, DATASET_ID, q.version, canonical(q.model_dump())),
                )
                advance_question_counter(db, q.id)
        return len(questions)
