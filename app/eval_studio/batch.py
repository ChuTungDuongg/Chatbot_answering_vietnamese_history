"""Validated, revision-aware all-or-nothing imports."""

import csv
import hashlib
import io
import json
from pydantic import ValidationError
from .models import Question, ResponseInput, now
from .store import ConflictError, canonical, advance_question_counter
from .v11 import RunInput
from uuid import uuid4

MAX_BYTES = 2_000_000
MAX_RECORDS = 1000


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse(kind, fmt, content):
    if len(content.encode("utf-8")) > MAX_BYTES:
        raise ValueError("File exceeds 2 MB")
    content = content.removeprefix("\ufeff")
    errors = []
    rows = []
    if fmt == "jsonl":
        for number, line in enumerate(content.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append((number, json.loads(line)))
            except (ValueError, TypeError):
                errors.append({"line": number, "message": "Invalid JSON"})
    elif fmt == "json":
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                data = data.get(
                    {"questions": "cases", "responses": "responses", "runs": "runs"}[kind], [data]
                )
            if not isinstance(data, list):
                raise ValueError("Expected an array")
            rows = list(enumerate(data, 1))
        except (ValueError, TypeError) as exc:
            errors.append({"line": 1, "message": str(exc)})
    elif fmt == "csv":
        try:
            reader = csv.DictReader(io.StringIO(content, newline=""), strict=True)
            if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError("Missing or duplicate CSV headers")
            for number, row in enumerate(reader, 2):
                if None in row:
                    raise ValueError(f"Invalid CSV at line {number}")
                row = {k: v for k, v in row.items() if v not in (None, "")}
                if "smoke" in row:
                    if row["smoke"].lower() not in ("true", "false", "1", "0"):
                        raise ValueError(f"Invalid smoke at line {number}")
                    row["smoke"] = row["smoke"].lower() in ("true", "1")
                if "tags" in row:
                    row["tags"] = [v.strip() for v in row["tags"].split(",") if v.strip()]
                rows.append((number, row))
        except (csv.Error, ValueError) as exc:
            errors.append({"line": 1, "message": str(exc)})
    else:
        raise ValueError("Unsupported format")
    if len(rows) > MAX_RECORDS:
        raise ValueError("File exceeds 1000 records")
    if not rows and not errors:
        errors.append({"line": 1, "message": "No records"})
    return rows, errors


def inspect(studio, db, kind, fmt, content):
    if kind not in ("questions", "responses", "runs"):
        raise ValueError("Unsupported import kind")
    rows, errors = parse(kind, fmt, content)
    model = {"questions": Question, "responses": ResponseInput, "runs": RunInput}[kind]
    valid = []
    seen = set()
    conflicts = []
    versions = []
    for line, raw in rows:
        try:
            if fmt == "csv" and kind == "questions" and isinstance(raw, dict) and raw.get("id"):
                existing = db.execute(
                    "SELECT data FROM evaluation_questions WHERE id=?", (raw["id"],)
                ).fetchone()
                if existing:
                    if "version" not in raw:
                        raise ValueError("CSV overwrite requires a version")
                    raw = {**json.loads(existing[0]), **raw}
            item = model.model_validate(raw).model_dump()
            key = (
                item.get("id")
                if kind == "questions"
                else (
                    (item["experiment_id"], item["external_id"])
                    if kind == "runs" and item.get("external_id")
                    else None
                )
            )
            if key and key in seen:
                raise ValueError("Duplicate ID in file")
            if key:
                seen.add(key)
            if kind == "questions":
                old = db.execute(
                    "SELECT version FROM evaluation_questions WHERE id=?", (item["id"],)
                ).fetchone()
                versions.append([item["id"], old[0] if old else None])
                if old:
                    conflicts.append({"line": line, "id": item["id"], "current_version": old[0]})
            elif kind == "responses":
                content_id = digest(canonical(item))
                if content_id in seen:
                    raise ValueError("Duplicate response in file")
                seen.add(content_id)
                if not db.execute(
                    "SELECT 1 FROM evaluation_questions WHERE id=?", (item["question_id"],)
                ).fetchone():
                    raise ValueError("Question not found")
            else:
                exp = db.execute(
                    "SELECT data FROM evaluation_experiments WHERE id=?", (item["experiment_id"],)
                ).fetchone()
                if not exp:
                    raise ValueError("Experiment not found")
                exp = json.loads(exp[0])
                versions.append([item["experiment_id"], exp])
                snapshot = db.execute(
                    "SELECT data FROM dataset_versions WHERE version=?", (exp["snapshot_id"],)
                ).fetchone()
                if item["question_id"] not in {q["id"] for q in json.loads(snapshot[0])["cases"]}:
                    raise ValueError("Question is not in the snapshot")
                if not db.execute(
                    "SELECT 1 FROM evaluation_questions WHERE id=?", (item["question_id"],)
                ).fetchone():
                    raise ValueError("Question missing from working dataset")
                if item["completion_status"] not in ("completed", "failed") or (
                    item["completion_status"] == "completed" and not item["answer"].strip()
                ):
                    raise ValueError("Invalid completion status or answer")
                if not item["external_id"]:
                    item["external_id"] = "sha256:" + digest(canonical(item))
                scoped_id = (item["experiment_id"], item["external_id"])
                if scoped_id in seen and not key:
                    raise ValueError("Duplicate content in file")
                seen.add(scoped_id)
                existing = db.execute(
                    "SELECT data FROM evaluation_runs WHERE experiment_id=? AND external_id=?",
                    (item["experiment_id"], item["external_id"]),
                ).fetchone()
                if existing:
                    old = json.loads(existing[0])
                    old_evidence = [
                        json.loads(r[0])
                        for r in db.execute(
                            "SELECT data FROM evaluation_evidence WHERE run_id=? ORDER BY rowid", (old["id"],)
                        )
                    ]
                    if {k: old.get(k) for k in item if k != "evidence"} != {
                        k: v for k, v in item.items() if k != "evidence"
                    } or old_evidence != item["evidence"]:
                        raise ValueError("External ID already exists with different content")
                    conflicts.append(
                        {"line": line, "id": item["external_id"], "current_version": "identical; retry safe"}
                    )
            valid.append((line, item))
        except ValidationError as exc:
            errors.append(
                {
                    "line": line,
                    "message": "; ".join(
                        f'{".".join(map(str,e["loc"]))}: {e["msg"]}'
                        for e in exc.errors(include_input=False, include_url=False)
                    ),
                }
            )
        except (ValueError, TypeError, KeyError) as exc:
            errors.append({"line": line, "message": str(exc)})
    token = digest(canonical({"kind": kind, "format": fmt, "content": content, "versions": versions}))
    return {
        "count": len(rows),
        "valid": len(valid),
        "errors": errors,
        "conflicts": conflicts,
        "token": token,
        "sample": [
            {
                "line": n,
                "id": v.get("id") or v.get("external_id") or v.get("question_id"),
                "question_id": v.get("question_id"),
                "text": (v.get("question") or v.get("answer") or "")[:160],
            }
            for n, v in valid[:10]
        ],
    }, valid


def preview_batch(studio, kind, fmt, content):
    with studio.store.connect() as db:
        db.execute("BEGIN")
        result, _ = inspect(studio, db, kind, fmt, content)
    return result


def record_fingerprint(db, table, record_id):
    if table not in ("evaluation_questions", "evaluation_responses", "evaluation_runs"):
        raise ConflictError("Unverifiable previous import; import a new batch")
    row = db.execute(f"SELECT data FROM {table} WHERE id=?", (record_id,)).fetchone()
    if not row:
        return None
    evidence = (
        [
            r[0]
            for r in db.execute(
                "SELECT data FROM evaluation_evidence WHERE run_id=? ORDER BY rowid", (record_id,)
            )
        ]
        if table == "evaluation_runs"
        else []
    )
    return digest(canonical([row[0], evidence]))


def apply_batch(studio, kind, fmt, content, token, policy="reject"):
    if policy not in ("reject", "skip", "overwrite"):
        raise ValueError("Invalid conflict policy")
    batch_id = digest(canonical([kind, fmt, content, policy]))
    with studio.store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        previous = db.execute(
            "SELECT result FROM evaluation_import_batches WHERE id=?", (batch_id,)
        ).fetchone()
        if previous:
            receipt = json.loads(previous[0])
            if "records" not in receipt or any(
                record_fingerprint(db, table, record_id) != checksum
                for table, record_id, checksum in receipt["records"]
            ):
                raise ConflictError(
                    "Records from the previous import changed or were deleted; use a new batch"
                )
            return receipt["result"]
        report, items = inspect(studio, db, kind, fmt, content)
        if report["token"] != token:
            raise ConflictError("Data changed since preview; preview again")
        if report["errors"]:
            raise ValueError("Import has invalid rows")
        if kind == "questions" and report["conflicts"] and policy == "reject":
            raise ConflictError("Choose a conflict policy")
        count = 0
        records = []
        for _, item in items:
            if kind == "questions":
                old = db.execute(
                    "SELECT data,version FROM evaluation_questions WHERE id=?", (item["id"],)
                ).fetchone()
                if old and policy == "skip":
                    records.append(("evaluation_questions", item["id"]))
                    continue
                if old and old[1] != item["version"]:
                    raise ConflictError("Question revision differs; refresh the file before overwrite")
                if old and json.loads(old[0]) != item:
                    item["version"] = old[1] + 1
                    item["updated_at"] = now()
                db.execute(
                    "INSERT INTO evaluation_questions VALUES (?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET version=excluded.version,data=excluded.data",
                    (item["id"], item["dataset_id"], item["version"], canonical(item)),
                )
                advance_question_counter(db, item["id"])
                records.append(("evaluation_questions", item["id"]))
            elif kind == "runs":
                exp = json.loads(
                    db.execute(
                        "SELECT data FROM evaluation_experiments WHERE id=?", (item["experiment_id"],)
                    ).fetchone()[0]
                )
                saved = studio._insert_run(db, item, exp)
                records.append(("evaluation_runs", saved["id"]))
            else:
                item.update(id=str(uuid4()), created_at=now())
                db.execute(
                    "INSERT INTO evaluation_responses VALUES (?, ?, ?)",
                    (item["id"], item["question_id"], canonical(item)),
                )
                records.append(("evaluation_responses", item["id"]))
            count += 1
        result = {"imported": count, "batch_id": batch_id}
        receipt = {
            "result": result,
            "records": [
                [table, record_id, record_fingerprint(db, table, record_id)] for table, record_id in records
            ],
        }
        db.execute(
            "INSERT INTO evaluation_import_batches VALUES (?, ?, ?)",
            (batch_id, digest(content), canonical(receipt)),
        )
        return result
