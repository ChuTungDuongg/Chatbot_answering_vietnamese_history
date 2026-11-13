"""Evaluation-only SQLite backup and non-destructive offline restore."""

import argparse
from contextlib import closing
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import zipfile
from pathlib import Path
from .models import now
from .privacy import reject_secrets

TABLES = (
    "evaluation_datasets",
    "evaluation_questions",
    "dataset_versions",
    "evaluation_responses",
    "human_reviews",
    "evaluation_experiments",
    "evaluation_runs",
    "evaluation_evidence",
    "evaluation_blind_sessions",
    "evaluation_blind_assignments",
    "evaluation_blind_reviews",
    "evaluation_import_batches",
    "evaluation_id_counters",
)
SCHEMA_VERSION = "1.1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def normalized_sql(sql):
    return re.sub(r"\s+", "", re.sub(r"\bIF\s+NOT\s+EXISTS\b", "", sql or "", flags=re.IGNORECASE)).casefold()


def table_schema(db, table, schema="main"):
    """Include constraints; nonunique lookup indexes are optional for old archives."""
    sql = db.execute(
        f"SELECT sql FROM {schema}.sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    indexes = []
    for _, name, unique, origin, partial in db.execute(f'PRAGMA {schema}.index_list("{table}")'):
        if not unique:
            continue
        index_name = name.replace('"', '""')
        index_sql = db.execute(
            f"SELECT sql FROM {schema}.sqlite_master WHERE type='index' AND name=?", (name,)
        ).fetchone()
        indexes.append(
            (
                name,
                unique,
                origin,
                partial,
                db.execute(f'PRAGMA {schema}.index_xinfo("{index_name}")').fetchall(),
                normalized_sql(index_sql[0]) if index_sql else "",
            )
        )
    return (
        normalized_sql(sql[0]) if sql else "",
        db.execute(f'PRAGMA {schema}.table_info("{table}")').fetchall(),
        db.execute(f'PRAGMA {schema}.foreign_key_list("{table}")').fetchall(),
        sorted(indexes),
    )


def reject_record_secrets(row):
    for value in row:
        reject_secrets(value)
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except (ValueError, TypeError):
                continue
            reject_secrets(decoded)


def create_backup(source, target):
    source, target = Path(source), Path(target)
    if not source.is_file():
        raise FileNotFoundError(source)
    with tempfile.TemporaryDirectory(prefix="eval-backup-") as td:
        snap = Path(td) / "snapshot.sqlite"
        cleaned = Path(td) / "evaluation.sqlite"
        with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as origin, closing(
            sqlite3.connect(snap)
        ) as copy:
            origin.backup(copy)
        counts = {}
        with closing(sqlite3.connect(snap)) as original, closing(sqlite3.connect(cleaned)) as output:
            original.execute("BEGIN")
            present = {
                r[0]: r[1]
                for r in original.execute("SELECT name,sql FROM sqlite_master WHERE type='table'")
                if r[0] in TABLES
            }
            if not set(TABLES[:5]).issubset(present):
                raise ValueError("Not an Eval Studio database")
            for table in TABLES:
                if table not in present:
                    continue
                output.execute(present[table])
                columns = [r[1] for r in original.execute(f'PRAGMA table_info("{table}")')]
                count = 0
                with closing(original.execute(f'SELECT * FROM "{table}"')) as records:
                    for row in records:
                        reject_record_secrets(row)
                        output.execute(
                            f'INSERT INTO "{table}" VALUES ({",".join("?" for _ in columns)})', row
                        )
                        count += 1
                counts[table] = count
            for table, sql in original.execute(
                "SELECT tbl_name,sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
            ):
                if table in present:
                    output.execute(sql)
            output.commit()
            if (
                output.execute("PRAGMA integrity_check").fetchone()[0] != "ok"
                or output.execute("PRAGMA foreign_key_check").fetchone()
            ):
                raise ValueError("Invalid database relationships")
        payload = cleaned.read_bytes()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": now(),
            "counts": counts,
            "sha256": sha(payload),
        }
        created = False
        try:
            with target.open("xb") as handle:
                created = True
                with zipfile.ZipFile(handle, "w", zipfile.ZIP_DEFLATED) as archive:
                    archive.writestr(
                        "manifest.json", json.dumps(manifest, ensure_ascii=False, sort_keys=True)
                    )
                    archive.writestr("evaluation.sqlite", payload)
        except BaseException:
            if created:
                target.unlink(missing_ok=True)
            raise
        return manifest


def restore_backup(archive_path, target):
    target = Path(target)
    if target.exists():
        raise FileExistsError("Restore target must not exist")
    with zipfile.ZipFile(archive_path) as archive:
        if sorted(archive.namelist()) != ["evaluation.sqlite", "manifest.json"]:
            raise ValueError("Unexpected backup contents")
        if (
            archive.getinfo("evaluation.sqlite").file_size > 200_000_000
            or archive.getinfo("manifest.json").file_size > 100_000
        ):
            raise ValueError("Backup too large")
        manifest = json.loads(archive.read("manifest.json"))
        payload = archive.read("evaluation.sqlite")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Incompatible backup schema")
    if sha(payload) != manifest.get("sha256"):
        raise ValueError("Backup checksum mismatch")
    target.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with target.open("xb") as handle:
            created = True
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        with closing(sqlite3.connect(target.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            tables = {
                r[0]
                for r in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            if not set(TABLES[:5]).issubset(tables) or tables - set(TABLES):
                raise ValueError("Unexpected backup schema")
            if (
                db.execute("PRAGMA integrity_check").fetchone()[0] != "ok"
                or db.execute("PRAGMA foreign_key_check").fetchone()
            ):
                raise ValueError("Corrupt backup database")
            if db.execute("SELECT 1 FROM sqlite_master WHERE type IN ('view','trigger')").fetchone():
                raise ValueError("Unexpected schema objects")
            from .store import EvalStore
            from .v11 import StudioV11

            with tempfile.TemporaryDirectory(prefix="eval-schema-") as directory:
                expected_path = Path(directory) / "expected.sqlite"
                StudioV11(EvalStore(expected_path))
                with closing(sqlite3.connect(expected_path)) as expected_db:
                    for table in tables:
                        if table_schema(db, table) != table_schema(expected_db, table):
                            raise ValueError("Incompatible table schema")
            if set(manifest.get("counts", {})) != tables:
                raise ValueError("Backup manifest does not match tables")
            for table, expected in manifest["counts"].items():
                if (
                    table not in TABLES
                    or table not in tables
                    or db.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] != expected
                ):
                    raise ValueError("Backup record count mismatch")
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise
    return manifest


def restore_to_copy(archive_path, existing, target, replace_evaluation=False):
    """Restore into a new copy; the existing database is never modified."""
    existing, target = Path(existing), Path(target)
    if not existing.is_file():
        raise FileNotFoundError(existing)
    if target.exists():
        raise FileExistsError("Restore target must not exist")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="eval-restore-") as directory:
        restored = Path(directory) / "evaluation.sqlite"
        copied = Path(directory) / "combined.sqlite"
        manifest = restore_backup(archive_path, restored)
        with closing(sqlite3.connect(existing.resolve().as_uri() + "?mode=ro", uri=True)) as source, closing(
            sqlite3.connect(copied)
        ) as dest:
            source.backup(dest)
        with closing(sqlite3.connect(copied)) as db:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("ATTACH DATABASE ? AS restored", (str(restored),))
            source_tables = {
                r[0]: r[1]
                for r in db.execute("SELECT name,sql FROM restored.sqlite_master WHERE type='table'")
                if r[0] in TABLES
            }
            target_tables = {
                r[0] for r in db.execute("SELECT name FROM main.sqlite_master WHERE type='table'")
            }
            for table in source_tables:
                if table in target_tables and table_schema(db, table) != table_schema(db, table, "restored"):
                    raise ValueError("Existing evaluation schema is incompatible")
            populated = any(
                db.execute(f'SELECT 1 FROM main."{table}" LIMIT 1').fetchone()
                for table in TABLES
                if table in target_tables
            )
            if populated and not replace_evaluation:
                raise ValueError(
                    "Existing evaluation data conflicts; explicit replace_evaluation is required on a copy"
                )
            db.execute("BEGIN IMMEDIATE")
            try:
                for table in reversed(TABLES):
                    if table in target_tables:
                        db.execute(f'DELETE FROM main."{table}"')
                for table in TABLES:
                    if table not in source_tables:
                        continue
                    if table not in target_tables:
                        db.execute(source_tables[table])
                    db.execute(f'INSERT INTO main."{table}" SELECT * FROM restored."{table}"')
                for name, table, sql in db.execute(
                    "SELECT name,tbl_name,sql FROM restored.sqlite_master WHERE type='index' AND sql IS NOT NULL"
                ).fetchall():
                    if (
                        table in source_tables
                        and not db.execute(
                            "SELECT 1 FROM main.sqlite_master WHERE name=?", (name,)
                        ).fetchone()
                    ):
                        db.execute(sql)
                if db.execute("PRAGMA foreign_key_check").fetchone():
                    raise ValueError("Restored relationships are invalid")
                db.commit()
            except BaseException:
                db.rollback()
                raise
        created = False
        try:
            with target.open("xb") as out, copied.open("rb") as source:
                created = True
                shutil.copyfileobj(source, out)
                out.flush()
                os.fsync(out.fileno())
        except BaseException:
            if created:
                target.unlink(missing_ok=True)
            raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Eval Studio backup and offline restore")
    parser.add_argument("action", choices=["backup", "restore", "restore-copy"])
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("--base", help="Existing SQLite file to copy when using restore-copy")
    parser.add_argument(
        "--replace-evaluation", action="store_true", help="Replace evaluation records only in the new copy"
    )
    args = parser.parse_args()
    if args.action == "restore-copy":
        if not args.base:
            parser.error("restore-copy requires --base")
        result = restore_to_copy(args.source, args.base, args.target, args.replace_evaluation)
    else:
        result = (
            create_backup(args.source, args.target)
            if args.action == "backup"
            else restore_backup(args.source, args.target)
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
