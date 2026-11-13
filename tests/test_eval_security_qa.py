"""Regressions for credential-safe backups and compatible restore schemas."""

from contextlib import closing
import hashlib
import json
import sqlite3
import zipfile

import pytest
from pydantic import ValidationError

from app.eval_studio.backup import TABLES, create_backup, restore_backup, restore_to_copy
from app.eval_studio.store import EvalStore
from app.eval_studio.v11 import StudioV11


@pytest.mark.parametrize(
    "config",
    [
        '{"api_key": "abcdefghijklmnopqrstuvwx"}',
        "{'session_secret': 'abcdefghijklmnopqrstuvwx'}",
    ],
)
def test_experiment_rejects_quoted_credential_keys(tmp_path, config):
    store = EvalStore(tmp_path / "studio.sqlite")
    store.snapshot("one")
    studio = StudioV11(store)
    with pytest.raises(ValidationError, match="credential"):
        studio.create_experiment({"name": "A", "snapshot_id": "one", "mcp_config": config})
    assert studio.experiments() == []


@pytest.mark.parametrize(
    "answer",
    [
        "Bearer\nabcdefghijklmnopqrstuvwx",
        "Bearer\tabcdefghijklmnopqrstuvwx",
        'api_key="abcdefghijklmnopqrstuvwx"',
        '{"password": "abcdefghijklmnopqrstuvwx"}',
    ],
)
def test_backup_rejects_credentials_hidden_by_json_escaping(tmp_path, answer):
    store = EvalStore(tmp_path / "studio.sqlite")
    store.create_response({"question_id": "HIST-EVAL-001", "answer": answer})
    target = tmp_path / "backup.zip"
    with pytest.raises(ValueError, match="credential"):
        create_backup(store.path, target)
    assert not target.exists()
    assert store.responses()[0]["answer"] == answer


def altered_backup(tmp_path, transform):
    """Build a checksummed archive with a changed SQLite schema."""
    store = EvalStore(tmp_path / "studio.sqlite")
    StudioV11(store)
    good = tmp_path / "good.zip"
    create_backup(store.path, good)
    with zipfile.ZipFile(good) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    changed = tmp_path / "changed.sqlite"
    with store.connect() as source, closing(sqlite3.connect(changed)) as target:
        for table in TABLES:
            sql = source.execute("SELECT sql FROM sqlite_master WHERE name=?", (table,)).fetchone()[0]
            target.execute(transform(table, sql))
            rows = source.execute(f'SELECT * FROM "{table}"').fetchall()
            if rows:
                target.executemany(f'INSERT INTO "{table}" VALUES ({",".join("?" for _ in rows[0])})', rows)
        target.commit()
    payload = changed.read_bytes()
    manifest["sha256"] = hashlib.sha256(payload).hexdigest()
    bad = tmp_path / "changed.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("evaluation.sqlite", payload)
    return bad, changed


@pytest.mark.parametrize("change", ["check", "unique"])
def test_restore_rejects_altered_table_constraints(tmp_path, change):
    def transform(table, sql):
        if table != "evaluation_runs":
            return sql
        if change == "check":
            return sql[:-1] + ", CHECK (external_id IS NULL))"
        return sql.replace(", UNIQUE(experiment_id, external_id)", "")

    archive, _ = altered_backup(tmp_path, transform)
    target = tmp_path / "restored.sqlite"
    with pytest.raises(ValueError, match="schema"):
        restore_backup(archive, target)
    assert not target.exists()


def test_restore_rejects_unexpected_unique_index(tmp_path):
    archive, changed = altered_backup(tmp_path, lambda table, sql: sql)
    with closing(sqlite3.connect(changed)) as db:
        db.execute("CREATE UNIQUE INDEX unexpected_run_question ON evaluation_runs(question_id)")
        db.commit()
    with zipfile.ZipFile(archive) as original:
        manifest = json.loads(original.read("manifest.json"))
    payload = changed.read_bytes()
    manifest["sha256"] = hashlib.sha256(payload).hexdigest()
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("manifest.json", json.dumps(manifest))
        output.writestr("evaluation.sqlite", payload)
    target = tmp_path / "restored.sqlite"
    with pytest.raises(ValueError, match="schema"):
        restore_backup(archive, target)
    assert not target.exists()


def test_restore_accepts_legacy_tables_without_v11(tmp_path):
    old = EvalStore(tmp_path / "legacy.sqlite")
    answer = old.create_response({"question_id": "HIST-EVAL-001", "answer": "Legacy answer"})
    backup = tmp_path / "legacy.zip"
    create_backup(old.path, backup)
    target = tmp_path / "restored.sqlite"
    restore_backup(backup, target)
    assert EvalStore(target).responses() == [answer]


def test_backup_preserves_lookup_indexes(tmp_path):
    store = EvalStore(tmp_path / "studio.sqlite")
    StudioV11(store)
    backup = tmp_path / "backup.zip"
    create_backup(store.path, backup)
    with zipfile.ZipFile(backup) as archive:
        restored = tmp_path / "backup.sqlite"
        restored.write_bytes(archive.read("evaluation.sqlite"))
    with closing(sqlite3.connect(restored)) as db:
        assert db.execute("SELECT 1 FROM sqlite_master WHERE name='evaluation_runs_question_idx'").fetchone()


def test_restore_accepts_schema_whitespace_and_missing_optional_indexes(tmp_path):
    def transform(table, sql):
        return sql.replace("CREATE TABLE ", "CREATE TABLE IF NOT EXISTS ").replace(", ", ",\n    ")

    archive, _ = altered_backup(tmp_path, transform)
    target = tmp_path / "restored.sqlite"
    restore_backup(archive, target)
    assert len(EvalStore(target).questions()) == 40


def test_restore_copy_retains_lookup_indexes_with_chat_only_base(tmp_path):
    store = EvalStore(tmp_path / "studio.sqlite")
    StudioV11(store)
    backup = tmp_path / "backup.zip"
    create_backup(store.path, backup)
    base = tmp_path / "chat.sqlite"
    with closing(sqlite3.connect(base)) as db:
        db.execute("CREATE TABLE conversations (id TEXT PRIMARY KEY)")
        db.execute("INSERT INTO conversations VALUES ('keep-chat')")
        db.commit()
    target = tmp_path / "copy.sqlite"
    restore_to_copy(backup, base, target)
    with closing(sqlite3.connect(target)) as db:
        assert db.execute("SELECT id FROM conversations").fetchall() == [("keep-chat",)]
        assert db.execute("SELECT 1 FROM sqlite_master WHERE name='evaluation_runs_question_idx'").fetchone()
