from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.corpus import audit_corpus, build_corpus, enrich_corpus
from scripts.retrieval import build_index


def _snapshot(tmp_path: Path) -> Path:
    corpus = tmp_path / "Dataset" / "chunks.jsonl"
    corpus.parent.mkdir()
    corpus.write_bytes(b'{"chunk_id":"a"}\n{"chunk_id":"b"}')
    index = tmp_path / "artifacts" / "retrieval" / "faiss" / "chunks.index"
    index.parent.mkdir(parents=True)
    index.write_bytes(b"\x00\x01\x02")
    files = []
    for path in (corpus, index):
        relative = path.relative_to(tmp_path).as_posix()
        record = {"path": relative, **audit_corpus.fingerprint(path, count_lines=path.suffix == ".jsonl")}
        if path.suffix == ".jsonl":
            record["jsonl_record_count_estimate"] = record["physical_lines"]
        files.append(record)
    manifest = tmp_path / "snapshot.json"
    manifest.write_text(json.dumps({"files": files, "roots": ["Dataset", "artifacts"]}), encoding="utf-8")
    return manifest


def test_audit_detects_byte_changes_missing_files_and_new_files_without_writing(tmp_path: Path) -> None:
    manifest = _snapshot(tmp_path)
    corpus = tmp_path / "Dataset" / "chunks.jsonl"
    index = tmp_path / "artifacts" / "retrieval" / "faiss" / "chunks.index"
    before = {path.relative_to(tmp_path): (path.read_bytes(), path.stat().st_mtime_ns) for path in tmp_path.rglob("*") if path.is_file()}

    clean = audit_corpus.audit_manifest(manifest, tmp_path)
    assert clean["preserved"] is True
    assert clean["unchanged_files"] == 2
    assert {path.relative_to(tmp_path): (path.read_bytes(), path.stat().st_mtime_ns) for path in tmp_path.rglob("*") if path.is_file()} == before

    corpus.write_bytes(b'{"chunk_id":"x"}\n{"chunk_id":"b"}')
    index.unlink()
    (tmp_path / "Dataset" / "new.jsonl").write_text("{}\n", encoding="utf-8")
    changed = audit_corpus.audit_manifest(manifest, tmp_path)
    assert changed["preserved"] is False
    assert changed["missing_files"] == ["artifacts/retrieval/faiss/chunks.index"]
    assert changed["modified_files"][0]["path"] == "Dataset/chunks.jsonl"
    assert "sha256" in changed["modified_files"][0]["differences"]
    assert changed["unexpected_files"] == ["Dataset/new.jsonl"]


def test_audit_rejects_manifest_path_escape(tmp_path: Path) -> None:
    manifest = tmp_path / "snapshot.json"
    manifest.write_text(json.dumps({"files": [{"path": "../outside.txt"}], "roots": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="not repository-relative"):
        audit_corpus.audit_manifest(manifest, tmp_path)


def test_default_audit_tracks_protected_corpus_separately_from_mutable_inventory(tmp_path: Path) -> None:
    manifest = _snapshot(tmp_path)
    snapshot = json.loads(manifest.read_text(encoding="utf-8"))
    mutable = tmp_path / "data" / "chat.sqlite3"
    mutable.parent.mkdir()
    mutable.write_bytes(b"before")
    snapshot["roots"].append("data")
    snapshot["files"].append({
        "path": "data/chat.sqlite3",
        **audit_corpus.fingerprint(mutable),
        "preservation_tier": "inventory_only",
    })
    manifest.write_text(json.dumps(snapshot), encoding="utf-8")
    mutable.write_bytes(b"after")

    protected = audit_corpus.audit_manifest(manifest, tmp_path)
    full = audit_corpus.audit_manifest(manifest, tmp_path, all_files=True)
    assert protected["preserved"] is True
    assert protected["expected_files"] == 2
    assert full["preserved"] is False
    assert full["modified_files"][0]["path"] == "data/chat.sqlite3"


def test_descriptive_audit_reports_quality_signals_without_touching_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "Dataset" / "chunks.jsonl"
    corpus.parent.mkdir()
    rows = [
        {"chunk_id": "a", "text": "Battle in 1789", "title": "Battle", "metadata": {"years": [1789], "source_sha1": "abc"}, "source_type": "hf_wikipedia", "url": "https://example.org/a"},
        {"chunk_id": "a", "text": "Battle in 1789", "title": "", "metadata": {}, "source_type": "web", "url": ""},
        {"chunk_id": "b", "text": "", "title": "Other", "source_type": "archive"},
    ]
    corpus.write_text("".join(json.dumps(row) + "\n" for row in rows) + "{bad json\n[1]\n\n", encoding="utf-8")
    before = (corpus.read_bytes(), corpus.stat().st_mtime_ns)
    report = audit_corpus.diagnose_corpus(corpus, near_empty_chars=5)

    assert report["chunk_count"] == 3
    assert report["physical_lines"] == 6
    assert report["malformed_json_rows"] == 1
    assert report["malformed_json_line_examples"] == [4]
    assert report["non_object_rows"] == 1
    assert report["blank_lines"] == 1
    assert report["chunk_ids"]["unique_count"] == 2
    assert report["chunk_ids"]["duplicate_rows"] == 1
    assert report["text"]["exact_duplicate_text_rate"] == 0.5
    assert report["text"]["near_empty_chunks"] == 1
    assert report["missing_titles"] == 1
    assert report["missing_source_metadata"] == 2
    assert report["source_ids"]["missing_rows"] == 2
    assert report["urls"]["missing_where_expected"] == 1
    assert report["year_date_metadata_coverage"]["either_rows"] == 1
    assert report["length_distribution"]["characters"]["min"] == 0
    assert (corpus.read_bytes(), corpus.stat().st_mtime_ns) == before

    output = tmp_path / "reports" / "audit.json"
    assert audit_corpus.main(["--corpus", str(corpus), "--output", str(output), "--repo-root", str(tmp_path)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["sha256"] == report["sha256"]
    with pytest.raises(ValueError, match="outside the corpus directory"):
        audit_corpus.main(["--corpus", str(corpus), "--output", str(corpus)])
    with pytest.raises(ValueError, match="outside repository data"):
        audit_corpus.main(["--corpus", str(corpus), "--output", str(tmp_path / "artifacts" / "audit.json"), "--repo-root", str(tmp_path)])
    assert (corpus.read_bytes(), corpus.stat().st_mtime_ns) == before


def test_neutral_corpus_build_and_enrich_keep_legacy_serialization(tmp_path: Path) -> None:
    source = tmp_path / "packs"
    source.mkdir()
    (source / "a.jsonl").write_text(
        '{"chunk_id": "a", "text": "Năm 1789", "metadata": {"era": "Tây Sơn"}}\n'
        '{"chunk_id": "a", "text": "duplicate"}\n'
        '{"chunk_id": "b", "text": "1945 và 1789"}\n',
        encoding="utf-8",
    )
    (source / "b.jsonl").write_text('{"chunk_id": "c", "text": "text"}\n', encoding="utf-8")
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    built = tmp_path / "built.jsonl"
    enriched = tmp_path / "enriched.jsonl"

    assert build_corpus.main(["--input-dir", str(source), "--output", str(built)]) == 0
    assert enrich_corpus.main(["--input", str(built), "--output", str(enriched)]) == 0
    assert [json.loads(line)["chunk_id"] for line in built.read_text(encoding="utf-8").splitlines()] == ["a", "b", "c"]
    assert enriched.read_text(encoding="utf-8").splitlines()[0] == json.dumps(
        {"chunk_id": "a", "text": "Năm 1789", "metadata": {"era": "Tây Sơn", "years": [1789], "char_len": 8}, "history_score": 1.0},
        ensure_ascii=False,
    )
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()} == source_hashes
    assert build_index.build_parser().parse_args([]).embedding_model == "intfloat/multilingual-e5-base"
