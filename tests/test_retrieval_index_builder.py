"""Offline tests for V1 index readiness; no model download or full index."""

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from scripts.retrieval import build_index as builder


def fixture(tmp_path, rows=None):
    path = tmp_path / "chunks.jsonl"
    if rows is None:
        rows = [
            {"schema_version": 2, "chunk_id": f"chunk-{i}", "document_id": f"doc-{i}",
             "title": f"Title {i}", "text": f"Historical text number {i}"}
            for i in range(5)
        ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path, rows


class FakeModel:
    def __init__(self):
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return 2

    def encode(self, passages, **kwargs):
        self.calls.append((list(passages), kwargs))
        return np.array([[float(p.split("Title ")[1][0]), 1.0] for p in passages], dtype="float32")


class FakeIndex:
    def __init__(self, dimension):
        assert dimension == 2
        self.vectors = []
        self.ntotal = 0

    def add(self, vectors):
        self.vectors.extend(vectors.tolist())
        self.ntotal += len(vectors)


class FakeFaiss:
    def __init__(self):
        self.index = None

    def IndexFlatIP(self, dimension):
        self.index = FakeIndex(dimension)
        return self.index

    def write_index(self, index, path):
        Path(path).write_bytes(b"tiny-index")


def test_preflight_is_read_only_and_keeps_order(tmp_path, capsys, monkeypatch):
    corpus, rows = fixture(tmp_path)
    output = tmp_path / "retrieval"
    monkeypatch.setattr(builder, "resolve_model_revision", lambda *_: pytest.fail("model resolved"))
    assert builder.main(["--corpus", str(corpus), "--output-dir", str(output),
                         "--device", "cpu", "--preflight"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert not output.exists()
    assert report["chunk_count"] == 5
    assert report["corpus_sha256"] == hashlib.sha256(corpus.read_bytes()).hexdigest()
    legacy = hashlib.sha256("".join(row["chunk_id"] + "\n" for row in rows).encode()).hexdigest()
    assert report["ordered_chunk_id_sha256"] == legacy
    assert builder.ordered_chunk_id_sha256(iter(rows)) == legacy
    assert report["estimated_faiss_vector_bytes"] == 5 * 768 * 4
    assert report["missing_ids"] == report["duplicate_ids"] == 0
    assert report["malformed_rows"] == report["missing_or_empty_text"] == 0
    assert report["schema_versions_observed"] == {"2": 5}
    assert report["embedding_model_requested_revision"] == "main"
    assert report["embedding_model_resolved_revision"] is None
    assert report["output_dir_exists"] is False
    assert report["components_already_present"] == {"faiss": False, "bm25": False}


@pytest.mark.parametrize("rows,error", [
    ([], "empty corpus"),
    ([{"text": "Text"}], "Missing chunk_id"),
    ([{"chunk_id": "a", "text": "Text"}, {"chunk_id": "a", "text": "More"}], "Duplicate chunk_id"),
    ([{"chunk_id": "a", "text": ""}], "Empty or invalid text"),
    ([{"schema_version": 3, "chunk_id": "a", "text": "Text"}], "Unsupported schema"),
])
def test_preflight_rejects_invalid_rows(tmp_path, rows, error):
    corpus, _ = fixture(tmp_path, rows)
    with pytest.raises(ValueError, match=error):
        builder.scan_corpus(corpus)


def test_malformed_jsonl_fails_with_line_number(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    corpus.write_text('{"chunk_id":"a","text":"Text"}\n{broken\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"Malformed JSONL.*:2"):
        builder.scan_corpus(corpus)


def test_faiss_batches_preserve_row_order_and_manifest(tmp_path):
    corpus, _ = fixture(tmp_path)
    scan = builder.scan_corpus(corpus)
    output = tmp_path / "retrieval"
    output.mkdir()
    model, faiss = FakeModel(), FakeFaiss()
    manifest = builder.build_faiss(
        corpus, output, scan, requested_revision="main", resolved_revision="a" * 40,
        device="cpu", batch_size=2, model_factory=lambda: model, faiss_module=faiss)
    assert [len(call[0]) for call in model.calls] == [2, 2, 1]
    assert all(call[1]["normalize_embeddings"] for call in model.calls)
    assert model.calls[0][0][0] == "passage: Title 0\nHistorical text number 0"
    assert faiss.index.vectors == [[float(i), 1.0] for i in range(5)]
    assert faiss.index.ntotal == manifest["count"] == 5
    assert manifest["corpus_sha256"] == scan["corpus_sha256"]
    assert manifest["ordered_chunk_id_sha256"] == scan["ordered_chunk_id_sha256"]
    assert manifest["embedding_model_resolved_revision"] == "a" * 40
    assert manifest["embedding_model_requested_revision"] == "main"
    assert manifest["passage_prefix"] == "passage: "
    assert manifest["query_prefix"] == "query: "
    assert manifest["embedding_dimension"] == 2
    assert (output / "faiss" / "chunks.index").is_file()
    assert not (output / "faiss.partial").exists()
    overall = builder.write_index_manifest(corpus, output, scan)
    assert overall["components_present"] == ["faiss"]
    assert overall["corpus_chunk_count"] == 5
    assert overall["component_status"] == {"faiss": True, "bm25": False}


def test_changed_corpus_is_rejected_before_faiss_commit(tmp_path):
    corpus, _ = fixture(tmp_path)
    scan = builder.scan_corpus(corpus)
    corpus.write_text(corpus.read_text(encoding="utf-8").replace("Historical", "Different"), encoding="utf-8")
    output = tmp_path / "retrieval"
    output.mkdir()
    with pytest.raises(RuntimeError, match="Corpus changed"):
        builder.build_faiss(corpus, output, scan, requested_revision="main",
                            resolved_revision="a" * 40, device="cpu", batch_size=2,
                            model_factory=FakeModel, faiss_module=FakeFaiss())
    assert not (output / "faiss").exists()


def test_bm25_small_fixture_matches_legacy_tokenization(tmp_path):
    import bm25s
    from app.rag.retrieval import match_norm

    corpus, rows = fixture(tmp_path)
    scan = builder.scan_corpus(corpus)
    output = tmp_path / "retrieval"
    output.mkdir()
    manifest = builder.build_bm25(corpus, output, scan, batch_size=2)
    assert manifest["count"] == 5
    assert manifest["corpus_sha256"] == scan["corpus_sha256"]
    assert not (output / "bm25s_index.partial").exists()
    legacy_texts = [match_norm(f"{row['title']} {row['title']} {row['text']}") for row in rows]
    legacy = bm25s.BM25()
    legacy.index(bm25s.tokenize(legacy_texts, stopwords=None, stemmer=None, show_progress=False),
                 show_progress=False)
    loaded = bm25s.BM25.load(str(output / "bm25s_index"), mmap=True, load_corpus=False)
    query = bm25s.tokenize(["historical title"], stopwords=None, stemmer=None, show_progress=False)
    old_ids, old_scores = legacy.retrieve(query, k=5, show_progress=False)
    new_ids, new_scores = loaded.retrieve(query, k=5, show_progress=False)
    assert np.array_equal(old_ids, new_ids)
    assert np.allclose(old_scores, new_scores)
    overall = builder.write_index_manifest(corpus, output, scan)
    assert overall["components_present"] == ["bm25"]


def test_separate_components_share_corpus_identity(tmp_path):
    corpus, _ = fixture(tmp_path)
    scan = builder.scan_corpus(corpus)
    output = tmp_path / "retrieval"
    output.mkdir()
    builder.build_faiss(corpus, output, scan, requested_revision="main",
                        resolved_revision="a" * 40, device="cpu", batch_size=2,
                        model_factory=FakeModel, faiss_module=FakeFaiss())
    builder.check_existing(output, scan)
    builder.build_bm25(corpus, output, scan, batch_size=2)
    manifest = builder.write_index_manifest(corpus, output, scan)
    assert manifest["components_present"] == ["bm25", "faiss"]
    assert manifest["corpus_chunk_count"] == scan["chunk_count"]
    assert manifest["embedding_dimension"] == 2
    assert manifest["bm25_configuration"]["title_repetitions"] == 2


def test_existing_component_mismatch_is_rejected(tmp_path):
    corpus, _ = fixture(tmp_path)
    scan = builder.scan_corpus(corpus)
    output = tmp_path / "retrieval"
    output.mkdir()
    builder.build_faiss(corpus, output, scan, requested_revision="main",
                        resolved_revision="a" * 40, device="cpu", batch_size=2,
                        model_factory=FakeModel, faiss_module=FakeFaiss())
    sidecar = output / "faiss" / "manifest.json"
    saved = json.loads(sidecar.read_text(encoding="utf-8"))
    saved["ordered_chunk_id_sha256"] = "wrong"
    sidecar.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from corpus"):
        builder.check_existing(output, scan)
    with pytest.raises(RuntimeError, match="differs from corpus"):
        builder.write_index_manifest(corpus, output, scan)


@pytest.mark.parametrize("component,expected", [
    ("faiss", ["faiss"]),
    ("bm25", ["bm25"]),
    ("all", ["bm25", "faiss"]),
])
def test_cli_components_use_tiny_fixture(tmp_path, monkeypatch, capsys, component, expected):
    corpus, _ = fixture(tmp_path)
    output = tmp_path / component
    original_faiss = builder.build_faiss
    monkeypatch.setattr(builder, "resolve_model_revision", lambda *_: "a" * 40)

    def fake_faiss(*args, **kwargs):
        return original_faiss(*args, **kwargs, model_factory=FakeModel,
                              faiss_module=FakeFaiss())

    monkeypatch.setattr(builder, "build_faiss", fake_faiss)
    assert builder.main(["--corpus", str(corpus), "--output-dir", str(output),
                         "--device", "cpu", "--embedding-batch-size", "2",
                         "--component", component]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["components_present"] == expected
    assert manifest["corpus_sha256"] == hashlib.sha256(corpus.read_bytes()).hexdigest()
    assert (output / "index_manifest.json").is_file()
    assert (output / "faiss" / "chunks.index").exists() == ("faiss" in expected)
    assert (output / "bm25s_index" / "params.index.json").exists() == ("bm25" in expected)


def test_partial_output_and_v0_paths_fail_safely(tmp_path):
    corpus, _ = fixture(tmp_path)
    scan = builder.scan_corpus(corpus)
    output = tmp_path / "retrieval"
    (output / "faiss.partial").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Incomplete index output"):
        builder.check_existing(output, scan)
    for part in ("vn_history_deployment", "vn_history_modal", "old_corpus"):
        with pytest.raises(ValueError, match="Refusing"):
            builder.safe_output(tmp_path / "artifacts" / part / "retrieval")


def test_ignore_rules_cover_local_corpus_and_archive():
    root = Path(__file__).resolve().parents[1]
    paths = ["artifacts/corpus_v1/chunks.jsonl", "artifacts/corpus_v1/documents.jsonl",
             "artifacts/corpus_v1/retrieval/faiss/chunks.index",
             "artifacts/old_corpus/vn_history_deployment/manifest.json"]
    result = subprocess.run(["git", "check-ignore", "-v", *paths], cwd=root,
                            capture_output=True, text=True, check=True)
    assert len(result.stdout.splitlines()) == len(paths)
