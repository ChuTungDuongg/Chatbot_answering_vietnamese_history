"""Small, offline fixtures for V1 metadata and resumable retrieval."""

import json
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.rag.artifact_identity import validate_identity, validate_v1_manifests
from scripts.retrieval import build_index as builder
from scripts.retrieval.prepare_v1_runtime import prepare
from tests.test_retrieval_index_builder import FakeFaiss, FakeModel, fixture


def _local_indexes(tmp_path):
    corpus, _ = fixture(tmp_path)
    scan = builder.scan_corpus(corpus)
    root = tmp_path / "retrieval"
    root.mkdir()
    builder.build_faiss(corpus, root, scan, requested_revision="main",
                        resolved_revision="a" * 40, device="cpu", batch_size=2,
                        model_factory=FakeModel, faiss_module=FakeFaiss())
    builder.build_bm25(corpus, root, scan, batch_size=2)
    builder.write_index_manifest(corpus, root, scan)
    return corpus, root, scan


def test_v1_prepare_ignores_partial_qdrant_and_preserves_v0_paths(tmp_path, monkeypatch):
    corpus, root, scan = _local_indexes(tmp_path)
    (root / "qdrant.partial").mkdir()
    runtime = prepare(corpus, root, tmp_path / "runtime")
    assert runtime["available_dense_backends"] == ["faiss"]
    assert (root / "qdrant.partial").is_dir()
    assert runtime["corpus"]["count"] == scan["chunk_count"]
    assert runtime["corpus"]["bytes"] == corpus.stat().st_size
    cfg = json.loads((tmp_path / "runtime/inference_config.json").read_text(encoding="utf-8"))
    assert cfg["retrieval"]["embedding_model_id"] == builder.MODEL_ID
    v1 = Settings(_env_file=None, corpus_path_override=corpus, retrieval_root=root,
                  inference_config_path_override=tmp_path / "runtime/inference_config.json",
                  runtime_manifest_path=tmp_path / "runtime/manifest.json")
    assert v1.corpus_path == corpus and v1.faiss_path == root / "faiss/chunks.index"
    assert v1.qdrant_manifest_path not in v1.required_retrieval_paths()
    direct = Settings(_env_file=None, corpus_path_override=corpus, retrieval_root=root)
    assert direct.inference_config_path == corpus.parent / "runtime/inference_config.json"
    assert direct.manifest_path == corpus.parent / "runtime/manifest.json"
    import app.services.rag_service as service_module
    monkeypatch.setattr(service_module, "settings", direct)
    service_module.RAGService()._validate_artifacts()
    v0 = Settings(_env_file=None, artifact_root=tmp_path)
    assert v0.corpus_path == tmp_path / "corpus/vn_history_rag_chunks_enriched.jsonl"


@pytest.mark.parametrize("field", ["count", "corpus_sha256", "ordered_chunk_id_sha256"])
def test_v1_identity_rejects_mismatch(tmp_path, field):
    corpus, root, scan = _local_indexes(tmp_path)
    expected = {"count": scan["chunk_count"], "corpus_sha256": scan["corpus_sha256"],
                "ordered_chunk_id_sha256": scan["ordered_chunk_id_sha256"]}
    manifest = root / "faiss/manifest.json"
    changed = json.loads(manifest.read_text(encoding="utf-8"))
    changed[field] = -1 if field == "count" else "wrong"
    manifest.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(RuntimeError, match=field):
        validate_identity(manifest, expected)


def test_qdrant_manifest_requires_matching_dimension_and_identity(tmp_path):
    corpus, root, scan = _local_indexes(tmp_path)
    expected = {"count": scan["chunk_count"], "corpus_sha256": scan["corpus_sha256"],
                "ordered_chunk_id_sha256": scan["ordered_chunk_id_sha256"]}
    qdrant = root / "qdrant"
    qdrant.mkdir()
    saved = json.loads((root / "faiss/manifest.json").read_text(encoding="utf-8"))
    saved.update(collection_name="test", distance="COSINE", quantization="none")
    path = qdrant / "manifest.json"
    path.write_text(json.dumps({**saved, "embedding_dimension": 99}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="embedding_dimension"):
        validate_v1_manifests(root, expected, include_qdrant=True)
    for field in ("corpus_sha256", "ordered_chunk_id_sha256"):
        path.write_text(json.dumps({**saved, field: "wrong"}), encoding="utf-8")
        with pytest.raises(RuntimeError, match=field):
            validate_v1_manifests(root, expected, include_qdrant=True)


def test_headless_resume_skips_flushed_rows_and_rejects_changed_dataset(tmp_path, monkeypatch):
    import evaluation.run_retrieval as runner

    dataset = tmp_path / "questions.jsonl"
    def write_questions(suffix=""):
        dataset.write_text("".join(json.dumps({"id": f"q{i}", "question": f"History {i}{suffix}",
                                               "category": "fact"}) + "\n" for i in range(3)), encoding="utf-8")
    write_questions()
    output = tmp_path / "run"
    monkeypatch.setattr(runner, "settings", Settings(_env_file=None, app_mode="retrieval-only"))
    monkeypatch.setattr(runner, "metadata", lambda dataset, *_args: {
        "dataset_sha256": runner.file_sha(dataset), "dense_backend": "faiss", "corpus_sha256": "fixed"})
    service = SimpleNamespace(load=lambda: None, shutdown=lambda: None)

    class Retriever:
        calls = []
        retrieval_config = {}
        interrupt = True

        def retrieve(self, question, final_k):
            self.calls.append(question)
            if self.interrupt and len(self.calls) == 2:
                raise KeyboardInterrupt()
            return {"final_context": [{"chunk_id": question, "title": "History", "text": "Text"}]}

    retriever = Retriever()
    with pytest.raises(KeyboardInterrupt):
        runner.run(dataset, output, service=service, retriever=retriever)
    assert len((output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    retriever.interrupt = False
    runner.run(dataset, output, resume=True, service=service, retriever=retriever)
    rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["question_id"] for row in rows] == ["q0", "q1", "q2"]
    assert retriever.calls.count("History 0") == 1
    assert (output / "summary.json").is_file()
    with pytest.raises(RuntimeError, match="already complete"):
        runner.run(dataset, output, resume=True, service=service, retriever=retriever)
    (output / "summary.json").unlink()
    write_questions(" changed")
    with pytest.raises(RuntimeError, match="dataset_sha256"):
        runner.run(dataset, output, resume=True, service=service, retriever=retriever)
