"""Offline two-lane tests; no Qdrant server or model download."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.config import Settings
from app.rag.dense_backend import FaissDenseRetriever, QdrantDenseRetriever
from app.rag.retrieval import HybridRetriever
from app.services.rag_service import RAGService
from scripts.retrieval import build_index as builder
from scripts.retrieval import compare_dense, validate_indexes
from scripts.retrieval.qdrant_index import PAYLOAD_FIELDS
from tests.test_retrieval_index_builder import FakeFaiss, FakeModel, fixture


class FakeQdrant:
    def __init__(self):
        self.points = {}
        self.created = None
        self.query_calls = []
        self.upsert_batches = []

    def collection_exists(self, name):
        return self.created == name

    def create_collection(self, *, collection_name, vectors_config):
        self.created = collection_name
        self.vector_config = vectors_config
        return True

    def upsert(self, *, collection_name, points, wait):
        assert collection_name == self.created and wait is True
        self.upsert_batches.append([point.id for point in points])
        self.points.update({point.id: point for point in points})

    def count(self, collection, exact):
        assert collection == self.created and exact is True
        return SimpleNamespace(count=len(self.points))

    def get_collection(self, collection):
        assert collection == self.created
        return SimpleNamespace(status="green", indexed_vectors_count=len(self.points), config=SimpleNamespace(
            params=SimpleNamespace(vectors=self.vector_config),
            hnsw_config=SimpleNamespace(m=16, ef_construct=100, full_scan_threshold=10000),
            quantization_config=None))

    def info(self):
        return SimpleNamespace(version="test-server")

    def query_points(self, **kwargs):
        self.query_calls.append(kwargs)
        vector = np.asarray(kwargs["query"])
        scored = sorted(self.points.values(),
                        key=lambda point: float(np.dot(vector, point.vector["dense_e5"])),
                        reverse=True)[:kwargs["limit"]]
        return SimpleNamespace(points=[SimpleNamespace(id=point.id,
                                    payload=point.payload if kwargs["with_payload"] else {},
                                    score=float(np.dot(vector, point.vector["dense_e5"])))
                                       for point in scored])


def write_dense_manifest_pair(out, scan):
    """Write a legacy FAISS and newer Qdrant sidecar without building indexes."""
    base = {"count": 5, "corpus_sha256": scan["corpus_sha256"],
            "ordered_chunk_id_sha256": scan["ordered_chunk_id_sha256"],
            "embedding_model_id": "intfloat/multilingual-e5-base",
            "embedding_model_resolved_revision": "a" * 40,
            "embedding_dimension": 2, "normalize_embeddings": True,
            "passage_prefix": "passage: ", "query_prefix": "query: ",
            "passage_template": "passage: {title}\n{text}"}
    faiss_path = out / "faiss" / "manifest.json"
    qdrant_path = out / "qdrant" / "manifest.json"
    faiss_path.parent.mkdir(parents=True)
    qdrant_path.parent.mkdir(parents=True)
    faiss_path.write_text(json.dumps(base), encoding="utf-8")
    qdrant_path.write_text(json.dumps({**base, "corpus_bytes": scan["corpus_bytes"]}),
                           encoding="utf-8")
    return faiss_path, qdrant_path


@pytest.mark.parametrize("missing_from", ["faiss", "qdrant"])
def test_legacy_dense_manifest_missing_corpus_bytes_matches_new_peer(tmp_path, missing_from):
    corpus, _ = fixture(tmp_path)
    scan = builder.scan_corpus(corpus)
    out = tmp_path / "retrieval"
    faiss_path, qdrant_path = write_dense_manifest_pair(out, scan)
    if missing_from == "qdrant":
        faiss = json.loads(faiss_path.read_text(encoding="utf-8"))
        faiss["corpus_bytes"] = scan["corpus_bytes"]
        faiss_path.write_text(json.dumps(faiss), encoding="utf-8")
        qdrant = json.loads(qdrant_path.read_text(encoding="utf-8"))
        qdrant.pop("corpus_bytes")
        qdrant_path.write_text(json.dumps(qdrant), encoding="utf-8")
    original_faiss = faiss_path.read_bytes()
    builder.assert_dense_manifests_match(out)
    assert faiss_path.read_bytes() == original_faiss
    assert qdrant_path.is_file()


@pytest.mark.parametrize("faiss_bytes,qdrant_bytes,compatible", [
    (100, 100, True),
    (100, 200, False),
])
def test_present_corpus_bytes_must_match(tmp_path, faiss_bytes, qdrant_bytes, compatible):
    corpus, _ = fixture(tmp_path)
    scan = builder.scan_corpus(corpus)
    out = tmp_path / "retrieval"
    faiss_path, qdrant_path = write_dense_manifest_pair(out, scan)
    faiss = json.loads(faiss_path.read_text(encoding="utf-8"))
    qdrant = json.loads(qdrant_path.read_text(encoding="utf-8"))
    faiss["corpus_bytes"] = faiss_bytes
    qdrant["corpus_bytes"] = qdrant_bytes
    faiss_path.write_text(json.dumps(faiss), encoding="utf-8")
    qdrant_path.write_text(json.dumps(qdrant), encoding="utf-8")
    if compatible:
        builder.assert_dense_manifests_match(out)
    else:
        with pytest.raises(RuntimeError, match="manifest mismatch: corpus_bytes"):
            builder.assert_dense_manifests_match(out)


@pytest.mark.parametrize("key,bad_value", [
    ("corpus_sha256", "wrong"),
    ("ordered_chunk_id_sha256", "wrong"),
    ("count", 6),
    ("embedding_model_resolved_revision", "b" * 40),
    ("embedding_dimension", 3),
    ("normalize_embeddings", False),
    ("embedding_model_id", "wrong"),
    ("passage_prefix", "wrong"),
    ("query_prefix", "wrong"),
    ("passage_template", "wrong"),
])
def test_legacy_optional_field_does_not_weaken_required_identity(tmp_path, key, bad_value):
    corpus, _ = fixture(tmp_path)
    scan = builder.scan_corpus(corpus)
    out = tmp_path / "retrieval"
    _, qdrant_path = write_dense_manifest_pair(out, scan)
    qdrant = json.loads(qdrant_path.read_text(encoding="utf-8"))
    qdrant[key] = bad_value
    qdrant_path.write_text(json.dumps(qdrant), encoding="utf-8")
    with pytest.raises(RuntimeError, match=f"manifest mismatch: {key}"):
        builder.assert_dense_manifests_match(out)


def test_required_identity_missing_from_both_manifests_is_rejected(tmp_path):
    corpus, _ = fixture(tmp_path)
    scan = builder.scan_corpus(corpus)
    out = tmp_path / "retrieval"
    faiss_path, qdrant_path = write_dense_manifest_pair(out, scan)
    for path in (faiss_path, qdrant_path):
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved.pop("embedding_model_resolved_revision")
        path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest mismatch: embedding_model_resolved_revision"):
        builder.assert_dense_manifests_match(out)


def test_legacy_and_new_dense_manifests_combine_with_bm25_without_rewrite(tmp_path):
    corpus, _ = fixture(tmp_path)
    scan = builder.scan_corpus(corpus)
    out = tmp_path / "retrieval"
    faiss_path, _ = write_dense_manifest_pair(out, scan)
    original_faiss = faiss_path.read_bytes()
    bm25_path = out / "bm25s_index" / "phase9_manifest.json"
    bm25_path.parent.mkdir(parents=True)
    bm25_path.write_text(json.dumps({"count": 5, "corpus_sha256": scan["corpus_sha256"],
        "ordered_chunk_id_sha256": scan["ordered_chunk_id_sha256"]}), encoding="utf-8")
    manifest = builder.write_index_manifest(corpus, out, scan)
    assert manifest["components_present"] == ["bm25", "faiss", "qdrant"]
    assert faiss_path.read_bytes() == original_faiss


def test_one_embedding_stream_feeds_identical_dense_rows(tmp_path):
    corpus, rows = fixture(tmp_path)
    scan = builder.scan_corpus(corpus)
    out = tmp_path / "retrieval"
    out.mkdir()
    client, model, faiss = FakeQdrant(), FakeModel(), FakeFaiss()
    dense, qdrant = builder.build_dense_pair(
        corpus, out, scan, requested_revision="main", resolved_revision="a" * 40,
        device="cpu", batch_size=2, url=None, collection="vn_history_v1_e5",
        model_factory=lambda: model, faiss_module=faiss, client=client)
    assert [len(call[0]) for call in model.calls] == [2, 2, 1]
    assert client.upsert_batches == [[0, 1], [2, 3], [4]]
    assert list(client.points) == list(range(5))
    for row_id, row in enumerate(rows):
        point = client.points[row_id]
        assert point.payload["chunk_id"] == row["chunk_id"]
        assert point.payload["document_id"] == row["document_id"]
        assert set(point.payload) == set(PAYLOAD_FIELDS)
        assert "text" not in point.payload
        assert point.vector["dense_e5"] == faiss.index.vectors[row_id]
    assert dense["corpus_sha256"] == qdrant["corpus_sha256"] == scan["corpus_sha256"]
    assert dense["ordered_chunk_id_sha256"] == qdrant["ordered_chunk_id_sha256"]
    assert dense["embedding_model_resolved_revision"] == qdrant["embedding_model_resolved_revision"]
    assert dense["shared_embedding_stream"] is qdrant["shared_embedding_stream"] is True
    assert dense["embedding_stream_sha256"] == qdrant["embedding_stream_sha256"]
    assert qdrant["hnsw_configuration"]["source"].startswith("server defaults")
    assert qdrant["hnsw_configuration"]["m"] == 16
    assert qdrant["distance"] == "COSINE" and qdrant["quantization"] == "none"
    assert qdrant["qdrant_client_version"] == "1.19.1"
    assert not (out / "qdrant.partial").exists()
    manifest = builder.write_index_manifest(corpus, out, scan)
    assert manifest["components_present"] == ["faiss", "qdrant"]


@pytest.mark.parametrize("component,expected", [
    ("qdrant", ["qdrant"]),
    ("dense-all", ["faiss", "qdrant"]),
    ("all-backends", ["bm25", "faiss", "qdrant"]),
])
def test_qdrant_cli_modes_use_tiny_fixture(tmp_path, monkeypatch, capsys, component, expected):
    corpus, _ = fixture(tmp_path)
    out = tmp_path / "retrieval"
    client = FakeQdrant()
    original_qdrant, original_pair = builder.build_qdrant, builder.build_dense_pair
    monkeypatch.setattr(builder, "resolve_model_revision", lambda *_: "a" * 40)
    monkeypatch.setattr(builder, "build_qdrant", lambda *args, **kw: original_qdrant(
        *args, **kw, model_factory=FakeModel, client=client))
    monkeypatch.setattr(builder, "build_dense_pair", lambda *args, **kw: original_pair(
        *args, **kw, model_factory=FakeModel, faiss_module=FakeFaiss(), client=client))
    monkeypatch.setenv("QDRANT_API_KEY", "private-test-key")
    assert builder.main(["--corpus", str(corpus), "--output-dir", str(out),
                         "--device", "cpu", "--embedding-batch-size", "2",
                         "--component", component]) == 0
    output = capsys.readouterr()
    assert "private-test-key" not in output.out + output.err
    assert json.loads(output.out)["components_present"] == expected


def test_manifest_mismatch_and_partial_qdrant_fail(tmp_path):
    corpus, _ = fixture(tmp_path)
    scan = builder.scan_corpus(corpus)
    out = tmp_path / "retrieval"
    out.mkdir()
    builder.build_dense_pair(corpus, out, scan, requested_revision="main",
                             resolved_revision="a" * 40, device="cpu", batch_size=2,
                             url=None, collection="test", model_factory=FakeModel,
                             faiss_module=FakeFaiss(), client=FakeQdrant())
    path = out / "qdrant" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["embedding_model_resolved_revision"] = "b" * 40
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="FAISS/Qdrant manifest mismatch"):
        builder.check_existing(out, scan)
    path.write_text(json.dumps({**manifest, "embedding_model_resolved_revision": "a" * 40}), encoding="utf-8")
    (out / "qdrant.partial").mkdir()
    with pytest.raises(RuntimeError, match="Incomplete index output"):
        builder.check_existing(out, scan)


def test_common_dense_result_contract_and_query_modes():
    ids = ["c0", "c1"]
    faiss = FakeFaiss().IndexFlatIP(2)
    faiss.add(np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))
    client = FakeQdrant()
    from qdrant_client import models
    client.create_collection(collection_name="test", vectors_config={
        "dense_e5": models.VectorParams(size=2, distance=models.Distance.COSINE)})
    client.upsert(collection_name="test", wait=True, points=[
        models.PointStruct(id=i, vector={"dense_e5": vector}, payload={"chunk_id": ids[i]})
        for i, vector in enumerate(([1.0, 0.0], [0.0, 1.0]))])
    local = FaissDenseRetriever(faiss, ids).search(np.array([1.0, 0.0]), 2)
    remote = QdrantDenseRetriever(client, "test", ids, hnsw_ef=128)
    ann = remote.search(np.array([1.0, 0.0]), 2)
    exact = remote.search(np.array([1.0, 0.0]), 2, exact=True)
    assert [(hit.row_id, hit.chunk_id, hit.rank) for hit in local] == [
        (hit.row_id, hit.chunk_id, hit.rank) for hit in ann]
    assert all(hit.backend == "faiss" for hit in local)
    assert all(hit.backend == "qdrant" for hit in ann)
    assert client.query_calls[-2]["search_params"].hnsw_ef == 128
    assert client.query_calls[-2]["search_params"].exact is False
    assert client.query_calls[-1]["search_params"].exact is True
    assert client.query_calls[-1]["search_params"].hnsw_ef is None
    assert exact == ann
    client.points[0].payload["chunk_id"] = "wrong"
    with pytest.raises(RuntimeError, match="point/chunk mismatch"):
        remote.search(np.array([1.0, 0.0]), 2)


def test_hybrid_uses_either_dense_backend_without_changing_fusion():
    class Embedder:
        def encode(self, *_args, **_kwargs):
            return np.array([[1.0, 0.0]], dtype="float32")

    chunks = [{"chunk_id": "c0", "title": "Zero"}, {"chunk_id": "c1", "title": "One"}]
    faiss = FakeFaiss().IndexFlatIP(2)
    faiss.add(np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))
    service = SimpleNamespace(config={"retrieval": {}}, chunks=chunks, embedder=Embedder(),
                              faiss_index=faiss, dense_retriever=FaissDenseRetriever(faiss, ["c0", "c1"]))
    retriever = HybridRetriever(service)
    local = retriever.dense_search("history", 2)
    client = FakeQdrant()
    from qdrant_client import models
    client.create_collection(collection_name="test", vectors_config={
        "dense_e5": models.VectorParams(size=2, distance=models.Distance.COSINE)})
    client.upsert(collection_name="test", wait=True, points=[
        models.PointStruct(id=i, vector={"dense_e5": vector}, payload={"chunk_id": f"c{i}"})
        for i, vector in enumerate(([1.0, 0.0], [0.0, 1.0]))])
    service.dense_retriever = QdrantDenseRetriever(client, "test", ["c0", "c1"], hnsw_ef=128)
    remote = retriever.dense_search("history", 2)
    assert local == remote
    runs = [{"query": "history", "dense": local, "bm25": [(1, 2.0)]}]
    a = retriever.multi_query_rrf(runs)
    runs[0]["dense"] = remote
    b = retriever.multi_query_rrf(runs)
    assert [(row["chunk_id"], row["rrf_score"]) for row in a] == [
        (row["chunk_id"], row["rrf_score"]) for row in b]


def test_faiss_mode_needs_no_qdrant_and_qdrant_needs_url(monkeypatch):
    baseline = Settings(_env_file=None, retrieval_dense_backend="faiss")
    assert baseline.faiss_path in baseline.required_retrieval_paths()
    assert baseline.qdrant_manifest_path not in baseline.required_retrieval_paths()
    import app.services.rag_service as service_module
    monkeypatch.setattr(service_module, "settings", Settings(
        _env_file=None, retrieval_dense_backend="qdrant", qdrant_url=None))
    with pytest.raises(RuntimeError, match="index is not finalized"):
        RAGService()._load_qdrant()


def test_v0_faiss_manifest_field_remains_accepted(tmp_path, monkeypatch):
    import app.services.rag_service as service_module
    import faiss
    manifest_path = tmp_path / "retrieval" / "faiss" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"count": 2,
        "embedding_model": builder.MODEL_ID}), encoding="utf-8")
    index = FakeFaiss().IndexFlatIP(2)
    index.add(np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))
    monkeypatch.setattr(faiss, "read_index", lambda *_: index)
    monkeypatch.setattr(service_module, "settings", Settings(
        _env_file=None, artifact_root=tmp_path, retrieval_dense_backend="faiss"))
    service = RAGService()
    service.chunks = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    service.config = {"retrieval": {"embedding_model_id": builder.MODEL_ID}}
    service._load_faiss()
    assert service.dense_retriever.search(np.array([1.0, 0.0]), 2)[0].chunk_id == "a"


def test_runtime_selects_qdrant_and_checks_collection(tmp_path, monkeypatch):
    corpus, rows = fixture(tmp_path)
    scan = builder.scan_corpus(corpus)
    out = tmp_path / "retrieval"
    out.mkdir()
    client = FakeQdrant()
    builder.build_qdrant(corpus, out, scan, requested_revision="main",
                         resolved_revision="a" * 40, device="cpu", batch_size=2,
                         url=None, collection="test", model_factory=FakeModel, client=client)
    import app.services.rag_service as service_module
    monkeypatch.setattr(service_module, "settings", Settings(
        _env_file=None, artifact_root=tmp_path, retrieval_dense_backend="qdrant",
            qdrant_url="http://fake", qdrant_api_key="placeholder",
            qdrant_collection="test", qdrant_hnsw_ef=128))
    monkeypatch.setattr("qdrant_client.QdrantClient", lambda **kwargs: client)
    service = RAGService()
    service.chunks = rows
    service.ordered_chunk_id_sha256 = scan["ordered_chunk_id_sha256"]
    service.corpus_sha256 = scan["corpus_sha256"]
    service.config = {"retrieval": {"embedding_model_id": builder.MODEL_ID}}
    service._load_qdrant()
    assert service.faiss_index is None
    assert service.embedding_revision == "a" * 40
    assert service.dense_retriever.search(np.array([1.0, 1.0]), 2)[0].backend == "qdrant"
    service_module.settings.qdrant_collection = "other"
    with pytest.raises(RuntimeError, match="collection differs"):
        service._load_qdrant()
    service_module.settings.qdrant_collection = "test"
    client.points.pop(4)
    with pytest.raises(RuntimeError, match="count mismatch"):
        service._load_qdrant()
    client.points[4] = SimpleNamespace(id=4)
    original_get_collection = client.get_collection
    def wrong_dimension(collection):
        info = original_get_collection(collection)
        info.config.params.vectors["dense_e5"].size = 3
        return info
    client.get_collection = wrong_dimension
    with pytest.raises(RuntimeError, match="vector configuration"):
        service._load_qdrant()
    client.get_collection = original_get_collection
    def leaking_error(*_args):
        raise RuntimeError("placeholder-secret")
    client.get_collection = leaking_error
    with pytest.raises(RuntimeError, match="unavailable or misconfigured") as error:
        service._load_qdrant()
    assert "placeholder-secret" not in str(error.value)
    client.get_collection = lambda *_: SimpleNamespace(status="yellow")
    with pytest.raises(RuntimeError, match="not ready"):
        service._load_qdrant()
    def unavailable(*_args, **_kwargs):
        raise RuntimeError("private-test-key")
    client.count = unavailable
    with pytest.raises(RuntimeError, match="unavailable or misconfigured") as error:
        service._load_qdrant()
    assert "private-test-key" not in str(error.value)


def test_index_validator_and_dense_metrics_on_tiny_fixture(tmp_path):
    corpus, _ = fixture(tmp_path)
    scan = builder.scan_corpus(corpus)
    out = tmp_path / "retrieval"
    out.mkdir()
    client, faiss = FakeQdrant(), FakeFaiss()
    builder.build_dense_pair(corpus, out, scan, requested_revision="main",
                             resolved_revision="a" * 40, device="cpu", batch_size=2,
                             url=None, collection="test", model_factory=FakeModel,
                             faiss_module=faiss, client=client)
    faiss_path = out / "faiss" / "manifest.json"
    legacy = json.loads(faiss_path.read_text(encoding="utf-8"))
    legacy.pop("corpus_bytes")
    faiss_path.write_text(json.dumps(legacy), encoding="utf-8")
    original_faiss = faiss_path.read_bytes()
    builder.build_bm25(corpus, out, scan, batch_size=2)
    builder.write_index_manifest(corpus, out, scan)
    assert faiss_path.read_bytes() == original_faiss

    class Reader:
        def read_index(self, path):
            assert Path(path).is_file()
            return faiss.index

    result = validate_indexes.validate(corpus, out, components=("faiss", "qdrant", "bm25"),
                                       qdrant_client=client, faiss_module=Reader())
    assert result["validated"] is True
    assert result["component_counts"] == {"faiss": 5, "bm25": 5, "qdrant": 5}
    assert faiss_path.read_bytes() == original_faiss
    metrics = compare_dense.compare_vector(np.array([1.0, 1.0]), faiss.index,
                                            client, "test", hnsw_ef=128)
    assert metrics["ann_recall@10"] == 1.0
    assert metrics["faiss_qdrant_exact_overlap@10"] == 1.0
    assert compare_dense.summarize([metrics])["query_count"] == 1
    original_get_collection = client.get_collection
    def changed_hnsw(collection):
        info = original_get_collection(collection)
        info.config.hnsw_config.m = 32
        return info
    client.get_collection = changed_hnsw
    with pytest.raises(RuntimeError, match="HNSW configuration"):
        validate_indexes.validate(corpus, out, components=("faiss", "qdrant", "bm25"),
                                  qdrant_client=client, faiss_module=Reader())
    client.get_collection = original_get_collection
    client.points.pop(4)
    with pytest.raises(RuntimeError, match="qdrant count mismatch"):
        validate_indexes.validate(corpus, out, components=("faiss", "qdrant", "bm25"),
                                  qdrant_client=client, faiss_module=Reader())


def test_dense_comparison_accepts_legacy_faiss_without_loading_real_model(tmp_path, monkeypatch, capsys):
    corpus, _ = fixture(tmp_path)
    scan = builder.scan_corpus(corpus)
    out = tmp_path / "retrieval"
    out.mkdir()
    client, faiss = FakeQdrant(), FakeFaiss()
    builder.build_dense_pair(corpus, out, scan, requested_revision="main",
                             resolved_revision="a" * 40, device="cpu", batch_size=2,
                             url=None, collection="test", model_factory=FakeModel,
                             faiss_module=faiss, client=client)
    faiss_path = out / "faiss" / "manifest.json"
    legacy = json.loads(faiss_path.read_text(encoding="utf-8"))
    legacy.pop("corpus_bytes")
    faiss_path.write_text(json.dumps(legacy), encoding="utf-8")
    queries = tmp_path / "queries.jsonl"
    queries.write_text('{"question":"Lịch sử Việt Nam?"}\n', encoding="utf-8")

    class QueryModel:
        def get_sentence_embedding_dimension(self):
            return 2

        def encode(self, *_args, **_kwargs):
            return np.array([[1.0, 1.0]], dtype="float32")

    monkeypatch.setitem(sys.modules, "faiss", SimpleNamespace(read_index=lambda _: faiss.index))
    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        SimpleNamespace(SentenceTransformer=lambda *_args, **_kwargs: QueryModel()))
    monkeypatch.setattr(compare_dense, "make_client", lambda *_args, **_kwargs: client)
    assert compare_dense.main(["--output-dir", str(out), "--corpus", str(corpus),
                               "--queries", str(queries), "--hnsw-ef", "128"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["metrics"]["query_count"] == 1
