"""Offline two-lane tests; no Qdrant server or model download."""

import json
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
    with pytest.raises(RuntimeError, match="requires QDRANT_URL"):
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
        qdrant_url="http://fake", qdrant_collection="test", qdrant_hnsw_ef=128))
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
    builder.build_bm25(corpus, out, scan, batch_size=2)
    builder.write_index_manifest(corpus, out, scan)

    class Reader:
        def read_index(self, path):
            assert Path(path).is_file()
            return faiss.index

    result = validate_indexes.validate(corpus, out, components=("faiss", "qdrant", "bm25"),
                                       qdrant_client=client, faiss_module=Reader())
    assert result["validated"] is True
    assert result["component_counts"] == {"faiss": 5, "bm25": 5, "qdrant": 5}
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
