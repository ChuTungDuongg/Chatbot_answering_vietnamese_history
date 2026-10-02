"""Request isolation, resource reuse and real stage events on tiny V1 fixtures."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings, CENTRAL_MODEL_ID
from app.central.runtime import CentralRuntime
from app.rag.backends import DenseBackendError, DenseBackendUnavailable, QdrantSearchError
from app.rag.retrieval import HybridRetriever
from app.schemas import ChatRequest, RetrieveRequest
from app.services.rag_service import RAGService
from app.tools.local_search import SearchHistoryTool
from app.tools.registry import ToolRegistry
from scripts.retrieval import build_index as builder
from scripts.retrieval.prepare_v1_runtime import prepare
from tests.test_baseline_runtime import _app, _events, FakeModel
from tests.test_qdrant_lanes import FakeQdrant
from tests.test_retrieval_index_builder import FakeFaiss, FakeModel as IndexModel, fixture


def loaded_service(tmp_path, monkeypatch, *, available="faiss,qdrant", default="faiss", key="private-key"):
    corpus, _ = fixture(tmp_path)
    scan = builder.scan_corpus(corpus)
    root = tmp_path / "retrieval"
    root.mkdir()
    client, faiss = FakeQdrant(), FakeFaiss()
    builder.build_dense_pair(corpus, root, scan, requested_revision="main", resolved_revision="a" * 40,
        device="cpu", batch_size=2, url=None, collection="test", model_factory=IndexModel,
        faiss_module=faiss, client=client)
    builder.build_bm25(corpus, root, scan, batch_size=2)
    builder.write_index_manifest(corpus, root, scan)
    prepare(corpus, root, tmp_path / "runtime")
    cfg = Settings(_env_file=None, app_mode="retrieval-only", corpus_path_override=corpus,
        retrieval_root=root, retrieval_available_backends=available, retrieval_dense_backend=default,
        qdrant_url="http://private-host", qdrant_api_key=key, qdrant_collection="test")
    import app.services.rag_service as module
    monkeypatch.setattr(module, "settings", cfg)
    faiss.index.d = 2
    loads = {"faiss": 0, "client": 0, "embedder": 0, "reranker": 0}
    def read_index(_):
        loads["faiss"] += 1; return faiss.index
    def make_client(**kwargs):
        assert kwargs["timeout"] == 30
        loads["client"] += 1; return client
    class Embedder:
        max_seq_length = 512
        def get_sentence_embedding_dimension(self): return 2
        def encode(self, rows, **_): return np.tile(np.array([1., 0.], dtype="float32"), (len(rows), 1))
    class Reranker:
        def predict(self, pairs, **_): return np.arange(len(pairs), dtype="float32")
    def embedder(*_, **__): loads["embedder"] += 1; return Embedder()
    def reranker(*_, **__): loads["reranker"] += 1; return Reranker()
    monkeypatch.setattr("faiss.read_index", read_index)
    monkeypatch.setattr("qdrant_client.QdrantClient", make_client)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=embedder, CrossEncoder=reranker))
    service = RAGService()
    return service, client, cfg, loads


def test_both_backends_load_once_share_resources_and_parallel_requests(tmp_path, monkeypatch):
    service, client, _, loads = loaded_service(tmp_path, monkeypatch)
    service.load(); service.load()
    retriever = HybridRetriever(service)
    resources = tuple(id(getattr(service, name)) for name in ("chunks", "bm25", "embedder", "reranker", "faiss_index"))
    barrier = threading.Barrier(2)
    for backend in service.dense_retrievers.values():
        search = backend.search
        first = [True]
        def synchronized(vector, k, *, exact=False, search=search, first=first):
            if first[0]: first[0] = False; barrier.wait(timeout=5)
            return search(vector, k, exact=exact)
        monkeypatch.setattr(backend, "search", synchronized)
    events = {name: [] for name in ("faiss", "qdrant")}
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(retriever.retrieve, "Bạch Đằng năm 938?", 2, dense_backend=name, progress=events[name].append)
                   for name in events]
        results = [f.result(timeout=10) for f in futures]
    assert [r["retrieval_backend"] for r in results] == ["faiss", "qdrant"]
    assert [r["final_context"] for r in results][0] == results[1]["final_context"]
    assert all(r["timings_ms"]["dense_search"] >= 0 for r in results)
    assert all({e["state"] for e in group if e["stage"] == "dense_search"} == {"started", "completed"} for group in events.values())
    assert loads == {"faiss": 1, "client": 1, "embedder": 1, "reranker": 1}
    assert resources == tuple(id(getattr(service, name)) for name in ("chunks", "bm25", "embedder", "reranker", "faiss_index"))
    assert service.dense_retriever is service.dense_retrievers["faiss"]
    assert service.dense_retrievers["qdrant"].client is client


def test_optional_qdrant_missing_credentials_keeps_faiss_and_default_fail_fast(tmp_path, monkeypatch):
    service, _, cfg, loads = loaded_service(tmp_path, monkeypatch, key=None)
    service.load()
    assert service.readiness()["retrieval"] == {"default_backend": "faiss", "available_backends": ["faiss"]}
    assert loads["client"] == 0
    with pytest.raises(DenseBackendUnavailable): service.get_dense_retriever("qdrant")
    cfg.retrieval_dense_backend = "qdrant"
    with pytest.raises(RuntimeError, match="QDRANT_API_KEY"): RAGService().load()


@pytest.mark.parametrize("field", ["corpus_sha256", "ordered_chunk_id_sha256", "embedding_dimension", "embedding_model_resolved_revision", "embedding_model_id"])
def test_optional_qdrant_wrong_identity_is_never_advertised(tmp_path, monkeypatch, field):
    service, _, cfg, _ = loaded_service(tmp_path, monkeypatch)
    value = json.loads(cfg.qdrant_manifest_path.read_text())
    value[field] = 99 if field == "embedding_dimension" else "wrong"
    cfg.qdrant_manifest_path.write_text(json.dumps(value))
    service.load()
    assert service.readiness()["dense_backends"]["qdrant"]["available"] is False
    assert service.get_dense_retriever("faiss") is not None


def test_faiss_only_and_request_validation():
    settings = Settings(_env_file=None)
    assert settings.retrieval_dense_backend == "faiss" and settings.requested_dense_backends == ("faiss",)
    with pytest.raises(ValidationError): Settings(_env_file=None, retrieval_available_backends="qdrant")
    with pytest.raises(ValidationError): RetrieveRequest(question="History?", retrieval_backend="unknown")
    with pytest.raises(ValidationError): ChatRequest(conversation_id="00000000-0000-0000-0000-000000000001", question="History?", retrieval_backend="unknown")


@pytest.mark.parametrize("fallback", [False, True])
def test_central_selected_backend_including_fallback(tmp_path, monkeypatch, fallback):
    service, _, _, _ = loaded_service(tmp_path, monkeypatch); service.load()
    retriever = HybridRetriever(service)
    class Planner(FakeModel):
        async def generate(self, *_args, **_kwargs):
            return ("" if fallback else '<tool_call>{"name":"search_history","arguments":{"query":"Bạch Đằng năm 938?","top_k":2}}</tool_call>'), None
    tools = ToolRegistry(); tools.register(SearchHistoryTool(retriever))
    central = CentralRuntime(model=Planner(CENTRAL_MODEL_ID), tools=tools, max_action_rounds=1)
    async def together():
        from app.rag.hybrid_runtime import HybridRuntime
        from app.config import HYBRID_MODEL_ID
        return await asyncio.gather(central.prepare("Bạch Đằng năm 938?", 2, [], retrieval_backend="qdrant"),
                                   HybridRuntime(retriever, FakeModel(HYBRID_MODEL_ID)).prepare("Bạch Đằng năm 938?", 2, [], retrieval_backend="faiss"))
    prepared = asyncio.run(together())
    assert [p.retrieval["retrieval_backend"] for p in prepared] == ["qdrant", "faiss"]
    assert prepared[0].retrieval["timings_ms"]["dense_search"] >= 0
    assert prepared[0].contexts
    from app.rag.prompting import SYSTEM_PROMPT
    for answer in prepared:
        assert answer.messages[0]["content"] == SYSTEM_PROMPT
        assert all(chunk["chunk_id"] in answer.messages[-1]["content"] for chunk in answer.contexts)


@pytest.mark.parametrize("backend", ["faiss", "qdrant"])
def test_http_retrieve_chat_sse_backend_progress_and_metrics(tmp_path, monkeypatch, backend):
    service, _, _, _ = loaded_service(tmp_path, monkeypatch); service.load()
    app, store, _, _ = _app(tmp_path)
    retriever = HybridRetriever(service)
    app.state.rag_service = service; app.state.retriever = retriever
    app.state.chat_mode_router.runtime_for("hybrid").retriever = retriever
    conversation = store.create_conversation("dynamic-test")
    with TestClient(app) as client:
        retrieval = client.post("/api/v1/retrieve", json={"question": "Bạch Đằng năm 938?", "retrieval_backend": backend})
        assert retrieval.status_code == 200 and retrieval.json()["retrieval_backend"] == backend
        response = client.post("/api/v1/chat/stream", headers={"X-Client-ID": "dynamic-test"}, json={
            "conversation_id": conversation["id"], "question": "Bạch Đằng năm 938?", "retrieval_backend": backend, "debug": True})
    events = _events(response.text)
    assert events[0][1]["stage"] == "request_preparation"
    statuses = [data for name, data in events if name == "status"]
    assert all(data["retrieval_backend"] == backend and data["request_id"] for data in statuses)
    assert any(data["stage"] == "dense_search" and data["state"] == "completed" for data in statuses)
    assert events[-1][1]["retrieval_backend"] == backend
    assert events[-1][1]["metrics"]["dense_search_ms"] >= 0
    debug = next(data for name, data in events if name == "debug_trace")
    assert debug["retrieval"]["backend"] == backend
    assert "private-key" not in response.text and "private-host" not in response.text


def test_qdrant_failure_never_falls_back_and_does_not_leak(tmp_path, monkeypatch):
    service, remote, _, _ = loaded_service(tmp_path, monkeypatch); service.load()
    def failure(**_): raise RuntimeError("https://private-host/api?key=private-key")
    remote.query_points = failure
    retriever = HybridRetriever(service)
    with pytest.raises(QdrantSearchError) as error:
        retriever.retrieve("Bạch Đằng năm 938?", 2, dense_backend="qdrant")
    assert "private-key" not in str(error.value)
    tools = ToolRegistry(); tools.register(SearchHistoryTool(retriever))
    from app.tools.registry import ToolExecutionContext
    with pytest.raises(DenseBackendError): asyncio.run(tools.call("search_history", {"query": "Bạch Đằng năm 938?"},
        context=ToolExecutionContext(retrieval_backend="qdrant")))


def test_first_sse_status_and_thread_progress_precede_retrieval_completion(tmp_path):
    app, store, _, retriever = _app(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = retriever.retrieve
    def slow(query, k, **kwargs):
        kwargs["progress"]({"stage": "dense_search", "state": "started", "message": "Truy vấn FAISS"})
        entered.set(); assert release.wait(5)
        return original(query, k, **kwargs)
    retriever.retrieve = slow
    async def exercise():
        from app.api.routes import _execute
        from app.chat_modes import ChatMode
        from app.telemetry import RequestTrace
        conversation = store.create_conversation("dynamic-test")
        payload = ChatRequest(conversation_id=conversation["id"], question="Bạch Đằng?")
        request = SimpleNamespace(app=app)
        stream = _execute(payload, request, "dynamic-test", store, app.state.chat_mode_router.runtime_for("hybrid"), ChatMode.HYBRID, RequestTrace("r1", "hybrid"), watch_disconnect=False)
        first = await anext(stream)
        assert first[1]["stage"] == "request_preparation" and not entered.is_set()
        assert (await anext(stream))[1]["state"] == "completed"
        dense = await asyncio.wait_for(anext(stream), 2)
        assert dense[1]["stage"] == "dense_search" and not release.is_set()
        release.set()
        rest = [event async for event in stream]
        assert any(name == "answer_delta" for name, _ in rest)
    asyncio.run(exercise())


@pytest.mark.parametrize("endpoint", ["chat", "chat/stream", "retrieve"])
def test_unavailable_backend_rejected_before_conversation_changes(tmp_path, monkeypatch, endpoint):
    service, _, _, _ = loaded_service(tmp_path, monkeypatch, key=None); service.load()
    app, store, _, _ = _app(tmp_path)
    app.state.rag_service = service
    conversation = store.create_conversation("dynamic-test")
    body = {"question": "Bạch Đằng?", "retrieval_backend": "qdrant"}
    if endpoint != "retrieve": body["conversation_id"] = conversation["id"]
    with TestClient(app) as client:
        response = client.post(f"/api/v1/{endpoint}", headers={"X-Client-ID": "dynamic-test"}, json=body)
        assert response.status_code == 503
        assert "Qdrant" in response.json()["detail"]
    assert store.get_recent_history("dynamic-test", conversation["id"], 6) == []


@pytest.mark.parametrize("backend", [None, "faiss", "qdrant"])
def test_nonstream_chat_default_and_explicit_backend(tmp_path, monkeypatch, backend):
    service, _, _, _ = loaded_service(tmp_path, monkeypatch); service.load()
    app, store, _, _ = _app(tmp_path)
    retriever = HybridRetriever(service)
    app.state.rag_service = service; app.state.retriever = retriever
    app.state.chat_mode_router.runtime_for("hybrid").retriever = retriever
    conversation = store.create_conversation("dynamic-test")
    body = {"conversation_id": conversation["id"], "question": "Bạch Đằng năm 938?"}
    if backend: body["retrieval_backend"] = backend
    with TestClient(app) as client:
        response = client.post("/api/v1/chat", headers={"X-Client-ID": "dynamic-test"}, json=body)
    assert response.status_code == 200 and response.json()["retrieval_backend"] == (backend or "faiss")
    assert "response_mode" not in response.json()


def test_live_http_qdrant_failure_is_safe_and_preserves_existing_conversation(tmp_path, monkeypatch):
    service, remote, _, _ = loaded_service(tmp_path, monkeypatch); service.load()
    app, store, _, _ = _app(tmp_path)
    retriever = HybridRetriever(service)
    app.state.rag_service = service; app.state.retriever = retriever
    app.state.chat_mode_router.runtime_for("hybrid").retriever = retriever
    conversation = store.create_conversation("dynamic-test")
    store.add_message("dynamic-test", conversation["id"], "user", "Earlier question", [])
    def fail(**_): raise RuntimeError("private-host private-key")
    remote.query_points = fail
    with TestClient(app) as client:
        response = client.post("/api/v1/chat/stream", headers={"X-Client-ID": "dynamic-test"}, json={
            "conversation_id": conversation["id"], "question": "Bạch Đằng năm 938?", "retrieval_backend": "qdrant"})
    events = _events(response.text)
    assert not any(name == "answer_delta" for name, _ in events)
    assert events[-1][1]["status"] == "error" and events[-1][1]["retrieval_backend"] == "qdrant"
    assert "private-host" not in response.text and "private-key" not in response.text
    assert store.get_recent_history("dynamic-test", conversation["id"], 6)[0]["content"] == "Earlier question"


def test_tool_input_validation_keeps_existing_central_recovery():
    tools = ToolRegistry(); tools.register(SearchHistoryTool(None))
    _, record = asyncio.run(tools.call("search_history", {"query": "", "top_k": -1}))
    assert record.error and record.result_count is None
