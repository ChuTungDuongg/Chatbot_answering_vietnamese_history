from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException, status

from app.api.routes import _get_generation_runtime
from app.config import settings
from app.services.rag_service import RAGService


def _request_for(service, *, orchestrator=None, generator=None):
    state = SimpleNamespace(
        rag_service=service,
        orchestrator=orchestrator,
        generator=generator,
    )
    return SimpleNamespace(app=SimpleNamespace(state=state))


def _generation_service(*, model=None, external_backend=False, loaded=True):
    return SimpleNamespace(
        loaded=loaded,
        model=model,
        external_generation_backend=external_backend,
    )


def _loaded_retrieval_service() -> RAGService:
    service = RAGService()
    service.loaded = True
    service.chunks = [{"chunk_id": "history:1"}]
    service.faiss_index = SimpleNamespace(ntotal=1)
    service.bm25 = object()
    service.embedder = object()
    service.reranker = object()
    return service


def test_generation_gate_accepts_shared_external_backend():
    service = _generation_service(model=None, external_backend=True)
    orchestrator = object()

    resolved_service, runtime = _get_generation_runtime(
        _request_for(service, orchestrator=orchestrator)
    )

    assert resolved_service is service
    assert runtime is orchestrator


def test_generation_gate_rejects_missing_model_and_external_backend():
    service = _generation_service(model=None, external_backend=False)

    with pytest.raises(HTTPException) as exc_info:
        _get_generation_runtime(_request_for(service, orchestrator=object()))

    assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert exc_info.value.detail == "Generation model is not ready."


def test_generation_gate_accepts_legacy_local_model():
    service = _generation_service(model=object(), external_backend=False)
    generator = object()

    resolved_service, runtime = _get_generation_runtime(
        _request_for(service, generator=generator)
    )

    assert resolved_service is service
    assert runtime is generator


def test_generation_gate_still_requires_runtime():
    service = _generation_service(model=None, external_backend=True)

    with pytest.raises(HTTPException) as exc_info:
        _get_generation_runtime(_request_for(service))

    assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert exc_info.value.detail.startswith("Generation runtime is not loaded.")


def test_full_readiness_accepts_shared_external_backend(monkeypatch):
    monkeypatch.setattr(settings, "app_mode", "full")
    service = _loaded_retrieval_service()
    service.tokenizer = object()
    service.model = None
    service.external_generation_backend = True

    readiness = service.readiness()

    assert readiness["ready"] is True
    assert readiness["model_loaded"] is True


def test_retrieval_only_readiness_does_not_require_generation(monkeypatch):
    monkeypatch.setattr(settings, "app_mode", "retrieval-only")
    service = _loaded_retrieval_service()

    readiness = service.readiness()

    assert readiness["ready"] is True
    assert readiness["model_loaded"] is False


def test_api_only_readiness_is_unchanged(monkeypatch):
    monkeypatch.setattr(settings, "app_mode", "api-only")
    service = RAGService()
    service.loaded = True

    readiness = service.readiness()

    assert readiness["ready"] is True
    assert readiness["model_loaded"] is False
