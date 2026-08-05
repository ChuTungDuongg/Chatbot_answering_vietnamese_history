from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.api.routes import router as api_router
from app.config import settings
from app.rag.retrieval import HybridRetriever
from app.schemas import HealthResponse, ReadyResponse
from app.services.rag_service import RAGService


# ============================================================
# Application lifecycle
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    service = RAGService()

    # Load resources according to APP_MODE.
    #
    # api-only:
    #   load nothing
    #
    # retrieval-only:
    #   corpus + FAISS + BM25 + E5 + reranker
    #
    # full:
    #   retrieval stack + Qwen
    service.load()

    app.state.rag_service = service

    if settings.should_load_retrieval:
        app.state.retriever = HybridRetriever(service)
    else:
        app.state.retriever = None

    yield

    service.shutdown()


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Vietnamese History Hybrid RAG API using Qwen2.5, "
        "multilingual E5, FAISS, BM25S, Reciprocal Rank Fusion "
        "and cross-encoder reranking."
    ),
    lifespan=lifespan,
)


# ============================================================
# Root
# ============================================================

@app.get(
    "/",
    tags=["System"],
)
async def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.app_env,
        "mode": settings.app_mode,
        "docs": "/docs",
    }


# ============================================================
# Health
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
async def health():
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
    }


# ============================================================
# Readiness
# ============================================================

@app.get(
    "/ready",
    response_model=ReadyResponse,
    tags=["System"],
)
async def ready(request: Request):
    service: RAGService = request.app.state.rag_service
    return service.readiness()


# ============================================================
# API routers
# ============================================================

app.include_router(api_router)