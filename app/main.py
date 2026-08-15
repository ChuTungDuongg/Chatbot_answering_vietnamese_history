from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.api.routes import router as api_router
from app.config import settings
from app.rag.generation import RAGGenerator
from app.rag.retrieval import HybridRetriever
from app.schemas import HealthResponse, ReadyResponse
from app.services.rag_service import RAGService
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# Application lifecycle
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    service = RAGService()
    service.load()

    retriever = None
    generator = None

    if settings.should_load_retrieval:
        retriever = HybridRetriever(service)

    if settings.should_load_model:
        if retriever is None:
            raise RuntimeError(
                "Full mode requires the retrieval runtime."
            )

        generator = RAGGenerator(
            service=service,
            retriever=retriever,
        )

    app.state.rag_service = service
    app.state.retriever = retriever
    app.state.generator = generator

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
        "multilingual E5, FAISS, BM25S, Reciprocal Rank Fusion, "
        "cross-encoder reranking and grounded-generation guards."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
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
# API router
# ============================================================

app.include_router(api_router)