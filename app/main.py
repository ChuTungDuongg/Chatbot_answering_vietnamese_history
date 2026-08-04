from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    Request,
)

from app.config import settings

from app.schemas import (
    HealthResponse,
    ReadyResponse,
)

from app.services.rag_service import (
    RAGService,
)


# ============================================================
# Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    service = RAGService()

    service.load()

    app.state.rag_service = service

    yield

    service.shutdown()


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(

    title=
        settings.app_name,

    version=
        settings.app_version,

    description=(
        "Vietnamese History Hybrid RAG API "
        "using Qwen2.5, FAISS, BM25S, "
        "RRF and Cross-Encoder reranking."
    ),

    lifespan=
        lifespan,
)


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

        "status":
            "ok",

        "service":
            settings.app_name,

        "version":
            settings.app_version,
    }


# ============================================================
# Ready
# ============================================================

@app.get(
    "/ready",
    response_model=ReadyResponse,
    tags=["System"],
)
async def ready(
    request: Request,
):

    service: RAGService = (
        request.app.state.rag_service
    )

    return service.readiness()