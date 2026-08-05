import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas import (
    RetrieveRequest,
    RetrieveResponse,
    RetrievalContextItem,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["RAG"],
)


def _context_to_api(chunk: dict[str, Any]) -> RetrievalContextItem:
    """Convert an internal retrieval chunk into a safe API response object."""
    return RetrievalContextItem(
        chunk_id=str(chunk.get("chunk_id", "")),
        title=chunk.get("title"),
        text=chunk.get("text"),
        final_retrieval_score=chunk.get("final_retrieval_score"),
        reranker_score=chunk.get("reranker_score"),
        rrf_score=chunk.get("rrf_score"),
        metadata_bonus=chunk.get("metadata_bonus"),
        metadata_hits=chunk.get("metadata_hits") or [],
    )


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    status_code=status.HTTP_200_OK,
)
async def retrieve(
    payload: RetrieveRequest,
    request: Request,
) -> RetrieveResponse:
    """
    Run the Phase 9 hybrid retrieval pipeline.

    Pipeline:
    question
        -> query analysis
        -> query expansion
        -> E5 + FAISS
        -> BM25
        -> weighted RRF
        -> cross-encoder reranking
        -> metadata soft boost
        -> diversity selection
        -> final contexts
    """

    service = request.app.state.rag_service
    retriever = request.app.state.retriever

    if retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Retrieval runtime is not loaded. "
                "Use APP_MODE=retrieval-only or APP_MODE=full."
            ),
        )

    if not service.loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service is not ready.",
        )

    started = time.perf_counter()

    try:
        # Retrieval + reranking are blocking operations.
        # Move them to a worker thread so the FastAPI event loop is not blocked.
        result = await asyncio.to_thread(
            retriever.retrieve,
            payload.question,
            payload.final_k,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        logger.exception("Retrieval runtime error")

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected retrieval error")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal retrieval error.",
        ) from exc

    latency_ms = (time.perf_counter() - started) * 1000

    final_context = [
        _context_to_api(chunk)
        for chunk in result.get("final_context", [])
    ]

    candidates = None
    tool_trace = None

    if payload.debug:
        candidates = [
            _context_to_api(chunk)
            for chunk in result.get("candidates20", [])
        ]

        tool_trace = result.get("tool_trace", [])

    return RetrieveResponse(
        question=result["question"],
        is_ood=result.get("is_ood", False),
        ood_reason=result.get("ood_reason", ""),
        intent=result.get("intent"),
        analysis=result.get("analysis"),
        query_variants=result.get("query_variants", []),
        final_context=final_context,
        candidates=candidates,
        tool_trace=tool_trace,
        max_dense=result.get("max_dense"),
        context_title_diversity=result.get(
            "context_title_diversity",
            0.0,
        ),
        latency_ms=latency_ms,
    )