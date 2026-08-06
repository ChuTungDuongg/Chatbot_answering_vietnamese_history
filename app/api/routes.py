import asyncio
import json
import logging
import re
import time
from typing import Any, Iterator

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.schemas import (
    ChatRequest,
    ChatResponse,
    RetrieveRequest,
    RetrieveResponse,
    RetrievalContextItem,
    SourceItem,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["RAG"])


# ============================================================
# Helpers
# ============================================================

def _context_to_api(chunk: dict[str, Any]) -> RetrievalContextItem:
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


def _source_to_api(service, source_id: str) -> SourceItem:
    chunk = service.chunk_by_id.get(str(source_id))
    return SourceItem(
        chunk_id=str(source_id),
        title=chunk.get("title") if chunk else None,
    )


def _sse(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _answer_chunks(text: str, words_per_chunk: int = 1) -> Iterator[str]:
    """
    Chia final answer thành các delta nhỏ nhưng vẫn giữ whitespace/newline.

    Ví dụ:
        "Bạch Đằng năm 938.\n\nChiến thắng này..."
    không bị biến thành một đoạn văn duy nhất khi stream.
    """
    pieces = re.findall(r"\S+\s*", text)

    for index in range(0, len(pieces), words_per_chunk):
        yield "".join(pieces[index:index + words_per_chunk])


# ============================================================
# Retrieve
# ============================================================

@router.post("/retrieve", response_model=RetrieveResponse, status_code=status.HTTP_200_OK)
async def retrieve(payload: RetrieveRequest, request: Request) -> RetrieveResponse:
    service = request.app.state.rag_service
    retriever = request.app.state.retriever

    if retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval runtime is not loaded. Use APP_MODE=retrieval-only or APP_MODE=full.",
        )

    if not service.loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service is not ready.",
        )

    started = time.perf_counter()

    try:
        result = await asyncio.to_thread(retriever.retrieve, payload.question, payload.final_k)
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
        context_title_diversity=result.get("context_title_diversity", 0.0),
        latency_ms=latency_ms,
    )


# ============================================================
# Chat
# ============================================================

@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    service = request.app.state.rag_service
    generator = request.app.state.generator

    if generator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Generation runtime is not loaded. Use APP_MODE=full to enable chat.",
        )

    if not service.loaded or service.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Generation model is not ready.",
        )

    started = time.perf_counter()

    try:
        result = await asyncio.to_thread(generator.chat, payload.question, payload.final_k)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        logger.exception("Generation runtime error")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected chat error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal generation error.",
        ) from exc

    latency_ms = (time.perf_counter() - started) * 1000

    sources = [
        _source_to_api(service, source_id)
        for source_id in result.get("source_ids", [])
    ]

    debug = None

    if payload.debug:
        retrieval = result.get("retrieval", {})

        debug = {
            "analysis": result.get("analysis"),
            "tool_trace": result.get("tool_trace", []),
            "prompt_budget": result.get("prompt_budget"),
            "support_score": result.get("support_score"),
            "quality_warnings": result.get("quality_warnings", []),
            "initial_quality_issues": result.get("initial_quality_issues", []),
            "repair_attempted": result.get("repair_attempted", False),
            "model_source_ids": result.get("model_source_ids", []),
            "invalid_source_ids": result.get("invalid_source_ids", []),
            "unsupported_years": result.get("unsupported_years", []),
            "format_ok": result.get("format_ok"),
            "is_ood": retrieval.get("is_ood", False),
            "ood_reason": retrieval.get("ood_reason", ""),
            "query_variants": retrieval.get("query_variants", []),
            "retrieval_latency_ms": result.get("retrieval_latency_sec", 0.0) * 1000,
        }

    return ChatResponse(
        answer=result["answer"],
        status=result["status"],
        sources=sources,
        latency_ms=latency_ms,
        rewrite_used=result.get("rewrite_used", False),
        debug=debug,
    )


# ============================================================
# Validated SSE Chat Stream
# ============================================================

@router.post("/chat/stream", status_code=status.HTTP_200_OK)
async def chat_stream(payload: ChatRequest, request: Request):
    service = request.app.state.rag_service
    generator = request.app.state.generator

    if generator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Generation runtime is not loaded. Use APP_MODE=full to enable chat streaming.",
        )

    if not service.loaded or service.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Generation model is not ready.",
        )

    async def event_stream():
        stream_started = time.perf_counter()

        yield _sse(
            "status",
            {
                "stage": "processing",
                "message": "Retrieving evidence, generating and validating answer.",
            },
        )

        task = asyncio.create_task(
            asyncio.to_thread(generator.chat, payload.question, payload.final_k)
        )

        # Giữ SSE connection sống trong lúc retrieval/generation/guard đang chạy.
        while not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=8.0)
            except asyncio.TimeoutError:
                yield _sse("ping", {"timestamp": time.time()})

        try:
            result = await task
        except ValueError as exc:
            yield _sse(
                "error",
                {
                    "type": "bad_request",
                    "message": str(exc),
                },
            )
            return
        except RuntimeError as exc:
            logger.exception("Streaming generation runtime error")
            yield _sse(
                "error",
                {
                    "type": "runtime_error",
                    "message": str(exc),
                },
            )
            return
        except Exception:
            logger.exception("Unexpected streaming chat error")
            yield _sse(
                "error",
                {
                    "type": "internal_error",
                    "message": "Internal generation error.",
                },
            )
            return

        # Chỉ tới đây mới bắt đầu gửi answer:
        # retrieval -> Qwen -> guards -> optional repair đã hoàn tất.
        yield _sse(
            "status",
            {
                "stage": "validated",
                "status": result.get("status"),
                "rewrite_used": result.get("rewrite_used", False),
            },
        )

        # Một từ / event, giữ nguyên whitespace.
        for delta in _answer_chunks(result["answer"], words_per_chunk=1):
            yield _sse(
                "answer_delta",
                {
                    "delta": delta,
                },
            )

            # Chỉ để tạo cảm giác progressive rendering.
            await asyncio.sleep(0.012)

        sources = [
            _source_to_api(service, source_id).model_dump()
            for source_id in result.get("source_ids", [])
        ]

        yield _sse(
            "sources",
            {
                "items": sources,
            },
        )

        if payload.debug:
            retrieval = result.get("retrieval", {})

            yield _sse(
                "debug",
                {
                    "tool_trace": result.get("tool_trace", []),
                    "support_score": result.get("support_score"),
                    "quality_warnings": result.get("quality_warnings", []),
                    "repair_attempted": result.get("repair_attempted", False),
                    "rewrite_used": result.get("rewrite_used", False),
                    "invalid_source_ids": result.get("invalid_source_ids", []),
                    "unsupported_years": result.get("unsupported_years", []),
                    "is_ood": retrieval.get("is_ood", False),
                    "ood_reason": retrieval.get("ood_reason", ""),
                },
            )

        elapsed_ms = (time.perf_counter() - stream_started) * 1000

        yield _sse(
            "done",
            {
                "status": result.get("status"),
                "latency_ms": elapsed_ms,
                "rewrite_used": result.get("rewrite_used", False),
            },
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )