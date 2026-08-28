import asyncio
import json
import logging
import re
import time
from typing import Any, Iterator
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.api.conversations import OwnerId, StoreDependency, require_conversation
from app.chat.store import ConversationStore
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

def _source_kind(chunk: dict[str, Any]) -> str:
    if chunk.get("source_kind") == "attachment":
        return "attachment"

    if str(chunk.get("chunk_id", "")).startswith("temp:"):
        return "attachment"

    return "history"


def _context_to_api(chunk: dict[str, Any]) -> RetrievalContextItem:
    return RetrievalContextItem(
        chunk_id=str(chunk.get("chunk_id", "")),
        title=chunk.get("title"),
        text=chunk.get("text"),
        source_kind=_source_kind(chunk),
        attachment_id=chunk.get("attachment_id"),
        page_number=chunk.get("page_number"),
        final_retrieval_score=chunk.get("final_retrieval_score"),
        reranker_score=chunk.get("reranker_score"),
        rrf_score=chunk.get("rrf_score"),
        metadata_bonus=chunk.get("metadata_bonus"),
        metadata_hits=chunk.get("metadata_hits") or [],
    )


def _source_to_api(
    service: Any,
    source_id: str,
    context_by_id: dict[str, dict[str, Any]] | None = None,
) -> SourceItem:
    source_id = str(source_id)
    context_by_id = context_by_id or {}

    chunk = context_by_id.get(source_id)
    if chunk is None:
        chunk = service.chunk_by_id.get(source_id)

    return SourceItem(
        chunk_id=source_id,
        title=chunk.get("title") if chunk else None,
        source_kind=_source_kind(chunk or {"chunk_id": source_id}),
        attachment_id=chunk.get("attachment_id") if chunk else None,
        page_number=chunk.get("page_number") if chunk else None,
    )


def _result_sources(service: Any, result: dict[str, Any]) -> list[SourceItem]:
    source_chunks = result.get("source_chunks") or []
    context_by_id = {
        str(chunk.get("chunk_id", "")): chunk
        for chunk in source_chunks
        if chunk.get("chunk_id")
    }

    return [
        _source_to_api(service, source_id, context_by_id)
        for source_id in result.get("source_ids", [])
    ]


def _sse(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _answer_chunks(text: str, words_per_chunk: int = 1) -> Iterator[str]:
    """Split an answer into small deltas while preserving whitespace and newlines."""
    pieces = re.findall(r"\S+\s*", text)

    for index in range(0, len(pieces), words_per_chunk):
        yield "".join(pieces[index:index + words_per_chunk])


def _get_generation_runtime(request: Request) -> tuple[Any, Any]:
    service = request.app.state.rag_service
    generator = getattr(request.app.state, "generator", None)
    runtime = getattr(request.app.state, "orchestrator", None) or generator

    if runtime is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Generation runtime is not loaded. Use APP_MODE=full to enable chat.",
        )

    generation_ready = (
        service.model is not None
        or bool(getattr(service, "external_generation_backend", False))
    )

    if not service.loaded or not generation_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Generation model is not ready.",
        )

    return service, runtime


def _build_debug(result: dict[str, Any]) -> dict[str, Any]:
    retrieval = result.get("retrieval") or {}

    return {
        "answer_provenance": result.get("answer_provenance", {}),
        "research": result.get("research_debug", {}),
        "evidence": result.get("evidence_debug", {}),
        "history": result.get("history_debug", {}),
        "analysis": result.get("analysis"),
        "tool_trace": result.get("tool_trace", []),
        "prompt_budget": result.get("prompt_budget"),
        "support_score": result.get("support_score"),
        "quality_warnings": result.get("quality_warnings", []),
        "initial_quality_issues": result.get("initial_quality_issues", []),
        "repair_attempted": result.get("repair_attempted", False),
        "repair_diagnostics": result.get("repair_diagnostics"),
        "structured_expansion_used": result.get("structured_expansion_used", False),
        "model_source_ids": result.get("model_source_ids", []),
        "invalid_source_ids": result.get("invalid_source_ids", []),
        "unsupported_years": result.get("unsupported_years", []),
        "format_ok": result.get("format_ok"),
        "is_ood": retrieval.get("is_ood", False),
        "ood_reason": retrieval.get("ood_reason", ""),
        "global_ood_reason": retrieval.get("global_ood_reason"),
        "query_variants": retrieval.get("query_variants", []),
        "retrieval_question": retrieval.get("retrieval_question"),
        "history_message_count": result.get("history_message_count", 0),
        "history_used_for_retrieval": retrieval.get("history_used_for_retrieval", False),
        "global_context_count": retrieval.get("global_context_count", 0),
        "temporary_context_count": retrieval.get("temporary_context_count", 0),
        "temporary_context_relevant": retrieval.get("temporary_context_relevant", False),
        "retrieval_latency_ms": result.get("retrieval_latency_sec", 0.0) * 1000,
    }


def _execute_chat(
    store: ConversationStore,
    generator: Any,
    service: Any,
    owner_id: str,
    payload: ChatRequest,
) -> dict[str, Any]:
    conversation_id = str(payload.conversation_id)
    history_limit = max(
        int(getattr(generator, "max_history_messages", 6)),
        int(getattr(generator, "retrieval_history_messages", 4)),
        6,
    )

    history = store.get_recent_history(owner_id, conversation_id, limit=history_limit)

    user_message = store.add_message(
        owner_id=owner_id,
        conversation_id=conversation_id,
        role="user",
        content=payload.question,
        status="done",
    )

    result = generator.chat(
        question=payload.question,
        final_k=payload.final_k,
        history=history,
        owner_id=owner_id,
        conversation_id=conversation_id,
    )

    sources = _result_sources(service, result)
    stored_sources = [source.model_dump(mode="json") for source in sources]

    assistant_message = store.add_message(
        owner_id=owner_id,
        conversation_id=conversation_id,
        role="assistant",
        content=result["answer"],
        sources=stored_sources,
        status="done",
    )

    result["conversation_id"] = conversation_id
    result["message_id"] = str(assistant_message["id"])
    result["user_message_id"] = str(user_message["id"])

    return result


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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
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
    final_context = [_context_to_api(chunk) for chunk in result.get("final_context", [])]

    candidates = None
    tool_trace = None

    if payload.debug:
        candidates = [_context_to_api(chunk) for chunk in result.get("candidates20", [])]
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
async def chat(
    payload: ChatRequest,
    request: Request,
    owner_id: OwnerId,
    store: StoreDependency,
) -> ChatResponse:
    service, generator = _get_generation_runtime(request)
    await require_conversation(store, owner_id, payload.conversation_id)

    started = time.perf_counter()

    try:
        result = await asyncio.to_thread(
            _execute_chat,
            store,
            generator,
            service,
            owner_id,
            payload,
        )
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
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

    return ChatResponse(
        conversation_id=result["conversation_id"],
        message_id=result["message_id"],
        answer=result["answer"],
        status=result["status"],
        sources=_result_sources(service, result),
        latency_ms=latency_ms,
        rewrite_used=result.get("rewrite_used", False),
        debug=_build_debug(result) if payload.debug else None,
    )


# ============================================================
# Validated SSE chat stream
# ============================================================

@router.post("/chat/stream", status_code=status.HTTP_200_OK)
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    owner_id: OwnerId,
    store: StoreDependency,
) -> StreamingResponse:
    service, generator = _get_generation_runtime(request)
    await require_conversation(store, owner_id, payload.conversation_id)

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
            asyncio.to_thread(
                _execute_chat,
                store,
                generator,
                service,
                owner_id,
                payload,
            )
        )

        while not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=8.0)
            except asyncio.TimeoutError:
                yield _sse("ping", {"timestamp": time.time()})

        try:
            result = await task
        except LookupError as exc:
            yield _sse("error", {"type": "not_found", "message": str(exc)})
            return
        except ValueError as exc:
            yield _sse("error", {"type": "bad_request", "message": str(exc)})
            return
        except RuntimeError as exc:
            logger.exception("Streaming generation runtime error")
            yield _sse("error", {"type": "runtime_error", "message": str(exc)})
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

        yield _sse(
            "status",
            {
                "stage": "validated",
                "status": result.get("status"),
                "rewrite_used": result.get("rewrite_used", False),
            },
        )

        for delta in _answer_chunks(result["answer"], words_per_chunk=1):
            yield _sse("answer_delta", {"delta": delta})
            await asyncio.sleep(0.012)

        retrieval = result.get("retrieval") or {}
        final_context = retrieval.get("final_context") or []
        cited_source_ids = [str(source_id) for source_id in result.get("source_ids", [])]
        cited_source_id_set = set(cited_source_ids)

        if final_context:
            sources = []

            for chunk in final_context:
                item = _context_to_api(chunk).model_dump(mode="json")
                item["cited"] = str(item.get("chunk_id", "")) in cited_source_id_set
                sources.append(item)
        else:
            sources = [
                {
                    **source.model_dump(mode="json"),
                    "cited": True,
                }
                for source in _result_sources(service, result)
            ]

        yield _sse(
            "sources",
            {
                "items": sources,
                "cited_source_ids": cited_source_ids,
                "final_context_count": len(final_context),
            },
        )

        if payload.debug:
            yield _sse("debug", _build_debug(result))

        elapsed_ms = (time.perf_counter() - stream_started) * 1000

        yield _sse(
            "done",
            {
                "conversation_id": result["conversation_id"],
                "message_id": result["message_id"],
                "user_message_id": result["user_message_id"],
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
