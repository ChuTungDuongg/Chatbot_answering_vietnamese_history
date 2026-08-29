import asyncio
import json
import logging
import re
import time
import uuid
from contextlib import suppress
from typing import Any, Iterator
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.api.conversations import OwnerId, StoreDependency, require_conversation
from app.agents.evidence_agent import EvidenceModelContractError
from app.chat.store import ConversationStore
from app.config import settings
from app.telemetry import RequestTelemetry, log_event, reset_request_telemetry, set_request_telemetry
from app.schemas import (
    ChatRequest,
    ChatResponse,
    InferenceMode,
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
    if chunk.get("source_kind") == "wikipedia":
        return "wikipedia"
    if chunk.get("source_kind") == "web":
        return "web"

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


def _consume_task_exception(task: asyncio.Task) -> None:
    try:
        task.exception()
    except asyncio.CancelledError:
        return
    except Exception:
        logger.exception("Could not consume background task exception")


def _evidence_contract_error_payload(exc: EvidenceModelContractError) -> dict[str, Any]:
    diagnostics = {
        "stage": exc.stage,
        "code": exc.code,
        "evidence_ids": exc.evidence_ids,
        "repair_attempted": exc.repair_attempted,
        "validation_errors": exc.validation_errors,
    }
    return {
        "type": "evidence_contract_error",
        "stage": exc.stage,
        "code": exc.code,
        "message": exc.user_message,
        "evidence_ids": exc.evidence_ids,
        "repair_attempted": exc.repair_attempted,
        "validation_errors": exc.validation_errors,
        "diagnostics": diagnostics,
    }


def _format_evidence_contract_error(prefix: str, exc: EvidenceModelContractError) -> str:
    return (
        f"{prefix}:\n"
        f"stage={exc.stage}\n"
        f"code={exc.code}\n"
        f"repair_attempted={json.dumps(exc.repair_attempted)}\n"
        f"evidence_ids={json.dumps(exc.evidence_ids, ensure_ascii=False)}\n"
        f"validation_errors={json.dumps(exc.validation_errors, ensure_ascii=False)}"
    )


def _answer_chunks(text: str, words_per_chunk: int = 1) -> Iterator[str]:
    """Split an answer into small deltas while preserving whitespace and newlines."""
    pieces = re.findall(r"\S+\s*", text)

    for index in range(0, len(pieces), words_per_chunk):
        yield "".join(pieces[index:index + words_per_chunk])


def _resolve_inference_mode(payload: ChatRequest) -> InferenceMode:
    return payload.mode or settings.default_inference_mode


def _get_generation_runtime(
    request: Request,
    payload: ChatRequest | None = None,
) -> tuple[Any, Any] | tuple[Any, Any, InferenceMode]:
    service = request.app.state.rag_service
    selected_mode = _resolve_inference_mode(payload) if payload is not None else settings.default_inference_mode
    if selected_mode == "hybrid_rag":
        runtime = getattr(request.app.state, "hybrid_orchestrator", None)
    else:
        runtime = getattr(request.app.state, "agentic_orchestrator", None) or getattr(request.app.state, "orchestrator", None)
    generator = getattr(request.app.state, "generator", None)
    runtime = runtime or generator

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

    if payload is None:
        return service, runtime
    return service, runtime, selected_mode


def _build_debug(result: dict[str, Any]) -> dict[str, Any]:
    retrieval = result.get("retrieval") or {}

    return {
        "mode": result.get("inference_mode"),
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
        "domain_gate_result": retrieval.get("domain_gate_result"),
        "domain_gate_reason": retrieval.get("domain_gate_reason"),
        "history_anchor": (retrieval.get("intent") or {}).get("history_anchor"),
        "ood_anchor": (retrieval.get("intent") or {}).get("ood_anchor"),
        "domain_margin": (retrieval.get("intent") or {}).get("margin"),
        "query_variants": retrieval.get("query_variants", []),
        "retrieval_question": retrieval.get("retrieval_question"),
        "history_message_count": result.get("history_message_count", 0),
        "history_used_for_retrieval": retrieval.get("history_used_for_retrieval", False),
        "global_context_count": retrieval.get("global_context_count", 0),
        "temporary_context_count": retrieval.get("temporary_context_count", 0),
        "temporary_context_relevant": retrieval.get("temporary_context_relevant", False),
        "retrieval_latency_ms": result.get("retrieval_latency_sec", 0.0) * 1000,
    }


def _gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        return None
    return None


def _failure_stage(exc: Exception) -> str:
    if isinstance(exc, LookupError):
        return "conversation"
    if isinstance(exc, ValueError):
        return "request"
    if isinstance(exc, RuntimeError):
        return "runtime"
    return "unknown"


def _execute_chat(
    store: ConversationStore,
    generator: Any,
    service: Any,
    owner_id: str,
    payload: ChatRequest,
    request_id: str,
    selected_mode: InferenceMode,
) -> dict[str, Any]:
    telemetry = RequestTelemetry(
        request_id=request_id,
        inference_mode=selected_mode,
        selected_inference_mode=selected_mode,
        deployment_id=getattr(service, "deployment_id", None),
        gpu=_gpu_name(),
        cold_start_included=False,
    )
    token = set_request_telemetry(telemetry)
    result_status = "failed"
    conversation_id = str(payload.conversation_id)
    try:
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
            request_id=request_id,
        )
        result.setdefault("inference_mode", selected_mode)
        result.setdefault("answer_provenance", {})["mode"] = selected_mode

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
        result_status = "success"
        return result
    except EvidenceModelContractError as exc:
        telemetry.failed_stage = exc.stage
        telemetry.failure_code = exc.code
        raise
    except Exception as exc:
        telemetry.failed_stage = _failure_stage(exc)
        telemetry.failure_code = type(exc).__name__
        raise
    finally:
        log_event("REQUEST_SUMMARY", **telemetry.summary(result=result_status))
        reset_request_telemetry(token)


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
    service, generator, selected_mode = _get_generation_runtime(request, payload)
    await require_conversation(store, owner_id, payload.conversation_id)
    request_id = str(uuid.uuid4())

    started = time.perf_counter()

    try:
        result = await asyncio.to_thread(
            _execute_chat,
            store,
            generator,
            service,
            owner_id,
            payload,
            request_id,
            selected_mode,
        )
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except EvidenceModelContractError as exc:
        logger.warning(
            _format_evidence_contract_error("Chat evidence contract error", exc),
            extra={
                "stage": exc.stage,
                "code": exc.code,
                "evidence_ids": exc.evidence_ids,
                "repair_attempted": exc.repair_attempted,
                "validation_errors": exc.validation_errors,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_evidence_contract_error_payload(exc),
        ) from exc
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
        mode=selected_mode,
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
    service, generator, selected_mode = _get_generation_runtime(request, payload)
    await require_conversation(store, owner_id, payload.conversation_id)
    request_id = str(uuid.uuid4())
    log_event(
        "CHAT_STREAM_START",
        request_id=request_id,
        deployment_id=getattr(service, "deployment_id", None),
        inference_mode=selected_mode,
    )

    async def event_stream():
        stream_started = time.perf_counter()
        task: asyncio.Task | None = None

        status_messages = (
            [
                ("agentic_analyzing", "Đang phân tích câu hỏi..."),
                ("agentic_local_search", "Đang tìm trong kho sử liệu..."),
                ("agentic_external_check", "Đang kiểm tra thêm nguồn ngoài..."),
                ("agentic_evidence_check", "Đang đối chiếu bằng chứng..."),
                ("agentic_answering", "Đang soạn câu trả lời..."),
            ]
            if selected_mode == "agentic_rag"
            else [
                ("hybrid_retrieval", "Đang truy xuất kho sử liệu..."),
                ("hybrid_answering", "Đang soạn câu trả lời..."),
            ]
        )
        yield _sse("status", {"stage": status_messages[0][0], "message": status_messages[0][1], "mode": selected_mode})

        task = asyncio.create_task(
            asyncio.to_thread(
                _execute_chat,
                store,
                generator,
                service,
                owner_id,
                payload,
                request_id,
                selected_mode,
            )
        )
        task.add_done_callback(_consume_task_exception)

        try:
            while not task.done():
                if await request.is_disconnected():
                    break
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=8.0)
                except asyncio.TimeoutError:
                    status_index = min(
                        int((time.perf_counter() - stream_started) // 8),
                        len(status_messages) - 1,
                    )
                    yield _sse(
                        "status",
                        {
                            "stage": status_messages[status_index][0],
                            "message": status_messages[status_index][1],
                            "mode": selected_mode,
                        },
                    )
                    yield _sse("ping", {"timestamp": time.time()})

            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                return

            result = await task
        except LookupError as exc:
            yield _sse("error", {"type": "not_found", "message": str(exc)})
            yield _sse("done", {"status": "error", "latency_ms": (time.perf_counter() - stream_started) * 1000})
            return
        except EvidenceModelContractError as exc:
            logger.warning(
                _format_evidence_contract_error("Streaming evidence contract error", exc),
                extra={
                    "stage": exc.stage,
                    "code": exc.code,
                    "evidence_ids": exc.evidence_ids,
                    "repair_attempted": exc.repair_attempted,
                    "validation_errors": exc.validation_errors,
                },
            )
            yield _sse("error", _evidence_contract_error_payload(exc))
            yield _sse("done", {"status": "error", "latency_ms": (time.perf_counter() - stream_started) * 1000})
            return
        except ValueError as exc:
            yield _sse("error", {"type": "bad_request", "message": str(exc)})
            yield _sse("done", {"status": "error", "latency_ms": (time.perf_counter() - stream_started) * 1000})
            return
        except RuntimeError as exc:
            logger.exception("Streaming generation runtime error")
            yield _sse("error", {"type": "runtime_error", "message": str(exc)})
            yield _sse("done", {"status": "error", "latency_ms": (time.perf_counter() - stream_started) * 1000})
            return
        except asyncio.CancelledError:
            if task is not None and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            raise
        except Exception:
            logger.exception("Unexpected streaming chat error")
            yield _sse(
                "error",
                {
                    "type": "internal_error",
                    "message": "Internal generation error.",
                },
            )
            yield _sse("done", {"status": "error", "latency_ms": (time.perf_counter() - stream_started) * 1000})
            return
        finally:
            if task is not None and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

        yield _sse(
            "status",
            {
                "stage": "validated",
                "status": result.get("status"),
                "rewrite_used": result.get("rewrite_used", False),
                "mode": selected_mode,
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
                "mode": selected_mode,
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
