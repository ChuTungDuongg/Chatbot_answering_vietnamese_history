import asyncio
import json
import logging
import re
import time
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, Iterator
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.api.conversations import OwnerId, StoreDependency, require_conversation
from app.agents.evidence_agent import EvidenceModelContractError
from app.chat.store import ConversationStore
from app.chat_modes import ChatMode, normalize_chat_mode
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
        url=chunk.get("url") if chunk else None,
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
    return normalize_chat_mode(
        payload.mode,
        default=normalize_chat_mode(settings.default_inference_mode, default=ChatMode.HYBRID),
    )


def _get_generation_runtime(
    request: Request,
    payload: ChatRequest | None = None,
) -> tuple[Any, Any] | tuple[Any, Any, InferenceMode]:
    service = request.app.state.rag_service
    selected_mode = _resolve_inference_mode(payload) if payload is not None else settings.default_inference_mode
    selected_mode = normalize_chat_mode(selected_mode, default=ChatMode.HYBRID)
    mode_router = getattr(request.app.state, "chat_mode_router", None)
    if mode_router is not None:
        try:
            runtime = mode_router.runtime_for(selected_mode)
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    elif selected_mode == ChatMode.HYBRID:
        runtime = getattr(request.app.state, "hybrid_runtime", None)
    elif selected_mode == ChatMode.THREE_LLM:
        runtime = getattr(request.app.state, "three_llm_runtime", None)
    else:
        runtime = getattr(request.app.state, "central_runtime", None)

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


_TRACE_SECRET_KEY_RE = re.compile(
    r"(?:authorization|cookie|api[_-]?key|secret|credential|password|environment|headers?|modal)",
    re.I,
)
_TRACE_OMITTED_KEYS = {
    "chain_of_thought",
    "developer_prompt",
    "hidden_reasoning",
    "messages",
    "prompt",
    "rationale",
    "raw_output",
    "reasoning",
    "scratchpad",
    "system_prompt",
    "user_prompt",
    "validated_source_text",
}


def _safe_trace_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        return "[bounded]"
    if isinstance(value, dict):
        return {
            str(key): _safe_trace_value(item, depth=depth + 1)
            for key, item in value.items()
            if not _TRACE_SECRET_KEY_RE.search(str(key))
            and str(key).lower() not in _TRACE_OMITTED_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_safe_trace_value(item, depth=depth + 1) for item in list(value)[:100]]
    if isinstance(value, str):
        safe = re.sub(
            r"(?i)\b(api[_ -]?key|authorization|bearer|token|secret|password)\b\s*[:=]\s*\S+",
            r"\1=[redacted]",
            value,
        )
        safe = re.sub(r"(?i)\b[A-Z]:\\(?:Users|ProgramData|Windows)\\[^\s\"']+", "[filesystem-path]", safe)
        safe = re.sub(r"/(?:home|root|etc|var)/[^\s\"']+", "[filesystem-path]", safe)
        return safe if len(safe) <= 800 else safe[:797].rstrip() + "..."
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:800]


def _trace_candidate(item: dict[str, Any], rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": str(item.get("chunk_id") or item.get("evidence_id") or ""),
        "title": item.get("title"),
        "source_kind": item.get("source_kind") or item.get("source_type") or "history",
        "retrieval_hits": item.get("retrieval_hits") or [],
        "retrieval_query_roles": item.get("retrieval_query_roles") or [],
        "best_dense_score": item.get("best_dense_score"),
        "best_bm25_score": item.get("best_bm25_score"),
        "rrf_score": item.get("rrf_score"),
        "reranker_score": item.get("reranker_score"),
        "final_retrieval_score": item.get("final_retrieval_score"),
        "comparison_target": item.get("comparison_target"),
        "incidental_target_penalty": item.get("incidental_target_penalty"),
        "text_preview": str(item.get("text_preview") or item.get("text") or "")[:260],
    }


def _research_retrieval_trace(research: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    plan: dict[str, Any] = {}
    merged: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = {"target_a": [], "target_b": [], "global": []}
    for step in research.get("tools", []) if isinstance(research, dict) else []:
        if not plan and isinstance(step.get("target_specific_queries"), dict):
            plan = dict(step["target_specific_queries"])
        for item in step.get("evidence", []) if isinstance(step, dict) else []:
            if not isinstance(item, dict):
                continue
            chunk_id = str(item.get("chunk_id") or "")
            if chunk_id:
                merged.setdefault(chunk_id, item)
            label = str(item.get("comparison_target") or "")
            roles = item.get("retrieval_query_roles") or []
            for role in ("target_a", "target_b", "global"):
                if label == role or role in roles:
                    grouped[role].append(item)
    ranked_groups = {
        role: [_trace_candidate(item, index) for index, item in enumerate(items[:10], 1)]
        for role, items in grouped.items()
    }
    ranked_groups["merged"] = [
        _trace_candidate(item, index)
        for index, item in enumerate(list(merged.values())[:30], 1)
    ]
    return plan, ranked_groups


def _build_debug(result: dict[str, Any]) -> dict[str, Any]:
    retrieval = result.get("retrieval") or {}
    research = result.get("research_debug") or {}
    evidence = result.get("evidence_debug") or {}
    history = result.get("history_debug") or {}
    provenance = result.get("answer_provenance") or {}
    analysis = result.get("analysis") or retrieval.get("analysis") or {}
    target_plan, research_rankings = _research_retrieval_trace(research)
    target_plan = retrieval.get("target_specific_queries") or target_plan
    retrieval_rankings = retrieval.get("target_retrieval_results") or research_rankings
    candidates = list(retrieval.get("candidates20") or [])
    merged_candidates = (
        [_trace_candidate(item, index) for index, item in enumerate(candidates[:30], 1)]
        if candidates
        else research_rankings.get("merged", [])
    )
    selected_contexts = [
        _trace_candidate(item, index)
        for index, item in enumerate((retrieval.get("final_context") or [])[:20], 1)
    ]
    normalized_rankings = {
        role: [
            _trace_candidate(item, index) if isinstance(item, dict) and "rank" not in item else item
            for index, item in enumerate(items[:10], 1)
        ]
        for role, items in retrieval_rankings.items()
        if isinstance(items, list)
    }
    source_chunks = result.get("source_chunks") or []
    sources = [
        {
            "chunk_id": str(item.get("chunk_id") or ""),
            "title": item.get("title"),
            "source_kind": item.get("source_kind") or item.get("source_type") or "history",
        }
        for item in source_chunks
        if item.get("chunk_id")
    ]
    research_latency_ms = sum(
        float(attempt.get("elapsed_ms") or 0.0)
        for attempt in research.get("attempts", [])
        if isinstance(attempt, dict)
    )
    performance_debug = result.get("performance_debug") or {}
    trace = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": result.get("inference_mode"),
        "request": {
            "mode": result.get("inference_mode"),
            "question": result.get("question") or retrieval.get("question"),
            "question_type": (
                history.get("question_type")
                or evidence.get("question_type")
                or analysis.get("question_type")
                or analysis.get("facet")
            ),
            "facet": analysis.get("facet"),
            "subject": analysis.get("subject"),
            "facets": analysis.get("facets") or [],
            "comparison_targets": evidence.get("comparison_targets") or analysis.get("comparison_targets") or [],
            "domain_result": retrieval.get("domain_gate_result"),
            "domain_reason": retrieval.get("domain_gate_reason"),
        },
        "retrieval": {
            "retrieval_question": retrieval.get("retrieval_question") or research.get("retrieval_question"),
            "query_variants": retrieval.get("query_variants") or [],
            "target_specific_queries": target_plan or {},
            "target_rankings": normalized_rankings,
            "merged_candidates": merged_candidates,
            "selected_contexts": selected_contexts,
            "comparison_balance": retrieval.get("comparison_balance") or {},
            "is_ood": retrieval.get("is_ood", False),
            "max_dense": retrieval.get("max_dense"),
        },
        "research": research if result.get("inference_mode") in {ChatMode.THREE_LLM, "agentic_rag"} else {},
        "evidence": evidence if result.get("inference_mode") in {ChatMode.THREE_LLM, "agentic_rag"} else {},
        "central": result.get("central_debug") if result.get("inference_mode") == ChatMode.CENTRAL else {},
        "history": {
            **history,
            "invalid_source_ids": result.get("invalid_source_ids", []),
            "unsupported_years": result.get("unsupported_years", []),
            "structured_expansion_used": result.get("structured_expansion_used", False),
        },
        "sources": sources,
        "performance": {
            "retrieval_latency_ms": float(result.get("retrieval_latency_sec") or 0.0) * 1000 or None,
            "research_latency_ms": research_latency_ms or None,
            "history_first_latency_ms": history.get("first_latency_ms"),
            "history_retry_latency_ms": history.get("retry_latency_ms"),
            "history_total_latency_ms": history.get("total_latency_ms"),
            "total_latency_ms": float(result.get("total_latency_sec") or result.get("latency_sec") or 0.0) * 1000 or None,
            "research_generation_calls": provenance.get("research_generation_calls"),
            "evidence_generation_calls": provenance.get("evidence_generation_calls"),
            "history_generation_calls": provenance.get("history_generation_calls"),
            "total_llm_calls": provenance.get("total_llm_calls"),
            **performance_debug,
        },
        "errors": [],
        "analysis": analysis,
        "tool_trace": result.get("tool_trace", []),
        "answer_provenance": provenance,
    }
    return _safe_trace_value(trace)


def _failure_debug_trace(
    *,
    payload: ChatRequest,
    mode: InferenceMode,
    stage: str,
    code: str,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _safe_trace_value({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "request": {"mode": mode, "question": payload.question},
        "retrieval": {},
        "research": {},
        "evidence": {},
        "history": {},
        "sources": [],
        "performance": {},
        "errors": [{"stage": stage, "code": code, **(diagnostics or {})}],
    })


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
    selected_mode = normalize_chat_mode(selected_mode)
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
        result["inference_mode"] = selected_mode
        result.setdefault("answer_provenance", {})["mode"] = selected_mode

        sources = _result_sources(service, result)
        stored_sources = [source.model_dump(mode="json") for source in sources]
        debug_trace = _build_debug(result) if bool(getattr(payload, "debug", False)) else None

        assistant_message = store.add_message(
            owner_id=owner_id,
            conversation_id=conversation_id,
            role="assistant",
            content=result["answer"],
            sources=stored_sources,
            debug_trace=debug_trace,
            status="done",
        )

        result["conversation_id"] = conversation_id
        result["message_id"] = str(assistant_message["id"])
        result["user_message_id"] = str(user_message["id"])
        result["debug_trace"] = debug_trace
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
        debug=result.get("debug_trace") if payload.debug else None,
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

        if selected_mode == ChatMode.CENTRAL:
            status_messages = [
                ("central_analyzing", "Central Agent đang phân tích câu hỏi..."),
                ("central_tools", "Central Agent đang thu thập bằng chứng..."),
                ("central_answering", "Central Agent đang tổng hợp câu trả lời..."),
            ]
        elif selected_mode == ChatMode.THREE_LLM:
            status_messages = [
                ("three_llm_research", "Research Agent đang thu thập bằng chứng..."),
                ("three_llm_evidence", "Evidence Agent đang kiểm tra bằng chứng..."),
                ("three_llm_answering", "History Answerer đang soạn câu trả lời..."),
            ]
        else:
            status_messages = [
                ("hybrid_retrieval", "Đang tìm trong kho sử liệu..."),
                ("hybrid_answering", "Đang chuẩn bị câu trả lời..."),
            ]
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
            error_payload = {"type": "not_found", "message": str(exc)}
            if payload.debug:
                error_payload["debug_trace"] = _failure_debug_trace(
                    payload=payload, mode=selected_mode, stage="conversation", code="not_found"
                )
            yield _sse("error", error_payload)
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
            error_payload = _evidence_contract_error_payload(exc)
            if payload.debug:
                error_payload["debug_trace"] = _failure_debug_trace(
                    payload=payload,
                    mode=selected_mode,
                    stage=exc.stage,
                    code=exc.code,
                    diagnostics={
                        "evidence_ids": exc.evidence_ids,
                        "repair_attempted": exc.repair_attempted,
                        "validation_errors": exc.validation_errors,
                    },
                )
            yield _sse("error", error_payload)
            yield _sse("done", {"status": "error", "latency_ms": (time.perf_counter() - stream_started) * 1000})
            return
        except ValueError as exc:
            error_payload = {"type": "bad_request", "message": str(exc)}
            if payload.debug:
                error_payload["debug_trace"] = _failure_debug_trace(
                    payload=payload, mode=selected_mode, stage="request", code="bad_request"
                )
            yield _sse("error", error_payload)
            yield _sse("done", {"status": "error", "latency_ms": (time.perf_counter() - stream_started) * 1000})
            return
        except RuntimeError as exc:
            logger.exception("Streaming generation runtime error")
            error_payload = {"type": "runtime_error", "message": str(exc)}
            if payload.debug:
                error_payload["debug_trace"] = _failure_debug_trace(
                    payload=payload,
                    mode=selected_mode,
                    stage="runtime",
                    code=type(exc).__name__,
                )
            yield _sse("error", error_payload)
            yield _sse("done", {"status": "error", "latency_ms": (time.perf_counter() - stream_started) * 1000})
            return
        except asyncio.CancelledError:
            if task is not None and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            raise
        except Exception as exc:
            logger.exception("Unexpected streaming chat error")
            error_payload = {
                "type": "internal_error",
                "message": "Internal generation error.",
            }
            if payload.debug:
                error_payload["debug_trace"] = _failure_debug_trace(
                    payload=payload,
                    mode=selected_mode,
                    stage="unknown",
                    code=type(exc).__name__,
                )
            yield _sse("error", error_payload)
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
            yield _sse("debug_trace", result.get("debug_trace") or _build_debug(result))

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
