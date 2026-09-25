"""Retrieval, chat, and genuine model-streamed SSE endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.api.conversations import OwnerId, StoreDependency, require_conversation
from app.chat_modes import ChatMode, normalize_chat_mode
from app.config import settings
from app.models.base import ModelDelta, ModelDone
from app.services.metadata import build_baseline_metadata
from app.schemas import (ChatRequest, ChatResponse, RetrieveRequest, RetrieveResponse,
                         RetrievalContextItem, SourceItem)
from app.telemetry import RequestTrace, reset_request_telemetry, set_request_telemetry


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["RAG"])


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def _source_kind(chunk: dict[str, Any]) -> str:
    value = str(chunk.get("source_kind") or "history")
    if value in {"history", "attachment", "wikipedia", "web"}:
        return value
    return "attachment" if str(chunk.get("chunk_id") or "").startswith("temp:") else "history"


def _source_id(chunk: dict[str, Any]) -> str | None:
    if chunk.get("source_id"):
        return str(chunk["source_id"])
    metadata = chunk.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("source_sha1"):
        return str(metadata["source_sha1"])
    if chunk.get("hf_dataset") is not None and chunk.get("raw_record_index") is not None:
        return f"{chunk['hf_dataset']}:{chunk['raw_record_index']}"
    return None


def _context_to_api(chunk: dict[str, Any], *, final_rank: int | None = None) -> RetrievalContextItem:
    dense_ranks = [int(match.group(1)) for hit in chunk.get("retrieval_hits") or []
                   if (match := re.match(r"dense:q\d+@(\d+)$", str(hit)))]
    bm25_ranks = [int(match.group(1)) for hit in chunk.get("retrieval_hits") or []
                  if (match := re.match(r"bm25:q\d+@(\d+)$", str(hit)))]
    return RetrievalContextItem(
        chunk_id=str(chunk.get("chunk_id") or ""), source_id=_source_id(chunk),
        display_index=chunk.get("display_index") or final_rank, title=chunk.get("title"),
        text=chunk.get("text"), url=chunk.get("url"), source_kind=_source_kind(chunk),
        attachment_id=chunk.get("attachment_id"), page_number=chunk.get("page_number"),
        final_retrieval_score=chunk.get("final_retrieval_score"),
        reranker_score=chunk.get("reranker_score"), rrf_score=chunk.get("rrf_score"),
        metadata_bonus=chunk.get("metadata_bonus"), metadata_hits=chunk.get("metadata_hits") or [],
        dense_rank=min(dense_ranks) if dense_ranks else None,
        bm25_rank=min(bm25_ranks) if bm25_ranks else None,
        rrf_rank=chunk.get("rrf_rank"), reranker_rank=chunk.get("reranker_rank"),
        final_rank=final_rank or chunk.get("final_rank"),
        best_dense_score=chunk.get("best_dense_score"), best_bm25_score=chunk.get("best_bm25_score"),
    )


def _cited_ids(answer: str, contexts: list[dict[str, Any]]) -> list[str]:
    known = {str(chunk["chunk_id"]): str(chunk["chunk_id"])
             for chunk in contexts if chunk.get("chunk_id")}
    for chunk in contexts:
        if chunk.get("chunk_id") and (source_id := _source_id(chunk)):
            known.setdefault(source_id, str(chunk["chunk_id"]))
    return list(dict.fromkeys(known[match] for match in re.findall(r"\[([^\]\n]+)\]", answer)
                              if match in known))


def _sources(answer: str, contexts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    cited = _cited_ids(answer, contexts)
    marked = []
    for rank, chunk in enumerate(contexts, 1):
        item = _context_to_api(chunk, final_rank=rank).model_dump(mode="json")
        item["cited"] = item["chunk_id"] in cited
        marked.append(item)
    return marked, cited


def _debug_trace(mode: ChatMode, trace: RequestTrace, prepared: Any,
                 source_items: list[dict[str, Any]], metrics: dict[str, Any],
                 model: Any) -> dict[str, Any]:
    diagnostics = []
    for item in source_items:
        diagnostics.append({key: item.get(key) for key in (
            "chunk_id", "source_id", "dense_rank", "bm25_rank", "rrf_rank",
            "reranker_rank", "final_rank", "best_dense_score", "best_bm25_score",
            "rrf_score", "reranker_score", "final_retrieval_score")})
    return {"schema_version": 1, "mode": mode.value,
            "request": {"request_id": trace.request_id},
            "retrieval": {"query_variants": prepared.retrieval.get("query_variants") or [],
                          "final_context": diagnostics},
            "tool_trace": prepared.tool_calls,
            "generation": {"model_id": model.model_id,
                           "model_revision": getattr(model, "resolved_revision", None),
                           "settings": model.generation_settings},
            "sources": [{"chunk_id": item["chunk_id"], "cited": item["cited"]}
                        for item in source_items],
            "performance": metrics}


def _model_metrics(completed: ModelDone | None, trace: RequestTrace,
                   prepared: Any, delta_times_ns: list[int]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "first_status_event_ms": trace.offset_ms("first_status_event"),
        "retrieval_ms": prepared.retrieval_ms if prepared else None,
        "prompt_build_ms": prepared.prompt_build_ms if prepared else None,
        "generation_start_ms": trace.offset_ms("generation_started"),
        "model_ttft_ms": None,
        "generation_ms": None,
        "input_tokens": None,
        "output_tokens": None,
        "tokens_per_second": None,
        "decode_tokens_per_second": None,
        "tpot_ms": None,
        "itl_p50_ms": None,
        "itl_p95_ms": None,
        "itl_p99_ms": None,
        "model_calls": None,
        "tool_calls": None,
        "tool_call_types": None,
        "tool_execution_ms": None,
        "tool_parse_failures": None,
        "action_rounds": None,
        "time_until_final_generation_ms": None,
        "final_answer_ttft_ms": trace.offset_ms("first_answer_delta_sent"),
        "e2e_ms": trace.offset_ms("request_finished"),
    }
    if completed:
        metrics.update(completed.metrics)
    if prepared:
        metrics["model_calls"] = prepared.model_calls_before_final + (1 if completed else 0)
        metrics["tool_calls"] = len(prepared.tool_calls)
        metrics["tool_call_types"] = [item["name"] for item in prepared.tool_calls]
        metrics["tool_execution_ms"] = sum(item["latency_ms"] for item in prepared.tool_calls)
        metrics["tool_parse_failures"] = prepared.tool_parse_failures
        metrics["action_rounds"] = prepared.action_rounds
        metrics["time_until_final_generation_ms"] = trace.offset_ms("generation_started")
    if len(delta_times_ns) > 1:
        gaps = sorted((b - a) / 1e6 for a, b in zip(delta_times_ns, delta_times_ns[1:]))
        for percentile in (50, 95, 99):
            index = (len(gaps) - 1) * percentile / 100
            lower = int(index)
            upper = min(lower + 1, len(gaps) - 1)
            metrics[f"itl_p{percentile}_ms"] = gaps[lower] + (gaps[upper] - gaps[lower]) * (index - lower)
    return metrics


def _runtime(request: Request, mode: ChatMode):
    router_instance = getattr(request.app.state, "chat_mode_router", None)
    if router_instance is None:
        raise HTTPException(status_code=503, detail="Generation runtime is not loaded; use APP_MODE=full")
    try:
        return router_instance.runtime_for(mode)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


async def _watch_disconnect(request: Request, cancel: threading.Event) -> None:
    while not cancel.is_set():
        if await request.is_disconnected():
            cancel.set()
            return
        await asyncio.sleep(0.25)


async def _execute(payload: ChatRequest, request: Request, owner_id: str, store: Any,
                   runtime: Any, mode: ChatMode, trace: RequestTrace,
                   *, watch_disconnect: bool) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    cancel = threading.Event()
    monitor = asyncio.create_task(_watch_disconnect(request, cancel)) if watch_disconnect else None
    prepared = None
    completed: ModelDone | None = None
    delta_times: list[int] = []
    answer_parts: list[str] = []
    token = set_request_telemetry(trace)
    try:
        trace.mark("first_status_event")
        yield "status", {"stage": "retrieval", "message": "Đang tìm tư liệu...", "mode": mode.value,
                         "request_id": trace.request_id}
        requested_ids = tuple(str(value) for value in payload.attachment_ids)
        if requested_ids:
            ready = {str(item["id"]): item for item in await asyncio.to_thread(
                store.list_attachments, owner_id, payload.conversation_id)
                if item["status"] == "ready" and int(item.get("chunk_count") or 0) > 0}
            if any(value not in ready for value in requested_ids):
                raise ValueError("Tài liệu không tồn tại, chưa đọc xong hoặc đã bị xóa.")
        else:
            ready = {}
        question = payload.question.strip() or "Phân tích nội dung tài liệu đính kèm."
        history = await asyncio.to_thread(store.get_recent_history, owner_id, payload.conversation_id, 6)
        attachment_sources = [{"chunk_id": f"attachment:{value}", "attachment_id": value,
                               "title": ready[value]["filename"], "source_kind": "attachment"}
                              for value in requested_ids]
        user_message = await asyncio.to_thread(store.add_message, owner_id, payload.conversation_id,
                                                "user", payload.question, attachment_sources)
        prepared = await runtime.prepare(question, payload.final_k, history,
                                         owner_id=owner_id, conversation_id=str(payload.conversation_id),
                                         attachment_ids=requested_ids, trace=trace, cancel=cancel)
        if cancel.is_set():
            return
        yield "status", {"stage": "generation", "message": "Đang tạo câu trả lời...", "mode": mode.value}
        max_tokens = (settings.hybrid_max_new_tokens if mode == ChatMode.HYBRID
                      else settings.central_final_max_new_tokens)
        async for item in runtime.model.stream(prepared.messages, max_new_tokens=max_tokens, cancel=cancel):
            if cancel.is_set():
                return
            if isinstance(item, ModelDelta):
                answer_parts.append(item.text)
                delta_times.append(item.produced_ns)
                if "first_answer_delta_sent" not in trace.marks_ns:
                    trace.mark("first_answer_delta_sent")
                yield "answer_delta", {"delta": item.text}
            else:
                completed = item
        if cancel.is_set():
            return
        if completed is None:
            raise RuntimeError("Model generation ended without completion metadata")
        trace.mark("generation_started", completed.started_ns)
        if completed.first_token_ns is not None:
            trace.mark("first_model_token", completed.first_token_ns)
        trace.mark("generation_finished", completed.finished_ns)
        answer = "".join(answer_parts)
        if not answer.strip():
            raise RuntimeError("Model returned an empty answer")
        source_items, cited_ids = _sources(answer, prepared.contexts)
        trace.mark("sources_ready")
        stored_sources = source_items
        debug_trace = (_debug_trace(mode, trace, prepared, source_items,
                                    _model_metrics(completed, trace, prepared, delta_times), runtime.model)
                       if payload.debug else None)
        assistant_message = await asyncio.to_thread(
            store.add_message, owner_id, payload.conversation_id, "assistant", answer, stored_sources,
            debug_trace
        )
        yield "sources", {"items": source_items, "cited_source_ids": cited_ids,
                          "final_context_count": len(prepared.contexts)}
        trace.mark("request_finished")
        metrics = _model_metrics(completed, trace, prepared, delta_times)
        done = {"request_id": trace.request_id, "conversation_id": str(payload.conversation_id),
                "message_id": str(assistant_message["id"]), "user_message_id": str(user_message["id"]),
                "answer": answer, "status": "done", "mode": mode.value,
                "latency_ms": metrics["e2e_ms"], "model_id": runtime.model.model_id,
                "model_revision": completed.model_revision,
                "generation_settings": {**runtime.model.generation_settings, "max_new_tokens": max_tokens},
                "retrieval_settings": getattr(request.app.state.retriever, "retrieval_config", {}),
                "metrics": metrics}
        if callable(done["retrieval_settings"]):
            done["retrieval_settings"] = done["retrieval_settings"]()
        trace.emit(metrics=metrics, model_id=runtime.model.model_id,
                   model_revision=completed.model_revision)
        if payload.debug:
            yield "debug_trace", _debug_trace(mode, trace, prepared, source_items, metrics, runtime.model)
        yield "done", done
    except asyncio.CancelledError:
        cancel.set()
        raise
    except Exception as exc:
        logger.exception("Chat request failed", extra={"request_id": trace.request_id})
        trace.mark("request_finished")
        metrics = _model_metrics(completed, trace, prepared, delta_times)
        trace.emit(metrics=metrics, model_id=runtime.model.model_id,
                   model_revision=getattr(runtime.model, "resolved_revision", None), error=type(exc).__name__)
        yield "error", {"type": type(exc).__name__, "message": str(exc), "request_id": trace.request_id}
        yield "done", {"request_id": trace.request_id, "status": "error", "mode": mode.value,
                       "model_id": runtime.model.model_id, "model_revision": getattr(runtime.model, "resolved_revision", None),
                       "metrics": metrics, "latency_ms": metrics["e2e_ms"]}
    finally:
        cancel.set()
        if monitor:
            monitor.cancel()
            try:
                await monitor
            except asyncio.CancelledError:
                pass
        reset_request_telemetry(token)


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(payload: RetrieveRequest, request: Request) -> RetrieveResponse:
    service = getattr(request.app.state, "rag_service", None)
    retriever = getattr(request.app.state, "retriever", None)
    if retriever is None or service is None or not service.loaded:
        raise HTTPException(status_code=503, detail="Retrieval runtime is not loaded")
    import time

    started = time.perf_counter_ns()
    result = await asyncio.to_thread(retriever.retrieve, payload.question, payload.final_k)
    return RetrieveResponse(
        question=result["question"], is_ood=result.get("is_ood", False),
        ood_reason=result.get("ood_reason", ""), intent=result.get("intent"),
        analysis=result.get("analysis"), query_variants=result.get("query_variants", []),
        final_context=[_context_to_api(item, final_rank=rank) for rank, item in enumerate(result.get("final_context", []), 1)],
        candidates=[_context_to_api(item) for item in result.get("candidates20", [])] if payload.debug else None,
        tool_trace=result.get("tool_trace") if payload.debug else None,
        max_dense=result.get("max_dense"),
        context_title_diversity=result.get("context_title_diversity", 0.0),
        latency_ms=(time.perf_counter_ns() - started) / 1e6,
    )


@router.get("/baseline/metadata")
async def baseline_metadata(request: Request, mode: ChatMode = ChatMode.HYBRID) -> dict[str, Any]:
    return await asyncio.to_thread(build_baseline_metadata, mode.value, request.app)


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request, owner_id: OwnerId,
               store: StoreDependency) -> ChatResponse:
    mode = normalize_chat_mode(payload.mode, default=settings.default_inference_mode)
    runtime = _runtime(request, mode)
    await require_conversation(store, owner_id, payload.conversation_id)
    trace = RequestTrace(str(uuid.uuid4()), mode.value)
    result = None
    failure = None
    sources: list[dict[str, Any]] = []
    debug_trace = None
    async for event, data in _execute(payload, request, owner_id, store, runtime, mode, trace,
                                      watch_disconnect=False):
        if event == "error":
            failure = data
        elif event == "sources":
            sources = data["items"]
        elif event == "debug_trace":
            debug_trace = data
        elif event == "done":
            result = data
    if failure or not result or result["status"] != "done":
        raise HTTPException(status_code=500, detail=(failure or {}).get("message", "Generation failed"))
    return ChatResponse(conversation_id=payload.conversation_id, message_id=result["message_id"],
                        answer=result["answer"], status="done", mode=mode,
                        sources=[SourceItem.model_validate(item) for item in sources],
                        latency_ms=result["latency_ms"], debug=debug_trace)


@router.post("/chat/stream", status_code=status.HTTP_200_OK)
async def chat_stream(payload: ChatRequest, request: Request, owner_id: OwnerId,
                      store: StoreDependency) -> StreamingResponse:
    mode = normalize_chat_mode(payload.mode, default=settings.default_inference_mode)
    runtime = _runtime(request, mode)
    await require_conversation(store, owner_id, payload.conversation_id)
    trace = RequestTrace(str(uuid.uuid4()), mode.value)

    async def events() -> AsyncIterator[str]:
        async for event, data in _execute(payload, request, owner_id, store, runtime, mode, trace,
                                          watch_disconnect=True):
            if event == "done":
                data = {key: value for key, value in data.items() if key != "answer"}
            yield _sse(event, data)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                                      "X-Request-ID": trace.request_id})
