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
from app.rag.backends import DenseBackendError
from app.tools.policy import ToolPolicyError
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
    if value in {"history", "attachment", "wikipedia", "web", "mcp"}:
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
            "request": {"request_id": trace.request_id, "mode": mode.value,
                        "retrieval_backend": prepared.retrieval.get("retrieval_backend"),
                        **({"steering": prepared.retrieval["steering"]} if "steering" in prepared.retrieval else {})},
            "retrieval": {"query_variants": prepared.retrieval.get("query_variants") or [],
                          "backend": prepared.retrieval.get("retrieval_backend"),
                          "timings_ms": prepared.retrieval.get("timings_ms", {}),
                          **({"history_retrieval_executed": prepared.retrieval["history_retrieval_executed"]}
                             if "history_retrieval_executed" in prepared.retrieval else {}),
                          "final_context": diagnostics},
            "tool_trace": prepared.tool_calls,
            **({"planning": prepared.planning} if prepared.planning else {}),
            **({"mcp": prepared.retrieval["mcp"]} if prepared.retrieval.get("mcp") else {}),
            "generation": {"model_id": model.model_id,
                           "model_revision": getattr(model, "resolved_revision", None),
                           "settings": {**model.generation_settings,
                                        "max_new_tokens": metrics.get("max_new_tokens")},
                           **{key: metrics.get(key) for key in
                              ("finish_reason", "hit_max_new_tokens", "truncated")}},
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
        "max_new_tokens": None,
        "finish_reason": None,
        "hit_max_new_tokens": None,
        "truncated": None,
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
        if prepared.planning:
            planning = prepared.planning
            metrics.update({"planning_model_ms": planning["total_model_ms"],
                            "planning_model_ttft_ms": planning["total_ttft_ms"],
                            "planning_input_tokens": planning["input_tokens"],
                            "planning_output_tokens": planning["output_tokens"],
                            "planning_wall_ms": planning["total_wall_ms"]})
        mcp_metrics = prepared.retrieval.get("mcp", [])
        if mcp_metrics:
            metrics["mcp_tool_ms"] = sum(item["mcp_tool_ms"] for item in mcp_metrics)
            metrics["mcp_calls"] = len(mcp_metrics)
        metrics.update({f"{name}_ms": value for name, value in prepared.retrieval.get("timings_ms", {}).items()})
        metrics["model_calls"] = prepared.model_calls_before_final + (1 if completed else 0)
        metrics["tool_calls"] = len(prepared.tool_calls)
        metrics["tool_call_types"] = [item["name"] for item in prepared.tool_calls]
        metrics["tool_execution_ms"] = sum(item["latency_ms"] for item in prepared.tool_calls)
        metrics["tool_parse_failures"] = prepared.tool_parse_failures
        metrics["action_rounds"] = prepared.action_rounds
        metrics["time_until_final_generation_ms"] = trace.offset_ms("generation_started")
        prepare_ms = trace.span_ms("prepare_started", "prompt_ready")
        planner_wall_ms = prepared.planning.get("total_wall_ms", 0.0)
        # Non-overlapping wall-time buckets, including model load/tokenization
        # and queueing which generation-only ModelDone metrics do not include.
        metrics["pre_final_breakdown_ms"] = {
            "request_preparation": trace.offset_ms("prepare_started"),
            "planning": planner_wall_ms,
            "tools": metrics["tool_execution_ms"],
            "prompt_build": prepared.prompt_build_ms,
            "orchestration": max(0.0, prepare_ms - planner_wall_ms - metrics["tool_execution_ms"]
                                 - (prepared.prompt_build_ms or 0.0)) if prepare_ms is not None else None,
            "prepare_to_model_request": trace.span_ms("prompt_ready", "final_model_requested"),
            "final_model_preparation_and_queue": trace.span_ms("final_model_requested", "generation_started"),
        } if prepared.planning else {}
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


def _resolve_backend(request: Request, selected: str | None):
    service = getattr(request.app.state, "rag_service", None)
    backend = selected or getattr(service, "default_dense_backend", settings.retrieval_dense_backend)
    if service is not None and callable(getattr(service, "get_dense_retriever", None)):
        try:
            service.get_dense_retriever(backend)
        except DenseBackendError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None
    return backend


def _resolve_tools(runtime, payload, mode):
    if mode != ChatMode.CENTRAL:
        if payload.steering is not None:
            raise HTTPException(status_code=422, detail="Tool steering chỉ hỗ trợ Central Agent.")
        return None
    resolver = getattr(runtime, "resolve_tools", None)
    if not callable(resolver):
        if payload.steering is not None:
            raise HTTPException(status_code=503, detail="Runtime chưa hỗ trợ tool steering.")
        return None
    try:
        return resolver(payload.steering, payload.question)
    except ToolPolicyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


async def _watch_disconnect(request: Request, cancel: threading.Event) -> None:
    while not cancel.is_set():
        if await request.is_disconnected():
            cancel.set()
            return
        await asyncio.sleep(0.25)


async def _execute(payload: ChatRequest, request: Request, owner_id: str, store: Any,
                   runtime: Any, mode: ChatMode, trace: RequestTrace,
                   *, watch_disconnect: bool, tool_view=None) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    cancel = threading.Event()
    monitor = asyncio.create_task(_watch_disconnect(request, cancel)) if watch_disconnect else None
    prepared = None
    completed: ModelDone | None = None
    delta_times: list[int] = []
    answer_parts: list[str] = []
    backend = _resolve_backend(request, payload.retrieval_backend)
    prepare_task = None
    progress_waiter = None
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    current_stage = "request_preparation"
    completed_detail_stages: set[str] = set()
    seen_detail_stages: set[str] = set()
    detail_stages = {"query_analysis", "embedding", "dense_search", "bm25_search", "fusion", "rerank", "context_selection"}

    def progress(event):
        if cancel.is_set():
            raise RuntimeError("Request cancelled")
        # Later Central searches retain tool events and aggregate metrics,
        # while detailed retrieval events are coalesced after the first pass.
        stage = event["stage"]
        if mode == ChatMode.CENTRAL and stage in completed_detail_stages and event["state"] != "failed":
            return
        if stage in detail_stages:
            seen_detail_stages.add(stage)
        if stage == "tool:search_history" and event["state"] == "completed":
            completed_detail_stages.update(seen_detail_stages)
        loop.call_soon_threadsafe(queue.put_nowait, {**event, "request_id": trace.request_id,
                                                   "retrieval_backend": backend, "mode": mode.value})
    token = set_request_telemetry(trace)
    try:
        trace.mark("first_status_event")
        yield "status", {"stage": "request_preparation", "state": "started",
                         "message": "Phân tích câu hỏi...", "mode": mode.value,
                         "retrieval_backend": backend, "request_id": trace.request_id}
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
        yield "status", {"stage": "request_preparation", "state": "completed", "message": "Đã nhận câu hỏi",
                         "mode": mode.value, "retrieval_backend": backend, "request_id": trace.request_id}
        trace.mark("prepare_started")
        prepare_task = asyncio.create_task(runtime.prepare(question, payload.final_k, history,
                                         owner_id=owner_id, conversation_id=str(payload.conversation_id),
                                         attachment_ids=requested_ids, trace=trace, cancel=cancel,
                                         retrieval_backend=backend, progress=progress,
                                         **({"tool_view": tool_view} if tool_view is not None else {})))
        # Wait for execution events or prepare completion, without polling or blocking.
        while not prepare_task.done():
            progress_waiter = asyncio.create_task(queue.get())
            await asyncio.wait({prepare_task, progress_waiter}, return_when=asyncio.FIRST_COMPLETED)
            if progress_waiter.done():
                event = progress_waiter.result()
                current_stage = event["stage"]
                yield "status", event
            else:
                progress_waiter.cancel()
                await asyncio.gather(progress_waiter, return_exceptions=True)
            progress_waiter = None
        while not queue.empty():
            event = queue.get_nowait()
            current_stage = event["stage"]
            yield "status", event
        prepared = prepare_task.result()
        if cancel.is_set():
            return
        current_stage = "generation"
        yield "status", {"stage": "generation", "state": "started", "message": "Tạo câu trả lời...",
                         "mode": mode.value, "retrieval_backend": backend, "request_id": trace.request_id}
        max_tokens = (settings.hybrid_max_new_tokens if mode == ChatMode.HYBRID
                      else settings.central_final_max_new_tokens)
        trace.mark("final_model_requested")
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
        yield "status", {"stage": "generation", "state": "completed", "message": "Đã tạo câu trả lời",
                         "mode": mode.value, "retrieval_backend": backend, "request_id": trace.request_id,
                         "latency_ms": completed.metrics.get("generation_ms")}
        answer = "".join(answer_parts)
        if not answer.strip():
            raise RuntimeError("Model returned an empty answer")
        source_items, cited_ids = _sources(answer, prepared.contexts)
        trace.mark("sources_ready")
        stored_sources = source_items
        assistant_message = await asyncio.to_thread(
            store.add_message, owner_id, payload.conversation_id, "assistant", answer, stored_sources
        )
        yield "sources", {"items": source_items, "cited_source_ids": cited_ids,
                          "final_context_count": len(prepared.contexts)}
        trace.mark("request_finished")
        metrics = _model_metrics(completed, trace, prepared, delta_times)
        # Legacy/unknown model metadata still reports the actual request budget,
        # but never guesses a finish reason from the decoded token count.
        metrics["max_new_tokens"] = max_tokens
        debug_trace = (_debug_trace(mode, trace, prepared, source_items, metrics, runtime.model)
                       if payload.debug else None)
        if debug_trace is not None:
            await asyncio.to_thread(store.update_message_debug_trace, owner_id, payload.conversation_id,
                                    assistant_message["id"], debug_trace)
        done = {"request_id": trace.request_id, "conversation_id": str(payload.conversation_id),
                "message_id": str(assistant_message["id"]), "user_message_id": str(user_message["id"]),
                "answer": answer, "status": "done", "mode": mode.value,
                "retrieval_backend": backend,
                "latency_ms": metrics["e2e_ms"], "model_id": runtime.model.model_id,
                "model_revision": completed.model_revision,
                "model_variant": getattr(runtime.model, "model_variant", None),
                "adapter_attached": getattr(runtime.model, "adapter_attached", None),
                "adapter_fingerprint": getattr(runtime.model, "adapter_fingerprint", None),
                "generation_settings": {**runtime.model.generation_settings, "max_new_tokens": max_tokens},
                "retrieval_settings": getattr(request.app.state.retriever, "retrieval_config", {}),
                "metrics": metrics}
        if callable(done["retrieval_settings"]):
            done["retrieval_settings"] = done["retrieval_settings"]()
        trace.emit(metrics=metrics, model_id=runtime.model.model_id,
                   model_revision=completed.model_revision)
        if payload.debug:
            yield "debug_trace", debug_trace
        yield "done", done
    except asyncio.CancelledError:
        cancel.set()
        raise
    except Exception as exc:
        if cancel.is_set():
            return
        logger.error("Chat request failed: type=%s", type(exc).__name__, extra={"request_id": trace.request_id})
        trace.mark("request_finished")
        metrics = _model_metrics(completed, trace, prepared, delta_times)
        trace.emit(metrics=metrics, model_id=runtime.model.model_id,
                   model_revision=getattr(runtime.model, "resolved_revision", None), error=type(exc).__name__)
        yield "status", {"stage": current_stage, "state": "failed", "message": "Không thể hoàn tất bước này",
                         "mode": mode.value, "retrieval_backend": backend, "request_id": trace.request_id}
        yield "error", {"type": type(exc).__name__, "message": str(exc), "request_id": trace.request_id,
                        "retrieval_backend": backend}
        yield "done", {"request_id": trace.request_id, "status": "error", "mode": mode.value,
                       "model_id": runtime.model.model_id, "model_revision": getattr(runtime.model, "resolved_revision", None),
                       "retrieval_backend": backend,
                       "metrics": metrics, "latency_ms": metrics["e2e_ms"]}
    finally:
        cancel.set()
        for task in (prepare_task, progress_waiter):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
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
    backend = _resolve_backend(request, payload.retrieval_backend)
    try:
        result = await asyncio.to_thread(retriever.retrieve, payload.question, payload.final_k, dense_backend=backend)
    except DenseBackendError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    return RetrieveResponse(
        retrieval_backend=result.get("retrieval_backend", backend), timings_ms=result.get("timings_ms", {}),
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
    _resolve_backend(request, payload.retrieval_backend)
    view = _resolve_tools(runtime, payload, mode)
    await require_conversation(store, owner_id, payload.conversation_id)
    trace = RequestTrace(str(uuid.uuid4()), mode.value)
    result = None
    failure = None
    sources: list[dict[str, Any]] = []
    debug_trace = None
    async for event, data in _execute(payload, request, owner_id, store, runtime, mode, trace,
                                      watch_disconnect=False, tool_view=view):
        if event == "error":
            failure = data
        elif event == "sources":
            sources = data["items"]
        elif event == "debug_trace":
            debug_trace = data
        elif event == "done":
            result = data
    if failure or not result or result["status"] != "done":
        raise HTTPException(status_code=503 if failure and failure.get("type") in {"QdrantSearchError", "DenseBackendError", "DenseBackendUnavailable", "MCPToolError"} else 500,
                            detail=(failure or {}).get("message", "Generation failed"))
    return ChatResponse(conversation_id=payload.conversation_id, message_id=result["message_id"],
                        answer=result["answer"], status="done", mode=mode,
                        retrieval_backend=result["retrieval_backend"],
                        sources=[SourceItem.model_validate(item) for item in sources],
                        latency_ms=result["latency_ms"], debug=debug_trace)


@router.post("/chat/stream", status_code=status.HTTP_200_OK)
async def chat_stream(payload: ChatRequest, request: Request, owner_id: OwnerId,
                      store: StoreDependency) -> StreamingResponse:
    mode = normalize_chat_mode(payload.mode, default=settings.default_inference_mode)
    runtime = _runtime(request, mode)
    _resolve_backend(request, payload.retrieval_backend)
    view = _resolve_tools(runtime, payload, mode)
    await require_conversation(store, owner_id, payload.conversation_id)
    trace = RequestTrace(str(uuid.uuid4()), mode.value)

    async def events() -> AsyncIterator[str]:
        async for event, data in _execute(payload, request, owner_id, store, runtime, mode, trace,
                                          watch_disconnect=True, tool_view=view):
            if event == "done":
                data = {key: value for key, value in data.items() if key != "answer"}
            yield _sse(event, data)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                                      "X-Request-ID": trace.request_id})
