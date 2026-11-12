"""FastAPI assembly for the configured inference modes."""

import hmac
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.conversations import router as conversations_router
from app.api.routes import router as api_router
from app.central.runtime import CentralRuntime
from app.central.budgets import AgentLimits
from app.chat.attachments import AttachmentService, TemporaryCorpusRetriever
from app.chat.store import ConversationStore
from app.config import settings
from app.observability import RequestMetricsMiddleware, cache_snapshot, serving_metrics
from app.models.factory import build_model_runtime
from app.models.openai_responses import OpenAIOptions, OpenAIResponsesRuntime
from app.mcp.manager import MCPManager
from app.tools.policy import builtin_capabilities
from app.rag.hybrid_runtime import HybridRuntime
from app.rag.retrieval import HybridRetriever
from app.services.chat_mode_router import ChatModeRouter
from app.services.openai_access import OpenAIAccessGuard
from app.security.auth import AuthManager, OwnerAuthMiddleware
from app.services.rag_service import RAGService
from app.tools.attachment_search import SearchUploadedDocumentsTool
from app.tools.local_search import SearchHistoryTool
from app.tools.page_fetcher import FetchPageTool
from app.tools.registry import ToolRegistry
from app.tools.web_search import SearchWebTool, build_web_search_provider
from app.tools.wikipedia import FetchWikipediaPageTool, SearchWikipediaTool
from app.prompts.loader import validate_prompt_resources
from app.tools.declarations import validate_builtin_contracts


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_prompt_resources()
    validate_builtin_contracts()
    service = RAGService()
    store = ConversationStore(settings.chat_database_path)
    app.state.rag_service = service
    app.state.chat_store = store
    app.state.retriever = None
    app.state.attachment_service = None
    app.state.hybrid_runtime = None
    app.state.central_runtime = None
    app.state.chat_mode_router = None
    app.state.openai_client = None
    app.state.openai_guard = None
    app.state.auth_manager = None
    manager = MCPManager()
    app.state.mcp_manager = manager
    try:
        if settings.app_env.lower() == "production" and not settings.auth_enabled:
            raise RuntimeError("Authentication must be enabled in production")
        if settings.auth_enabled:
            app.state.auth_manager = AuthManager(settings)
        legacy_openai_access = (settings.openai_trusted_internal_only
                                and settings.openai_access_token
                                and len(settings.openai_access_token.get_secret_value()) >= 32
                                and settings.openai_api_key
                                and not hmac.compare_digest(settings.openai_api_key.get_secret_value().encode(),
                                                            settings.openai_access_token.get_secret_value().encode()))
        if (settings.openai_enabled and (settings.auth_enabled or legacy_openai_access) and settings.openai_api_key
                and settings.openai_api_key.get_secret_value()
                ):
            from openai import AsyncOpenAI
            app.state.openai_client = AsyncOpenAI(
                api_key=settings.openai_api_key.get_secret_value(),
                timeout=settings.openai_timeout_seconds, max_retries=0)
            app.state.openai_guard = OpenAIAccessGuard(
                settings.openai_access_token.get_secret_value() if settings.openai_access_token else "",
                settings.openai_max_requests_per_hour,
                settings.openai_max_requests_per_day, settings.openai_max_concurrent_requests,
                database=settings.chat_database_path,
                lease_seconds=max(7200, settings.openai_timeout_seconds * (settings.central_max_action_rounds + 1) + 600))
        service.load()
        if settings.should_load_retrieval:
            retriever = HybridRetriever(service)
            attachment_service = AttachmentService(store=store, rag_service=service)
            temporary_retriever = TemporaryCorpusRetriever(store=store, rag_service=service)
            app.state.retriever = retriever
            app.state.attachment_service = attachment_service
            app.state.temporary_retriever = temporary_retriever

        if settings.should_load_model:
            if app.state.retriever is None:
                raise RuntimeError("Full mode requires retrieval artifacts")
            if settings.enable_central_mode:
                manager = MCPManager.from_settings(settings)
                app.state.mcp_manager = manager
                await manager.start()
            common = dict(device=settings.device, dtype=settings.dtype,
                          cache_dir=str(settings.model_cache_dir) if settings.model_cache_dir else None,
                          local_files_only=settings.model_local_files_only,
                          do_sample=settings.do_sample, enable_thinking=settings.enable_thinking,
                          temperature=settings.model_temperature, top_p=settings.model_top_p)
            hybrid = central = None
            def remote_model():
                return OpenAIResponsesRuntime(app.state.openai_client, OpenAIOptions(
                    settings.openai_default_model, settings.openai_default_reasoning_effort,
                    settings.openai_max_output_tokens))
            if settings.enable_hybrid_mode:
                hybrid_model = build_model_runtime(settings=settings, model_id=settings.hybrid_model_id,
                                           revision=settings.hybrid_model_revision,
                                           adapter_path=(settings.model_adapter_path if settings.model_variant == "sft" else None),
                                           **common) if settings.local_inference_enabled else (
                                               remote_model() if app.state.openai_client else None)
            if settings.enable_hybrid_mode and hybrid_model:
                hybrid = HybridRuntime(app.state.retriever, hybrid_model, temporary_retriever)
                if settings.local_inference_enabled:
                    await hybrid_model.aload()
                logger.info("Hybrid model ready: base=%s variant=%s adapter=%s peft_attached=%s adapter_fingerprint=%s",
                            hybrid_model.model_id, settings.model_variant.upper(),
                            hybrid_model.adapter_path,
                            hybrid_model.adapter_attached, hybrid_model.adapter_fingerprint)
            if settings.enable_central_mode:
                central_model = build_model_runtime(settings=settings, model_id=settings.central_model_id,
                                            revision=settings.central_model_revision, **common) if settings.local_inference_enabled else (
                                                remote_model() if app.state.openai_client else None)
                registry = ToolRegistry()
                registry.register(SearchHistoryTool(app.state.retriever))
                if settings.central_enable_documents:
                    registry.register(SearchUploadedDocumentsTool(temporary_retriever))
                if settings.central_enable_wikipedia:
                    registry.register(SearchWikipediaTool())
                    registry.register(FetchWikipediaPageTool())
                if settings.central_enable_web:
                    registry.register(SearchWebTool(build_web_search_provider(
                        settings.web_search_provider, settings.web_search_api_key)))
                    registry.register(FetchPageTool())
                manager.register(registry)
                central = CentralRuntime(model=central_model, tools=registry,
                                         limits=AgentLimits.from_settings(settings),
                                         max_action_rounds=settings.central_max_action_rounds,
                                         action_max_new_tokens=settings.central_action_max_new_tokens,
                                         mcp_manager=manager, max_mcp_tools=settings.mcp_max_tools_per_request,
                                         mcp_schema_budget=settings.mcp_schema_budget_bytes) if central_model else None
                if central_model and settings.local_inference_enabled and settings.runtime_loading_strategy == "eager":
                    await central_model.aload()
            app.state.hybrid_runtime = hybrid
            app.state.central_runtime = central
            app.state.chat_mode_router = ChatModeRouter(hybrid=hybrid, central=central)
        yield
    finally:
        for name in ("hybrid_runtime", "central_runtime"):
            runtime = getattr(app.state, name, None)
            if runtime is not None:
                await runtime.model.aclose()
        await manager.close()
        if app.state.openai_client is not None:
            await app.state.openai_client.close()
        if app.state.openai_guard is not None:
            app.state.openai_guard.close()
        if app.state.auth_manager is not None:
            app.state.auth_manager.close()
        service.shutdown()


app = FastAPI(title=settings.app_name, version=settings.app_version,
              description="Vietnamese history: Hybrid RAG (configured Qwen3-4B variant) and Central Agent (vanilla Qwen3-8B).",
              lifespan=lifespan)
app.add_middleware(RequestMetricsMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                   allow_credentials=False, allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
                   allow_headers=["Content-Type", "Accept", "X-Client-ID", "Authorization"])
app.add_middleware(OwnerAuthMiddleware, settings=settings)
app.include_router(api_router)
app.include_router(conversations_router)


@app.get("/")
async def root():
    index = Path(os.getenv("FRONTEND_DIST_DIR", "/workspace/frontend_dist")) / "index.html"
    if index.is_file():
        return FileResponse(index, headers={"Cache-Control": "no-cache"})
    return {"service": settings.app_name, "version": settings.app_version,
            "environment": settings.app_env, "mode": settings.app_mode,
            "features": {"chat_modes": ["hybrid", "central"],
                         "conversations": True,
                         "attachment_retrieval": settings.should_load_retrieval,
                         "hybrid": bool(getattr(app.state, "hybrid_runtime", None)),
                         "central": bool(getattr(app.state, "central_runtime", None))}}


@app.get("/health")
async def health():
    return {"status": "ok", "service": settings.app_name, "version": settings.app_version}


@app.get("/ready")
async def ready():
    service = getattr(app.state, "rag_service", None)
    if service is None:
        return {"ready": False}
    state = service.readiness()
    manager = getattr(app.state, "mcp_manager", None)
    central = getattr(app.state, "central_runtime", None)
    state.update({"mcp": manager.capabilities() if manager else {"enabled": False, "servers": []},
                  "tools": builtin_capabilities(central.tools) if central else [],
                  "tool_policy": {"max_mcp_tools": settings.mcp_max_tools_per_request,
                                  "schema_budget_bytes": settings.mcp_schema_budget_bytes}})
    state["providers"] = {
        "local": {"available": bool(settings.is_full and settings.local_inference_enabled)},
        "openai": {"available": bool(settings.is_full and getattr(app.state, "openai_client", None)),
                   "models": ["gpt-6-luna", "gpt-6.1-terra"],
                   "reasoning_efforts": ["low", "medium", "high", "xhigh"]},
    }
    state["authentication"] = {"enabled": settings.auth_enabled}
    if settings.is_full:
        hybrid = getattr(app.state, "hybrid_runtime", None)
        central = getattr(app.state, "central_runtime", None)
        hybrid_model = hybrid.model if hybrid is not None else None
        central_model = central.model if central is not None else None
        state = {**state,
                 "hybrid_loaded": bool(hybrid_model and hybrid_model.is_loaded),
                 "hybrid_model_variant": getattr(hybrid_model, "model_variant", None),
                 "hybrid_adapter_attached": getattr(hybrid_model, "adapter_attached", None),
                 "central_loaded": bool(central_model and central_model.is_loaded),
                 "inference_backend": settings.inference_backend}
    return state


@app.get("/api/v1/ops/summary", include_in_schema=False)
async def observability_summary(request: Request):
    """Process-local aggregate metrics; unavailable unless an admin secret is provisioned."""
    expected = os.getenv("OBSERVABILITY_ADMIN_TOKEN", "")
    supplied = request.headers.get("Authorization", "")
    if not expected:
        raise HTTPException(status_code=404, detail="Monitoring API is disabled")
    if not hmac.compare_digest(supplied.encode("utf-8"), f"Bearer {expected}".encode("utf-8")):
        raise HTTPException(status_code=403, detail="Forbidden")
    summary = serving_metrics.snapshot()
    state = getattr(getattr(request, "app", None), "state", None)
    for mode in ("hybrid", "central"):
        model = getattr(getattr(state, f"{mode}_runtime", None), "model", None)
        if model is not None:
            summary["cache"][mode] = cache_snapshot(model)
    return JSONResponse(summary, headers={"Cache-Control": "no-store"})


frontend_dist = Path(os.getenv("FRONTEND_DIST_DIR", "/workspace/frontend_dist")).resolve()
app.mount("/assets", StaticFiles(directory=frontend_dist / "assets", check_dir=False), name="assets")


@app.get("/{path:path}", include_in_schema=False)
async def spa_fallback(path: str):
    if path == "api" or path.startswith(("api/", "assets/")) or path in {"docs", "redoc", "openapi.json"}:
        raise HTTPException(status_code=404, detail="Not found")
    candidate = (frontend_dist / path).resolve()
    if candidate.is_relative_to(frontend_dist) and candidate.is_file():
        return FileResponse(candidate)
    index = frontend_dist / "index.html"
    if index.is_file():
        return FileResponse(index, headers={"Cache-Control": "no-cache"})
    raise HTTPException(status_code=404, detail="Frontend assets are unavailable")
