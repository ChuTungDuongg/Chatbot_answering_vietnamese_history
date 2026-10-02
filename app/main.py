"""FastAPI assembly for the configured inference modes."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.conversations import router as conversations_router
from app.api.routes import router as api_router
from app.central.runtime import CentralRuntime
from app.chat.attachments import AttachmentService, TemporaryCorpusRetriever
from app.chat.store import ConversationStore
from app.config import settings
from app.models.qwen import QwenRuntime
from app.mcp.manager import MCPManager
from app.tools.policy import builtin_capabilities
from app.rag.hybrid_runtime import HybridRuntime
from app.rag.retrieval import HybridRetriever
from app.services.chat_mode_router import ChatModeRouter
from app.services.rag_service import RAGService
from app.tools.attachment_search import SearchUploadedDocumentsTool
from app.tools.local_search import SearchHistoryTool
from app.tools.page_fetcher import FetchPageTool
from app.tools.registry import ToolRegistry
from app.tools.web_search import SearchWebTool, build_web_search_provider
from app.tools.wikipedia import FetchWikipediaPageTool, SearchWikipediaTool


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = RAGService()
    store = ConversationStore(settings.chat_database_path)
    app.state.rag_service = service
    app.state.chat_store = store
    app.state.retriever = None
    app.state.attachment_service = None
    app.state.hybrid_runtime = None
    app.state.central_runtime = None
    app.state.chat_mode_router = None
    manager = MCPManager()
    app.state.mcp_manager = manager
    try:
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
            if settings.enable_hybrid_mode:
                hybrid_model = QwenRuntime(model_id=settings.hybrid_model_id,
                                           revision=settings.hybrid_model_revision,
                                           adapter_path=(settings.model_adapter_path if settings.model_variant == "sft" else None),
                                           **common)
                hybrid = HybridRuntime(app.state.retriever, hybrid_model, temporary_retriever)
                hybrid_model.load()
                logger.info("Hybrid model ready: base=%s variant=%s adapter=%s peft_attached=%s adapter_fingerprint=%s",
                            hybrid_model.model_id, settings.model_variant.upper(),
                            hybrid_model.adapter_path,
                            hybrid_model.adapter_attached, hybrid_model.adapter_fingerprint)
            if settings.enable_central_mode:
                central_model = QwenRuntime(model_id=settings.central_model_id,
                                            revision=settings.central_model_revision, **common)
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
                                         max_action_rounds=settings.central_max_action_rounds,
                                         action_max_new_tokens=settings.central_action_max_new_tokens,
                                         mcp_manager=manager, max_mcp_tools=settings.mcp_max_tools_per_request,
                                         mcp_schema_budget=settings.mcp_schema_budget_bytes)
                if settings.runtime_loading_strategy == "eager":
                    central_model.load()
            app.state.hybrid_runtime = hybrid
            app.state.central_runtime = central
            app.state.chat_mode_router = ChatModeRouter(hybrid=hybrid, central=central)
        yield
    finally:
        await manager.close()
        service.shutdown()


app = FastAPI(title=settings.app_name, version=settings.app_version,
              description="Vietnamese history: Hybrid RAG (configured Qwen3-4B variant) and Central Agent (vanilla Qwen3-8B).",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                   allow_credentials=False, allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
                   allow_headers=["Content-Type", "Accept", "X-Client-ID"])
app.include_router(api_router)
app.include_router(conversations_router)


@app.get("/")
async def root():
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
    if settings.is_full:
        hybrid = getattr(app.state, "hybrid_runtime", None)
        central = getattr(app.state, "central_runtime", None)
        hybrid_model = hybrid.model if hybrid is not None else None
        central_model = central.model if central is not None else None
        state = {**state,
                 "hybrid_loaded": bool(hybrid_model and hybrid_model.model is not None),
                 "hybrid_model_variant": getattr(hybrid_model, "model_variant", None),
                 "hybrid_adapter_attached": getattr(hybrid_model, "adapter_attached", None),
                 "central_loaded": bool(central_model and central_model.model is not None)}
    return state
