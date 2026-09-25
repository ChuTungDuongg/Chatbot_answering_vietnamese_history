"""FastAPI assembly for the two baseline inference modes."""

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
            common = dict(device=settings.device, dtype=settings.dtype,
                          cache_dir=str(settings.model_cache_dir) if settings.model_cache_dir else None,
                          local_files_only=settings.model_local_files_only,
                          do_sample=settings.do_sample, enable_thinking=settings.enable_thinking)
            hybrid = central = None
            if settings.enable_hybrid_mode:
                hybrid_model = QwenRuntime(model_id=settings.hybrid_model_id,
                                           revision=settings.hybrid_model_revision, **common)
                hybrid = HybridRuntime(app.state.retriever, hybrid_model, temporary_retriever)
                if settings.runtime_loading_strategy == "eager":
                    hybrid_model.load()
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
                central = CentralRuntime(model=central_model, tools=registry,
                                         max_action_rounds=settings.central_max_action_rounds,
                                         action_max_new_tokens=settings.central_action_max_new_tokens)
                if settings.runtime_loading_strategy == "eager":
                    central_model.load()
            app.state.hybrid_runtime = hybrid
            app.state.central_runtime = central
            app.state.chat_mode_router = ChatModeRouter(hybrid=hybrid, central=central)
        yield
    finally:
        service.shutdown()


app = FastAPI(title=settings.app_name, version=settings.app_version,
              description="Vietnamese history: Hybrid RAG (vanilla Qwen3-4B) and Central Agent (vanilla Qwen3-8B).",
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
    return service.readiness()
