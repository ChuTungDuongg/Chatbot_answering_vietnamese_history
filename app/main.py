from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.conversations import router as conversations_router
from app.api.routes import router as api_router
from app.agents.evidence_agent import EvidenceCriticAgent
from app.agents.history_answerer import HistoryAnswererAgent
from app.agents.model_runtime import SharedAgentModelRuntime
from app.agents.orchestrator import AgentOrchestrator
from app.agents.research_agent import ResearchAgent
from app.chat.attachments import AttachmentService, TemporaryCorpusRetriever
from app.chat.store import ConversationStore
from app.config import settings
from app.rag.generation import RAGGenerator
from app.rag.retrieval import HybridRetriever
from app.schemas import HealthResponse, ReadyResponse
from app.services.rag_service import RAGService
from app.tools.evidence_tools import InspectEvidenceTool, RetrieveEvidenceTool, SessionEvidenceStore
from app.tools.local_search import SearchHistoryTool
from app.tools.page_fetcher import FetchPageTool
from app.tools.registry import ToolRegistry
from app.tools.web_search import SearchWebTool, build_web_search_provider


# ============================================================
# Application lifecycle
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    service = RAGService()
    chat_store = ConversationStore(settings.chat_database_path)

    retriever = None
    generator = None
    orchestrator = None
    agent_model_runtime = None
    attachment_service = None
    temporary_retriever = None

    try:
        service.load()

        if settings.should_load_retrieval:
            retriever = HybridRetriever(service)

            attachment_service = AttachmentService(
                store=chat_store,
                rag_service=service,
            )

            temporary_retriever = TemporaryCorpusRetriever(
                store=chat_store,
                rag_service=service,
            )

        if settings.should_load_model:
            if retriever is None:
                raise RuntimeError("Full mode requires the retrieval runtime.")

            generator = RAGGenerator(
                service=service,
                retriever=retriever,
                temporary_retriever=temporary_retriever,
            )

            if settings.agent_controller == "model":
                if settings.research_agent_model != settings.evidence_agent_model:
                    raise ValueError("Shared Qwen runtime requires matching Research/Evidence base model IDs.")
                if settings.research_agent_adapter_path is None or settings.evidence_agent_adapter_path is None:
                    raise ValueError("Model agent controller requires both Research and Evidence adapter paths.")
                agent_model_runtime = SharedAgentModelRuntime(
                    model_id=settings.research_agent_model,
                    research_adapter=settings.research_agent_adapter_path,
                    evidence_adapter=settings.evidence_agent_adapter_path,
                    dtype=settings.dtype,
                )

            evidence_store = SessionEvidenceStore()
            tool_registry = ToolRegistry()
            tool_registry.register(SearchHistoryTool(retriever))
            tool_registry.register(RetrieveEvidenceTool(evidence_store))
            tool_registry.register(InspectEvidenceTool(evidence_store))
            tool_registry.register(
                SearchWebTool(
                    build_web_search_provider(
                        settings.web_search_provider,
                        settings.web_search_api_key,
                    )
                )
            )
            tool_registry.register(FetchPageTool())

            orchestrator = AgentOrchestrator(
                research_agent=ResearchAgent(
                    registry=tool_registry,
                    evidence_store=evidence_store,
                    generator=generator,
                    model_runtime=agent_model_runtime,
                    max_steps=settings.max_agent_steps,
                    max_web_searches=settings.max_web_searches,
                    max_page_fetches=settings.max_page_fetches,
                ),
                evidence_agent=EvidenceCriticAgent(model_runtime=agent_model_runtime),
                answerer=HistoryAnswererAgent(generator),
            )

        app.state.rag_service = service
        app.state.retriever = retriever
        app.state.generator = generator
        app.state.orchestrator = orchestrator
        app.state.agent_model_runtime = agent_model_runtime
        app.state.chat_store = chat_store
        app.state.attachment_service = attachment_service
        app.state.temporary_retriever = temporary_retriever

        yield
    finally:
        app.state.agent_model_runtime = None
        service.shutdown()


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Vietnamese History Hybrid RAG API using Qwen2.5, multilingual E5, FAISS, BM25S, "
        "Reciprocal Rank Fusion, cross-encoder reranking, conversation memory, temporary "
        "document retrieval and grounded-generation guards."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "X-Client-ID"],
)


# ============================================================
# Root
# ============================================================

@app.get("/", tags=["System"])
async def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.app_env,
        "mode": settings.app_mode,
        "docs": "/docs",
        "features": {
            "conversations": True,
            "conversation_memory": settings.should_load_model,
            "attachment_retrieval": settings.should_load_retrieval,
            "agentic_rag": bool(getattr(app.state, "orchestrator", None)),
        },
    }


# ============================================================
# Health
# ============================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
    }


# ============================================================
# Readiness
# ============================================================

@app.get("/ready", response_model=ReadyResponse, tags=["System"])
async def ready(request: Request):
    service: RAGService = request.app.state.rag_service
    return service.readiness()


# ============================================================
# API routers
# ============================================================

app.include_router(conversations_router)
app.include_router(api_router)
