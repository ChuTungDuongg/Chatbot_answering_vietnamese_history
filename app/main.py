from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.conversations import router as conversations_router
from app.api.routes import router as api_router
from app.agents.evidence_agent import EvidenceCriticAgent
from app.agents.central_agent import CentralAgent
from app.agents.central_model_runtime import CentralModelRuntime
from app.agents.config import AgentConfig, CentralAgentConfig
from app.agents.history_answerer import HistoryAnswererAgent
from app.agents.lazy_runtime import LazyRuntime
from app.agents.model_runtime import SharedAgentModelRuntime, VLLMOpenAIBackend
from app.agents.model_registry import SHARED_BASE_MODEL_ID, validate_central_adapter
from app.agents.orchestrator import AgentOrchestrator, HybridRAGOrchestrator
from app.agents.research_agent import ResearchAgent
from app.chat.attachments import AttachmentService, TemporaryCorpusRetriever
from app.chat.store import ConversationStore
from app.config import settings
from app.rag.research_runtime import ResearchRetrievalRuntime
from app.rag.retrieval import HybridRetriever
from app.schemas import HealthResponse, ReadyResponse
from app.services.chat_mode_router import ChatModeRouter
from app.services.fast_service import FastChatService
from app.services.rag_service import RAGService
from app.tools.attachment_search import SearchUploadedDocumentsTool
from app.tools.evidence_tools import InspectEvidenceTool, RetrieveEvidenceTool, SessionEvidenceStore
from app.tools.local_search import SearchHistoryTool
from app.tools.page_fetcher import FetchPageTool
from app.tools.registry import ToolRegistry
from app.tools.web_search import SearchWebTool, build_web_search_provider
from app.tools.wikipedia import FetchWikipediaPageTool, SearchWikipediaTool


# ============================================================
# Application lifecycle
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    service = RAGService()
    chat_store = ConversationStore(settings.chat_database_path)

    retriever = None
    generator = None
    three_llm_runtime = None
    hybrid_runtime = None
    role_model_runtime = None
    central_model_runtime = None
    attachment_service = None
    temporary_retriever = None
    research_runtime = None
    central_runtime = None
    chat_mode_router = None

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
            research_runtime = ResearchRetrievalRuntime(service, retriever)

        if settings.should_load_model:
            if retriever is None:
                raise RuntimeError("Full mode requires the retrieval runtime.")

            needs_role_runtime = settings.enable_hybrid_mode or settings.enable_three_llm_mode
            if needs_role_runtime and settings.uses_shared_backend:
                if settings.shared_base_model_id != SHARED_BASE_MODEL_ID:
                    raise ValueError("Active shared backend must use the canonical Qwen3 base model ID.")
                if settings.llm_backend == "transformers":
                    if settings.history_agent_adapter_path is None:
                        raise ValueError("Hybrid/three_llm Transformers runtime requires the History adapter path.")
                    if settings.enable_three_llm_mode and any(path is None for path in (
                        settings.research_agent_adapter_path, settings.evidence_agent_adapter_path,
                    )):
                        raise ValueError("three_llm requires Research and Evidence adapter paths.")
                    role_model_runtime = LazyRuntime(
                        lambda: SharedAgentModelRuntime(
                            model_id=settings.shared_base_model_id,
                            research_adapter=(
                                settings.research_agent_adapter_path if settings.enable_three_llm_mode else None
                            ),
                            evidence_adapter=(
                                settings.evidence_agent_adapter_path if settings.enable_three_llm_mode else None
                            ),
                            history_adapter=settings.history_agent_adapter_path,
                            dtype=settings.dtype,
                        ),
                        name="qwen3-4b-role-runtime",
                    )
                else:
                    role_model_runtime = VLLMOpenAIBackend(
                        base_url=settings.vllm_base_url,
                        api_key=settings.vllm_api_key,
                        tokenizer_id=settings.shared_base_model_id,
                    )
            elif needs_role_runtime:
                raise ValueError("Hybrid/three_llm production modes require the shared Qwen3 role backend.")

            if settings.enable_central_mode:
                if settings.llm_backend != "transformers":
                    raise ValueError("The standalone Central runtime currently requires LLM_BACKEND=transformers.")
                if settings.central_adapter_path is not None:
                    validate_central_adapter(settings.central_adapter_path)
                central_model_runtime = LazyRuntime(
                    lambda: CentralModelRuntime(
                        model_id=settings.central_agent_model_id,
                        adapter_path=settings.central_adapter_path,
                        dtype=settings.dtype,
                        device=settings.device,
                        cache_dir=settings.central_agent_hf_cache_dir,
                        local_files_only=settings.central_agent_local_files_only,
                    ),
                    name="qwen3-8b-central-runtime",
                )

            if settings.runtime_loading_strategy == "eager":
                if isinstance(role_model_runtime, LazyRuntime):
                    role_model_runtime.get()
                if isinstance(central_model_runtime, LazyRuntime):
                    central_model_runtime.get()
            service.external_generation_backend = bool(role_model_runtime or central_model_runtime)

            if research_runtime is None:
                raise RuntimeError("Full mode requires the Research retrieval runtime.")

            answerer = HistoryAnswererAgent(model_runtime=role_model_runtime) if role_model_runtime is not None else None

            agent_config = AgentConfig(
                max_steps=settings.max_agent_steps,
                max_tool_results=settings.agent_max_tool_results,
                observation_char_budget=settings.agent_observation_char_budget,
                timeout_seconds=settings.agent_timeout_seconds,
                enable_web=settings.agent_enable_web,
                enable_wikipedia=settings.agent_enable_wikipedia,
                enable_document_search=settings.agent_enable_document_search,
            )

            evidence_store = SessionEvidenceStore()
            tool_registry = ToolRegistry()
            tool_registry.register(SearchHistoryTool(retriever))
            documents_enabled = agent_config.enable_document_search or (
                settings.enable_central_mode and settings.central_agent_enable_documents
            )
            wikipedia_enabled = agent_config.enable_wikipedia or (
                settings.enable_central_mode and settings.central_agent_enable_wikipedia
            )
            web_enabled = agent_config.enable_web or (
                settings.enable_central_mode and settings.central_agent_enable_web
            )
            if documents_enabled and temporary_retriever is not None:
                tool_registry.register(SearchUploadedDocumentsTool(temporary_retriever))
            tool_registry.register(RetrieveEvidenceTool(evidence_store))
            tool_registry.register(InspectEvidenceTool(evidence_store))
            if wikipedia_enabled:
                tool_registry.register(SearchWikipediaTool())
                tool_registry.register(FetchWikipediaPageTool())
            if web_enabled:
                tool_registry.register(
                    SearchWebTool(
                        build_web_search_provider(
                            settings.web_search_provider,
                            settings.web_search_api_key,
                        )
                    )
                )
                tool_registry.register(FetchPageTool())

            if settings.enable_three_llm_mode:
                assert role_model_runtime is not None and answerer is not None
                three_llm_runtime = AgentOrchestrator(
                    research_agent=ResearchAgent(
                        registry=tool_registry,
                        evidence_store=evidence_store,
                        retrieval_runtime=research_runtime,
                        model_runtime=role_model_runtime,
                        max_steps=agent_config.max_steps,
                        max_wikipedia_searches=settings.max_wikipedia_searches,
                        max_web_searches=settings.max_web_searches,
                        max_page_fetches=settings.max_page_fetches,
                    ),
                    evidence_agent=EvidenceCriticAgent(model_runtime=role_model_runtime),
                    answerer=answerer,
                )
            if settings.enable_hybrid_mode:
                assert answerer is not None
                hybrid_runtime = FastChatService(HybridRAGOrchestrator(
                    retriever=retriever,
                    retrieval_runtime=research_runtime,
                    answerer=answerer,
                ))
            if settings.enable_central_mode:
                assert central_model_runtime is not None
                central_runtime = CentralAgent(
                    model_runtime=central_model_runtime,
                    tool_registry=tool_registry,
                    config=CentralAgentConfig(
                        max_action_rounds=settings.central_agent_max_action_rounds,
                        repair_max_generations=settings.central_agent_repair_max_generations,
                        action_max_new_tokens=settings.central_action_max_new_tokens,
                        final_max_new_tokens=settings.central_final_max_new_tokens,
                        repair_max_new_tokens=settings.central_repair_max_new_tokens,
                        repair_min_new_tokens=settings.central_repair_min_new_tokens,
                        repair_token_margin=settings.central_repair_token_margin,
                        biography_max_sources=settings.central_biography_max_sources,
                        biography_min_exact_hits=settings.central_biography_min_exact_hits,
                        reranker_tail_gap_ratio=settings.central_reranker_tail_gap_ratio,
                        reranker_score_mode=settings.central_reranker_score_mode,
                        reranker_score_floor=settings.central_reranker_score_floor,
                        reranker_strong_score=settings.central_reranker_strong_score,
                        max_tool_results=settings.central_agent_max_tool_results,
                        observation_char_budget=settings.central_agent_observation_char_budget,
                        timeout_seconds=settings.central_agent_timeout_seconds,
                        model_load_timeout_seconds=settings.central_model_load_timeout_seconds,
                        tool_timeout_seconds=settings.central_tool_timeout_seconds,
                        enable_history=settings.central_agent_enable_history,
                        enable_documents=settings.central_agent_enable_documents,
                        enable_wikipedia=settings.central_agent_enable_wikipedia,
                        enable_web=settings.central_agent_enable_web,
                        web_search_provider=settings.web_search_provider,
                    ),
                    has_uploaded_documents=lambda owner, conversation: any(
                        item.get("status") == "ready" and int(item.get("chunk_count") or 0) > 0
                        for item in chat_store.list_attachments(owner, conversation)
                    ),
                )
            chat_mode_router = ChatModeRouter(
                hybrid=hybrid_runtime,
                three_llm=three_llm_runtime,
                central=central_runtime,
            )

        app.state.rag_service = service
        app.state.retriever = retriever
        app.state.generator = generator
        app.state.hybrid_runtime = hybrid_runtime
        app.state.three_llm_runtime = three_llm_runtime
        app.state.central_runtime = central_runtime
        app.state.agentic_orchestrator = three_llm_runtime
        app.state.hybrid_orchestrator = getattr(hybrid_runtime, "hybrid_service", None)
        app.state.orchestrator = three_llm_runtime
        app.state.central_agent = central_runtime
        app.state.fast_service = hybrid_runtime
        app.state.chat_mode_router = chat_mode_router
        app.state.agent_model_runtime = role_model_runtime
        app.state.central_model_runtime = central_model_runtime
        app.state.chat_store = chat_store
        app.state.attachment_service = attachment_service
        app.state.temporary_retriever = temporary_retriever

        yield
    finally:
        app.state.agent_model_runtime = None
        app.state.central_model_runtime = None
        service.shutdown()


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Vietnamese History RAG API with isolated Hybrid, three-LLM Qwen3-4B, and Central Qwen3-8B modes; "
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
            "chat_modes": ["hybrid", "three_llm", "central"],
            "conversations": True,
            "conversation_memory": settings.should_load_model,
            "attachment_retrieval": settings.should_load_retrieval,
            "hybrid": bool(getattr(app.state, "hybrid_runtime", None)),
            "three_llm": bool(getattr(app.state, "three_llm_runtime", None)),
            "central": bool(getattr(app.state, "central_runtime", None)),
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
