from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

import modal


app = modal.App("vn-history-agentic-smoke")
artifacts = modal.Volume.from_name("vn-history-artifacts", create_if_missing=False)
hf_cache = modal.Volume.from_name("vn-history-hf-cache", create_if_missing=False)
chat_data = modal.Volume.from_name("vn-history-chat-data", create_if_missing=True)

image = modal.Image.from_dockerfile("Dockerfile", context_dir=".").env(
    {
        "APP_ENV": "production",
        "APP_MODE": "full",
        "DEVICE": "cuda",
        "DTYPE": "bfloat16",
        "ARTIFACT_ROOT": "/artifacts",
        "LLM_BACKEND": "transformers",
        "SHARED_BASE_MODEL_ID": "Qwen/Qwen3-4B-Instruct-2507",
        "RESEARCH_AGENT_MODEL": "Qwen/Qwen3-4B-Instruct-2507",
        "RESEARCH_AGENT_ADAPTER_PATH": "/artifacts/adapters/research",
        "EVIDENCE_AGENT_MODEL": "Qwen/Qwen3-4B-Instruct-2507",
        "EVIDENCE_AGENT_ADAPTER_PATH": "/artifacts/adapters/evidence",
        "HISTORY_AGENT_ADAPTER_PATH": "/artifacts/adapters/history",
        "MAX_AGENT_STEPS": "6",
        "MAX_WIKIPEDIA_SEARCHES": "2",
        "MAX_WEB_SEARCHES": "3",
        "MAX_PAGE_FETCHES": "5",
        "WEB_SEARCH_PROVIDER": "local-only",
        "DEFAULT_INFERENCE_MODE": "agentic_rag",
        "CHAT_DATABASE_PATH": "/data/chat.sqlite3",
        "HF_HOME": "/hf-cache",
        "CORS_ORIGINS": "http://localhost:5173,http://127.0.0.1:5173",
    }
)

SMOKE_QUESTIONS = [
    "Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?",
    "Nguyên nhân Mỹ thua chiến tranh Việt Nam?",
    "So sánh Cách mạng Tháng Tám và chiến thắng Điện Biên Phủ.",
]


def _gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        return None
    return None


def _word_count(text: str) -> int:
    return len(re.findall(r"[0-9A-Za-zÀ-ỹĐđ]+", str(text)))


def _build_orchestrators() -> tuple[Any, Any, Any]:
    from app.agents.evidence_agent import EvidenceCriticAgent
    from app.agents.history_answerer import HistoryAnswererAgent
    from app.agents.model_runtime import SharedAgentModelRuntime
    from app.agents.orchestrator import AgentOrchestrator, HybridRAGOrchestrator
    from app.agents.research_agent import ResearchAgent
    from app.config import settings
    from app.rag.research_runtime import ResearchRetrievalRuntime
    from app.rag.retrieval import HybridRetriever
    from app.services.rag_service import RAGService
    from app.tools.evidence_tools import InspectEvidenceTool, RetrieveEvidenceTool, SessionEvidenceStore
    from app.tools.local_search import SearchHistoryTool
    from app.tools.page_fetcher import FetchPageTool
    from app.tools.registry import ToolRegistry
    from app.tools.web_search import SearchWebTool, build_web_search_provider
    from app.tools.wikipedia import FetchWikipediaPageTool, SearchWikipediaTool

    service = RAGService()
    service.load()
    retriever = HybridRetriever(service)
    research_runtime = ResearchRetrievalRuntime(service, retriever)
    model_runtime = SharedAgentModelRuntime(
        model_id=settings.shared_base_model_id,
        research_adapter=settings.research_agent_adapter_path,
        evidence_adapter=settings.evidence_agent_adapter_path,
        history_adapter=settings.history_agent_adapter_path,
        dtype=settings.dtype,
    )
    service.tokenizer = model_runtime.tokenizer
    service.external_generation_backend = True

    evidence_store = SessionEvidenceStore()
    registry = ToolRegistry()
    registry.register(SearchHistoryTool(retriever))
    registry.register(RetrieveEvidenceTool(evidence_store))
    registry.register(InspectEvidenceTool(evidence_store))
    registry.register(SearchWikipediaTool())
    registry.register(FetchWikipediaPageTool())
    registry.register(
        SearchWebTool(
            build_web_search_provider(settings.web_search_provider, settings.web_search_api_key)
        )
    )
    registry.register(FetchPageTool())

    orchestrator = AgentOrchestrator(
        research_agent=ResearchAgent(
            registry=registry,
            evidence_store=evidence_store,
            retrieval_runtime=research_runtime,
            model_runtime=model_runtime,
            max_steps=settings.max_agent_steps,
            max_wikipedia_searches=settings.max_wikipedia_searches,
            max_web_searches=settings.max_web_searches,
            max_page_fetches=settings.max_page_fetches,
        ),
        evidence_agent=EvidenceCriticAgent(model_runtime=model_runtime),
        answerer=HistoryAnswererAgent(model_runtime=model_runtime),
    )
    hybrid_orchestrator = HybridRAGOrchestrator(
        retriever=retriever,
        retrieval_runtime=research_runtime,
        answerer=HistoryAnswererAgent(model_runtime=model_runtime),
    )
    return service, orchestrator, hybrid_orchestrator


def _build_orchestrator() -> tuple[Any, Any]:
    service, orchestrator, _ = _build_orchestrators()
    return service, orchestrator


def _run_question(orchestrator: Any, service: Any, question: str, *, mode: str = "agentic_rag") -> dict[str, Any]:
    from app.telemetry import RequestTelemetry, reset_request_telemetry, set_request_telemetry

    request_id = f"smoke-{uuid.uuid4()}"
    telemetry = RequestTelemetry(
        request_id=request_id,
        inference_mode=mode,
        selected_inference_mode=mode,
        deployment_id=getattr(service, "deployment_id", None),
        gpu=_gpu_name(),
    )
    token = set_request_telemetry(telemetry)
    result: dict[str, Any] = {}
    failure: dict[str, Any] | None = None
    try:
        result = orchestrator.chat(question, final_k=6, request_id=request_id)
        status = "success"
    except Exception as exc:
        status = "failed"
        failure = {
            "stage": getattr(exc, "stage", "unknown"),
            "code": getattr(exc, "code", type(exc).__name__),
            "evidence_ids": getattr(exc, "evidence_ids", []),
            "repair_attempted": getattr(exc, "repair_attempted", False),
            "validation_errors": getattr(exc, "validation_errors", []),
            "message": str(exc),
        }
        telemetry.failed_stage = failure["stage"]
        telemetry.failure_code = failure["code"]
    finally:
        reset_request_telemetry(token)

    answer = str(result.get("answer") or "")
    evidence_debug = result.get("evidence_debug") or {}
    history_debug = result.get("history_debug") or {}
    provenance = result.get("answer_provenance") or {}
    summary = telemetry.summary(result=status)
    return {
        "question": question,
        "mode": mode,
        "status": status,
        "failure": failure,
        "answer_preview": answer[:700],
        "answer_length_chars": len(answer),
        "answer_length_words": _word_count(answer),
        "evidence_validation": evidence_debug.get("final_validation_result"),
        "history_called": bool(history_debug.get("generation_calls")),
        "answer_depth": history_debug.get("answer_depth") or provenance.get("answer_depth"),
        "question_type": history_debug.get("question_type") or evidence_debug.get("question_type"),
        "research_calls": provenance.get("research_generation_calls") or telemetry.research_llm_calls,
        "evidence_calls": provenance.get("evidence_generation_calls") or telemetry.evidence_generation_calls,
        "evidence_reconsideration": evidence_debug.get("reconsideration_used", telemetry.evidence_reconsideration_used),
        "history_calls": provenance.get("history_generation_calls") or telemetry.history_generation_calls,
        "history_retry_used": provenance.get("history_retry_used", telemetry.history_retry_used),
        "history_retry_reason": provenance.get("history_retry_reason", telemetry.history_retry_reason),
        "evidence_selected_ids": evidence_debug.get("selected_ids", []),
        "research_latency_ms": summary.get("research_latency_ms"),
        "evidence_first_pass_latency_ms": summary.get("evidence_first_pass_latency_ms"),
        "evidence_guard_latency_ms": summary.get("evidence_guard_latency_ms"),
        "evidence_reconsideration_latency_ms": summary.get("evidence_reconsideration_latency_ms"),
        "history_latency_ms": summary.get("history_latency_ms"),
        "total_latency_ms": summary.get("total_latency_ms"),
        "history_input_evidence_count": history_debug.get("input_evidence_ids") and len(history_debug.get("input_evidence_ids")),
        "history_input_claim_count": history_debug.get("input_claim_count"),
        "first_history_words": history_debug.get("first_answer_words") or provenance.get("history_first_answer_words"),
        "final_history_words": history_debug.get("final_answer_words") or provenance.get("history_final_answer_words"),
        "first_quality_issues": history_debug.get("first_quality_issues") or provenance.get("history_first_quality_issues"),
        "final_quality_issues": history_debug.get("final_quality_issues") or provenance.get("history_final_quality_issues"),
        "comparison_targets": evidence_debug.get("comparison_targets"),
        "comparison_target_coverage": evidence_debug.get("comparison_target_coverage"),
        "comparison_target_map": evidence_debug.get("comparison_target_map"),
        "target_a_selected_evidence": evidence_debug.get("target_a_selected_evidence"),
        "target_b_selected_evidence": evidence_debug.get("target_b_selected_evidence"),
        "shared_selected_evidence": evidence_debug.get("shared_selected_evidence"),
        "unknown_selected_evidence": evidence_debug.get("unknown_selected_evidence"),
        "comparison_evidence_groups": history_debug.get("comparison_evidence_groups"),
        "quality_warnings": result.get("quality_warnings", []),
        "structured_expansion_used": result.get("structured_expansion_used", False),
        "source_ids": result.get("source_ids", []),
        "final_answer": answer,
        "telemetry_summary": summary,
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=4.0,
    memory=32768,
    timeout=1800,
    startup_timeout=600,
    volumes={
        "/artifacts": artifacts,
        "/hf-cache": hf_cache,
        "/data": chat_data,
    },
)
def run_agentic_smoke() -> dict[str, Any]:
    started = time.perf_counter()
    service, orchestrator = _build_orchestrator()
    load_ms = (time.perf_counter() - started) * 1000
    try:
        results = [_run_question(orchestrator, service, question) for question in SMOKE_QUESTIONS]
    finally:
        try:
            service.shutdown()
        except Exception:
            pass
    return {
        "load_ms": load_ms,
        "questions": results,
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=4.0,
    memory=32768,
    timeout=1800,
    startup_timeout=600,
    volumes={
        "/artifacts": artifacts,
        "/hf-cache": hf_cache,
        "/data": chat_data,
    },
)
def run_bounded_acceptance_smoke() -> dict[str, Any]:
    started = time.perf_counter()
    service, agentic_orchestrator, hybrid_orchestrator = _build_orchestrators()
    load_ms = (time.perf_counter() - started) * 1000
    try:
        results = [
            _run_question(
                agentic_orchestrator,
                service,
                "So sánh Cách mạng Tháng Tám và chiến thắng Điện Biên Phủ.",
                mode="agentic_rag",
            ),
            _run_question(
                hybrid_orchestrator,
                service,
                "Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?",
                mode="hybrid_rag",
            ),
        ]
    finally:
        try:
            service.shutdown()
        except Exception:
            pass
    return {
        "load_ms": load_ms,
        "questions": results,
    }


@app.local_entrypoint()
def main(output: str = "artifacts/agentic_smoke_result.json", bounded: bool = False):
    result = run_bounded_acceptance_smoke.remote() if bounded else run_agentic_smoke.remote()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
