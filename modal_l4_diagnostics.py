from __future__ import annotations

import json
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

import modal


app = modal.App("vn-history-l4-diagnostics")
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


def _gpu_snapshot() -> dict[str, Any]:
    import torch

    payload: dict[str, Any] = {
        "gpu_name": None,
        "total_vram_gb": None,
        "allocated_gb": None,
        "reserved_gb": None,
        "peak_allocated_gb": None,
        "torch_version": getattr(torch, "__version__", None),
        "cuda_version": getattr(torch.version, "cuda", None),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        payload.update(
            {
                "gpu_name": props.name,
                "total_vram_gb": props.total_memory / 1024**3,
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            }
        )
    return payload


def _device_profile(runtime: Any, service: Any) -> dict[str, Any]:
    device_map = getattr(runtime.model, "hf_device_map", None)
    counts: dict[str, int] = {}
    cpu_offload = False
    disk_offload = False
    if isinstance(device_map, dict):
        for value in device_map.values():
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
            cpu_offload = cpu_offload or key == "cpu"
            disk_offload = disk_offload or key == "disk"
    return {
        "hf_device_map_summary": counts,
        "cpu_offload": cpu_offload,
        "disk_offload": disk_offload,
        "qwen_input_embedding_device": str(runtime.model.get_input_embeddings().weight.device),
        "embedder_device": str(getattr(getattr(service.embedder, "_target_device", None), "type", None)),
        "reranker_device": str(getattr(service.reranker, "device", None)),
    }


def _messages(adapter: str) -> list[dict[str, str]]:
    if adapter == "research":
        payload = {
            "question": "Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?",
            "step": 1,
            "limits": {"max_steps": 6},
            "observations": [],
        }
        return [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    if adapter == "evidence":
        evidence = "Chiến thắng Bạch Đằng năm 938 chấm dứt hơn một nghìn năm Bắc thuộc. " * 80
        payload = {
            "question": "Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?",
            "max_selected": 6,
            "evidence": [{"evidence_id": "ev_01", "text": evidence}],
        }
        return [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    context = "Chiến thắng Bạch Đằng năm 938 chấm dứt hơn một nghìn năm Bắc thuộc. " * 30
    return [{"role": "user", "content": f"[ev_01] {context}\n\nTrả lời có dẫn nguồn."}]


def _benchmark_adapter(runtime: Any, adapter: str) -> dict[str, Any]:
    import torch

    from app.telemetry import RequestTelemetry, reset_request_telemetry, set_request_telemetry

    runtime.generate_text(adapter=adapter, messages=_messages(adapter))
    measurements = []
    for _ in range(3):
        telemetry = RequestTelemetry(request_id=f"bench-{adapter}-{uuid.uuid4()}")
        token = set_request_telemetry(telemetry)
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            runtime.generate_text(adapter=adapter, messages=_messages(adapter))
        finally:
            reset_request_telemetry(token)
        metric = telemetry.generation_metrics[-1]
        measurements.append(metric)
    generation_ms = [item.generation_ms for item in measurements]
    tokens_per_sec = [item.tokens_per_sec for item in measurements]
    return {
        "adapter": adapter,
        "input_tokens": int(statistics.median(item.input_tokens for item in measurements)),
        "output_tokens": int(statistics.median(item.output_tokens for item in measurements)),
        "median_generation_ms": statistics.median(generation_ms),
        "tokens_per_sec": statistics.median(tokens_per_sec),
        "peak_vram": max(
            (item.peak_allocated_gb or 0.0)
            for item in measurements
        ),
    }


def _build_runtime_stack() -> tuple[Any, Any, Any]:
    from app.agents.evidence_agent import EvidenceCriticAgent
    from app.agents.history_answerer import HistoryAnswererAgent
    from app.agents.model_runtime import SharedAgentModelRuntime
    from app.agents.orchestrator import AgentOrchestrator
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
    agent_model_runtime = SharedAgentModelRuntime(
        model_id=settings.shared_base_model_id,
        research_adapter=settings.research_agent_adapter_path,
        evidence_adapter=settings.evidence_agent_adapter_path,
        history_adapter=settings.history_agent_adapter_path,
        dtype=settings.dtype,
    )
    service.tokenizer = agent_model_runtime.tokenizer
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
            model_runtime=agent_model_runtime,
            max_steps=settings.max_agent_steps,
            max_wikipedia_searches=settings.max_wikipedia_searches,
            max_web_searches=settings.max_web_searches,
            max_page_fetches=settings.max_page_fetches,
        ),
        evidence_agent=EvidenceCriticAgent(model_runtime=agent_model_runtime),
        answerer=HistoryAnswererAgent(model_runtime=agent_model_runtime),
    )
    return service, agent_model_runtime, orchestrator


@app.function(
    image=image,
    gpu="L4",
    cpu=4.0,
    memory=32768,
    timeout=1200,
    startup_timeout=600,
    volumes={
        "/artifacts": artifacts,
        "/hf-cache": hf_cache,
        "/data": chat_data,
    },
)
def run_diagnostics(run_benchmark: bool = True) -> dict[str, Any]:
    from app.telemetry import RequestTelemetry, reset_request_telemetry, set_request_telemetry

    cold_started = time.perf_counter()
    service, runtime, orchestrator = _build_runtime_stack()
    cold_start_ms = (time.perf_counter() - cold_started) * 1000
    benchmarks = (
        [_benchmark_adapter(runtime, adapter) for adapter in ("research", "evidence", "history")]
        if run_benchmark
        else []
    )
    request_id = f"diagnostic-{uuid.uuid4()}"
    telemetry = RequestTelemetry(
        request_id=request_id,
        deployment_id=getattr(service, "deployment_id", None),
        gpu=_gpu_snapshot().get("gpu_name"),
    )
    token = set_request_telemetry(telemetry)
    request_started = time.perf_counter()
    response_json: dict[str, Any]
    try:
        response_json = orchestrator.chat(
            "Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?",
            final_k=6,
            request_id=request_id,
        )
        bach_dang_status = "success"
        failure = None
    except Exception as exc:
        response_json = {}
        bach_dang_status = "failed"
        failure = {"stage": getattr(exc, "stage", "unknown"), "code": getattr(exc, "code", type(exc).__name__)}
        telemetry.failed_stage = failure["stage"]
        telemetry.failure_code = failure["code"]
    finally:
        reset_request_telemetry(token)
    warm_request_ms = (time.perf_counter() - request_started) * 1000
    deployment_id = getattr(service, "deployment_id", None) or telemetry.deployment_id
    startup_timings_ms = getattr(service, "startup_timings_ms", {})
    gpu = _gpu_snapshot()
    device_profile = _device_profile(runtime, service)
    try:
        service.shutdown()
    except Exception:
        pass
    return {
        "deployment_id": deployment_id,
        "startup_timings_ms": startup_timings_ms,
        "cold_start_total_ms": cold_start_ms,
        "gpu": gpu,
        "device_profile": device_profile,
        "benchmarks": benchmarks,
        "bach_dang": {
            "status": bach_dang_status,
            "failure": failure,
            "warm_request_ms": warm_request_ms,
            "telemetry_summary": telemetry.summary(result=bach_dang_status),
            "generation_metrics": [
                {
                    "adapter": item.adapter,
                    "input_tokens": item.input_tokens,
                    "output_tokens": item.output_tokens,
                    "max_new_tokens": item.max_new_tokens,
                    "lock_wait_ms": item.lock_wait_ms,
                    "adapter_switch_ms": item.adapter_switch_ms,
                    "generation_ms": item.generation_ms,
                    "decode_ms": item.decode_ms,
                    "tokens_per_sec": item.tokens_per_sec,
                    "peak_allocated_gb": item.peak_allocated_gb,
                }
                for item in telemetry.generation_metrics
            ],
            "response": response_json,
        },
    }


@app.local_entrypoint()
def main(output: str = "artifacts/l4_diagnostics_result.json", run_benchmark: bool = True):
    result = run_diagnostics.remote(run_benchmark)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
