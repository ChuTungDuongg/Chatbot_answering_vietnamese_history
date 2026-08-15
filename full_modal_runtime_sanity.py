import modal


app = modal.App("vn-history-full-runtime-sanity")

artifacts = modal.Volume.from_name(
    "vn-history-artifacts",
    create_if_missing=False,
)

hf_cache = modal.Volume.from_name(
    "vn-history-hf-cache",
    create_if_missing=False,
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "numpy",
        "torch",
        "transformers>=4.48,<5",
        "sentence-transformers>=3.3,<6",
        "sentencepiece",
        "accelerate",
        "faiss-cpu",
        "bm25s",
        "fastapi",
        "pydantic",
        "pydantic-settings",
        "PyMuPDF",
        "Pillow",
        "pytesseract",
    )
    .env(
        {
            "APP_ENV": "production",
            "APP_MODE": "full",
            "DEVICE": "cuda",
            "ARTIFACT_ROOT": "/artifacts/vn_history_deployment",
            "HF_HOME": "/hf-cache",
        }
    )
    .add_local_python_source("app")
)


@app.function(
    image=image,
    gpu="L4",
    cpu=4.0,
    memory=32768,
    timeout=1800,
    volumes={
        "/artifacts": artifacts,
        "/hf-cache": hf_cache,
    },
)
def full_runtime_sanity():
    import time

    import torch

    from app.config import settings
    from app.rag.generation import RAGGenerator
    from app.rag.retrieval import HybridRetriever
    from app.services.rag_service import RAGService

    print("=" * 72)
    print("VN HISTORY FULL RAG RUNTIME SANITY")
    print("=" * 72)

    print("\n[0] GPU CHECK")
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("GPU            :", torch.cuda.get_device_name(0))
    print(
        "VRAM GB        :",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    )

    service = RAGService()

    try:
        print("\n[1] LOADING FULL RAG SERVICE")
        started = time.perf_counter()

        service.load()

        load_seconds = time.perf_counter() - started

        print("Service loaded :", service.loaded)
        print("Runtime mode   :", settings.app_mode)
        print("Target device  :", settings.device)
        print("Corpus chunks  :", len(service.chunks))
        print(
            "FAISS vectors  :",
            service.faiss_index.ntotal if service.faiss_index is not None else None,
        )
        print("BM25 loaded    :", service.bm25 is not None)
        print("E5 loaded      :", service.embedder is not None)
        print("Reranker loaded:", service.reranker is not None)
        print("Tokenizer      :", service.tokenizer is not None)
        print("Qwen loaded    :", service.model is not None)
        print("Load time      :", round(load_seconds, 2), "s")

        print(
            "CUDA allocated :",
            round(torch.cuda.memory_allocated() / 1024**3, 2),
            "GB",
        )
        print(
            "CUDA reserved  :",
            round(torch.cuda.memory_reserved() / 1024**3, 2),
            "GB",
        )

        checks = {
            "service": service.loaded,
            "runtime_mode": settings.app_mode == "full",
            "device": settings.device == "cuda",
            "corpus": len(service.chunks) == 58603,
            "faiss": (
                service.faiss_index is not None
                and service.faiss_index.ntotal == 58603
            ),
            "bm25": service.bm25 is not None,
            "embedder": service.embedder is not None,
            "reranker": service.reranker is not None,
            "tokenizer": service.tokenizer is not None,
            "qwen": service.model is not None,
        }

        print("\n[2] RESOURCE CHECKS")

        for name, passed in checks.items():
            print(f"{'PASS' if passed else 'FAIL'}  {name}")

        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise RuntimeError(
                "One or more full runtime resources failed: "
                + ", ".join(failed)
            )

        print("\n[3] BUILDING FULL PIPELINE")

        retriever = HybridRetriever(service)
        generator = RAGGenerator(
            service=service,
            retriever=retriever,
        )

        print("PASS  HybridRetriever initialized")
        print("PASS  RAGGenerator initialized")

        history_question = "So sánh Cách mạng Tháng Tám và chiến thắng Điện Biên Phủ."

        print("\n[4] FULL HISTORY CHAT")
        print("Question:", history_question)

        started = time.perf_counter()

        result = generator.chat(
            history_question,
            final_k=6,
        )

        generation_seconds = time.perf_counter() - started

        print("\nStatus          :", result.get("status"))
        print("Rewrite used    :", result.get("rewrite_used"))
        print("Expansion used  :", result.get("structured_expansion_used"))
        print("Source IDs      :", result.get("source_ids"))
        print("Model source IDs:", result.get("model_source_ids"))
        print("Invalid sources :", result.get("invalid_source_ids"))
        print("Unsupported yrs :", result.get("unsupported_years"))
        print("Format OK       :", result.get("format_ok"))
        print("Support score   :", result.get("support_score"))
        print("Quality warnings:", result.get("quality_warnings"))
        print("Repair candidate:", result.get("repair_diagnostics"))
        print("Latency         :", round(generation_seconds, 2), "s")

        print("\nFINAL ANSWER")
        print("-" * 72)
        print(result.get("answer"))
        print("-" * 72)

        if not result.get("answer"):
            raise RuntimeError("Full pipeline returned an empty answer.")

        if not generator.guards.has_required_answer_structure(result["answer"]):
            raise RuntimeError("History answer did not contain the required Markdown sections.")

        blocking_quality_issues = {
            "missing_required_sections",
            "repeated_answer_sections",
            "answer_too_short",
            "multi_part_answer_too_short",
        }
        remaining_quality_issues = set(result.get("quality_warnings", []))
        if blocking_quality_issues & remaining_quality_issues:
            raise RuntimeError(
                "Structured answer quality checks failed: "
                + ", ".join(sorted(blocking_quality_issues & remaining_quality_issues))
            )

        if result.get("status") in {
            "blocked_invalid_source",
            "blocked_unsupported_year",
            "insufficient_evidence",
        }:
            raise RuntimeError(
                f"History answer was blocked with status={result.get('status')}"
            )

        if result.get("invalid_source_ids"):
            raise RuntimeError(
                f"Invalid source IDs detected: {result.get('invalid_source_ids')}"
            )

        if result.get("unsupported_years"):
            raise RuntimeError(
                f"Unsupported years detected: {result.get('unsupported_years')}"
            )

        if not result.get("source_ids"):
            raise RuntimeError("History answer returned no valid source IDs.")

        print("\nPASS  full history generation")

        print("\n[5] FULL OOD SHORT-CIRCUIT")

        ood_question = "Hãy viết một chương trình Python sắp xếp danh sách."
        print("Question:", ood_question)

        started = time.perf_counter()

        ood_result = generator.chat(
            ood_question,
            final_k=6,
        )

        ood_seconds = time.perf_counter() - started

        print("Status    :", ood_result.get("status"))
        print("Answer    :", ood_result.get("answer"))
        print("Source IDs:", ood_result.get("source_ids"))
        print("Latency   :", round(ood_seconds, 2), "s")

        retrieval = ood_result.get("retrieval", {})

        if not retrieval.get("is_ood"):
            raise RuntimeError("OOD question was not detected.")

        if ood_result.get("source_ids"):
            raise RuntimeError("OOD response unexpectedly returned sources.")

        print("PASS  OOD request blocked before factual generation")

        print("\n[6] FINAL GPU MEMORY")
        print(
            "CUDA allocated:",
            round(torch.cuda.memory_allocated() / 1024**3, 2),
            "GB",
        )
        print(
            "CUDA reserved :",
            round(torch.cuda.memory_reserved() / 1024**3, 2),
            "GB",
        )

        print("\n" + "=" * 72)
        print("FINAL RESULT: FULL RAG RUNTIME SANITY PASS")
        print("=" * 72)

    finally:
        try:
            hf_cache.commit()
            print("\nHugging Face cache committed.")
        except Exception as exc:
            print("HF cache commit warning:", exc)

        service.shutdown()


@app.local_entrypoint()
def main():
    full_runtime_sanity.remote()
