import modal


app = modal.App("vn-history-retrieval-runtime-sanity")

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
        "faiss-cpu",
        "bm25s",
        "fastapi",
        "pydantic",
        "pydantic-settings",
    )
    .env(
        {
            "APP_ENV": "production",
            "APP_MODE": "retrieval-only",
            "DEVICE": "cpu",
            "ARTIFACT_ROOT": "/artifacts",
            "HF_HOME": "/hf-cache",
        }
    )
    .add_local_python_source("app")
)


@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    timeout=1800,
    volumes={
        "/artifacts": artifacts,
        "/hf-cache": hf_cache,
    },
)
def runtime_sanity():
    from app.config import settings
    from app.rag.retrieval import HybridRetriever
    from app.services.rag_service import RAGService

    print("=" * 72)
    print("VN HISTORY RETRIEVAL RUNTIME SANITY")
    print("=" * 72)

    service = RAGService()

    try:
        print("\n[1] LOADING RAG SERVICE")
        service.load()

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
        print("Qwen loaded    :", service.model is not None)

        checks = {
            "service": service.loaded,
            "runtime_mode": settings.app_mode == "retrieval-only",
            "corpus": len(service.chunks) == 58603,
            "faiss": (
                service.faiss_index is not None
                and service.faiss_index.ntotal == 58603
            ),
            "bm25": service.bm25 is not None,
            "embedder": service.embedder is not None,
            "reranker": service.reranker is not None,
            "qwen_not_loaded": service.model is None,
        }

        print("\n[2] RESOURCE CHECKS")

        for name, passed in checks.items():
            print(f"{'PASS' if passed else 'FAIL'}  {name}")

        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise RuntimeError(
                "One or more retrieval resources failed to load: "
                + ", ".join(failed)
            )

        print("\n[3] BUILDING HYBRID RETRIEVER")
        retriever = HybridRetriever(service)
        print("PASS  HybridRetriever initialized")

        history_question = "Chiến thắng Bạch Đằng năm 938 có ý nghĩa gì?"

        print("\n[4] HISTORY RETRIEVAL")
        print("Question:", history_question)

        result = retriever.retrieve(
            history_question,
            final_k=6,
        )

        print("is_ood        :", result.get("is_ood"))
        print("ood_reason    :", result.get("ood_reason"))
        print("query variants:", result.get("query_variants"))
        print("max_dense     :", result.get("max_dense"))
        print("contexts      :", len(result.get("final_context", [])))

        if result.get("is_ood"):
            raise RuntimeError(
                "History question was incorrectly classified as OOD."
            )

        contexts = result.get("final_context", [])

        if not contexts:
            raise RuntimeError(
                "History retrieval returned no contexts."
            )

        print("\nTop contexts:")

        for index, chunk in enumerate(contexts, start=1):
            print(
                f"{index}. [{chunk.get('chunk_id')}] "
                f"{chunk.get('title')} | "
                f"score={chunk.get('final_retrieval_score')}"
            )

        print("\n[5] OOD GUARD")

        ood_question = "Hãy viết một chương trình Python sắp xếp danh sách."
        print("Question:", ood_question)

        ood_result = retriever.retrieve(
            ood_question,
            final_k=6,
        )

        print("is_ood    :", ood_result.get("is_ood"))
        print("ood_reason:", ood_result.get("ood_reason"))

        if not ood_result.get("is_ood"):
            raise RuntimeError(
                "OOD question was not blocked."
            )

        print("\n" + "=" * 72)
        print("FINAL RESULT: RETRIEVAL RUNTIME SANITY PASS")
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
    runtime_sanity.remote()
