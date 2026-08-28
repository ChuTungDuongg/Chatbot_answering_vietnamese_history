from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from app.agents.model_registry import (
    ROLE_MODELS,
    SHARED_BASE_MODEL_ID,
    registry_manifest,
    validate_role_adapter,
)


def _copy(src: str | Path, dst: str | Path) -> None:
    src_path = Path(src)
    dst_path = Path(dst)
    if src_path.is_dir():
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    else:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shared-Qwen3 multi-adapter deployment artifact exporter.")
    parser.add_argument("--model-dir", default=None, help="Optional legacy Qwen2.5 History baseline model.")
    parser.add_argument("--research-agent", required=True, help="Research Agent adapter directory.")
    parser.add_argument("--evidence-agent", required=True, help="Evidence Agent adapter directory.")
    parser.add_argument("--history-agent", required=True, help="Fresh Qwen3 History Answerer adapter directory.")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--retrieval-dir", required=True)
    parser.add_argument("--output-root", default="artifacts/vn_history_deployment")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.output_root)
    adapter_sources = {
        "research": args.research_agent,
        "evidence": args.evidence_agent,
        "history": args.history_agent,
    }
    for role, source in adapter_sources.items():
        validate_role_adapter(role, source)
        _copy(source, root / ROLE_MODELS[role].adapter_path)
    if args.model_dir:
        _copy(args.model_dir, root / "legacy" / "qwen25_history" / "model")
    _copy(args.corpus, root / "corpus" / "vn_history_rag_chunks_enriched.jsonl")
    _copy(Path(args.retrieval_dir) / "faiss", root / "retrieval" / "faiss")
    _copy(Path(args.retrieval_dir) / "bm25s_index", root / "retrieval" / "bm25s_index")
    (root / "config").mkdir(parents=True, exist_ok=True)
    config = {
        "llm": {
            "backend": "transformers",
            "shared_base_model_id": SHARED_BASE_MODEL_ID,
            "tokenizer_model_id": SHARED_BASE_MODEL_ID,
            "role_models": {
                role: spec.model_name for role, spec in ROLE_MODELS.items()
            },
            "vllm_base_url": "http://127.0.0.1:8001/v1",
        },
        "retrieval": {
            "embedding_model_id": "intfloat/multilingual-e5-base",
            "reranker_model_id": "BAAI/bge-reranker-v2-m3",
        },
        "generation": {role: spec.generation for role, spec in ROLE_MODELS.items()},
    }
    (root / "config" / "inference_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    corpus_count = sum(1 for _ in Path(args.corpus).open("r", encoding="utf-8"))
    model_registry = registry_manifest()
    (root / "config" / "model_registry.json").write_text(
        json.dumps(model_registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        **model_registry,
        "corpus": {"count": corpus_count},
        "retrieval": {
            "faiss": "retrieval/faiss",
            "bm25s": "retrieval/bm25s_index",
        },
        "base_weights_bundled": False,
        "legacy_history_model": "legacy/qwen25_history/model" if args.model_dir else None,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "EXPORT_SUCCESS.txt").write_text("ok\n", encoding="utf-8")
    print(f"Exported deployment artifacts to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
