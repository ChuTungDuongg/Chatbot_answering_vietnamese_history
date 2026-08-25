from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _copy(src: str | Path, dst: str | Path) -> None:
    src_path = Path(src)
    dst_path = Path(dst)
    if src_path.is_dir():
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    else:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 10 deployment artifact exporter.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--research-agent", required=True, help="Research Agent adapter directory.")
    parser.add_argument("--evidence-agent", required=True, help="Evidence Agent adapter directory.")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--retrieval-dir", required=True)
    parser.add_argument("--output-root", default="artifacts/vn_history_deployment")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.output_root)
    _copy(args.model_dir, root / "history_answerer" / "model")
    _copy(args.research_agent, root / "research_agent" / "adapter")
    _copy(args.evidence_agent, root / "evidence_agent" / "adapter")
    _copy(args.corpus, root / "corpus" / "vn_history_rag_chunks_enriched.jsonl")
    _copy(Path(args.retrieval_dir) / "faiss", root / "retrieval" / "faiss")
    _copy(Path(args.retrieval_dir) / "bm25s_index", root / "retrieval" / "bm25s_index")
    (root / "config").mkdir(parents=True, exist_ok=True)
    config = {
        "retrieval": {
            "embedding_model_id": "intfloat/multilingual-e5-base",
            "reranker_model_id": "BAAI/bge-reranker-v2-m3",
        },
        "generation": {"max_new_tokens": 300, "temperature": 0.0, "top_p": 1.0},
    }
    (root / "config" / "inference_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    corpus_count = sum(1 for _ in Path(args.corpus).open("r", encoding="utf-8"))
    manifest = {
        "corpus": {"count": corpus_count},
        "models": {
            "history_answerer": "history_answerer/model",
            "research_agent": "research_agent/adapter",
            "evidence_agent": "evidence_agent/adapter",
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (root / "EXPORT_SUCCESS.txt").write_text("ok\n", encoding="utf-8")
    print(f"Exported deployment artifacts to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
