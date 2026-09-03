from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path

from app.artifact_contract import (
    build_artifact_lock,
    inference_config_payload,
    manifest_payload,
    validate_artifact_lock,
    write_artifact_lock,
)
from app.agents.common.model_registry import (
    FUTURE_CENTRAL_V2_ADAPTER_PATH,
    ROLE_MODELS,
    registry_manifest,
    validate_central_adapter,
    validate_role_adapter,
)


PRODUCTION_ADAPTER_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
}
FORBIDDEN_NAMES = {
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "rng_state.pth",
}


def _copy(src: str | Path, dst: str | Path) -> None:
    src_path = Path(src)
    dst_path = Path(dst)
    if src_path.is_dir():
        shutil.copytree(src_path, dst_path)
    else:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)


def _copy_adapter(src: str | Path, dst: str | Path) -> None:
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.mkdir(parents=True, exist_ok=True)
    for name in sorted(PRODUCTION_ADAPTER_FILES):
        source_file = src_path / name
        if not source_file.is_file():
            raise FileNotFoundError(f"Missing production adapter file: {source_file}")
        shutil.copy2(source_file, dst_path / name)


def _assert_no_forbidden_files(root: Path) -> None:
    violations: list[str] = []
    for path in root.rglob("*"):
        if not path.exists():
            continue
        if path.is_dir() and path.name.startswith("checkpoint-"):
            violations.append(str(path.relative_to(root)))
        if path.is_file() and (path.name in FORBIDDEN_NAMES or "qwen2.5" in str(path).lower()):
            violations.append(str(path.relative_to(root)))
    if violations:
        raise RuntimeError("Forbidden deployment artifacts found:\n" + "\n".join(sorted(violations)))


def _replace_directory_atomically_enough(src: Path, dst: Path) -> list[str]:
    removed: list[str] = []
    backup = dst.with_name(f".{dst.name}.backup-{uuid.uuid4().hex}")
    if backup.exists():
        raise RuntimeError(f"Unexpected backup path exists: {backup}")
    try:
        if dst.exists():
            removed = [
                str(path.relative_to(dst))
                for path in dst.rglob("*")
                if path.is_file() or path.is_dir()
            ]
            shutil.move(str(dst), str(backup))
        shutil.move(str(src), str(dst))
        if backup.exists():
            shutil.rmtree(backup)
        return sorted(removed)
    except Exception:
        if dst.exists() and backup.exists():
            shutil.rmtree(dst)
        if backup.exists():
            shutil.move(str(backup), str(dst))
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shared-Qwen3 multi-adapter deployment artifact exporter.")
    parser.add_argument("--model-dir", default=None, help="Optional legacy Qwen2.5 History baseline model.")
    parser.add_argument("--research-agent", required=True, help="Research Agent adapter directory.")
    parser.add_argument("--evidence-agent", required=True, help="Evidence Agent adapter directory.")
    parser.add_argument("--history-agent", required=True, help="Fresh Qwen3 History Answerer adapter directory.")
    parser.add_argument("--central-agent", default=None, help="Optional future Qwen3-8B Central V2 adapter directory.")
    parser.add_argument(
        "--central-adapter-relative-path",
        default=FUTURE_CENTRAL_V2_ADAPTER_PATH,
        help="Artifact-relative destination used only when --central-agent is configured.",
    )
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--retrieval-dir", required=True)
    parser.add_argument("--output-root", default="artifacts/vn_history_deployment")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.output_root)
    tmp_root = root.with_name(f".{root.name}.tmp-{uuid.uuid4().hex}")
    adapter_sources = {
        "research": args.research_agent,
        "evidence": args.evidence_agent,
        "history": args.history_agent,
    }
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    try:
        for role, source in adapter_sources.items():
            validate_role_adapter(role, source)
            _copy_adapter(source, tmp_root / ROLE_MODELS[role].adapter_path)
        central_adapter_path = None
        if args.central_agent:
            central_adapter_path = args.central_adapter_relative_path
            validate_central_adapter(args.central_agent)
            _copy_adapter(args.central_agent, tmp_root / central_adapter_path)
        if args.model_dir:
            legacy_dir = tmp_root / "legacy" / "qwen25_history" / "benchmark_only_model"
            _copy(args.model_dir, legacy_dir)
        _copy(args.corpus, tmp_root / "corpus" / "vn_history_rag_chunks_enriched.jsonl")
        _copy(Path(args.retrieval_dir) / "faiss", tmp_root / "retrieval" / "faiss")
        _copy(Path(args.retrieval_dir) / "bm25s_index", tmp_root / "retrieval" / "bm25s_index")
        (tmp_root / "config").mkdir(parents=True, exist_ok=True)
        (tmp_root / "config" / "inference_config.json").write_text(
            json.dumps(inference_config_payload(central_adapter_path=central_adapter_path), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        model_registry = registry_manifest(central_adapter_path=central_adapter_path)
        (tmp_root / "config" / "model_registry.json").write_text(
            json.dumps(model_registry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        corpus_count = sum(1 for line in Path(args.corpus).open("r", encoding="utf-8") if line.strip())
        provisional_manifest = manifest_payload(corpus_count=corpus_count, central_adapter_path=central_adapter_path)
        (tmp_root / "manifest.json").write_text(
            json.dumps(provisional_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        provisional_lock = build_artifact_lock(tmp_root)
        manifest = manifest_payload(
            corpus_count=corpus_count,
            deployment_id=provisional_lock["deployment_id"],
            central_adapter_path=central_adapter_path,
        )
        (tmp_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        lock = write_artifact_lock(tmp_root)
        if lock["deployment_id"] != provisional_lock["deployment_id"]:
            raise RuntimeError("deployment_id derivation changed after manifest finalization")
        (tmp_root / "EXPORT_SUCCESS.txt").write_text("ok\n", encoding="utf-8")
        _assert_no_forbidden_files(tmp_root)
        validate_artifact_lock(tmp_root)
        removed = _replace_directory_atomically_enough(tmp_root, root)
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
    print(f"Exported canonical deployment artifacts to {root}")
    print(f"deployment_id={lock['deployment_id']}")
    print(f"stale_files_removed={len(removed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
