from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.artifact_contract import (
    build_artifact_lock,
    count_jsonl,
    inference_config_payload,
    manifest_payload,
    validate_artifact_lock,
    write_artifact_lock,
)
from app.agents.common.model_registry import registry_manifest


def migrate_base_only(root: str | Path) -> dict:
    """Regenerate only canonical metadata; retain every adapter byte on disk."""
    root_path = Path(root).resolve()
    corpus = root_path / "corpus" / "vn_history_rag_chunks_enriched.jsonl"
    config_dir = root_path / "config"
    if not corpus.is_file() or not config_dir.is_dir():
        raise FileNotFoundError(f"Not a canonical deployment artifact root: {root_path}")

    (config_dir / "inference_config.json").write_text(
        json.dumps(inference_config_payload(central_adapter_path=None), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (config_dir / "model_registry.json").write_text(
        json.dumps(registry_manifest(central_adapter_path=None), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    corpus_count = count_jsonl(corpus)
    (root_path / "manifest.json").write_text(
        json.dumps(manifest_payload(corpus_count=corpus_count), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    provisional = build_artifact_lock(root_path)
    (root_path / "manifest.json").write_text(
        json.dumps(
            manifest_payload(corpus_count=corpus_count, deployment_id=provisional["deployment_id"]),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    lock = write_artifact_lock(root_path)
    validate_artifact_lock(root_path)
    return {
        "root": str(root_path),
        "deployment_id": lock["deployment_id"],
        "central_adapter_present": lock["central_adapter_present"],
        "retained_legacy_central_adapter_directory": (root_path / "adapters" / "central").is_dir(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate artifact metadata to base-only Central V2.")
    parser.add_argument("root")
    args = parser.parse_args(argv)
    print(json.dumps(migrate_base_only(args.root), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
