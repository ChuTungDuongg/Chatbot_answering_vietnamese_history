import json
from pathlib import Path

import modal


app = modal.App("vn-history-artifact-sanity")

volume = modal.Volume.from_name(
    "vn-history-artifacts",
    create_if_missing=False,
)

ROOT = Path("/artifacts")
EXPECTED_CORPUS_COUNT = 58603
EXPECTED_SHARED_BASE = "Qwen/Qwen3-4B-Instruct-2507"
ROLE_ADAPTERS = {
    "research": ROOT / "adapters" / "research",
    "evidence": ROOT / "adapters" / "evidence",
    "history": ROOT / "adapters" / "history",
}


def size_mb(path: Path) -> float:
    return round(path.stat().st_size / 1024**2, 2)


@app.function(volumes={"/artifacts": volume})
def sanity_check():
    print("=" * 72)
    print("VN HISTORY DEPLOYMENT ARTIFACT SANITY CHECK")
    print("=" * 72)
    print("Artifact root:", ROOT)
    print()

    failures = []
    warnings = []

    required_paths = {
        "deployment manifest": ROOT / "manifest.json",
        "export success marker": ROOT / "EXPORT_SUCCESS.txt",
        "inference config": ROOT / "config" / "inference_config.json",
        "corpus": ROOT / "corpus" / "vn_history_rag_chunks_enriched.jsonl",
        "FAISS index": ROOT / "retrieval" / "faiss" / "chunks.index",
        "FAISS manifest": ROOT / "retrieval" / "faiss" / "manifest.json",
        "BM25 directory": ROOT / "retrieval" / "bm25s_index",
        "research adapter directory": ROLE_ADAPTERS["research"],
        "evidence adapter directory": ROLE_ADAPTERS["evidence"],
        "history adapter directory": ROLE_ADAPTERS["history"],
    }

    print("[1] REQUIRED PATHS")

    for label, path in required_paths.items():
        if path.exists():
            print(f"PASS  {label}: {path}")
        else:
            print(f"FAIL  {label}: {path}")
            failures.append(f"Missing {label}: {path}")

    print()

    manifest_path = required_paths["deployment manifest"]

    print("[2] DEPLOYMENT MANIFEST")

    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as file:
                manifest = json.load(file)

            print("PASS  manifest.json parsed successfully")
            print("Keys :", list(manifest.keys()))

            manifest_base = manifest.get("shared_base_model_id")
            if manifest_base == EXPECTED_SHARED_BASE:
                print("PASS  shared base model:", manifest_base)
            else:
                failures.append(
                    f"Manifest shared base {manifest_base!r} != {EXPECTED_SHARED_BASE!r}"
                )

            if manifest.get("base_weights_bundled") is not False:
                failures.append("Manifest must declare base_weights_bundled=false")

            try:
                manifest_count = int(manifest["corpus"]["count"])
                print("Corpus count in manifest:", manifest_count)

                if manifest_count != EXPECTED_CORPUS_COUNT:
                    failures.append(
                        f"Manifest corpus count {manifest_count} != {EXPECTED_CORPUS_COUNT}"
                    )
            except Exception:
                warnings.append("Could not read manifest['corpus']['count']")

        except Exception as exc:
            print("FAIL  manifest.json could not be parsed:", exc)
            failures.append("Invalid deployment manifest")

    print()

    config_path = required_paths["inference config"]

    print("[3] INFERENCE CONFIG")

    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as file:
                config = json.load(file)

            print("PASS  inference_config.json parsed successfully")
            print("Keys :", list(config.keys()))
            llm = config.get("llm", {})
            if llm.get("shared_base_model_id") != EXPECTED_SHARED_BASE:
                failures.append("Inference config shared base does not match Qwen3 registry")
            role_models = llm.get("role_models", {})
            if set(role_models) != set(ROLE_ADAPTERS):
                failures.append(
                    "Inference config must route research, evidence, and history roles"
                )
        except Exception as exc:
            print("FAIL  inference_config.json could not be parsed:", exc)
            failures.append("Invalid inference config")

    print()

    corpus_path = required_paths["corpus"]

    print("[4] CORPUS")

    if corpus_path.exists():
        corpus_count = 0
        duplicate_ids = 0
        missing_ids = 0
        seen_ids = set()

        try:
            with corpus_path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        failures.append(f"Invalid JSONL at corpus line {line_number}")
                        continue

                    corpus_count += 1
                    chunk_id = row.get("chunk_id")

                    if not chunk_id:
                        missing_ids += 1
                        continue

                    chunk_id = str(chunk_id)

                    if chunk_id in seen_ids:
                        duplicate_ids += 1
                    else:
                        seen_ids.add(chunk_id)

            print("Corpus size MB :", size_mb(corpus_path))
            print("Corpus chunks  :", corpus_count)
            print("Unique IDs     :", len(seen_ids))
            print("Missing IDs    :", missing_ids)
            print("Duplicate IDs  :", duplicate_ids)

            if corpus_count == EXPECTED_CORPUS_COUNT:
                print(f"PASS  corpus count = {EXPECTED_CORPUS_COUNT}")
            else:
                failures.append(
                    f"Corpus count {corpus_count} != expected {EXPECTED_CORPUS_COUNT}"
                )

            if missing_ids == 0:
                print("PASS  no missing chunk_id")
            else:
                failures.append(f"{missing_ids} chunks missing chunk_id")

            if duplicate_ids == 0:
                print("PASS  chunk_id values are unique")
            else:
                failures.append(f"{duplicate_ids} duplicate chunk_id values")

        except Exception as exc:
            print("FAIL  corpus check failed:", exc)
            failures.append("Corpus validation failed")

    print()

    faiss_path = required_paths["FAISS index"]

    print("[5] FAISS")

    if faiss_path.exists():
        print("FAISS size MB:", size_mb(faiss_path))

        if faiss_path.stat().st_size > 0:
            print("PASS  FAISS index is non-empty")
        else:
            failures.append("FAISS index is empty")

    print()

    bm25_dir = required_paths["BM25 directory"]

    print("[6] BM25")

    if bm25_dir.exists():
        bm25_files = [path for path in bm25_dir.rglob("*") if path.is_file()]
        bm25_size = sum(path.stat().st_size for path in bm25_files)

        print("BM25 files   :", len(bm25_files))
        print("BM25 size MB :", round(bm25_size / 1024**2, 2))

        if bm25_files:
            print("PASS  BM25 directory contains files")
        else:
            failures.append("BM25 directory is empty")

    print()

    print("[7] SHARED QWEN3 ROLE ADAPTERS")

    for role, adapter_dir in ROLE_ADAPTERS.items():
        config_file = adapter_dir / "adapter_config.json"
        weights = sorted(adapter_dir.glob("adapter_model*.safetensors"))

        if not config_file.is_file():
            failures.append(f"Missing {role} adapter_config.json: {config_file}")
            continue
        if not weights:
            failures.append(f"Missing {role} adapter safetensors: {adapter_dir}")
            continue

        try:
            adapter_config = json.loads(config_file.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"Invalid {role} adapter_config.json: {exc}")
            continue

        declared_base = str(adapter_config.get("base_model_name_or_path", "")).rstrip("/")
        if declared_base != EXPECTED_SHARED_BASE:
            failures.append(
                f"{role} adapter base mismatch: {declared_base!r} != {EXPECTED_SHARED_BASE!r}"
            )
            continue

        total_bytes = sum(path.stat().st_size for path in weights)
        print(
            f"PASS  {role}: base={declared_base}, "
            f"weights={len(weights)}, size={round(total_bytes / 1024**2, 2)} MB"
        )

    legacy_model = ROOT / "legacy" / "qwen25_history" / "model"
    if legacy_model.exists():
        warnings.append(
            "Legacy Qwen2.5 History baseline is present for comparison only; "
            "it is not part of the active shared-base runtime."
        )

    print()

    print("[8] EXPORT SUCCESS MARKER")

    success_path = required_paths["export success marker"]

    if success_path.exists():
        try:
            content = success_path.read_text(encoding="utf-8").strip()
            print("PASS  EXPORT_SUCCESS.txt exists")
            print("Content:", content[:300])
        except Exception as exc:
            warnings.append(f"Could not read EXPORT_SUCCESS.txt: {exc}")

    print()
    print("=" * 72)

    if warnings:
        print("WARNINGS")

        for warning in warnings:
            print(" -", warning)

        print()

    if failures:
        print("FINAL RESULT: ARTIFACT SANITY FAILED")

        for failure in failures:
            print(" -", failure)

        raise RuntimeError(
            f"Artifact sanity failed with {len(failures)} error(s)."
        )

    print("FINAL RESULT: ARTIFACT SANITY PASS")
    print("=" * 72)


@app.local_entrypoint()
def main():
    sanity_check.remote()
