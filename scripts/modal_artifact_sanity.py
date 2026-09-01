import hashlib
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
EXPECTED_CENTRAL_BASE = "Qwen/Qwen3-8B"
ROLE_ADAPTERS = {
    "research": ROOT / "adapters" / "research",
    "evidence": ROOT / "adapters" / "evidence",
    "history": ROOT / "adapters" / "history",
}
CENTRAL_ADAPTER = ROOT / "adapters" / "central"
EXPECTED_WEIGHT_HASHES = {
    "research": "0d36e09fb947a6b077ee493f3589a36bf68dba0403f7eac91f684d070d399086",
    "evidence": "39385ca7c82b57b5ff8c9a531b5359509ea185b15e5a0adb0724626c58ed7ff6",
    "history": "70d873e15c48f5802e26d0c32eab7c63ea7b83f713be3192092476a0dac746a3",
}


def size_mb(path: Path) -> float:
    return round(path.stat().st_size / 1024**2, 2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.function(volumes={"/artifacts": volume})
def inventory_json():
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        files.append(
            {
                "path": "/" + str(path.relative_to(ROOT)).replace("\\", "/"),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "root": str(ROOT),
        "files": files,
    }


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
        "artifact lock": ROOT / "artifact_lock.json",
        "export success marker": ROOT / "EXPORT_SUCCESS.txt",
        "inference config": ROOT / "config" / "inference_config.json",
        "model registry": ROOT / "config" / "model_registry.json",
        "corpus": ROOT / "corpus" / "vn_history_rag_chunks_enriched.jsonl",
        "FAISS index": ROOT / "retrieval" / "faiss" / "chunks.index",
        "FAISS manifest": ROOT / "retrieval" / "faiss" / "manifest.json",
        "BM25 directory": ROOT / "retrieval" / "bm25s_index",
        "research adapter directory": ROLE_ADAPTERS["research"],
        "evidence adapter directory": ROLE_ADAPTERS["evidence"],
        "history adapter directory": ROLE_ADAPTERS["history"],
        "central adapter directory": CENTRAL_ADAPTER,
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
    artifact_lock = None

    print("[2] DEPLOYMENT MANIFEST")

    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as file:
                manifest = json.load(file)

            print("PASS  manifest.json parsed successfully")
            print("Keys :", list(manifest.keys()))
            print("Deployment ID:", manifest.get("deployment_id"))

            manifest_base = manifest.get("shared_base_model_id")
            if manifest_base == EXPECTED_SHARED_BASE:
                print("PASS  shared base model:", manifest_base)
            else:
                failures.append(
                    f"Manifest shared base {manifest_base!r} != {EXPECTED_SHARED_BASE!r}"
                )
            if manifest.get("central", {}).get("expected_base_model_id") != EXPECTED_CENTRAL_BASE:
                failures.append("Manifest Central base does not match Qwen3-8B")

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

    lock_path = required_paths["artifact lock"]

    print("[2b] ARTIFACT LOCK")

    if lock_path.exists():
        try:
            with lock_path.open("r", encoding="utf-8") as file:
                artifact_lock = json.load(file)
            print("PASS  artifact_lock.json parsed successfully")
            print("Deployment ID:", artifact_lock.get("deployment_id"))
            print("Base:", artifact_lock.get("shared_base_model_id"))
            if artifact_lock.get("shared_base_model_id") != EXPECTED_SHARED_BASE:
                failures.append("Artifact lock shared base does not match Qwen3")
            if artifact_lock.get("central_base_model_id") != EXPECTED_CENTRAL_BASE:
                failures.append("Artifact lock Central base does not match Qwen3-8B")
        except Exception as exc:
            print("FAIL  artifact_lock.json could not be parsed:", exc)
            failures.append("Invalid artifact lock")

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
            if llm.get("central", {}).get("model_id") != EXPECTED_CENTRAL_BASE:
                failures.append("Inference config Central base does not match Qwen3-8B")
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
        weight_file = adapter_dir / "adapter_model.safetensors"

        if not config_file.is_file():
            failures.append(f"Missing {role} adapter_config.json: {config_file}")
            continue
        if not weight_file.is_file():
            failures.append(f"Missing {role} adapter safetensors: {weight_file}")
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

        total_bytes = weight_file.stat().st_size
        weight_hash = sha256_file(weight_file)
        expected_hash = EXPECTED_WEIGHT_HASHES[role]
        hash_status = "MATCH" if weight_hash == expected_hash else "MISMATCH"
        if hash_status != "MATCH":
            failures.append(
                f"{role} adapter weight hash {weight_hash} != expected {expected_hash}"
            )
        if artifact_lock is not None:
            locked_hash = (
                artifact_lock.get("roles", {})
                .get(role, {})
                .get("adapter_model_sha256")
            )
            if locked_hash and locked_hash != weight_hash:
                failures.append(f"{role} remote hash does not match artifact lock")
        print(
            f"PASS  {role}: base={declared_base}, "
            f"weights=1, size={round(total_bytes / 1024**2, 2)} MB, "
            f"sha256={weight_hash}, status={hash_status}"
        )

    central_config_file = CENTRAL_ADAPTER / "adapter_config.json"
    central_weights = CENTRAL_ADAPTER / "adapter_model.safetensors"
    if central_config_file.is_file() and central_weights.is_file():
        central_config = json.loads(central_config_file.read_text(encoding="utf-8"))
        central_base = str(central_config.get("base_model_name_or_path", "")).rstrip("/")
        central_hash = sha256_file(central_weights)
        if central_base != EXPECTED_CENTRAL_BASE:
            failures.append(
                f"central adapter base mismatch: {central_base!r} != {EXPECTED_CENTRAL_BASE!r}"
            )
        locked_hash = (artifact_lock or {}).get("central", {}).get("adapter_model_sha256")
        if locked_hash and locked_hash != central_hash:
            failures.append("central remote hash does not match artifact lock")
        print(
            f"PASS  central: base={central_base}, size={size_mb(central_weights)} MB, "
            f"sha256={central_hash}"
        )
    else:
        failures.append("Missing Central adapter config or weights")

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
def main(inventory_json_output: bool = False):
    if inventory_json_output:
        print("REMOTE_INVENTORY_JSON_BEGIN")
        print(json.dumps(inventory_json.remote(), ensure_ascii=False, sort_keys=True))
        print("REMOTE_INVENTORY_JSON_END")
        return
    sanity_check.remote()
