"""LEGACY ONLY: repair shards in the old merged Qwen2.5 History baseline."""

import json
from pathlib import Path

import modal


app = modal.App("vn-history-fix-model-shards")

artifacts = modal.Volume.from_name(
    "vn-history-artifacts",
    create_if_missing=False,
)

MODEL_DIR = Path(
    "/artifacts/history_answerer/model"
)


@app.function(volumes={"/artifacts": artifacts})
def fix_model_shards():
    print("=" * 72)
    print("VN HISTORY MODEL SHARD REPAIR")
    print("=" * 72)

    index_path = MODEL_DIR / "model.safetensors.index.json"

    if not index_path.exists():
        raise RuntimeError(f"Missing model index: {index_path}")

    with index_path.open("r", encoding="utf-8") as file:
        index_data = json.load(file)

    expected_shards = sorted(set(index_data["weight_map"].values()))
    actual_shards = sorted(path.name for path in MODEL_DIR.glob("*.safetensors"))

    print("\nExpected by index:")
    for name in expected_shards:
        print(" -", name)

    print("\nActual files:")
    for name in actual_shards:
        path = MODEL_DIR / name
        print(f" - {name} ({path.stat().st_size / 1024**3:.2f} GB)")

    print("\n[1] CHECKING SHARDS")

    renamed = []

    for expected_name in expected_shards:
        expected_path = MODEL_DIR / expected_name

        if expected_path.exists():
            print("PASS ", expected_name)
            continue

        stem = expected_name.removesuffix(".safetensors")
        candidates = sorted(MODEL_DIR.glob(f"{stem}-*.safetensors"))

        if len(candidates) == 0:
            raise RuntimeError(
                f"Missing shard '{expected_name}' and no rename candidate was found."
            )

        if len(candidates) > 1:
            names = [path.name for path in candidates]
            raise RuntimeError(
                f"Multiple candidates found for '{expected_name}': {names}"
            )

        source = candidates[0]

        print(f"RENAME {source.name}")
        print(f"    -> {expected_name}")

        source.rename(expected_path)
        renamed.append((source.name, expected_name))

    print("\n[2] VERIFYING INDEX")

    missing_after_repair = [
        name
        for name in expected_shards
        if not (MODEL_DIR / name).exists()
    ]

    if missing_after_repair:
        raise RuntimeError(
            f"Still missing shards after repair: {missing_after_repair}"
        )

    for name in expected_shards:
        path = MODEL_DIR / name
        size_gb = path.stat().st_size / 1024**3
        print(f"PASS  {name}: {size_gb:.2f} GB")

        if path.stat().st_size == 0:
            raise RuntimeError(f"Shard is empty: {name}")

    artifacts.commit()

    print("\nVolume changes committed.")

    print("\n[3] FINAL MODEL FILES")
    for path in sorted(MODEL_DIR.iterdir()):
        if path.is_file():
            if path.suffix == ".safetensors":
                print(f"{path.name}: {path.stat().st_size / 1024**3:.2f} GB")
            else:
                print(path.name)

    print("\n" + "=" * 72)
    print("FINAL RESULT: MODEL SHARD REPAIR PASS")
    print("=" * 72)


@app.local_entrypoint()
def main():
    fix_model_shards.remote()
