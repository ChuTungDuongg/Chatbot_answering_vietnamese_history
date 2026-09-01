from __future__ import annotations

import json

import modal


app = modal.App("vn-history-hf-cache-seed")

hf_cache = modal.Volume.from_name(
    "vn-history-hf-cache",
    create_if_missing=False,
)

image = (
    modal.Image.from_dockerfile(
        "Dockerfile",
        context_dir=".",
    )
    .env(
        {
            "HF_HOME": "/hf-cache",
            "HF_HUB_CACHE": "/hf-cache/hub",
        }
    )
    .add_local_python_source("app")
)


@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    timeout=7200,
    volumes={"/hf-cache": hf_cache},
)
def seed_hf_cache_remote(model_ids: list[str], validate_only: bool = False) -> dict:
    from app.agents.hf_cache import hf_cache_status, seed_hf_cache

    reports = []
    for model_id in model_ids:
        report = (
            hf_cache_status(model_id, cache_dir="/hf-cache/hub")
            if validate_only
            else seed_hf_cache(model_id, cache_dir="/hf-cache/hub")
        )
        reports.append(report)
    if not validate_only:
        hf_cache.commit()
    return {"ok": all(item.get("cache_hit") for item in reports), "models": reports}


@app.local_entrypoint()
def main(
    include_shared_4b: bool = False,
    validate_only: bool = False,
):
    from app.agents.model_registry import CENTRAL_BASE_MODEL_ID, SHARED_BASE_MODEL_ID

    models = [CENTRAL_BASE_MODEL_ID]
    if include_shared_4b:
        models.append(SHARED_BASE_MODEL_ID)
    result = seed_hf_cache_remote.remote(models, validate_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))
