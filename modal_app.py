import modal


app = modal.App("vn-history-rag-api")

artifacts = modal.Volume.from_name(
    "vn-history-artifacts",
    create_if_missing=False,
)

hf_cache = modal.Volume.from_name(
    "vn-history-hf-cache",
    create_if_missing=False,
)

image = modal.Image.from_dockerfile(
    "Dockerfile",
    context_dir=".",
).env(
    {
        "APP_ENV": "production",
        "APP_MODE": "full",
        "DEVICE": "cuda",
        "ARTIFACT_ROOT": "/artifacts/vn_history_deployment",
        "HF_HOME": "/hf-cache",
    }
)


@app.function(
    image=image,
    gpu="L4",
    cpu=4.0,
    memory=32768,
    timeout=600,
    startup_timeout=300,
    min_containers=0,
    max_containers=1,
    scaledown_window=120,
    volumes={
        "/artifacts": artifacts,
        "/hf-cache": hf_cache,
    },
)
@modal.asgi_app()
def fastapi_app():
    from app.main import app as fastapi_application

    return fastapi_application