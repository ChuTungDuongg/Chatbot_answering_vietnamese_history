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

chat_data = modal.Volume.from_name(
    "vn-history-chat-data",
    create_if_missing=True,
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
        "CHAT_DATABASE_PATH": "/data/chat.sqlite3",
        "HF_HOME": "/hf-cache",
        "CORS_ORIGINS": "http://localhost:5173,http://127.0.0.1:5173",
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
        "/data": chat_data,
    },
)
@modal.asgi_app()
def fastapi_app():
    from app.main import app as fastapi_application

    return fastapi_application