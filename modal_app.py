import os

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

web_search_secret_name = os.getenv("MODAL_WEB_SEARCH_SECRET_NAME", "").strip()
runtime_secrets = (
    [modal.Secret.from_name(web_search_secret_name)]
    if web_search_secret_name
    else []
)

image = modal.Image.from_dockerfile(
    "Dockerfile",
    context_dir=".",
).env(
    {
        "APP_ENV": "production",
        "APP_MODE": "full",
        "DEVICE": "cuda",
        "DTYPE": "bfloat16",
        "ARTIFACT_ROOT": "/artifacts",
        "LLM_BACKEND": "transformers",
        "SHARED_BASE_MODEL_ID": "Qwen/Qwen3-4B-Instruct-2507",
        "RESEARCH_AGENT_MODEL": "Qwen/Qwen3-4B-Instruct-2507",
        "RESEARCH_AGENT_ADAPTER_PATH": "/artifacts/adapters/research",
        "EVIDENCE_AGENT_MODEL": "Qwen/Qwen3-4B-Instruct-2507",
        "EVIDENCE_AGENT_ADAPTER_PATH": "/artifacts/adapters/evidence",
        "HISTORY_AGENT_ADAPTER_PATH": "/artifacts/adapters/history",
        "CENTRAL_AGENT_MODEL_ID": "Qwen/Qwen3-8B",
        "CENTRAL_AGENT_ADAPTER_PATH": "/artifacts/adapters/central",
        "CENTRAL_AGENT_MAX_STEPS": "3",
        "CENTRAL_AGENT_MAX_NEW_TOKENS": "1536",
        "CENTRAL_AGENT_TIMEOUT_SECONDS": "120",
        "CENTRAL_AGENT_OBSERVATION_CHAR_BUDGET": "12000",
        "CENTRAL_AGENT_MAX_TOOL_RESULTS": "6",
        "RUNTIME_LOADING_STRATEGY": "lazy",
        "ENABLE_HYBRID_MODE": os.getenv("ENABLE_HYBRID_MODE", "true"),
        "ENABLE_THREE_LLM_MODE": os.getenv("ENABLE_THREE_LLM_MODE", "true"),
        "ENABLE_CENTRAL_MODE": os.getenv("ENABLE_CENTRAL_MODE", "true"),
        "MAX_AGENT_STEPS": "6",
        "MAX_WIKIPEDIA_SEARCHES": "2",
        "MAX_WEB_SEARCHES": "3",
        "MAX_PAGE_FETCHES": "5",
        "WEB_SEARCH_PROVIDER": os.getenv("WEB_SEARCH_PROVIDER", "local-only"),
        "DEFAULT_INFERENCE_MODE": "central",
        "CHAT_DATABASE_PATH": "/data/chat.sqlite3",
        "HF_HOME": "/hf-cache",
        "HF_HUB_CACHE": "/hf-cache/hub",
        "CENTRAL_AGENT_HF_CACHE_DIR": "/hf-cache/hub",
        "CENTRAL_AGENT_LOCAL_FILES_ONLY": os.getenv("CENTRAL_AGENT_LOCAL_FILES_ONLY", "false"),
        "CORS_ORIGINS": "http://localhost:5173,http://127.0.0.1:5173",
    }
)


@app.function(
    image=image,
    gpu="A100",
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
    secrets=runtime_secrets,
)
@modal.asgi_app()
def fastapi_app():
    from app.main import app as fastapi_application

    return fastapi_application
