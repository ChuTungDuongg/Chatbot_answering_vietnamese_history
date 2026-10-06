import os
from pathlib import Path

import modal
from dotenv import load_dotenv


repo_root = Path(__file__).resolve().parent
# Read the repo's local settings before constructing the Modal image/client.
# Explicit shell/deployment environment variables keep precedence over .env.
load_dotenv(repo_root / ".env", override=False, encoding="utf-8")

app = modal.App("vn-history-rag-api")
adapter_path = "/artifacts/models/qwen3_4b_sft_v1/run_best_b4_ga4_e2/adapter"
dense_backend = os.getenv("RETRIEVAL_DENSE_BACKEND", "faiss")
available_backends = os.getenv("RETRIEVAL_AVAILABLE_BACKENDS", dense_backend)
if dense_backend not in {"faiss", "qdrant"}:
    raise ValueError("RETRIEVAL_DENSE_BACKEND must be faiss or qdrant")
qdrant_secret_name = os.getenv("MODAL_QDRANT_SECRET_NAME", "vn-history-qdrant").strip()

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
mcp_enabled = os.getenv("MCP_ENABLED", "false").lower() == "true"
mcp_secret_name = os.getenv("MODAL_MCP_SECRET_NAME", "").strip()
runtime_secrets = (
    [modal.Secret.from_name(web_search_secret_name)]
    if web_search_secret_name
    else []
)
if "qdrant" in [name.strip() for name in available_backends.split(",")]:
    if not qdrant_secret_name:
        raise ValueError("Qdrant requires MODAL_QDRANT_SECRET_NAME")
    runtime_secrets.append(modal.Secret.from_name(
        qdrant_secret_name, required_keys=["QDRANT_URL", "QDRANT_API_KEY", "QDRANT_COLLECTION"]))
if mcp_enabled and mcp_secret_name:
    runtime_secrets.append(modal.Secret.from_name(mcp_secret_name))

image = modal.Image.from_dockerfile(
    str(repo_root / "Dockerfile"),
    context_dir=str(repo_root),
).env(
    {
        "APP_ENV": "production",
        "APP_MODE": "full",
        "DEVICE": "cuda",
        "DTYPE": "bfloat16",
        "ARTIFACT_ROOT": "/artifacts/corpus_v1",
        "CORPUS_PATH": "/artifacts/corpus_v1/chunks.jsonl",
        "RETRIEVAL_ROOT": "/artifacts/corpus_v1/retrieval",
        "INFERENCE_CONFIG_PATH": "/artifacts/corpus_v1/runtime/inference_config.json",
        "RUNTIME_MANIFEST_PATH": "/artifacts/corpus_v1/runtime/manifest.json",
        "RETRIEVAL_DENSE_BACKEND": dense_backend,
        "RETRIEVAL_AVAILABLE_BACKENDS": available_backends,
        "HYBRID_MODEL_ID": "Qwen/Qwen3-4B-Instruct-2507",
        "CENTRAL_MODEL_ID": "Qwen/Qwen3-8B",
        "MODEL_VARIANT": os.getenv("MODEL_VARIANT", "vanilla"),
        "MODEL_ADAPTER_PATH": adapter_path,
        "DO_SAMPLE": "false",
        "ENABLE_THINKING": "false",
        "CENTRAL_MAX_ACTION_ROUNDS": "2",
        "CENTRAL_ACTION_MAX_NEW_TOKENS": "256",
        "CENTRAL_FINAL_MAX_NEW_TOKENS": "1536",
        # Omit the default so container Settings is the single source of truth.
        **({"HYBRID_MAX_NEW_TOKENS": os.environ["HYBRID_MAX_NEW_TOKENS"]}
           if os.getenv("HYBRID_MAX_NEW_TOKENS") else {}),
        "RUNTIME_LOADING_STRATEGY": "lazy",
        "ENABLE_HYBRID_MODE": os.getenv("ENABLE_HYBRID_MODE", "true"),
        "ENABLE_CENTRAL_MODE": os.getenv("ENABLE_CENTRAL_MODE", "true"),
        "CENTRAL_ENABLE_DOCUMENTS": "true",
        "CENTRAL_ENABLE_WIKIPEDIA": "true",
        "CENTRAL_ENABLE_WEB": "false",
        "MCP_ENABLED": str(mcp_enabled).lower(),
        "MCP_CONFIG_PATH": "/etc/vn-history/mcp_servers.json",
        "MCP_MAX_TOOLS_PER_REQUEST": os.getenv("MCP_MAX_TOOLS_PER_REQUEST", "8"),
        "MCP_SCHEMA_BUDGET_BYTES": os.getenv("MCP_SCHEMA_BUDGET_BYTES", "16384"),
        "WEB_SEARCH_PROVIDER": os.getenv("WEB_SEARCH_PROVIDER", "local-only"),
        "DEFAULT_INFERENCE_MODE": "hybrid",
        "CHAT_DATABASE_PATH": "/data/chat.sqlite3",
        "HF_HOME": "/hf-cache",
        "HF_HUB_CACHE": "/hf-cache/hub",
        "MODEL_CACHE_DIR": "/hf-cache/hub",
        "MODEL_LOCAL_FILES_ONLY": os.getenv("MODEL_LOCAL_FILES_ONLY", "false"),
        "CORS_ORIGINS": "http://localhost:5173,http://127.0.0.1:5173",
    }
)


if mcp_enabled:
    mcp_config_path = Path(os.getenv("MCP_CONFIG_PATH", "config/mcp_servers.local.json"))
    if not mcp_config_path.is_absolute():
        mcp_config_path = repo_root / mcp_config_path
    if not mcp_config_path.is_file():
        raise ValueError("MCP_ENABLED=true requires an existing MCP_CONFIG_PATH")
    image = image.add_local_file(str(mcp_config_path), "/etc/vn-history/mcp_servers.json", copy=True)


@app.function(
    image=image,
    gpu="A100",
    cpu=4.0,
    memory=32768,
    timeout=600,
    startup_timeout=900,
    min_containers=0,
    max_containers=1,
    scaledown_window=int(os.getenv("CENTRAL_SCALEDOWN_WINDOW_SECONDS", "120")),
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


@app.function(
    image=image, gpu="A100", cpu=4.0, memory=32768, timeout=1800,
    startup_timeout=900,
    volumes={"/artifacts": artifacts, "/hf-cache": hf_cache},
    secrets=runtime_secrets,
)

def runtime_smoke(generate: bool = False, central: bool = False,
                  question: str = "Chiến thắng Bạch Đằng năm 938 có ý nghĩa gì?") -> str:
    """Exercise real V1 retrieval and optionally Hybrid/Central generation once."""
    import json
    import logging
    import tempfile
    logging.basicConfig(level=logging.ERROR, force=True)

    # The smoke process uses a temporary chat store, leaving user conversations alone.
    os.environ["APP_MODE"] = "full" if generate else "retrieval-only"
    os.environ["CHAT_DATABASE_PATH"] = str(Path(tempfile.mkdtemp()) / "smoke.sqlite3")
    from fastapi.testclient import TestClient
    from app.config import settings
    from app.main import app as api

    if str(settings.corpus_path) != "/artifacts/corpus_v1/chunks.jsonl":
        raise RuntimeError("Modal smoke resolved a non-V1 corpus")
    with TestClient(api) as client:
        ready = client.get("/ready").json()
        if not ready.get("ready") or ready.get("corpus_chunks") != 624288:
            raise RuntimeError(f"V1 readiness failed: {ready}")
        response = client.post("/api/v1/retrieve", json={"question": question, "final_k": 3})
        response.raise_for_status()
        retrieval = response.json()
        if retrieval.get("is_ood") or not retrieval.get("final_context"):
            raise RuntimeError("Historical smoke question returned no V1 evidence")
        result = {"corpus_path": str(settings.corpus_path), "ready": ready,
                  "question": question, "retrieval": retrieval, "generation": {}}
        if generate:
            headers = {"X-Client-ID": "modal-restore-smoke"}
            for mode in (["hybrid", "central"] if central else ["hybrid"]):
                conversation = client.post("/api/v1/conversations", headers=headers, json={})
                conversation.raise_for_status()
                response = client.post("/api/v1/chat", headers=headers, json={
                    "conversation_id": conversation.json()["id"], "question": question,
                    "mode": mode, "final_k": 3})
                if response.is_error:
                    raise RuntimeError(f"{mode} HTTP {response.status_code}: {response.text}")
                response.raise_for_status()
                answer = response.json()
                if not answer.get("answer"):
                    raise RuntimeError(f"{mode} returned an empty answer")
                result["generation"][mode] = answer
        print(f"PASS V1 runtime: corpus={settings.corpus_path}; backend={settings.retrieval_dense_backend}; chunks=624288; generation={list(result['generation'])}")
        return json.dumps(result, ensure_ascii=False, indent=2)
