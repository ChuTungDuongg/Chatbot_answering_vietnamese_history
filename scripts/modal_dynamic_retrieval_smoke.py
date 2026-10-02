"""Read-only V1 paired latency checks using Modal Secrets and shared runtime."""

import json
import os
from pathlib import Path
import tempfile

from modal_app import app, image, artifacts, hf_cache, runtime_secrets

# The CLI entrypoint lives under scripts/; include its imported repo modules.
image = image.add_local_python_source("modal_app", "scripts")


@app.function(image=image, gpu="A100", cpu=4, memory=32768, timeout=1800, startup_timeout=900,
              volumes={"/artifacts": artifacts, "/hf-cache": hf_cache}, secrets=runtime_secrets)
def dynamic_smoke(repeats: int = 3, central: bool = True) -> str:
    from fastapi.testclient import TestClient
    os.environ["CHAT_DATABASE_PATH"] = str(Path(tempfile.mkdtemp()) / "dynamic-smoke.sqlite3")
    from app.main import app as api
    from scripts.benchmark_dynamic_retrieval import run
    with TestClient(api) as client:
        from app.config import settings
        service = api.state.rag_service
        resources = tuple(id(getattr(service, name)) for name in ("chunks", "bm25", "embedder", "reranker", "faiss_index"))
        if str(settings.corpus_path) != "/artifacts/corpus_v1/chunks.jsonl":
            raise RuntimeError("Smoke must use restored V1 corpus")
        result = {"hybrid": run(client, repeats=repeats)}
        if central:
            result["central"] = run(client, mode="central", repeats=1, warmup=1)
        if resources != tuple(id(getattr(service, name)) for name in ("chunks", "bm25", "embedder", "reranker", "faiss_index")):
            raise RuntimeError("Shared retrieval resources were replaced during requests")
        result["resources_reused"] = True
        print("PASS: V1 shared runtime; Hybrid/Central request backends; real progress; latency measurements")
        return json.dumps(result, ensure_ascii=False, indent=2)
