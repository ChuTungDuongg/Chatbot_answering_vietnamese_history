"""Run read-only Qdrant checks using Modal Secret; no credentials printed locally."""

import json
import os

import modal

app = modal.App("vn-history-qdrant-smoke")
volume = modal.Volume.from_name("vn-history-artifacts", create_if_missing=False)
secret = modal.Secret.from_name(os.getenv("MODAL_QDRANT_SECRET_NAME", "vn-history-qdrant"),
                               required_keys=["QDRANT_URL", "QDRANT_API_KEY", "QDRANT_COLLECTION"])
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("qdrant-client==1.19.1", "numpy==2.4.6", "faiss-cpu==1.15.0",
                      "pydantic==2.13.4", "pydantic-settings==2.14.2")
         .env({"ARTIFACT_ROOT": "/artifacts/corpus_v1", "CORPUS_PATH": "/artifacts/corpus_v1/chunks.jsonl",
               "RETRIEVAL_ROOT": "/artifacts/corpus_v1/retrieval"})
         .add_local_python_source("app", "scripts"))


@app.function(image=image, volumes={"/artifacts": volume}, secrets=[secret],
              cpu=2, memory=8192, timeout=600)
def qdrant_smoke() -> str:
    from app.config import settings
    from scripts.smoke_test_qdrant import check_qdrant
    result = check_qdrant(settings)
    print(f"PASS Qdrant: host={result['hostname']}; collection={result['collection']}; points={result['count']}")
    return json.dumps(result, indent=2)


@app.function(image=image, secrets=[secret], cpu=1, memory=1024, timeout=120)
def connection_probe() -> str:
    from urllib.parse import urlparse
    import socket
    from qdrant_client import QdrantClient
    host = urlparse(os.environ["QDRANT_URL"]).hostname
    collection = os.environ["QDRANT_COLLECTION"]
    try:
        socket.getaddrinfo(host, 443)
        client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"], timeout=30)
        info = client.get_collection(collection)
        count = client.count(collection, exact=True).count
        vectors = info.config.params.vectors
        result = {"hostname": host, "collection": collection, "count": count,
                  "vectors": {name: {"dimension": item.size, "distance": str(item.distance)} for name, item in vectors.items()},
                  "status": str(info.status), "network_authentication": "PASS"}
        client.close()
    except Exception as exc:
        raise RuntimeError(f"Qdrant connection probe failed ({type(exc).__name__})") from None
    print(json.dumps(result, indent=2))
    return json.dumps(result, indent=2)
