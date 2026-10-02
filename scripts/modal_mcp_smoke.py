"""Dev-only SDK stdio + real Central model smoke on existing V1 Modal artifacts."""

import json
import os
from pathlib import Path
import tempfile

from modal_app import app, artifacts, hf_cache, image, runtime_secrets

image = image.add_local_python_source("modal_app", "scripts")


@app.function(image=image, gpu="A100", cpu=4, memory=32768, timeout=1200, startup_timeout=900,
              volumes={"/artifacts": artifacts, "/hf-cache": hf_cache}, secrets=runtime_secrets)
def mcp_smoke() -> str:
    from scripts import mcp_demo_server
    # Demo is intentionally confined to this development function/private temp config.
    root = Path(tempfile.mkdtemp())
    config = root / "mcp.json"
    config.write_text(json.dumps({"servers": {"demo": {"enabled": True, "required": True,
        "label": "Demo Research", "transport": "stdio", "command": "python",
        "cwd": str(Path(mcp_demo_server.__file__).resolve().parent.parent),
        "args": ["-m", "scripts.mcp_demo_server"], "allowed_tools": ["lookup"],
        "connect_timeout_seconds": 30}}}), encoding="utf-8")
    os.environ.update(MCP_ENABLED="true", MCP_CONFIG_PATH=str(config),
                      CHAT_DATABASE_PATH=str(root / "chat.sqlite3"))
    from fastapi.testclient import TestClient
    from app.main import app as api
    from app.mcp.adapters import tool_name
    with TestClient(api) as client:
        ready = client.get("/ready").json()
        if not ready["mcp"]["servers"][0]["available"]:
            raise RuntimeError("Demo SDK discovery failed")
        manager = api.state.mcp_manager
        connection = manager.connections["demo"].client
        service = api.state.rag_service
        resources = tuple(id(getattr(service, name)) for name in ("chunks", "bm25", "embedder", "reranker", "faiss_index"))
        headers = {"X-Client-ID": "mcp-dev-smoke"}
        responses = {}
        for lane, steering, question in (
            ("mcp", {"mcp_enabled": True, "allowed_mcp_servers": ["demo"],
                "allowed_tools": [tool_name("demo", "lookup")], "mcp_failure_policy": "fail"},
                "Dùng công cụ lookup để tra cứu bằng chứng về Bạch Đằng năm 938; sau đó tóm tắt nguồn và giới hạn của nguồn."),
            ("history", None, "Ngô Quyền giành chiến thắng trên sông Bạch Đằng vào năm nào?"),
        ):
            conversation = client.post("/api/v1/conversations", headers=headers, json={})
            conversation.raise_for_status()
            cid = conversation.json()["id"]
            response = client.post("/api/v1/chat", headers=headers, json={
                "conversation_id": cid, "question": question, "mode": "central",
                "retrieval_backend": "qdrant", "debug": True, "final_k": 3,
                **({"steering": steering} if steering else {})})
            response.raise_for_status()
            result = response.json()
            if not result["answer"]: raise RuntimeError("Central returned an empty response")
            if lane == "mcp":
                if not any(record.get("provider") == "mcp" and record.get("success") for record in result["debug"]["tool_trace"]):
                    raise RuntimeError("Real Central planner did not execute permitted MCP lookup")
                if not any(source["source_kind"] == "mcp" for source in result["sources"]):
                    raise RuntimeError("MCP provenance missing")
            else:
                if result["debug"].get("mcp") or not result["sources"]:
                    raise RuntimeError("Default Central must use history without MCP exposure")
            responses[lane] = result
            client.delete(f"/api/v1/conversations/{cid}", headers=headers).raise_for_status()
        assert manager.connections["demo"].client is connection
        assert resources == tuple(id(getattr(service, name)) for name in ("chunks", "bm25", "embedder", "reranker", "faiss_index"))
        print("PASS: real Qwen3-8B called configured SDK MCP; default Central retained V1 Qdrant evidence; resources reused")
        return json.dumps({"ready": ready, "responses": responses, "connection_reused": True,
                           "resources_reused": True}, ensure_ascii=False, indent=2)
