from pathlib import Path
from tempfile import TemporaryDirectory
from argon2 import PasswordHasher
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

with TemporaryDirectory(prefix="su-viet-eval-browser-") as directory:
    os.environ.update(
        APP_MODE="api-only",
        APP_ENV="development",
        AUTH_ENABLED="true",
        EVAL_STUDIO_ENABLED="true",
        OPENAI_ENABLED="false",
        LOCAL_INFERENCE_ENABLED="false",
        MCP_ENABLED="false",
        AUTH_PASSWORD_HASH=PasswordHasher().hash("test-password"),
        AUTH_SESSION_SECRET="test-session-secret-at-least-32-characters",
        CHAT_DATABASE_PATH=str(Path(directory) / "eval.sqlite3"),
        FRONTEND_DIST_DIR=str(Path.cwd() / "frontend" / "dist"),
    )
    import uvicorn
    from app.main import app

    # A loopback-only testing adapter. Production has no HTTP-selectable fake
    # provider and no environment switch that enables this adapter.
    class OfflineExecutor:
        def freeze(self, experiment, options):
            import hashlib
            from app.eval_studio.store import canonical

            identity = {
                "experiment_id": experiment["id"],
                "snapshot_id": experiment["snapshot_id"],
                "provider": experiment.get("provider"),
                "model_id": experiment.get("model_id"),
                "chat_mode": experiment.get("chat_mode"),
                "prompts": options["prompt_overrides"],
                "tool_descriptions": options["tool_descriptions"],
                "runtime_provenance": {"environment": "offline-browser-fixture"},
            }
            digest = hashlib.sha256(canonical(identity).encode()).hexdigest()
            return {
                **identity,
                "configuration_hash": digest,
                "prompt_hash": digest,
                "tool_hash": "offline-fixture-tools",
            }

        def verify(self, configuration):
            pass

        async def execute(self, question, configuration, cancel):
            import asyncio
            import hashlib
            import time

            started = time.perf_counter()
            await asyncio.sleep(0.15)
            return {
                "answer": "Offline benchmark fixture answer [fixture-chunk]",
                "completion_status": "completed",
                "latency_ms": (time.perf_counter() - started) * 1000,
                "evidence": [
                    {
                        "chunk_id": "fixture-chunk",
                        "title": "Offline fixture source",
                        "excerpt": "Captured offline benchmark evidence.",
                        "selected": True,
                        "cited": True,
                    }
                ],
                "trace": {
                    "retrieval_backend": "faiss",
                    "final_context_chunk_ids": ["fixture-chunk"],
                    "cited_chunk_ids": ["fixture-chunk"],
                    "tools": [{"name": "search_history", "status": "completed"}],
                },
                "trace_captured": True,
                "diagnostics": {
                    "tool_calls": [
                        {
                            "name": "search_history",
                            "status": "completed",
                            "arguments_valid": True,
                            "argument_fingerprint": hashlib.sha256(question.encode()).hexdigest(),
                        }
                    ],
                    "planning": {"exit_reason": "evidence_sufficient"},
                    "capture": {"tool_trace": True, "retrieved_evidence": True, "final_context": True},
                },
            }

    app.state.eval_benchmark_executor = OfflineExecutor()

    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
