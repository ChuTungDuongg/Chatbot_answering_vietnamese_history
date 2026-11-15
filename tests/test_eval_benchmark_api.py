import json
import time

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.config import settings
from app.eval_studio.store import EvalStore
from app.eval_studio.v11 import StudioV11


def test_benchmark_api_requires_owner_flag_and_explicit_live_confirmation(tmp_path, monkeypatch):
    from app.eval_studio.benchmark_api import router

    store = EvalStore(tmp_path / "api.sqlite")
    q = store.questions()[0]
    store.save_question({**q, "question": "Question for an offline fake model"})
    store.snapshot("api")
    studio = StudioV11(store)
    exp = studio.create_experiment({"name": "API", "snapshot_id": "api"})

    class Fake:
        def __init__(self):
            self.calls = []

        async def execute(self, question, configuration, cancel):
            self.calls.append(question)
            return {"answer": "Offline fixture answer", "completion_status": "completed"}

        def freeze(self, experiment, options):
            return {
                "experiment_id": experiment["id"],
                "snapshot_id": experiment["snapshot_id"],
                "configuration_hash": "fixture",
                "prompt_hash": "prompt",
                "tool_hash": "tools",
            }

        def verify(self, configuration):
            pass

    app = FastAPI()
    app.state.eval_store, app.state.eval_v11 = store, studio
    app.state.eval_benchmark_executor = Fake()

    @app.middleware("http")
    async def owner(request: Request, call_next):
        request.state.owner_authenticated = request.headers.get("x-owner") == "yes"
        return await call_next(request)

    app.include_router(router)
    monkeypatch.setattr(settings, "eval_studio_enabled", True)
    with TestClient(app) as client:
        assert client.get("/api/v1/eval/benchmarks").status_code == 401
        headers = {"x-owner": "yes"}
        created = client.post(
            "/api/v1/eval/benchmarks",
            headers=headers,
            json={"experiment_id": exp["id"], "question_ids": [q["id"]]},
        )
        assert created.status_code == 201, created.text
        job_id = created.json()["id"]
        assert not app.state.eval_benchmark_executor.calls
        path = f"/api/v1/eval/benchmarks/{job_id}/start"
        assert client.post(path, headers=headers, json={}).status_code == 422
        assert client.post(path, headers=headers, json={"confirm_live": True}).status_code == 200
        for _ in range(100):
            detail = client.get(f"/api/v1/eval/benchmarks/{job_id}", headers=headers).json()
            if detail["status"] == "completed":
                break
            time.sleep(0.01)
        assert detail["counts"]["completed"] == 1
        run_id = detail["items"][0]["run_id"]
        assert client.get(f"/api/v1/eval/runs/{run_id}/measurements", headers=headers).json()["metrics"]
        monkeypatch.setattr(settings, "eval_studio_enabled", False)
        assert client.get("/api/v1/eval/benchmarks", headers=headers).status_code == 404
