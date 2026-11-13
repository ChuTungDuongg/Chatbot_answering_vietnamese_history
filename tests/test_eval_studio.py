import pytest
from app.eval_studio.store import EvalStore, ConflictError
from app.eval_studio.models import Question, Review


def test_seed_persistence_and_quota(tmp_path):
    path = tmp_path / "eval.sqlite"
    store = EvalStore(path)
    questions = store.questions()
    assert len(questions) == 40
    assert sum(q["smoke"] for q in questions) == 20
    assert list(store.validation()["category_counts"].values()) == [6, 12, 8, 8, 4, 2]
    assert len(EvalStore(path).questions()) == 40


def test_approval_and_optimistic_concurrency(tmp_path):
    store = EvalStore(tmp_path / "eval.sqlite")
    q = store.questions()[0]
    with pytest.raises(ValueError):
        store.save_question({**q, "status": "approved"})
    saved = store.save_question({**q, "question": "Ai?"})
    with pytest.raises(ConflictError):
        store.save_question({**q, "question": "Ghi đè"})
    assert store.get_question(q["id"])["question"] == "Ai?"
    assert saved["version"] == q["version"] + 1


def test_validation_and_snapshot(tmp_path):
    store = EvalStore(tmp_path / "eval.sqlite")
    first, second = store.questions()[:2]
    store.save_question({**first, "question": "Câu hỏi giống nhau?"})
    store.save_question({**second, "question": "Câu hỏi giống nhau?"})
    assert store.validation()["duplicates"]
    snapshot = store.snapshot("1.0.0")
    q = store.get_question(first["id"])
    store.save_question({**q, "question": "Đã thay đổi"})
    assert store.get_snapshot("1.0.0") == snapshot
    with pytest.raises(ConflictError):
        store.snapshot("1.0.0")


def test_review_score_and_roundtrip(tmp_path):
    store = EvalStore(tmp_path / "one.sqlite")
    q = store.questions()[0]
    response = store.create_response({"question_id": q["id"], "answer": "Một câu trả lời"})
    review = store.create_review(
        {
            "response_id": response["id"],
            "scores": {"accuracy": 5, "coverage": 4, "reasoning": 3, "evidence": 2, "clarity": 1},
            "decision": "needs_review",
        }
    )
    assert review["weighted_score"] == 68
    exported = store.export()
    other = EvalStore(tmp_path / "two.sqlite")
    other.import_questions(exported["cases"], overwrite=True)

    def content(cases):
        return [{k: v for k, v in q.items() if k not in ("version", "updated_at")} for q in cases]

    assert content(other.export()["cases"]) == content(exported["cases"])


def test_api_requires_feature_and_owner(tmp_path, monkeypatch):
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    from app.eval_studio.api import router
    from app.config import settings

    app = FastAPI()
    app.state.eval_store = EvalStore(tmp_path / "api.sqlite")

    @app.middleware("http")
    async def owner(request: Request, call_next):
        request.state.owner_authenticated = request.headers.get("x-test-owner") == "yes"
        return await call_next(request)

    app.include_router(router)
    client = TestClient(app)
    monkeypatch.setattr(settings, "eval_studio_enabled", False)
    assert client.get("/api/v1/eval/questions", headers={"x-test-owner": "yes"}).status_code == 404
    monkeypatch.setattr(settings, "eval_studio_enabled", True)
    assert client.get("/api/v1/eval/questions").status_code == 401
    assert len(client.get("/api/v1/eval/questions", headers={"x-test-owner": "yes"}).json()) == 40


def test_approval_ranking_uncertainty_and_export_filters(tmp_path):
    store = EvalStore(tmp_path / "rules.sqlite")
    q = store.questions()[18]
    complete = {
        **q,
        "question": "So sánh?",
        "reference_answer": "Tham chiếu",
        "expected_facts": ["Sự kiện"],
        "must_cover_points": ["Luận điểm"],
        "reference_sources": ["Nguồn"],
        "status": "approved",
    }
    with pytest.raises(ValueError):
        store.save_question(complete)
    approved = store.save_question({**complete, "ranking_criteria": ["Tiêu chí"]})
    assert approved["status"] == "approved"
    q = store.questions()[34]
    with pytest.raises(ValueError):
        store.save_question(
            {
                **complete,
                **q,
                "question": "Tranh luận?",
                "reference_answer": "Đáp án",
                "expected_facts": ["Sự kiện"],
                "must_cover_points": ["Luận điểm"],
                "reference_sources": ["Nguồn"],
                "status": "approved",
            }
        )
    store.create_question()
    assert len(store.export(base_only=True)["cases"]) == 40
    assert len(store.export(smoke=True)["cases"]) == 20
    assert [item["id"] for item in store.export(approved=True)["cases"]] == [approved["id"]]


def test_import_is_atomic_and_preserves_linked_reviews(tmp_path):
    store = EvalStore(tmp_path / "atomic.sqlite")
    first, second = store.questions()[:2]
    response = store.create_response({"question_id": first["id"], "answer": "Output"})
    with pytest.raises(ConflictError):
        store.import_questions([{**first, "id": "extra", "question": "New"}, second])
    with pytest.raises(KeyError):
        store.get_question("extra")
    store.import_questions([{**first, "question": "Nhập mới"}], overwrite=True)
    assert store.responses()[0]["id"] == response["id"]
    assert store.get_question(first["id"])["version"] == first["version"] + 1


def test_review_scores_are_validated(tmp_path):
    from pydantic import ValidationError

    store = EvalStore(tmp_path / "scores.sqlite")
    response = store.create_response({"question_id": store.questions()[0]["id"], "answer": "Output"})
    with pytest.raises(ValidationError):
        store.create_review(
            {
                "response_id": response["id"],
                "scores": {"accuracy": 6, "coverage": 4, "reasoning": 3, "evidence": 2, "clarity": 1},
                "decision": "accept",
            }
        )


def test_api_export_import_and_snapshots(tmp_path, monkeypatch):
    import json
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    from app.eval_studio.api import router
    from app.config import settings

    app = FastAPI()
    app.state.eval_store = EvalStore(tmp_path / "api-roundtrip.sqlite")

    @app.middleware("http")
    async def owner(request: Request, call_next):
        request.state.owner_authenticated = True
        return await call_next(request)

    app.include_router(router)
    monkeypatch.setattr(settings, "eval_studio_enabled", True)
    client = TestClient(app)
    document = client.get("/api/v1/eval/export?smoke=true").json()
    lines = client.get("/api/v1/eval/export?format=jsonl&smoke=true").text.splitlines()
    assert [json.loads(line) for line in lines] == document["cases"]
    assert document["manifest"]["case_count"] == 20
    assert client.post("/api/v1/eval/import", json={"cases": document["cases"]}).status_code == 409
    assert client.post(
        "/api/v1/eval/import", json={"cases": document["cases"], "overwrite": True}
    ).json() == {"imported": 20}
    created = client.post("/api/v1/eval/snapshots", json={"version": "1.0.0"})
    assert created.status_code == 201
    assert client.get("/api/v1/eval/snapshots/1.0.0").json() == created.json()
    assert client.get("/api/v1/eval/snapshots").json()[0]["version"] == "1.0.0"
    assert client.post("/api/v1/eval/snapshots", json={"version": "1.0.0"}).status_code == 409


def test_api_question_filters(tmp_path, monkeypatch):
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    from app.eval_studio.api import router
    from app.config import settings

    app = FastAPI()
    app.state.eval_store = EvalStore(tmp_path / "filter.sqlite")

    @app.middleware("http")
    async def owner(request: Request, call_next):
        request.state.owner_authenticated = True
        return await call_next(request)

    app.include_router(router)
    monkeypatch.setattr(settings, "eval_studio_enabled", True)
    client = TestClient(app)
    response = client.get("/api/v1/eval/questions", params={"category": "Factual QA", "smoke": "true"})
    assert len(response.json()) == 3
    assert all(q["category"] == "Factual QA" and q["smoke"] for q in response.json())
    assert len(client.get("/api/v1/eval/questions", params={"search": "HIST-EVAL-001"}).json()) == 1
