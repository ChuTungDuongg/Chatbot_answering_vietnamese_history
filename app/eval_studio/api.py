import csv
import io
import json
import os
import sqlite3
import threading
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.routing import APIRoute
from fastapi.responses import Response
from pydantic import BaseModel, Field, ValidationError
from app.config import settings
from .models import Question, ResponseInput, Review
from .store import ConflictError, EvalStore


class BoundedRoute(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()

        async def handler(request):
            if request.url.path.endswith(("/batch/preview", "/batch/apply", "/import")):
                limit = 2_100_000
                try:
                    declared = int(request.headers.get("content-length", "0"))
                except ValueError:
                    raise HTTPException(400, "Invalid Content-Length")
                if declared > limit:
                    raise HTTPException(413, "Import request too large")
                chunks = []
                size = 0
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, "Import request too large")
                    chunks.append(chunk)
                request._body = b"".join(chunks)
            return await original(request)

        return handler


router = APIRouter(prefix="/api/v1/eval", dependencies=[], route_class=BoundedRoute)
_lock = threading.RLock()


def store_for(request: Request):
    if not settings.eval_studio_enabled:
        raise HTTPException(404, "Eval Studio is disabled")
    if not getattr(request.state, "owner_authenticated", False):
        raise HTTPException(401, "Owner session required")
    if not getattr(request.app.state, "eval_store", None):
        with _lock:
            if not getattr(request.app.state, "eval_store", None):
                request.app.state.eval_store = EvalStore(settings.chat_database_path)
                persist()
    return request.app.state.eval_store


def persist():
    if os.getenv("EVAL_STUDIO_MODAL_VOLUME"):
        import modal

        modal.Volume.from_name(os.environ["EVAL_STUDIO_MODAL_VOLUME"]).commit()


def run(fn, *args, mutation=False, **kwargs):
    try:
        if mutation:
            with _lock:
                result = fn(*args, **kwargs)
                persist()
                return result
        return fn(*args, **kwargs)
    except ConflictError as e:
        raise HTTPException(409, str(e)) from e
    except KeyError as e:
        raise HTTPException(404, "Not found") from e
    except sqlite3.IntegrityError as e:
        raise HTTPException(409, "This record has linked responses or reviews") from e
    except ValidationError as e:
        details = [
            {"field": ".".join(map(str, error["loc"])), "message": error["msg"]}
            for error in e.errors(include_input=False, include_url=False)
        ]
        raise HTTPException(422, details) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@router.get("/questions")
def questions(
    search: str = "",
    category: str | None = None,
    difficulty: str | None = None,
    status: str | None = None,
    smoke: bool | None = None,
    sort: str = "id",
    store: EvalStore = Depends(store_for),
):
    items = [
        q
        for q in store.questions()
        if (not category or q["category"] == category)
        and (not difficulty or q["difficulty"] == difficulty)
        and (not status or q["status"] == status)
        and (smoke is None or q["smoke"] == smoke)
        and search.casefold() in (q["id"] + " " + q["question"] + " " + " ".join(q["tags"])).casefold()
    ]
    if sort not in ("id", "updated_at", "status"):
        raise HTTPException(422, "Invalid sort")
    return sorted(items, key=lambda q: (q[sort], q["id"]), reverse=sort == "updated_at")


@router.post("/questions", status_code=201)
def create_question(data: dict | None = None, store: EvalStore = Depends(store_for)):
    return run(store.create_question, data, mutation=True)


@router.get("/questions/{question_id}")
def get_question(question_id: str, store: EvalStore = Depends(store_for)):
    return run(store.get_question, question_id)


@router.patch("/questions/{question_id}")
def save_question(question_id: str, question: Question, store: EvalStore = Depends(store_for)):
    if question.id != question_id:
        raise HTTPException(422, "ID mismatch")
    return run(store.save_question, question.model_dump(), mutation=True)


@router.delete("/questions/{question_id}", status_code=204)
def delete_question(question_id: str, store: EvalStore = Depends(store_for)):
    run(store.delete_question, question_id, mutation=True)
    return Response(status_code=204)


@router.get("/validation")
def validation(store: EvalStore = Depends(store_for)):
    return store.validation()


@router.get("/responses")
def responses(question_id: str | None = None, store: EvalStore = Depends(store_for)):
    return store.responses(question_id)


@router.post("/responses", status_code=201)
def create_response(response: ResponseInput, store: EvalStore = Depends(store_for)):
    return run(store.create_response, response.model_dump(), mutation=True)


@router.get("/reviews")
def reviews(store: EvalStore = Depends(store_for)):
    return store.reviews()


@router.post("/reviews", status_code=201)
def create_review(review: Review, store: EvalStore = Depends(store_for)):
    return run(store.create_review, review.model_dump(), mutation=True)


@router.get("/export")
def export(
    format: str = "json",
    smoke: bool = False,
    approved: bool = False,
    base_only: bool = False,
    store: EvalStore = Depends(store_for),
):
    data = store.export(smoke, approved, base_only)
    if format == "json":
        return data
    if format == "jsonl":
        body = "\n".join(json.dumps(q, ensure_ascii=False) for q in data["cases"]) + (
            "\n" if data["cases"] else ""
        )
        return Response(
            body,
            media_type="application/x-ndjson",
            headers={"Content-Disposition": "attachment; filename=history-eval.jsonl"},
        )
    if format == "csv":
        buf = io.StringIO(newline="")
        fields = [
            "id",
            "dataset_id",
            "question",
            "category",
            "difficulty",
            "tags",
            "smoke",
            "status",
            "created_at",
            "updated_at",
            "version",
        ]
        writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for q in data["cases"]:
            writer.writerow({**q, "tags": ", ".join(q["tags"])})
        return Response(
            "\ufeff" + buf.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=history-eval.csv"},
        )
    raise HTTPException(400, "Unsupported format")


@router.get("/export/reviews")
def export_reviews(store: EvalStore = Depends(store_for)):
    return {
        "responses": store.responses(),
        "reviews": store.reviews(),
        "export_timestamp": __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .isoformat(),
    }


class ImportPayload(BaseModel):
    cases: list[Question] = Field(max_length=1000)
    overwrite: bool = False


@router.post("/import")
def import_questions(payload: ImportPayload, store: EvalStore = Depends(store_for)):
    return {
        "imported": run(
            store.import_questions, [q.model_dump() for q in payload.cases], payload.overwrite, mutation=True
        )
    }


class VersionPayload(BaseModel):
    version: str


@router.get("/snapshots")
def list_snapshots(store: EvalStore = Depends(store_for)):
    return store.list_snapshots()


@router.post("/snapshots", status_code=201)
def snapshot(payload: VersionPayload, store: EvalStore = Depends(store_for)):
    return run(store.snapshot, payload.version, mutation=True)


@router.get("/snapshots/{version}")
def get_snapshot(version: str, store: EvalStore = Depends(store_for)):
    return run(store.get_snapshot, version)


# v1.1: additive experiment and analytics routes; all share the owner-only dependency.
from pathlib import Path
from tempfile import TemporaryDirectory
from .v11 import StudioV11, ExperimentInput, RunInput, BlindSubmission
from .batch import preview_batch, apply_batch
from .backup import create_backup


def v11_for(request: Request, store: EvalStore = Depends(store_for)):
    if not getattr(request.app.state, "eval_v11", None):
        with _lock:
            if not getattr(request.app.state, "eval_v11", None):
                request.app.state.eval_v11 = StudioV11(store)
                persist()
    return request.app.state.eval_v11


@router.get("/experiments")
def list_experiments(studio: StudioV11 = Depends(v11_for)):
    return studio.experiments()


@router.post("/experiments", status_code=201)
def create_experiment(payload: ExperimentInput, studio: StudioV11 = Depends(v11_for)):
    return run(studio.create_experiment, payload.model_dump(), mutation=True)


@router.get("/runs")
def list_runs(experiment_id: str | None = None, studio: StudioV11 = Depends(v11_for)):
    return studio.runs(experiment_id)


@router.post("/runs", status_code=201)
def create_run(payload: RunInput, studio: StudioV11 = Depends(v11_for)):
    return run(studio.create_run, payload.model_dump(), mutation=True)


@router.get("/runs/{run_id}/evidence")
def get_run_evidence(run_id: str, studio: StudioV11 = Depends(v11_for)):
    return run(studio.evidence, run_id)


class CompareInput(BaseModel):
    experiment_ids: list[str] = Field(min_length=2, max_length=20)
    category: str | None = None
    difficulty: str | None = None
    smoke: bool | None = None
    diagnosis: str | None = None


@router.post("/compare")
def compare(payload: CompareInput, studio: StudioV11 = Depends(v11_for)):
    return run(
        studio.compare,
        payload.experiment_ids,
        payload.category,
        payload.difficulty,
        payload.smoke,
        payload.diagnosis,
    )


class BlindInput(BaseModel):
    snapshot_id: str
    experiment_a: str
    experiment_b: str


@router.get("/blind")
def blind_sessions(studio: StudioV11 = Depends(v11_for)):
    return studio.blind_sessions()


@router.post("/blind", status_code=201)
def create_blind(payload: BlindInput, studio: StudioV11 = Depends(v11_for)):
    return run(
        studio.create_blind, payload.snapshot_id, payload.experiment_a, payload.experiment_b, mutation=True
    )


@router.get("/blind/{session_id}")
def get_blind(session_id: str, studio: StudioV11 = Depends(v11_for)):
    return run(studio.get_blind, session_id)


@router.post("/blind/{session_id}/{assignment_id}/review")
def blind_review(
    session_id: str, assignment_id: str, payload: BlindSubmission, studio: StudioV11 = Depends(v11_for)
):
    return run(studio.submit_blind, session_id, assignment_id, payload.model_dump(), mutation=True)


@router.get("/blind/{session_id}/{assignment_id}/reveal")
def blind_reveal(session_id: str, assignment_id: str, studio: StudioV11 = Depends(v11_for)):
    return run(studio.reveal_blind, session_id, assignment_id)


class BatchPayload(BaseModel):
    kind: str
    format: str
    content: str = Field(max_length=2_000_000)
    token: str = ""
    policy: str = "reject"


@router.post("/batch/preview")
def batch_preview(payload: BatchPayload, studio: StudioV11 = Depends(v11_for)):
    return run(preview_batch, studio, payload.kind, payload.format, payload.content)


@router.post("/batch/apply")
def batch_apply(payload: BatchPayload, studio: StudioV11 = Depends(v11_for)):
    return run(
        apply_batch,
        studio,
        payload.kind,
        payload.format,
        payload.content,
        payload.token,
        payload.policy,
        mutation=True,
    )


@router.get("/backup")
def download_backup(studio: StudioV11 = Depends(v11_for)):
    with TemporaryDirectory(prefix="eval-api-backup-") as directory:
        target = Path(directory) / "evaluation-backup.zip"
        run(create_backup, studio.store.path, target)
        data = target.read_bytes()
    return Response(
        data,
        media_type="application/zip",
        headers={
            "Content-Disposition": "attachment; filename=eval-studio-backup.zip",
            "Cache-Control": "no-store",
        },
    )


@router.patch("/experiments/{experiment_id}")
def update_experiment(experiment_id: str, payload: ExperimentInput, studio: StudioV11 = Depends(v11_for)):
    return run(studio.update_experiment, experiment_id, payload.model_dump(), mutation=True)


@router.delete("/experiments/{experiment_id}", status_code=204)
def delete_experiment(experiment_id: str, studio: StudioV11 = Depends(v11_for)):
    run(studio.delete_experiment, experiment_id, mutation=True)
    return Response(status_code=204)


@router.delete("/runs/{run_id}", status_code=204)
def delete_run(run_id: str, studio: StudioV11 = Depends(v11_for)):
    run(studio.delete_run, run_id, mutation=True)
    return Response(status_code=204)


@router.patch("/runs/{run_id}")
def update_run(run_id: str, payload: RunInput, studio: StudioV11 = Depends(v11_for)):
    return run(studio.update_run, run_id, payload.model_dump(), mutation=True)
