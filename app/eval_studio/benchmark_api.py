"""Owner-only benchmark controls and separate deterministic/judge artifacts."""

import asyncio
import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from .api import BoundedRoute, _lock, persist, run, v11_for
from .benchmark import BenchmarkManager, BenchmarkOptions, BenchmarkStore
from .judge import JudgeProfileInput, JudgeScoreInput, JudgeStore
from .store import ConflictError
from .v11 import StudioV11

router = APIRouter(prefix="/api/v1/eval", route_class=BoundedRoute)


def jobs_for(request: Request, studio: StudioV11 = Depends(v11_for)):
    with _lock:
        if not getattr(request.app.state, "eval_benchmarks", None):
            request.app.state.eval_benchmarks = BenchmarkStore(studio)
            persist()
    return request.app.state.eval_benchmarks


def judges_for(studio: StudioV11 = Depends(v11_for)):
    return JudgeStore(studio.store)


def executor_for(request):
    # Test fixtures may inject an in-process fake, never an HTTP-selectable provider.
    if getattr(request.app.state, "eval_benchmark_executor", None) is not None:
        return request.app.state.eval_benchmark_executor
    from .execution import ChatBenchmarkExecutor

    return ChatBenchmarkExecutor(request.app)


def manager_for(request, jobs):
    with _lock:
        if not getattr(request.app.state, "eval_benchmark_manager", None):
            request.app.state.eval_benchmark_manager = BenchmarkManager(jobs, executor_for(request), persist)
        return request.app.state.eval_benchmark_manager


def frozen_for(request, experiment, options):
    options = {
        k: options[k]
        for k in ("prompt_overrides", "tool_descriptions", "mcp_enabled", "allowed_mcp_servers")
        if k in options
    }
    executor = executor_for(request)
    if callable(getattr(executor, "freeze", None)):
        return executor.freeze(experiment, options)
    from .execution import freeze_configuration

    return freeze_configuration(request.app, experiment, options)


def verify_for(request, configuration):
    executor = executor_for(request)
    if callable(getattr(executor, "verify", None)):
        return executor.verify(configuration)
    from .execution import verify_configuration

    return verify_configuration(request.app, configuration)


@router.get("/benchmark/capabilities")
def capabilities(request: Request, studio: StudioV11 = Depends(v11_for)):
    from app.models.openai_responses import OPENAI_MODELS
    from app.prompts.loader import PROMPT_NAMES

    state = request.app.state
    runtime = getattr(state, "central_runtime", None)
    injected = getattr(state, "eval_benchmark_executor", None) is not None
    available = injected or (
        getattr(state, "chat_mode_router", None) is not None
        and (
            getattr(state, "openai_client", None) is not None
            or getattr(state, "hybrid_runtime", None) is not None
        )
    )
    return {
        "live_available": bool(available),
        "reason": (
            None
            if available
            else "Runtime unavailable: use the authenticated full-mode backend with a configured provider. Creating a job never invokes a model."
        ),
        "models": sorted(OPENAI_MODELS),
        "prompt_resources": [n for n in PROMPT_NAMES if n.startswith(("shared/", "hybrid/", "central/"))],
        "tools": runtime.tools.names() if runtime else ["search_history"] if injected else [],
    }


@router.get("/benchmarks")
def list_jobs(jobs: BenchmarkStore = Depends(jobs_for)):
    return run(jobs.list)


@router.post("/benchmarks", status_code=201)
def create_job(payload: BenchmarkOptions, request: Request, jobs: BenchmarkStore = Depends(jobs_for)):
    experiment = run(jobs.studio.get_experiment, payload.experiment_id)
    frozen = run(frozen_for, request, experiment, payload.model_dump())
    return run(jobs.create, payload.model_dump(), frozen, experiment, mutation=True)


@router.get("/benchmarks/{job_id}")
def get_job(job_id: str, jobs: BenchmarkStore = Depends(jobs_for)):
    return run(jobs.get, job_id)


class LiveConfirmation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirm_live: StrictBool = False


async def controlled(awaitable):
    try:
        return await awaitable
    except ConflictError as exc:
        raise HTTPException(409, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, "Not found") from exc
    except (ValueError, sqlite3.IntegrityError) as exc:
        raise HTTPException(422, "Invalid benchmark operation") from exc


@router.post("/benchmarks/{job_id}/start")
async def start_job(
    job_id: str, payload: LiveConfirmation, request: Request, jobs: BenchmarkStore = Depends(jobs_for)
):
    if not payload.confirm_live:
        raise HTTPException(422, "Explicit confirm_live=true is required; this may invoke paid model APIs")
    job = await asyncio.to_thread(run, jobs.get, job_id)
    await asyncio.to_thread(run, verify_for, request, job["configuration"])
    return await controlled(manager_for(request, jobs).start(job_id, True))


@router.post("/benchmarks/{job_id}/cancel")
async def cancel_job(job_id: str, request: Request, jobs: BenchmarkStore = Depends(jobs_for)):
    return await controlled(manager_for(request, jobs).cancel(job_id))


@router.get("/runs/{run_id}/measurements")
def measurements(run_id: str, jobs: BenchmarkStore = Depends(jobs_for)):
    return run(jobs.measurements, run_id)


class AutomatedComparisonInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experiment_ids: list[str] = Field(min_length=2, max_length=20)
    benchmark_ids: list[str] | None = Field(default=None, max_length=20)
    judge_profile_id: str | None = None


@router.post("/benchmark/compare")
def automated_comparison(payload: AutomatedComparisonInput, jobs: BenchmarkStore = Depends(jobs_for)):
    from .analytics import compare_benchmarks

    return run(
        compare_benchmarks, jobs, payload.experiment_ids, payload.benchmark_ids, payload.judge_profile_id
    )


@router.get("/judge/profiles")
def judge_profiles(judges: JudgeStore = Depends(judges_for)):
    return run(judges.profiles)


@router.post("/judge/profiles", status_code=201)
def create_judge_profile(payload: JudgeProfileInput, judges: JudgeStore = Depends(judges_for)):
    return run(judges.create_profile, payload.model_dump(), mutation=True)


@router.get("/judge/scores")
def judge_scores(run_id: str | None = None, judges: JudgeStore = Depends(judges_for)):
    return run(judges.scores, run_id)


@router.post("/judge/scores", status_code=201)
def import_judge_score(payload: JudgeScoreInput, judges: JudgeStore = Depends(judges_for)):
    return run(judges.record_score, payload.model_dump(), mutation=True)


@router.get("/judge/profiles/{profile_id}/calibration")
def judge_calibration(profile_id: str, judges: JudgeStore = Depends(judges_for)):
    return run(judges.calibration, profile_id)


@router.get("/judge/profiles/{profile_id}/contract/{run_id}")
def judge_contract(profile_id: str, run_id: str, judges: JudgeStore = Depends(judges_for)):
    return run(judges.contract, profile_id, run_id)
