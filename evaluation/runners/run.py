"""Future local Central runs. Default validates inputs only; --execute is explicit."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from pydantic import Field

from evaluation.io import file_sha256, fingerprint_paths, read_jsonl, write_json
from evaluation.recording import from_result
from evaluation.schema import Contract, Question, RunMetadata

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


class RunConfig(Contract):
    dataset: str
    dataset_version: str
    seed: int = 42
    adapter_path: str = "/artifacts/adapters/central-v2"
    # Shared settings are applied identically to both variants in separate processes.
    settings: dict[str, str | int | float | bool] = Field(default_factory=dict)


async def run_questions(agent, questions, metadata, output_dir, *, seed_question=None):
    """The injectable seam is for tiny fake tests; production uses the app lifespan."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "metadata.json", metadata.model_dump(mode="json"))
    with (output / "records.jsonl").open("x", encoding="utf-8") as handle:
        for index, question in enumerate(questions):
            if seed_question:
                seed_question(metadata.seed + index)
            start = time.perf_counter()
            try:
                result = await agent.run(question=question.question, history=[], request_id=f"{metadata.run_id}:{question.id}")
            except Exception as exc:
                # A failed question is still paired and auditable. Do not invent
                # counters/quality labels for a request that never returned them.
                result = {"status": "runner_error", "final_failure_reason": "runner_error",
                          "runner_error": {"type": type(exc).__name__, "message": str(exc)}}
            record = from_result(question, metadata, result, latency_ms=(time.perf_counter() - start) * 1000)
            handle.write(record.model_dump_json() + "\n")
            handle.flush()
    return output


async def execute(config, questions, dataset, variant, run_id):
    # This imports the real production composition root only on explicit execution.
    # No alternate agent implementation or legacy fallback exists in this runner.
    if "app.main" in sys.modules or "app.config" in sys.modules:
        raise RuntimeError("run each variant in a fresh process")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=REPO, text=True).strip():
        raise ValueError("evaluation execution requires a clean committed host checkout")
    env = {key.upper(): str(value).lower() if isinstance(value, bool) else str(value)
           for key, value in config.settings.items()}
    protected = {"APP_MODE": "full", "DEVICE": "cuda", "ENABLE_CENTRAL_MODE": "true",
        "ENABLE_HYBRID_MODE": "false", "ENABLE_THREE_LLM_MODE": "false", "DEFAULT_INFERENCE_MODE": "central",
        "CENTRAL_AGENT_MODEL_ID": "Qwen/Qwen3-8B", "CENTRAL_AGENT_LOCAL_FILES_ONLY": "true",
        "CENTRAL_AGENT_ENABLE_WEB": "false", "CENTRAL_AGENT_ENABLE_WIKIPEDIA": "false",
        "CENTRAL_AGENT_ENABLE_DOCUMENTS": "false", "AGENT_ENABLE_WEB": "false",
        "AGENT_ENABLE_WIKIPEDIA": "false", "AGENT_ENABLE_DOCUMENT_SEARCH": "false",
        "CENTRAL_AGENT_ADAPTER_PATH": config.adapter_path if variant == "adapted" else "",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    if any(key in protected and value != protected[key] for key, value in env.items()):
        raise ValueError("shared settings cannot override paired-run invariants")
    os.environ.update({**env, **protected})
    from app.config import settings
    unknown = set(key.lower() for key in config.settings) - set(type(settings).model_fields)
    if unknown:
        raise ValueError(f"unknown host settings: {sorted(unknown)}")
    from app.agents.common.hf_cache import hf_cache_status
    cached = hf_cache_status(settings.central_agent_model_id, cache_dir=settings.central_agent_hf_cache_dir)
    if not cached["cache_hit"]:
        raise ValueError("Qwen must already be cached; evaluation never downloads it")
    import torch
    from transformers import set_seed
    if not torch.cuda.is_available():
        raise ValueError("future Qwen execution requires an available GPU")
    hardware = {"name": torch.cuda.get_device_name(0), "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
                "cuda_version": torch.version.cuda, "device_count": torch.cuda.device_count()}
    snapshot = settings.model_dump(mode="json")
    # Secrets are not reproducibility settings and must not enter raw run logs.
    snapshot = {key: value for key, value in snapshot.items() if not any(word in key for word in ("api_key", "secret", "password", "token"))
                or key.endswith("tokens") or "token_margin" in key}
    snapshot.pop("central_agent_adapter_path", None)
    environment = {"python": platform.python_version(), "platform": platform.system(),
        **{name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "numpy")}}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    from app.main import app, lifespan
    async with lifespan(app):
        agent = app.state.central_agent
        if agent is None:
            raise RuntimeError("production Central runtime was not constructed")
        metadata = RunMetadata(run_id=run_id, variant=variant, timestamp=datetime.now(timezone.utc), git_commit=commit,
            model_id=settings.central_agent_model_id, model_revision=Path(cached["snapshot_path"]).name,
            adapter_path=config.adapter_path if variant == "adapted" else None,
            adapter_sha256=fingerprint_paths([config.adapter_path]) if variant == "adapted" else None,
            adapter_enabled=variant == "adapted", dataset_version=config.dataset_version, dataset_sha256=file_sha256(dataset),
            retrieval_index_sha256=fingerprint_paths(settings.required_retrieval_paths()),
            prompt_sha256=fingerprint_paths([*sorted((REPO / "app/agents/central").glob("*.py")), REPO / "app/agents/common/hermes_function_call.py"]),
            generation_settings={key: value for key, value in snapshot.items() if "tokens" in key or key == "dtype"},
            retrieval_settings={key: value for key, value in snapshot.items() if any(word in key for word in ("retrieval", "rerank", "top_k", "embedding"))},
            tools=sorted(agent._allowed_tools(None, None)),
            context_budgets={key: value for key, value in snapshot.items() if "budget" in key or "excerpt" in key},
            host_config=snapshot, seed=config.seed, hardware=hardware, hardware_class=json.dumps(hardware, sort_keys=True), environment=environment)
        return await run_questions(agent, questions, metadata, ROOT / "logs" / run_id, seed_question=set_seed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/paired.json")
    parser.add_argument("--variant", choices=("base", "adapted"), required=True)
    parser.add_argument("--run-id", help="Directory name under evaluation/logs")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    config = RunConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    dataset = Path(config.dataset)
    if not dataset.is_absolute():
        dataset = REPO / dataset
    questions = read_jsonl(dataset, Question)
    if not questions:
        raise ValueError("question set is empty")
    if not args.execute:
        print(json.dumps({"validation": "ok", "questions": len(questions), "dataset_sha256": file_sha256(dataset), "executed": False}))
        return 0
    if not args.run_id or args.run_id in {".", ".."} or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in args.run_id):
        raise ValueError("--execute requires a simple unique --run-id")
    asyncio.run(execute(config, questions, dataset, args.variant, args.run_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
