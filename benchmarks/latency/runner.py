"""Run external HTTP SSE latency measurements; never imports application runtime."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from benchmarks.latency.metrics import summarize
from benchmarks.latency.report import render_markdown, write_json
from benchmarks.latency.sse_client import measure_request
from evaluation.schema import load_questions


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL, timeout=3).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _client_hardware() -> dict[str, object]:
    # These describe the benchmark client. Remote server hardware must be supplied
    # separately; never label the client GPU as the server GPU.
    try:
        import psutil  # type: ignore[import-not-found]
        ram_bytes = psutil.virtual_memory().total
    except ImportError:
        ram_bytes = None
    return {"os": platform.platform(), "python": platform.python_version(),
            "cpu": platform.processor() or None, "cpu_count": os.cpu_count(),
            "ram_bytes": ram_bytes,
            "torch_version": _package_version("torch"),
            "transformers_version": _package_version("transformers")}


def _server_metadata(base_url: str, client_id: str, timeout: float, mode: str) -> dict:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/v1/baseline/metadata?{urllib.parse.urlencode({'mode': mode})}",
        headers={"Accept": "application/json", "X-Client-ID": client_id},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=min(timeout, 30)) as response:
            value = json.load(response)
    except Exception as exc:
        raise RuntimeError(f"could not fetch server baseline metadata: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("server baseline metadata must be a JSON object")
    return value


def _for_mode(value, mode: str):
    return value[mode] if isinstance(value, dict) and mode in value else value


def _merge_observed(observed, declared, field: str):
    """Fill server-null fields from a file and reject contradictory claims."""
    if observed is None:
        return declared
    if declared is None:
        return observed
    if isinstance(observed, dict) and isinstance(declared, dict):
        return {key: _merge_observed(observed.get(key), declared.get(key), f"{field}.{key}")
                for key in sorted(observed.keys() | declared.keys())}
    if observed != declared:
        raise ValueError(f"metadata-file {field} conflicts with server metadata")
    return observed


def run(args: argparse.Namespace) -> Path:
    questions = load_questions(args.dataset)
    if not questions:
        raise ValueError("dataset is empty")
    if args.cold_start and args.concurrency != 1:
        raise ValueError("a cold-start sample requires --concurrency 1")
    if args.runs < 1 or args.warmup < 0 or args.concurrency < 1 or args.timeout <= 0:
        raise ValueError("runs/concurrency must be positive; warmup cannot be negative")
    metadata_extra = json.loads(args.metadata_file.read_text(encoding="utf-8")) if args.metadata_file else {}
    if not isinstance(metadata_extra, dict):
        raise ValueError("metadata file must be a JSON object")
    client_id = args.client_id or f"benchmark-{uuid4().hex}"
    observed_metadata = _server_metadata(args.base_url, client_id, args.timeout, args.mode)
    if observed_metadata.get("schema_version") not in (None, 1):
        raise ValueError("unsupported server baseline metadata schema version")
    if observed_metadata.get("mode") not in (None, args.mode):
        raise ValueError("server metadata mode differs from requested benchmark mode")
    model_ids = observed_metadata.get("model_ids")
    model_revisions = observed_metadata.get("model_revisions")
    selected_server_metadata = {
        "git_commit": observed_metadata.get("git_commit"),
        "server_hardware": observed_metadata.get("server_hardware", observed_metadata.get("hardware")),
        "model_id": model_ids.get(args.mode) if isinstance(model_ids, dict) else observed_metadata.get("model_id"),
        "model_revision": model_revisions.get(args.mode) if isinstance(model_revisions, dict) else observed_metadata.get("model_revision"),
        "corpus_hash": observed_metadata.get("corpus_hash"),
        "retrieval_index_hash": observed_metadata.get("retrieval_index_hash"),
        "generation_settings": _for_mode(observed_metadata.get("generation_settings"), args.mode),
        "retrieval_settings": _for_mode(observed_metadata.get("retrieval_settings"), args.mode),
        "server_environment": observed_metadata.get("server_environment"),
    }
    selected_metadata = {
        key: _merge_observed(value, metadata_extra.get(key), key)
        for key, value in selected_server_metadata.items()
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema_version": 1, "run_id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": selected_metadata["git_commit"],
        "client_git_commit": _git_commit(), "dataset": str(args.dataset.resolve()),
        "dataset_sha256": _sha256(args.dataset),
        "dataset_schema_version": 1, "mode": args.mode, "base_url": args.base_url,
        "warmup": args.warmup, "runs_per_question": args.runs,
        "concurrency": args.concurrency, "cold_start_requested": args.cold_start,
        "client_hardware": _client_hardware(),
        "server_hardware": selected_metadata["server_hardware"],
        "model_id": selected_metadata["model_id"],
        "model_revision": selected_metadata["model_revision"],
        "corpus_hash": selected_metadata["corpus_hash"],
        "retrieval_index_hash": selected_metadata["retrieval_index_hash"],
        "generation_settings": selected_metadata["generation_settings"],
        "retrieval_settings": selected_metadata["retrieval_settings"],
        "server_environment": selected_metadata["server_environment"],
        "server_metadata_schema_version": observed_metadata.get("schema_version"),
    }
    rows: list[dict] = []
    raw_path = output / "latency_records.jsonl"
    with raw_path.open("x", encoding="utf-8") as handle:
        def save(row: dict) -> None:
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()

        def measure(item: tuple[dict, str, int]) -> dict:
            question, phase, index = item
            return measure_request(base_url=args.base_url, question=question, mode=args.mode,
                                   client_id=client_id, timeout=args.timeout,
                                   phase=phase, run_index=index)

        dataset = [q.model_dump(mode="json") for q in questions]
        if args.cold_start:
            save(measure((dataset[0], "cold", 0)))
        for index in range(args.warmup):
            save(measure((dataset[index % len(dataset)], "warmup", index)))
        work = [(question, "warm", repeat) for repeat in range(args.runs) for question in dataset]
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for row in pool.map(measure, work):
                save(row)
    observed_model_ids = {row["model_id"] for row in rows if row.get("model_id")}
    observed_revisions = {row["model_revision"] for row in rows if row.get("model_revision")}
    if len(observed_model_ids) > 1 or len(observed_revisions) > 1:
        raise ValueError("server changed model ID or revision during the benchmark")
    if observed_model_ids:
        observed = next(iter(observed_model_ids))
        if metadata["model_id"] and metadata["model_id"] != observed:
            raise ValueError("declared model_id differs from server telemetry")
        metadata["model_id"] = observed
    if observed_revisions:
        observed = next(iter(observed_revisions))
        if metadata["model_revision"] and metadata["model_revision"] != observed:
            raise ValueError("declared model_revision differs from server telemetry")
        metadata["model_revision"] = observed
    for setting in ("generation_settings", "retrieval_settings"):
        observed = [row[setting] for row in rows if row.get(setting) is not None]
        if observed:
            if any(value != observed[0] for value in observed[1:]):
                raise ValueError(f"server changed {setting} during the benchmark")
            if metadata[setting] is not None and metadata[setting] != observed[0]:
                raise ValueError(f"declared {setting} differs from server telemetry")
            metadata[setting] = observed[0]
    summary = summarize(rows)
    write_json(output / "run_metadata.json", metadata)
    write_json(output / "latency_summary.json", summary)
    (output / "latency_summary.md").write_text(render_markdown(summary, metadata), encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("hybrid", "central"), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation/datasets/fixtures/questions.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--cold-start", action="store_true")
    parser.add_argument("--client-id")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--metadata-file", type=Path,
                        help="Optional server hardware and artifact/config hashes JSON")
    args = parser.parse_args(argv)
    print(run(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
