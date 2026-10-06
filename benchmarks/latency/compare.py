"""Run an explicit HTTP/SSE experiment matrix against already deployed endpoints."""
from __future__ import annotations

import argparse
from argparse import Namespace
import json
import os
from pathlib import Path
import re
from urllib.parse import urlparse

from benchmarks.latency.runner import ROOT, _server_metadata, _sha256, run
from benchmarks.latency.report import write_json

LATENCY = ("answer_ttft_ms", "model_ttft_ms", "e2e_ms", "tpot_ms")
CENTRAL = ("planning_model_ms", "planning_wall_ms", "tool_execution_ms", "retrieval_ms")
IDENTITY = ("dataset_sha256", "model_id", "model_revision", "model_variant", "adapter_attached",
            "adapter_fingerprint", "corpus_hash", "retrieval_index_hash", "generation_settings",
            "retrieval_settings", "mode", "population", "concurrency", "request_options",
            "git_commit", "deployment_source_sha256")


def validate_invariants(reference: dict, candidate: dict) -> None:
    """Unknown critical identities are invalid comparisons, never proof of parity."""
    for key in IDENTITY:
        left, right = reference.get(key), candidate.get(key)
        if left != right:
            raise ValueError(f"Comparison invariant differs: {key}")
        if left is None and key not in {"adapter_fingerprint"}:
            raise ValueError(f"Comparison invariant is unknown: {key}")
    for key in ("gpu_model", "gpu_memory_bytes", "torch_version", "transformers_version", "cuda_version"):
        left = (reference.get("server_hardware") or {}).get(key)
        right = (candidate.get("server_hardware") or {}).get(key)
        if left is None or left != right:
            raise ValueError(f"Hardware/software class differs or is unknown: {key}")
    for key in ("fastapi_version", "numpy_version"):
        if (reference.get("server_hardware") or {}).get(key) != (candidate.get("server_hardware") or {}).get(key):
            raise ValueError(f"Serving/retrieval dependency version differs: {key}")
    for record in (reference, candidate):
        environment = record.get("server_environment") or {}
        if (environment.get("enable_hybrid_mode") != (record["mode"] == "hybrid")
                or environment.get("enable_central_mode") != (record["mode"] == "central")):
            raise ValueError("Comparison requires isolated single-mode deployments")
        capacity = environment.get("modal_max_inputs")
        if capacity is not None and capacity < record["concurrency"]:
            raise ValueError("Modal input capacity is below requested concurrency")
    for key in ("modal_max_inputs", "modal_gpu_class"):
        if (reference.get("server_environment") or {}).get(key) != (candidate.get("server_environment") or {}).get(key):
            raise ValueError(f"Deployment topology differs: {key}")
    if reference["mode"] == "central" and reference.get("central_settings") != candidate.get("central_settings"):
        raise ValueError("Central server tool/planner policies differ")


def difference(reference: float | None, candidate: float | None, *, higher_is_better: bool = False) -> dict:
    if reference is None or candidate is None:
        return {"absolute_difference": None, "percentage": None,
                "percentage_label": "throughput_increase_pct" if higher_is_better else "latency_reduction_pct"}
    absolute = candidate - reference
    percent = (absolute if higher_is_better else -absolute) / reference * 100 if reference > 0 else None
    return {"absolute_difference": absolute, "percentage": percent,
            "percentage_label": "throughput_increase_pct" if higher_is_better else "latency_reduction_pct"}


def _expected(observed: dict, expected: dict, path="metadata"):
    for key, value in expected.items():
        actual = observed.get(key)
        if isinstance(value, dict):
            if not isinstance(actual, dict):
                raise ValueError(f"{path}.{key} missing effective configuration")
            _expected(actual, value, f"{path}.{key}")
        elif actual != value:
            raise ValueError(f"{path}.{key}: expected {value!r}, observed {actual!r}")


def _identity(metadata, config, concurrency):
    mode = config["mode"]
    normalized = dict(metadata)
    for name in ("model_id", "model_revision"):
        plural = "model_ids" if name == "model_id" else "model_revisions"
        if not normalized.get(name) and isinstance(metadata.get(plural), dict):
            normalized[name] = metadata[plural].get(mode)
    normalized.update(dataset_sha256=config["dataset_sha256"], mode=mode,
                      population=config["population"], request_options=config.get("request_options", {}),
                      concurrency=concurrency)
    return normalized


def load_experiment(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("mode") not in {"hybrid", "central"}:
        raise ValueError("Invalid experiment schema/mode")
    if config.get("population") not in {"hybrid", "controlled_central", "realistic_central"}:
        raise ValueError("Explicit benchmark population required")
    if (config["mode"] == "hybrid") != (config["population"] == "hybrid"):
        raise ValueError("Population and mode differ")
    options = config.get("request_options", {})
    if set(options) - {"retrieval_backend", "steering", "debug"}:
        raise ValueError("Experiment cannot change retrieval quality or generation settings through request options")
    if config["population"] == "controlled_central" and options.get("steering") != {
        "mcp_enabled": False, "allowed_mcp_servers": [], "allowed_tools": ["search_history"]}:
        raise ValueError("Controlled Central requires deterministic local history-only steering")
    dataset = Path(config["dataset"])
    dataset = dataset if dataset.is_absolute() else ROOT / dataset
    if _sha256(dataset) != config["dataset_sha256"]:
        raise ValueError("Frozen dataset SHA256 mismatch")
    ids = []
    for variant in config["variants"]:
        identifier = variant["id"]
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", identifier) or identifier in ids:
            raise ValueError("Variant IDs must be unique safe directory names")
        ids.append(identifier)
        url = os.path.expandvars(variant["base_url"])
        parsed = urlparse(url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or "$" in url
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError(f"Set the base URL environment variable for {identifier}; credentials in URLs are refused")
        variant["base_url"] = url.rstrip("/")
    if config["reference_variant"] not in ids:
        raise ValueError("Missing reference variant")
    levels = config["concurrency"]
    if not levels or len(levels) != len(set(levels)) or any(level not in {1, 2, 4, 8} for level in levels):
        raise ValueError("Concurrency must explicitly select separate levels from 1,2,4,8")
    config["dataset"] = str(dataset.resolve())
    return config


def compare_results(config: dict, runs: list[dict]) -> dict:
    groups = []
    for concurrency in config["concurrency"]:
        population = [item for item in runs if item["metadata"]["concurrency"] == concurrency]
        reference = next(item for item in population if item["variant_id"] == config["reference_variant"])
        ref_group = reference["summary"]["groups"]["warm"]
        candidates = []
        for item in population:
            current = item["summary"]["groups"]["warm"]
            violations = []
            for label, record in (("reference", reference), ("candidate", item)):
                failure = record["metadata"].get("comparison_failure")
                if failure:
                    violations.append(f"{label} run failed validation: {failure}")
            try:
                validate_invariants(reference["metadata"], item["metadata"])
            except ValueError as error:
                violations.append(str(error))
            if current["count"] != ref_group["count"]:
                violations.append("Different measured workload size")
            if current["error_rate"] is None or ref_group["error_rate"] is None:
                violations.append("No measured requests")
            elif current["error_rate"] - ref_group["error_rate"] > config.get("max_error_rate_increase", 0):
                violations.append("Error rate increased; inspect raw failures before accepting latency results")
            metrics = {}
            names = [*LATENCY, "decode_tokens_per_second", "output_tokens", *(CENTRAL if config["mode"] == "central" else ())]
            for name in names:
                baseline = ref_group["metrics"][name]
                measured = current["metrics"][name]
                metrics[name] = {
                    "candidate": measured,
                    "reference": baseline,
                    "differences": {p: difference(baseline[p], measured[p], higher_is_better=name == "decode_tokens_per_second")
                                    for p in ("p50", "p95", "p99")} if not violations and name != "output_tokens" else None,
                }
            tokens = metrics["output_tokens"]
            ref_tokens, candidate_tokens = tokens["reference"]["p50"], tokens["candidate"]["p50"]
            length_review = bool(ref_tokens and candidate_tokens is not None
                                 and abs(candidate_tokens - ref_tokens) / ref_tokens > 0.20)
            candidates.append({"variant_id": item["variant_id"], "run_directory": item["directory"],
                               "comparable": not violations, "violations": violations,
                               "output_length_review_required": length_review,
                               "success_rate": current["success_rate"], "error_count": current["error_count"],
                               "metrics": metrics,
                               "aggregate_throughput": {key: {"candidate": current[key], "reference": ref_group[key],
                                   **difference(ref_group[key] if not violations else None, current[key], higher_is_better=True)}
                                   for key in ("successful_requests_per_second", "output_tokens_per_second")}})
        groups.append({"mode": config["mode"], "population": config["population"],
                       "concurrency": concurrency, "candidates": candidates})
    return {"schema_version": 1, "experiment_id": config["experiment_id"],
            "dataset_sha256": config["dataset_sha256"], "reference_variant": config["reference_variant"],
            "groups": groups, "interpretation": "Structural guardrails are necessary; Silver labels and output-length differences require quality review."}


def render_comparison(comparison: dict) -> str:
    def fmt(value):
        return "N/A" if value is None else f"{value:.3f}"
    lines = ["# Inference latency experiment", "", f"Experiment: `{comparison['experiment_id']}`",
             f"Dataset SHA256: `{comparison['dataset_sha256']}`", "",
             "Latency reduction = (reference - candidate) / reference. Throughput increase = (candidate - reference) / reference.",
             "Positive percentages indicate improvement. These are not claims of semantic equivalence.", ""]
    for group in comparison["groups"]:
        lines += [f"## {group['mode']} / {group['population']} / concurrency {group['concurrency']}", "",
                  "| Variant | Success | Answer TTFT p50/p95/p99 ms | Model TTFT p50/p95/p99 ms | E2E p50/p95/p99 ms | TPOT p50/p95/p99 ms | Decode tokens/s p50 | Successful requests/s |",
                  "|---|---:|---|---|---|---|---:|---:|"]
        for item in group["candidates"]:
            triples = [" / ".join(fmt(item["metrics"][name]["candidate"][p]) for p in ("p50", "p95", "p99")) for name in LATENCY]
            lines.append("| " + " | ".join([item["variant_id"], fmt(item["success_rate"]), *triples,
                fmt(item["metrics"]["decode_tokens_per_second"]["candidate"]["p50"]),
                fmt(item["aggregate_throughput"]["successful_requests_per_second"]["candidate"])]) + " |")
        lines += ["", "| Variant | Metric | Percentile | Absolute Δ (candidate − reference) | Latency reduction / throughput increase % |", "|---|---|---|---:|---:|"]
        for item in group["candidates"]:
            if item["violations"]:
                lines += ["", f"**Invalid comparison: {item['variant_id']}** — " + "; ".join(item["violations"])]
            if item["output_length_review_required"]:
                lines += ["", f"**Output length review required: {item['variant_id']}** (median output tokens differ by >20%)."]
            for name, values in item["metrics"].items():
                if values["differences"]:
                    for percentile, change in values["differences"].items():
                        lines.append(f"| {item['variant_id']} | {name} | {percentile} | {fmt(change['absolute_difference'])} | {fmt(change['percentage'])} |")
            for name, change in item["aggregate_throughput"].items():
                lines.append(f"| {item['variant_id']} | {name} | aggregate | {fmt(change['absolute_difference'])} | {fmt(change['percentage'])} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def execute(config_path: Path, output: Path) -> Path:
    config = load_experiment(config_path)
    # Preflight observed server identity/config before issuing any measured work.
    observed = []
    for variant in config["variants"]:
        metadata = _server_metadata(variant["base_url"], "experiment-preflight", config.get("timeout", 900), config["mode"])
        _expected(metadata, variant["expected"])
        observed.append(_identity(metadata, config, 1))
    reference = observed[next(index for index, variant in enumerate(config["variants"])
                              if variant["id"] == config["reference_variant"])]
    for candidate in observed:
        for concurrency in config["concurrency"]:
            validate_invariants({**reference, "concurrency": concurrency}, {**candidate, "concurrency": concurrency})
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "experiment_manifest.json", {**config, "server_preflight": observed})
    results = []
    for variant in config["variants"]:
        for concurrency in config["concurrency"]:
            directory = output / variant["id"] / f"c{concurrency}"
            args = Namespace(dataset=Path(config["dataset"]), output=directory, mode=config["mode"],
                             base_url=variant["base_url"], concurrency=concurrency,
                             runs=config.get("runs", 1), warmup=config.get("warmup", 2),
                             cold_start=False, timeout=config.get("timeout", 900),
                             client_id=None, metadata_file=None, strict_contract=True,
                             experiment_id=config["experiment_id"], variant_id=variant["id"],
                             population=config["population"], request_options=config.get("request_options", {}))
            failure = None
            try:
                run(args)
            except Exception as error:
                failure = f"{type(error).__name__}: {error}"
                write_json(directory / "run_failure.json", {"error": failure})
                if not (directory / "latency_summary.json").is_file():
                    raise
            metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
            if failure:
                metadata["comparison_failure"] = failure
            else:
                try:
                    _expected(metadata, variant["expected"])
                except ValueError as error:
                    metadata["comparison_failure"] = str(error)
                    write_json(directory / "run_failure.json", {"error": str(error)})
            if config.get("offline_evaluation", True):
                from evaluation.runner import run as evaluate

                evaluate(Path(config["dataset"]), directory / "latency_records.jsonl", directory / "quality")
            results.append({"variant_id": variant["id"], "directory": str(directory), "metadata": metadata,
                            "summary": json.loads((directory / "latency_summary.json").read_text(encoding="utf-8"))})
    comparison = compare_results(config, results)
    write_json(output / "comparison.json", comparison)
    (output / "comparison.md").write_text(render_comparison(comparison), encoding="utf-8")
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(execute(args.experiment, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
