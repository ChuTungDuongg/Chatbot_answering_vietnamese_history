from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

from .adapters import normalize_agent_flan, normalize_multihop, normalize_vietnam_history
from .adapters.common import AdapterError, get_messages, semantic_messages
from .audit import audit_rows, tokenizer_audit
from .builders.custom_history import (
    CustomBuildConfig,
    build_custom_trajectories,
    build_no_tool_trajectories,
    inspect_corpus,
    load_seed_records,
)
from .dedup import deduplicate, first_user_question, normalized_question
from .drive import resolve_corpus_path
from .final_gate import final_dataset_gate
from .io_utils import IncrementalJsonlWriter, atomic_write_json, atomic_write_jsonl, iter_jsonl, read_jsonl
from .mix import DEFAULT_MIX_RATIOS, mix_capacity_report, mix_sources
from .retrieval import FixtureRetriever, PrecomputedRetriever, PrecomputedToolRetriever, ProjectRetriever
from .split import TrajectorySplits, source_groups, split_coverage_report, split_trajectories
from .stats import dataset_stats
from .teacher.enhance import DEFAULT_TEACHER_TASKS, enhance_rows
from .teacher.local_hf import LocalHFTeacher
from .validate import validate_rows


PUBLIC_SOURCES: dict[str, tuple[str, Callable[..., dict[str, Any]]]] = {
    "agent_flan": ("internlm/Agent-FLAN", normalize_agent_flan),
    "multihop": ("khaimaitien/multi-hop-qa-function-calling-format-V1.0", normalize_multihop),
    "vietnam_history": ("minhxthanh/Vietnam-History-200K-Vi", normalize_vietnam_history),
}
PUBLIC_SOURCE_DEFAULT_SPLITS = {
    "agent_flan": "agent_instruct_react",
    "multihop": "train",
    "vietnam_history": "train",
}


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _ratio(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("ratio must be between 0 and 1")
    return parsed


def _key_value(values: list[str] | None, *, value_type: Callable[[str], Any] = str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"expected NAME=VALUE, received: {value}")
        key, raw = value.split("=", 1)
        if not key.strip() or not raw.strip():
            raise ValueError(f"expected non-empty NAME=VALUE, received: {value}")
        result[key.strip()] = value_type(raw.strip())
    return result


def _add_corpus_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus-path", default=None, help="Repo-relative or absolute enriched corpus directory/file.")
    parser.add_argument("--drive-corpus-path", default=None, help="Corpus directory/file on a mounted Google Drive.")
    parser.add_argument("--mount-drive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--drive-mount-point", default="/content/drive")


def _corpus(args: argparse.Namespace, *, must_exist: bool = True) -> Path:
    return resolve_corpus_path(
        args.corpus_path,
        drive_corpus_path=args.drive_corpus_path,
        mount_drive=args.mount_drive,
        drive_mount_point=args.drive_mount_point,
        must_exist=must_exist,
    )


def _public_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.input_jsonl:
        yield from iter_jsonl(args.input_jsonl)
        return
    dataset_id = args.dataset_id or PUBLIC_SOURCES[args.source][0]
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install requirements-training.txt to load public Hugging Face datasets") from exc
    split = args.split or PUBLIC_SOURCE_DEFAULT_SPLITS[args.source]
    kwargs: dict[str, Any] = {
        "path": dataset_id,
        "split": split,
        "streaming": True,
    }
    if args.dataset_config:
        kwargs["name"] = args.dataset_config
    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir
    dataset = load_dataset(**kwargs)
    yield from dataset


def _make_retriever(args: argparse.Namespace, corpus: Path | None = None):
    backend = args.retrieval_backend
    if backend == "precomputed":
        if not args.retrieval_results:
            raise ValueError("--retrieval-results is required for retrieval-backend=precomputed")
        return PrecomputedRetriever(args.retrieval_results)
    if corpus is None:
        raise ValueError(f"retrieval-backend={backend} requires --corpus-path")
    if backend == "fixture":
        records = load_seed_records(corpus, limit=args.max_corpus_records, seed=args.seed)
        return FixtureRetriever(records)
    if backend == "project":
        return ProjectRetriever.load(
            corpus,
            device=args.device,
            rerank_batch_size=getattr(args, "rerank_batch_size", None),
        )
    raise ValueError(f"unknown retrieval backend: {backend}")


def _normalize_public(args: argparse.Namespace) -> dict[str, Any]:
    _, adapter = PUBLIC_SOURCES[args.source]
    resolved_split = args.split or PUBLIC_SOURCE_DEFAULT_SPLITS[args.source]
    output = Path(args.output)
    rejected_path = Path(args.rejected_output or output.with_name(f"{output.stem}.rejected.jsonl"))
    if args.dry_run:
        return {
            "dry_run": True,
            "source": args.source,
            "dataset_id": args.dataset_id or PUBLIC_SOURCES[args.source][0],
            "max_samples": args.max_samples,
            "max_attempts": getattr(args, "max_attempts", None),
            "split": resolved_split,
            "output": str(output),
            "no_model_or_dataset_loaded": True,
        }
    retriever = None
    rejected: list[dict[str, Any]] = []
    attempted = 0
    seen_questions: set[str] = set()
    if args.resume and output.exists():
        seen_questions = {
            normalized_question(first_user_question(row)) for row in iter_jsonl(output)
        }
    max_attempts = getattr(args, "max_attempts", None) or max(args.max_samples * 10, args.max_samples + 100)
    hit_max_attempts = False
    try:
        if args.source == "vietnam_history" and args.history_mode == "rag_grounded":
            retriever = _make_retriever(args, _corpus(args))
        with IncrementalJsonlWriter(output, resume=args.resume, checkpoint_every=args.checkpoint_every) as writer:
            existing = len(writer.completed_ids)
            target_new = max(0, args.max_samples - existing)
            raw_rows = _public_rows(args) if target_new else ()
            for index, raw_row in enumerate(raw_rows):
                if writer.written >= target_new or attempted >= max_attempts:
                    break
                attempted += 1
                try:
                    if args.source == "agent_flan":
                        row = adapter(raw_row, index=index, split=resolved_split, include_reasoning=args.include_reasoning)
                    elif args.source == "multihop":
                        row = adapter(raw_row, index=index, split=resolved_split, include_reasoning=args.include_reasoning)
                    else:
                        retrieval_results = None
                        if retriever is not None:
                            raw_messages = semantic_messages(get_messages(raw_row), include_reasoning=False)
                            question = next(str(item["content"]) for item in raw_messages if item["role"] == "user")
                            retrieval_results = retriever.search(question, top_k=args.top_k)
                        row = adapter(
                            raw_row,
                            index=index,
                            split=resolved_split,
                            mode=args.history_mode,
                            retrieval_results=retrieval_results,
                            preferred_only=args.history_preferred_only,
                        )
                    if str(row.get("id", "")) in writer.completed_ids:
                        writer.write(row)
                        continue
                    question_key = normalized_question(first_user_question(row))
                    if question_key and question_key in seen_questions:
                        raise AdapterError("duplicate normalized user question")
                    validation = validate_rows([row])
                    if validation.rejected:
                        raise AdapterError(validation.rejected[0]["reason"])
                    writer.write(row)
                    if question_key:
                        seen_questions.add(question_key)
                except (AdapterError, ValueError, KeyError, TypeError) as exc:
                    rejected.append({"id": raw_row.get("id"), "reason": str(exc), "source_index": index})
            hit_max_attempts = attempted >= max_attempts and writer.written < target_new
    finally:
        if hasattr(retriever, "close"):
            retriever.close()
    atomic_write_jsonl(rejected_path, rejected)
    return {
        "attempted": attempted,
        "written": writer.written,
        "resume_skipped": writer.skipped,
        "rejected": len(rejected),
        "rejected_by_reason": dict(sorted(Counter(str(item.get("reason") or "unknown") for item in rejected).items())),
        "split": resolved_split,
        "max_attempts": max_attempts,
        "hit_max_attempts": hit_max_attempts,
        "output": str(output),
        "rejected_output": str(rejected_path),
    }


def _custom_config(args: argparse.Namespace) -> CustomBuildConfig:
    counts = {task: getattr(args, f"num_{task}") for task in (
        "factual", "cause", "significance", "compare", "summary", "multihop", "verification", "hard_negative", "insufficient_evidence"
    )}
    return CustomBuildConfig(
        task_counts=counts,
        top_k=args.top_k,
        seed=args.seed,
        max_corpus_records=args.max_corpus_records,
        observation_char_budget=args.observation_char_budget,
        trajectory_observation_char_budget=args.trajectory_observation_char_budget,
        max_result_text_chars=args.max_result_text_chars,
        max_candidate_attempts_per_task=args.max_candidate_attempts_per_task,
    )


def _build_custom(args: argparse.Namespace) -> dict[str, Any]:
    corpus = _corpus(args)
    output_dir = Path(args.output_dir)
    output = output_dir / "custom_history.jsonl"
    rejected_path = output_dir / "custom_history.rejected.jsonl"
    config = _custom_config(args)
    if args.teacher_backend == "local_hf" and not args.teacher_model:
        raise ValueError("--teacher-model is required for teacher-backend=local_hf")
    if args.dry_run:
        return {
            "dry_run": True,
            "corpus": str(corpus),
            "corpus_read_only": True,
            "retrieval_backend": args.retrieval_backend,
            "teacher_backend": args.teacher_backend,
            "task_counts": config.task_counts,
            "output": str(output),
            "no_model_or_retriever_loaded": True,
        }
    deterministic_output = (
        output_dir / "custom_history.deterministic.jsonl"
        if args.teacher_backend == "local_hf"
        else output
    )
    external_retriever = PrecomputedToolRetriever(args.external_results) if args.external_results else None
    rejected: list[dict[str, Any]] = []
    retriever = None
    try:
        retriever = _make_retriever(args, corpus)
        with IncrementalJsonlWriter(
            deterministic_output,
            resume=args.resume,
            checkpoint_every=args.checkpoint_every,
        ) as writer:
            for row in build_custom_trajectories(
                corpus,
                retriever,
                config=config,
                completed_ids=writer.completed_ids,
                external_retriever=external_retriever,
            ):
                validation = validate_rows([row])
                if validation.rejected:
                    rejected.extend(validation.rejected)
                else:
                    writer.write(row)
            if args.include_no_tool:
                for row in build_no_tool_trajectories():
                    writer.write(row)
    finally:
        if hasattr(retriever, "close"):
            retriever.close()

    if args.teacher_backend == "local_hf":
        teacher = LocalHFTeacher(
            args.teacher_model,
            device=args.teacher_device,
            batch_size=args.teacher_batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        enhanced = enhance_rows(
            read_jsonl(deterministic_output),
            teacher,
            task_types=DEFAULT_TEACHER_TASKS,
            failure_policy=args.teacher_failure_policy,
            seed=args.seed,
        )
        atomic_write_jsonl(output, enhanced.rows)
        rejected.extend(enhanced.rejected)
        final_written = len(enhanced.rows)
    else:
        final_written = len(read_jsonl(output))
    atomic_write_jsonl(rejected_path, rejected)
    return {
        "written": final_written,
        "resume_skipped": writer.skipped,
        "rejected": len(rejected),
        "output": str(output),
        "deterministic_output": str(deterministic_output),
        "rejected_output": str(rejected_path),
        "corpus_read_only": True,
    }


def _enhance_teacher(args: argparse.Namespace) -> dict[str, Any]:
    if args.teacher_backend != "local_hf":
        raise ValueError("enhance-teacher currently requires --teacher-backend local_hf")
    if not args.teacher_model:
        raise ValueError("--teacher-model is required")
    rows = read_jsonl(args.input)
    teacher = LocalHFTeacher(
        args.teacher_model,
        device=args.teacher_device,
        batch_size=args.teacher_batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    enhanced = enhance_rows(
        rows,
        teacher,
        task_types=args.task_type or DEFAULT_TEACHER_TASKS,
        failure_policy=args.failure_policy,
        seed=args.seed,
    )
    atomic_write_jsonl(args.output, enhanced.rows)
    rejected_output = args.rejected_output or str(Path(args.output).with_name(f"{Path(args.output).stem}.rejected.jsonl"))
    atomic_write_jsonl(rejected_output, enhanced.rejected)
    return {
        "input": len(rows),
        "written": len(enhanced.rows),
        "enhanced": enhanced.enhanced,
        "fallback": enhanced.fallback,
        "rejected": len(enhanced.rejected),
        "output": args.output,
        "rejected_output": rejected_output,
    }


def _audit_command(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.input)
    report = audit_rows(rows, strict_custom=args.strict_custom)
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required for --tokenizer audit") from exc
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
        report["tokenizer"] = tokenizer_audit(rows, tokenizer, max_seq_length=args.max_seq_length)
        if args.strict_custom and (
            report["tokenizer"]["rows_initial_user_lost"]
            or report["tokenizer"]["rows_any_tool_call_supervision_lost"]
        ):
            report["valid"] = False
    if args.output:
        atomic_write_json(args.output, report)
    return report


def _final_gate_command(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for the final tokenizer safety gate") from exc
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    splits = TrajectorySplits(
        train=read_jsonl(args.train_file),
        validation=read_jsonl(args.validation_file),
        test=read_jsonl(args.test_file),
    )
    report = final_dataset_gate(splits, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    if args.output:
        atomic_write_json(args.output, report)
    return report


def _validate_command(args: argparse.Namespace) -> dict[str, Any]:
    result = validate_rows(iter_jsonl(args.input))
    if args.output:
        atomic_write_jsonl(args.output, result.valid)
    rejected_path = args.rejected_output or str(Path(args.input).with_name("rejected.jsonl"))
    atomic_write_jsonl(rejected_path, result.rejected)
    return {"valid": len(result.valid), "rejected": len(result.rejected), "rejected_output": rejected_path}


def _mix_command(args: argparse.Namespace) -> dict[str, Any]:
    paths = _key_value(args.input)
    ratios = _key_value(args.ratio, value_type=float) if args.ratio else DEFAULT_MIX_RATIOS
    sources = {name: read_jsonl(path) for name, path in paths.items()}
    capacity = mix_capacity_report(sources, ratios, requested_total=args.max_samples)
    mixed = mix_sources(sources, ratios, seed=args.seed, max_total=args.max_samples)
    deduped = deduplicate(mixed)
    validation = validate_rows(deduped.rows)
    rejected = [*deduped.rejected, *validation.rejected]
    if not args.dry_run:
        atomic_write_jsonl(args.output, validation.valid)
        atomic_write_jsonl(args.rejected_output or str(Path(args.output).with_name("rejected.jsonl")), rejected)
        atomic_write_json(str(Path(args.output).with_suffix(".stats.json")), dataset_stats(validation.valid))
    return {
        "mixed": len(mixed), "valid": len(validation.valid), "rejected": len(rejected),
        "source_capacity": capacity, "dry_run": args.dry_run,
    }


def _split_command(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.input)
    splits = split_trajectories(
        rows,
        train_ratio=args.train_ratio,
        validation_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    output_dir = Path(args.output_dir)
    stats = dataset_stats(rows)
    manifest = {
        "seed": args.seed,
        "ratios": {"train": args.train_ratio, "validation": args.val_ratio, "test": args.test_ratio},
        "rows": {name: len(getattr(splits, name)) for name in ("train", "validation", "test")},
        "source_groups": {
            name: sorted({group for row in getattr(splits, name) for group in source_groups(row)})
            for name in ("train", "validation", "test")
        },
        "coverage": split_coverage_report(splits),
    }
    if not args.dry_run:
        atomic_write_jsonl(output_dir / "train.jsonl", splits.train)
        atomic_write_jsonl(output_dir / "validation.jsonl", splits.validation)
        atomic_write_jsonl(output_dir / "test.jsonl", splits.test)
        atomic_write_json(output_dir / "dataset_stats.json", stats)
        atomic_write_json(output_dir / "manifest.json", manifest)
    return {
        **manifest["rows"], "coverage": manifest["coverage"],
        "output_dir": str(output_dir), "dry_run": args.dry_run,
    }


def _build_all(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if args.dry_run:
        corpus = _corpus(args)
        return {
            "dry_run": True,
            "corpus": str(corpus),
            "output_dir": str(output_dir),
            "max_samples_per_source": args.max_samples_per_source,
            "no_download_model_or_retrieval_load": True,
        }
    intermediate = output_dir / "intermediate"
    normalized_paths: dict[str, Path] = {}
    normalization_reports: dict[str, dict[str, Any]] = {}
    for source in PUBLIC_SOURCES:
        namespace = argparse.Namespace(**vars(args))
        namespace.source = source
        namespace.output = str(intermediate / f"{source}.jsonl")
        namespace.rejected_output = str(intermediate / f"{source}.rejected.jsonl")
        namespace.max_samples = args.max_samples_per_source
        namespace.input_jsonl = None
        namespace.dataset_id = None
        namespace.dataset_config = None
        namespace.split = None
        namespace.include_reasoning = args.include_reasoning
        namespace.history_mode = args.history_mode
        namespace.history_preferred_only = args.history_preferred_only
        normalization_report = _normalize_public(namespace)
        canonical_source = {
            "agent_flan": "agent_flan",
            "multihop": "multi_hop_function_calling",
            "vietnam_history": "vietnam_history_200k",
        }[source]
        normalized_paths[canonical_source] = Path(namespace.output)
        normalization_reports[canonical_source] = normalization_report
    custom_args = argparse.Namespace(**vars(args))
    _build_custom(custom_args)
    normalized_paths["custom_history"] = output_dir / "custom_history.jsonl"
    ratios = json.loads(Path(args.mix_config).read_text(encoding="utf-8"))
    sources = {name: read_jsonl(path) for name, path in normalized_paths.items()}
    capacity = mix_capacity_report(sources, ratios, requested_total=args.max_samples)
    mixed = mix_sources(sources, ratios, seed=args.seed, max_total=args.max_samples)
    deduped = deduplicate(mixed)
    validated = validate_rows(deduped.rows)
    rejected = [*deduped.rejected, *validated.rejected]
    mixed_path = output_dir / "mixed.validated.jsonl"
    atomic_write_jsonl(mixed_path, validated.valid)
    atomic_write_jsonl(output_dir / "rejected.jsonl", rejected)
    split_args = argparse.Namespace(
        input=str(mixed_path),
        output_dir=str(output_dir / "final"),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        dry_run=False,
    )
    split_result = _split_command(split_args)
    return {
        "normalized": {key: str(value) for key, value in normalized_paths.items()},
        "public_normalization": normalization_reports,
        "rejected": len(rejected), "source_capacity": capacity, **split_result,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build canonical tool-use trajectories for a future central Qwen3-8B agent.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a bounded read-only corpus sample.")
    _add_corpus_arguments(inspect_parser)
    inspect_parser.add_argument("--max-samples", type=_positive, default=1000)

    public = subparsers.add_parser("normalize-public", help="Normalize one bounded public source.")
    public.add_argument("--source", choices=sorted(PUBLIC_SOURCES), required=True)
    public.add_argument("--dataset-id", default=None)
    public.add_argument("--dataset-config", default=None)
    public.add_argument(
        "--split",
        default=None,
        help="Dataset split/config. Defaults per source (Agent-FLAN: agent_instruct_react).",
    )
    public.add_argument("--input-jsonl", default=None, help="Offline raw fixture instead of Hugging Face.")
    public.add_argument("--output", required=True)
    public.add_argument("--rejected-output", default=None)
    public.add_argument("--cache-dir", default=None)
    public.add_argument("--max-samples", type=_positive, default=1000)
    public.add_argument("--max-attempts", type=_positive, default=None)
    public.add_argument("--include-reasoning", action=argparse.BooleanOptionalAction, default=False)
    public.add_argument("--history-mode", choices=("style_only", "rag_grounded"), default="style_only")
    public.add_argument("--history-preferred-only", action=argparse.BooleanOptionalAction, default=False)
    public.add_argument("--resume", action="store_true")
    public.add_argument("--checkpoint-every", type=_positive, default=100)
    public.add_argument("--dry-run", action="store_true")
    public.add_argument("--retrieval-backend", choices=("project", "precomputed", "fixture"), default="project")
    public.add_argument("--retrieval-results", default=None)
    public.add_argument("--top-k", type=_positive, default=6)
    public.add_argument("--max-corpus-records", type=_positive, default=10_000)
    public.add_argument("--seed", type=int, default=42)
    public.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    public.add_argument("--rerank-batch-size", type=_positive, default=None)
    _add_corpus_arguments(public)

    custom = subparsers.add_parser("build-custom", help="Build read-only corpus-grounded custom trajectories.")
    _add_corpus_arguments(custom)
    custom.add_argument("--output-dir", required=True)
    custom.add_argument("--retrieval-backend", choices=("project", "precomputed", "fixture"), default="project")
    custom.add_argument("--retrieval-results", default=None)
    custom.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    custom.add_argument("--top-k", type=_positive, default=6)
    custom.add_argument("--max-corpus-records", type=_positive, default=10_000)
    custom.add_argument("--observation-char-budget", type=_positive, default=12_000)
    custom.add_argument("--trajectory-observation-char-budget", type=_positive, default=6_000)
    custom.add_argument("--max-result-text-chars", type=_positive, default=1_600)
    custom.add_argument("--max-candidate-attempts-per-task", type=_positive, default=10_000)
    custom.add_argument("--rerank-batch-size", type=_positive, default=None)
    for name, default in (("factual", 20), ("cause", 10), ("significance", 10), ("compare", 8), ("summary", 10), ("multihop", 10), ("verification", 6), ("hard-negative", 6), ("insufficient-evidence", 6)):
        custom.add_argument(f"--num-{name}", dest=f"num_{name.replace('-', '_')}", type=int, default=default)
    custom.add_argument("--external-results", default=None, help="Optional precomputed search_wikipedia/search_web results.")
    custom.add_argument("--include-no-tool", action=argparse.BooleanOptionalAction, default=True)
    custom.add_argument("--teacher-backend", choices=("none", "local_hf"), default="none")
    custom.add_argument("--teacher-model", default=None)
    custom.add_argument("--teacher-device", choices=("auto", "cpu", "cuda"), default="auto")
    custom.add_argument("--teacher-batch-size", type=_positive, default=1)
    custom.add_argument("--max-new-tokens", type=_positive, default=512)
    custom.add_argument("--temperature", type=float, default=0.0)
    custom.add_argument("--teacher-failure-policy", choices=("fallback", "reject"), default="fallback")
    custom.add_argument("--seed", type=int, default=42)
    custom.add_argument("--resume", action="store_true")
    custom.add_argument("--checkpoint-every", type=_positive, default=25)
    custom.add_argument("--dry-run", action="store_true")

    validate = subparsers.add_parser("validate", help="Validate canonical JSONL and emit rejected rows with reasons.")
    validate.add_argument("--input", required=True)
    validate.add_argument("--output", default=None)
    validate.add_argument("--rejected-output", default=None)

    mix = subparsers.add_parser("mix", help="Mix canonical sources with configurable ratios.")
    mix.add_argument("--input", action="append", required=True, metavar="SOURCE=PATH")
    mix.add_argument("--ratio", action="append", default=None, metavar="SOURCE=RATIO")
    mix.add_argument("--output", required=True)
    mix.add_argument("--rejected-output", default=None)
    mix.add_argument("--max-samples", type=_positive, default=None)
    mix.add_argument("--seed", type=int, default=42)
    mix.add_argument("--dry-run", action="store_true")

    split = subparsers.add_parser("split", help="Create deterministic source-group-safe train/validation/test files.")
    split.add_argument("--input", required=True)
    split.add_argument("--output-dir", required=True)
    split.add_argument("--train-ratio", type=_ratio, default=0.90)
    split.add_argument("--val-ratio", type=_ratio, default=0.05)
    split.add_argument("--test-ratio", type=_ratio, default=0.05)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--dry-run", action="store_true")

    enhance = subparsers.add_parser("enhance-teacher", help="Post-process canonical rows with an answer-only local teacher.")
    enhance.add_argument("--input", required=True)
    enhance.add_argument("--output", required=True)
    enhance.add_argument("--rejected-output", default=None)
    enhance.add_argument("--teacher-backend", choices=("local_hf",), default="local_hf")
    enhance.add_argument("--teacher-model", required=True)
    enhance.add_argument("--teacher-device", choices=("auto", "cpu", "cuda"), default="auto")
    enhance.add_argument("--teacher-batch-size", type=_positive, default=1)
    enhance.add_argument("--max-new-tokens", type=_positive, default=512)
    enhance.add_argument("--temperature", type=float, default=0.0)
    enhance.add_argument("--task-type", action="append", choices=sorted(DEFAULT_TEACHER_TASKS), default=None)
    enhance.add_argument("--failure-policy", choices=("fallback", "reject"), default="fallback")
    enhance.add_argument("--seed", type=int, default=42)

    audit = subparsers.add_parser("audit", help="Audit semantic, grounding, compactness, and optional token safety.")
    audit.add_argument("--input", required=True)
    audit.add_argument("--output", default=None)
    audit.add_argument("--strict-custom", action="store_true")
    audit.add_argument(
        "--tokenizer", "--tokenizer-model-id", dest="tokenizer", default=None,
        help="Optional local/Hugging Face tokenizer id for token-only audit.",
    )
    audit.add_argument("--max-seq-length", type=_positive, default=8192)
    audit.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)

    gate = subparsers.add_parser("gate-final", help="Run all deterministic all-split GO-TRAIN gates.")
    gate.add_argument("--train-file", required=True)
    gate.add_argument("--validation-file", required=True)
    gate.add_argument("--test-file", required=True)
    gate.add_argument("--tokenizer", required=True)
    gate.add_argument("--max-seq-length", type=_positive, default=4096)
    gate.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    gate.add_argument("--output", default=None)

    all_parser = subparsers.add_parser("build-all", help="Normalize, build, mix, validate, deduplicate, and split.")
    _add_corpus_arguments(all_parser)
    all_parser.add_argument("--output-dir", required=True)
    all_parser.add_argument("--cache-dir", default=None)
    all_parser.add_argument("--max-samples-per-source", type=_positive, default=1000)
    all_parser.add_argument("--max-attempts", type=_positive, default=None)
    all_parser.add_argument("--max-samples", type=_positive, default=None)
    all_parser.add_argument("--mix-config", default=str(Path(__file__).with_name("configs") / "mix_v1.json"))
    all_parser.add_argument("--history-mode", choices=("style_only", "rag_grounded"), default="style_only")
    all_parser.add_argument("--history-preferred-only", action=argparse.BooleanOptionalAction, default=False)
    all_parser.add_argument("--include-reasoning", action=argparse.BooleanOptionalAction, default=False)
    all_parser.add_argument("--retrieval-backend", choices=("project", "precomputed", "fixture"), default="project")
    all_parser.add_argument("--retrieval-results", default=None)
    all_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    all_parser.add_argument("--top-k", type=_positive, default=6)
    all_parser.add_argument("--max-corpus-records", type=_positive, default=10_000)
    all_parser.add_argument("--observation-char-budget", type=_positive, default=12_000)
    all_parser.add_argument("--trajectory-observation-char-budget", type=_positive, default=6_000)
    all_parser.add_argument("--max-result-text-chars", type=_positive, default=1_600)
    all_parser.add_argument("--max-candidate-attempts-per-task", type=_positive, default=10_000)
    all_parser.add_argument("--rerank-batch-size", type=_positive, default=None)
    for name, default in (("factual", 20), ("cause", 10), ("significance", 10), ("compare", 8), ("summary", 10), ("multihop", 10), ("verification", 6), ("hard-negative", 6), ("insufficient-evidence", 6)):
        all_parser.add_argument(f"--num-{name}", dest=f"num_{name.replace('-', '_')}", type=int, default=default)
    all_parser.add_argument("--external-results", default=None)
    all_parser.add_argument("--include-no-tool", action=argparse.BooleanOptionalAction, default=True)
    all_parser.add_argument("--teacher-backend", choices=("none", "local_hf"), default="none")
    all_parser.add_argument("--teacher-model", default=None)
    all_parser.add_argument("--teacher-device", choices=("auto", "cpu", "cuda"), default="auto")
    all_parser.add_argument("--teacher-batch-size", type=_positive, default=1)
    all_parser.add_argument("--max-new-tokens", type=_positive, default=512)
    all_parser.add_argument("--temperature", type=float, default=0.0)
    all_parser.add_argument("--teacher-failure-policy", choices=("fallback", "reject"), default="fallback")
    all_parser.add_argument("--checkpoint-every", type=_positive, default=25)
    all_parser.add_argument("--resume", action="store_true")
    all_parser.add_argument("--train-ratio", type=_ratio, default=0.90)
    all_parser.add_argument("--val-ratio", type=_ratio, default=0.05)
    all_parser.add_argument("--test-ratio", type=_ratio, default=0.05)
    all_parser.add_argument("--seed", type=int, default=42)
    all_parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        result = inspect_corpus(_corpus(args), max_records=args.max_samples)
    elif args.command == "normalize-public":
        result = _normalize_public(args)
    elif args.command == "build-custom":
        result = _build_custom(args)
    elif args.command == "validate":
        result = _validate_command(args)
    elif args.command == "mix":
        result = _mix_command(args)
    elif args.command == "split":
        result = _split_command(args)
    elif args.command == "enhance-teacher":
        result = _enhance_teacher(args)
    elif args.command == "audit":
        result = _audit_command(args)
    elif args.command == "gate-final":
        result = _final_gate_command(args)
    elif args.command == "build-all":
        result = _build_all(args)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not (args.command in {"audit", "gate-final"} and not result.get("valid", True)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
