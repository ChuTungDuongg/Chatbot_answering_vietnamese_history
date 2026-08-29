from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterable

from .adapters import normalize_agent_flan, normalize_multihop, normalize_vietnam_history
from .adapters.common import AdapterError, get_messages, semantic_messages
from .builders.custom_history import (
    CustomBuildConfig,
    build_custom_trajectories,
    build_no_tool_trajectories,
    inspect_corpus,
    load_seed_records,
)
from .dedup import deduplicate, first_user_question, normalized_question
from .drive import resolve_corpus_path
from .io_utils import IncrementalJsonlWriter, atomic_write_json, atomic_write_jsonl, iter_jsonl, read_jsonl
from .mix import DEFAULT_MIX_RATIOS, mix_sources
from .retrieval import FixtureRetriever, PrecomputedRetriever, PrecomputedToolRetriever, ProjectRetriever
from .split import source_group, split_trajectories
from .stats import dataset_stats
from .teacher.local_hf import LocalHFTeacher
from .validate import validate_rows


PUBLIC_SOURCES: dict[str, tuple[str, Callable[..., dict[str, Any]]]] = {
    "agent_flan": ("internlm/Agent-FLAN", normalize_agent_flan),
    "multihop": ("khaimaitien/multi-hop-qa-function-calling-format-V1.0", normalize_multihop),
    "vietnam_history": ("minhxthanh/Vietnam-History-200K-Vi", normalize_vietnam_history),
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
    kwargs: dict[str, Any] = {
        "path": dataset_id,
        "split": args.split,
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
        return ProjectRetriever.load(corpus, device=args.device)
    raise ValueError(f"unknown retrieval backend: {backend}")


def _normalize_public(args: argparse.Namespace) -> dict[str, Any]:
    _, adapter = PUBLIC_SOURCES[args.source]
    output = Path(args.output)
    rejected_path = Path(args.rejected_output or output.with_name(f"{output.stem}.rejected.jsonl"))
    if args.dry_run:
        return {
            "dry_run": True,
            "source": args.source,
            "dataset_id": args.dataset_id or PUBLIC_SOURCES[args.source][0],
            "max_samples": args.max_samples,
            "output": str(output),
            "no_model_or_dataset_loaded": True,
        }
    retriever = None
    if args.source == "vietnam_history" and args.history_mode == "rag_grounded":
        retriever = _make_retriever(args, _corpus(args))
    rejected: list[dict[str, Any]] = []
    attempted = 0
    seen_questions: set[str] = set()
    if args.resume and output.exists():
        seen_questions = {
            normalized_question(first_user_question(row)) for row in iter_jsonl(output)
        }
    with IncrementalJsonlWriter(output, resume=args.resume, checkpoint_every=args.checkpoint_every) as writer:
        for index, raw_row in enumerate(_public_rows(args)):
            if attempted >= args.max_samples:
                break
            attempted += 1
            try:
                if args.source == "agent_flan":
                    row = adapter(raw_row, index=index, split=args.split, include_reasoning=args.include_reasoning)
                elif args.source == "multihop":
                    row = adapter(raw_row, index=index, split=args.split, include_reasoning=args.include_reasoning)
                else:
                    retrieval_results = None
                    if retriever is not None:
                        raw_messages = semantic_messages(get_messages(raw_row), include_reasoning=False)
                        question = next(str(item["content"]) for item in raw_messages if item["role"] == "user")
                        retrieval_results = retriever.search(question, top_k=args.top_k)
                    row = adapter(
                        raw_row,
                        index=index,
                        split=args.split,
                        mode=args.history_mode,
                        retrieval_results=retrieval_results,
                        preferred_only=args.history_preferred_only,
                    )
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
    if hasattr(retriever, "close"):
        retriever.close()
    atomic_write_jsonl(rejected_path, rejected)
    return {
        "attempted": attempted,
        "written": writer.written,
        "resume_skipped": writer.skipped,
        "rejected": len(rejected),
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
    )


def _build_custom(args: argparse.Namespace) -> dict[str, Any]:
    corpus = _corpus(args)
    output_dir = Path(args.output_dir)
    output = output_dir / "custom_history.jsonl"
    rejected_path = output_dir / "custom_history.rejected.jsonl"
    config = _custom_config(args)
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
    retriever = _make_retriever(args, corpus)
    teacher = None
    external_retriever = PrecomputedToolRetriever(args.external_results) if args.external_results else None
    if args.teacher_backend == "local_hf":
        if not args.teacher_model:
            raise ValueError("--teacher-model is required for teacher-backend=local_hf")
        teacher = LocalHFTeacher(
            args.teacher_model,
            device=args.teacher_device,
            batch_size=args.teacher_batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    rejected: list[dict[str, Any]] = []
    with IncrementalJsonlWriter(output, resume=args.resume, checkpoint_every=args.checkpoint_every) as writer:
        for row in build_custom_trajectories(
            corpus,
            retriever,
            config=config,
            completed_ids=writer.completed_ids,
            teacher=teacher,
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
    if hasattr(retriever, "close"):
        retriever.close()
    atomic_write_jsonl(rejected_path, rejected)
    return {
        "written": writer.written,
        "resume_skipped": writer.skipped,
        "rejected": len(rejected),
        "output": str(output),
        "rejected_output": str(rejected_path),
        "corpus_read_only": True,
    }


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
    mixed = mix_sources(sources, ratios, seed=args.seed, max_total=args.max_samples)
    deduped = deduplicate(mixed)
    validation = validate_rows(deduped.rows)
    rejected = [*deduped.rejected, *validation.rejected]
    if not args.dry_run:
        atomic_write_jsonl(args.output, validation.valid)
        atomic_write_jsonl(args.rejected_output or str(Path(args.output).with_name("rejected.jsonl")), rejected)
        atomic_write_json(str(Path(args.output).with_suffix(".stats.json")), dataset_stats(validation.valid))
    return {"mixed": len(mixed), "valid": len(validation.valid), "rejected": len(rejected), "dry_run": args.dry_run}


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
            name: sorted({source_group(row) for row in getattr(splits, name)})
            for name in ("train", "validation", "test")
        },
    }
    if not args.dry_run:
        atomic_write_jsonl(output_dir / "train.jsonl", splits.train)
        atomic_write_jsonl(output_dir / "validation.jsonl", splits.validation)
        atomic_write_jsonl(output_dir / "test.jsonl", splits.test)
        atomic_write_json(output_dir / "dataset_stats.json", stats)
        atomic_write_json(output_dir / "manifest.json", manifest)
    return {**manifest["rows"], "output_dir": str(output_dir), "dry_run": args.dry_run}


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
    for source in PUBLIC_SOURCES:
        namespace = argparse.Namespace(**vars(args))
        namespace.source = source
        namespace.output = str(intermediate / f"{source}.jsonl")
        namespace.rejected_output = str(intermediate / f"{source}.rejected.jsonl")
        namespace.max_samples = args.max_samples_per_source
        namespace.input_jsonl = None
        namespace.dataset_id = None
        namespace.dataset_config = None
        namespace.split = "train"
        namespace.include_reasoning = args.include_reasoning
        namespace.history_mode = args.history_mode
        namespace.history_preferred_only = args.history_preferred_only
        _normalize_public(namespace)
        canonical_source = {
            "agent_flan": "agent_flan",
            "multihop": "multi_hop_function_calling",
            "vietnam_history": "vietnam_history_200k",
        }[source]
        normalized_paths[canonical_source] = Path(namespace.output)
    custom_args = argparse.Namespace(**vars(args))
    _build_custom(custom_args)
    normalized_paths["custom_history"] = output_dir / "custom_history.jsonl"
    ratios = json.loads(Path(args.mix_config).read_text(encoding="utf-8"))
    sources = {name: read_jsonl(path) for name, path in normalized_paths.items()}
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
    return {"normalized": {key: str(value) for key, value in normalized_paths.items()}, "rejected": len(rejected), **split_result}


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
    public.add_argument("--split", default="train")
    public.add_argument("--input-jsonl", default=None, help="Offline raw fixture instead of Hugging Face.")
    public.add_argument("--output", required=True)
    public.add_argument("--rejected-output", default=None)
    public.add_argument("--cache-dir", default=None)
    public.add_argument("--max-samples", type=_positive, default=1000)
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
    _add_corpus_arguments(public)

    custom = subparsers.add_parser("build-custom", help="Build read-only corpus-grounded custom trajectories.")
    _add_corpus_arguments(custom)
    custom.add_argument("--output-dir", required=True)
    custom.add_argument("--retrieval-backend", choices=("project", "precomputed", "fixture"), default="project")
    custom.add_argument("--retrieval-results", default=None)
    custom.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    custom.add_argument("--top-k", type=_positive, default=6)
    custom.add_argument("--max-corpus-records", type=_positive, default=10_000)
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

    all_parser = subparsers.add_parser("build-all", help="Normalize, build, mix, validate, deduplicate, and split.")
    _add_corpus_arguments(all_parser)
    all_parser.add_argument("--output-dir", required=True)
    all_parser.add_argument("--cache-dir", default=None)
    all_parser.add_argument("--max-samples-per-source", type=_positive, default=1000)
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
    elif args.command == "build-all":
        result = _build_all(args)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
