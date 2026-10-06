"""Safe file transfer to a mounted Google Drive directory (no Drive API)."""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "evaluation/datasets/v1_silver/questions_3000.jsonl": "datasets/canonical/questions_3000.jsonl",
    "evaluation/datasets/v1_silver/splits/v1_seed42/train.jsonl": "datasets/splits/v1_seed42/train.jsonl",
    "evaluation/datasets/v1_silver/splits/v1_seed42/validation.jsonl": "datasets/splits/v1_seed42/validation.jsonl",
    "evaluation/datasets/v1_silver/splits/v1_seed42/test.jsonl": "datasets/splits/v1_seed42/test.jsonl",
    "evaluation/datasets/v1_silver/splits/v1_seed42/split_manifest.json": "datasets/splits/v1_seed42/split_manifest.json",
    "evaluation/datasets/v1_silver/splits/v1_seed42/leakage_report.json": "datasets/splits/v1_seed42/leakage_report.json",
    "training/datasets/vn_history_rag_sft_v1/train_sft.jsonl": "datasets/sft/train_sft.jsonl",
    "training/datasets/vn_history_rag_sft_v1/validation_sft.jsonl": "datasets/sft/validation_sft.jsonl",
    "training/datasets/vn_history_rag_sft_v1/manifest.json": "datasets/sft/manifest.json",
    "training/datasets/vn_history_rag_sft_v1/stats.json": "datasets/sft/stats.json",
}
# SFT consumers also need the frozen split IDs beside the SFT files.
SFT_SPLIT_MANIFEST = "datasets/sft/split_manifest.json"
LAYOUT = ("datasets/canonical", "datasets/splits/v1_seed42", "datasets/sft",
          "models/qwen3_4b_sft_v1", "evaluation/predictions", "evaluation/reports", "logs",
          "benchmarks/datasets/latency_v1", "benchmarks/experiments", "benchmarks/results")
BENCHMARK_ASSETS = {
    "evaluation/datasets/latency_v1/questions_100.jsonl": "benchmarks/datasets/latency_v1/questions_100.jsonl",
    "evaluation/datasets/latency_v1/manifest.json": "benchmarks/datasets/latency_v1/manifest.json",
    **{f"configs/benchmarks/{name}.json": f"benchmarks/experiments/{name}.json" for name in (
        "vllm_latency_hybrid_v1", "vllm_latency_central_controlled_v1", "vllm_latency_central_realistic_v1")},
}


def secret_like(path: Path) -> bool:
    name = path.name.lower()
    return (name == ".env" or name.startswith(".env.")
            or any(part.lower() in {".aws", ".ssh", ".config", "secrets", "credentials"} for part in path.parts)
            or name.endswith((".key", ".pem", ".p12", ".pfx"))
            or name in {"credentials.json", "token.json", "secrets.json", ".modal.toml"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_verified(source: Path, destination: Path, *, overwrite: bool = False) -> str:
    if secret_like(source):
        raise ValueError(f"Secret-like file refused: {source}")
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"Regular source file required: {source}")
    if destination.is_symlink():
        raise ValueError(f"Destination symlink refused: {destination}")
    digest = sha256(source)
    print(f"{source} -> {destination} ({source.stat().st_size} bytes; sha256={digest})")
    if destination.exists():
        if not destination.is_file():
            raise ValueError(f"Destination is not a file: {destination}")
        if sha256(destination) == digest:
            return "unchanged"
        if not overwrite:
            raise FileExistsError(f"Hash mismatch; add --overwrite to replace: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.is_symlink():
        raise ValueError(f"Temporary destination symlink refused: {temporary}")
    try:
        shutil.copyfile(source, temporary)
        if sha256(temporary) != digest:
            raise IOError(f"Copy verification failed: {destination}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return "copied"


def transfer_tree(source: Path, destination: Path, *, overwrite: bool = False) -> int:
    if not source.is_dir() or source.is_symlink():
        raise FileNotFoundError(f"Regular directory required: {source}")
    if destination.is_symlink():
        raise ValueError(f"Destination symlink refused: {destination}")
    paths = sorted(source.rglob("*"))
    # Validate the whole tree before copying any file.
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"Symlink refused: {path}")
        if path.is_file() and secret_like(path):
            raise ValueError(f"Secret-like file refused: {path}")
        target = destination / path.relative_to(source)
        if not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError(f"Destination escapes transfer root: {target}")
    count = 0
    for path in paths:
        if path.is_file():
            copy_verified(path, destination / path.relative_to(source), overwrite=overwrite)
            count += 1
    return count


def verify_tree(source: Path, destination: Path) -> None:
    if source.is_symlink() or destination.is_symlink() or not source.is_dir() or not destination.is_dir():
        raise FileNotFoundError("Regular local and Drive result directories are required")
    def files(root):
        result = {}
        for path in root.rglob("*"):
            if path.is_symlink() or secret_like(path):
                raise ValueError(f"Unsafe result asset: {path}")
            if path.is_file():
                result[path.relative_to(root).as_posix()] = sha256(path)
        return result
    if files(source) != files(destination):
        raise ValueError("Benchmark results path/SHA256 mismatch")
    print("Benchmark results verified by relative path and SHA256")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "push-datasets", "pull-datasets", "verify", "push-adapter", "pull-adapter",
                 "push-evaluation", "pull-evaluation", "push-benchmark-datasets", "pull-benchmark-datasets",
                 "verify-benchmark-datasets", "push-benchmark-results", "pull-benchmark-results", "verify-benchmark-results"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--drive-root", required=True, type=Path)
        if name.startswith(("push", "pull")):
            cmd.add_argument("--overwrite", action="store_true")
        if name.endswith(("adapter", "evaluation", "results")):
            cmd.add_argument("--local-path", required=True, type=Path)
            cmd.add_argument("--name", required=True, help="One run directory name, without path separators")
    return parser


def run(args: argparse.Namespace) -> int:
    drive = args.drive_root.expanduser().resolve()
    if args.command == "init":
        drive.mkdir(parents=True, exist_ok=True)
        for item in LAYOUT:
            (drive / item).mkdir(parents=True, exist_ok=True)
        print(f"Initialized {drive}")
        return 0
    if not drive.is_dir():
        raise FileNotFoundError(f"Mounted drive root not found: {drive}")
    if args.command in ("push-datasets", "pull-datasets", "verify", "push-benchmark-datasets", "pull-benchmark-datasets", "verify-benchmark-datasets"):
        benchmark = "benchmark" in args.command
        pairs = list(BENCHMARK_ASSETS.items()) if benchmark else [*DATASETS.items(),
                 ("evaluation/datasets/v1_silver/splits/v1_seed42/split_manifest.json", SFT_SPLIT_MANIFEST)]
        for local, remote in pairs:
            left, right = ROOT / local, drive / remote
            if not right.resolve().is_relative_to(drive):
                raise ValueError(f"Drive asset escapes mount root: {right}")
            if args.command.startswith("verify"):
                if not left.is_file() or not right.is_file():
                    raise FileNotFoundError(f"Dataset missing on one side: {local} / {remote}")
                if sha256(left) != sha256(right):
                    raise ValueError(f"Dataset SHA mismatch: {remote}")
                path = right
                print(f"{path} ({path.stat().st_size} bytes; sha256={sha256(path)})")
            else:
                source, dest = (left, right) if args.command.startswith("push") else (right, left)
                copy_verified(source, dest, overwrite=args.overwrite)
        return 0
    name = args.name
    if name in ("", ".", "..") or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("--name must be one directory name")
    kind = ("benchmarks/results" if args.command.endswith("results") else
            "models/qwen3_4b_sft_v1" if args.command.endswith("adapter") else "evaluation/reports")
    remote = drive / kind / name
    if not remote.resolve().is_relative_to(drive):
        raise ValueError("Drive run directory escapes mount root")
    if args.command == "verify-benchmark-results":
        verify_tree(args.local_path, remote)
        return 0
    source, dest = (args.local_path, remote) if args.command.startswith("push") else (remote, args.local_path)
    print(f"Transferred {transfer_tree(source, dest, overwrite=args.overwrite)} files")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except (ValueError, FileNotFoundError, FileExistsError, IOError) as exc:
        print(f"[drive] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
