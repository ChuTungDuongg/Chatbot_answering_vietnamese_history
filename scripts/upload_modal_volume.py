from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.artifact_contract import sha256_file


@dataclass(frozen=True)
class Upload:
    source: Path
    remote: str


@dataclass(frozen=True)
class LocalFile:
    source: Path
    remote: str
    size: int
    sha256: str


@dataclass(frozen=True)
class RemoteFile:
    remote: str
    size: int
    sha256: str | None = None


@dataclass(frozen=True)
class SyncPlan:
    unchanged: list[LocalFile]
    upload: list[LocalFile]
    replace: list[LocalFile]
    delete_stale: list[RemoteFile]

    def has_adapter_weight_replacement(self) -> bool:
        return any(item.remote.endswith("/adapter_model.safetensors") for item in self.replace)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and upload the three 4B role adapters, Central 8B adapter, and retrieval artifacts."
    )
    parser.add_argument("--volume", required=True, help="Existing Modal Volume name.")
    parser.add_argument("--history-model", help="Optional legacy Qwen2.5 benchmark model directory.")
    parser.add_argument("--history-adapter", help="Fresh Qwen3 History Answerer adapter directory.")
    parser.add_argument("--research-agent", help="Research Agent LoRA adapter directory.")
    parser.add_argument("--evidence-agent", help="Evidence Agent LoRA adapter directory.")
    parser.add_argument("--central-agent", help="Qwen3-8B Central Agent LoRA adapter directory.")
    parser.add_argument("--retrieval-dir", help="Directory containing faiss/ and bm25s_index/.")
    parser.add_argument("--corpus", help="Enriched corpus JSONL file.")
    parser.add_argument("--config-dir", help="Directory containing inference/runtime config files.")
    parser.add_argument("--manifest", help="Deployment manifest JSON file.")
    parser.add_argument("--local-dir", help="Upload a prebuilt artifact tree as-is (legacy/full-bundle mode).")
    parser.add_argument("--remote-dir", default="/", help="Remote root for --local-dir mode.")
    parser.add_argument("--exact-sync", action="store_true", help="Safely sync a prebuilt canonical tree against managed Modal files.")
    parser.add_argument("--remote-inventory-json", help="Precomputed Modal inventory JSON for exact-sync tests/offline planning.")
    parser.add_argument(
        "--allow-replace-adapter-weights",
        action="store_true",
        help="Allow replacing adapter_model.safetensors when the exact-sync plan detects a hash mismatch.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands after validating paths.")
    return parser


def _validated(path: str, *, label: str, directory: bool | None = None) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if directory is True and not resolved.is_dir():
        raise NotADirectoryError(f"{label} must be a directory: {resolved}")
    if directory is False and not resolved.is_file():
        raise FileNotFoundError(f"{label} must be a file: {resolved}")
    return resolved


def collect_uploads(args: argparse.Namespace) -> list[Upload]:
    if args.local_dir:
        component_flags = (
            args.history_model,
            args.history_adapter,
            args.research_agent,
            args.evidence_agent,
            getattr(args, "central_agent", None),
            args.retrieval_dir,
            args.corpus,
            args.config_dir,
            args.manifest,
        )
        if any(component_flags):
            raise ValueError("--local-dir cannot be combined with component upload flags.")
        bundle = _validated(args.local_dir, label="local bundle", directory=True)
        remote_root = args.remote_dir.rstrip("/")
        children = sorted(bundle.iterdir(), key=lambda path: path.name)
        if not children:
            raise ValueError(f"Local bundle is empty: {bundle}")
        return [
            Upload(child, f"{remote_root}/{child.name}" if remote_root else f"/{child.name}")
            for child in children
        ]

    required = {
        "--history-adapter": args.history_adapter,
        "--research-agent": args.research_agent,
        "--evidence-agent": args.evidence_agent,
        "--central-agent": args.central_agent,
        "--retrieval-dir": args.retrieval_dir,
        "--corpus": args.corpus,
        "--config-dir": args.config_dir,
        "--manifest": args.manifest,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"Missing component upload flags: {', '.join(missing)}")

    uploads = [
        Upload(_validated(args.history_adapter, label="history adapter", directory=True), "/adapters/history"),
        Upload(_validated(args.research_agent, label="research adapter", directory=True), "/adapters/research"),
        Upload(_validated(args.evidence_agent, label="evidence adapter", directory=True), "/adapters/evidence"),
        Upload(_validated(args.central_agent, label="central adapter", directory=True), "/adapters/central"),
        Upload(_validated(args.retrieval_dir, label="retrieval directory", directory=True), "/retrieval"),
        Upload(_validated(args.corpus, label="corpus", directory=False), "/corpus/vn_history_rag_chunks_enriched.jsonl"),
        Upload(_validated(args.config_dir, label="config directory", directory=True), "/config"),
        Upload(_validated(args.manifest, label="manifest", directory=False), "/manifest.json"),
    ]
    success_marker = Path(args.manifest).expanduser().resolve().with_name("EXPORT_SUCCESS.txt")
    if success_marker.is_file():
        uploads.append(Upload(success_marker, "/EXPORT_SUCCESS.txt"))
    if args.history_model:
        uploads.append(
            Upload(_validated(args.history_model, label="legacy history model", directory=True), "/legacy/qwen25_history/model")
        )
    return uploads


def _remote_join(root: str, relative: Path) -> str:
    root = root.strip() or "/"
    prefix = "/" if root == "/" else "/" + root.strip("/")
    suffix = str(relative).replace("\\", "/")
    return f"{prefix.rstrip('/')}/{suffix}" if suffix else prefix


def collect_local_files(local_dir: str | Path, remote_dir: str = "/") -> list[LocalFile]:
    root = _validated(str(local_dir), label="local canonical bundle", directory=True)
    files: list[LocalFile] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        remote = _remote_join(remote_dir, path.relative_to(root))
        files.append(LocalFile(source=path, remote=remote, size=path.stat().st_size, sha256=sha256_file(path)))
    return files


def _coerce_remote_inventory(payload: dict[str, Any]) -> dict[str, RemoteFile]:
    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("Remote inventory JSON must contain a files list.")
    output: dict[str, RemoteFile] = {}
    for item in raw_files:
        if not isinstance(item, dict):
            continue
        remote = "/" + str(item.get("path") or item.get("remote") or "").strip().lstrip("/")
        if remote == "/":
            continue
        size = int(item.get("size") or 0)
        digest = item.get("sha256")
        output[remote] = RemoteFile(remote=remote, size=size, sha256=str(digest) if digest else None)
    return output


def load_remote_inventory(path: str | Path) -> dict[str, RemoteFile]:
    return _coerce_remote_inventory(json.loads(Path(path).read_text(encoding="utf-8")))


def fetch_remote_inventory() -> dict[str, RemoteFile]:
    command = ["modal", "run", "scripts/modal_artifact_sanity.py", "--inventory-json-output"]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(completed.stderr or completed.stdout or "Could not fetch Modal remote inventory.")
    marker_start = "REMOTE_INVENTORY_JSON_BEGIN"
    marker_end = "REMOTE_INVENTORY_JSON_END"
    text = completed.stdout
    try:
        start = text.index(marker_start) + len(marker_start)
        end = text.index(marker_end, start)
    except ValueError as exc:
        raise RuntimeError("Modal inventory output did not contain JSON markers.") from exc
    return _coerce_remote_inventory(json.loads(text[start:end].strip()))


def build_sync_plan(
    local_files: list[LocalFile],
    remote_files: dict[str, RemoteFile],
    *,
    remote_dir: str = "/",
) -> SyncPlan:
    local_by_remote = {item.remote: item for item in local_files}
    remote_prefix = "/" if remote_dir.strip() in {"", "/"} else "/" + remote_dir.strip("/").rstrip("/")
    managed_remote = {
        path: item
        for path, item in remote_files.items()
        if remote_prefix == "/" or path == remote_prefix or path.startswith(remote_prefix + "/")
    }
    unchanged: list[LocalFile] = []
    upload: list[LocalFile] = []
    replace: list[LocalFile] = []
    for local in local_files:
        remote = managed_remote.get(local.remote)
        if remote is None:
            upload.append(local)
        elif remote.sha256 == local.sha256:
            unchanged.append(local)
        else:
            replace.append(local)
    delete_stale = [
        remote
        for path, remote in sorted(managed_remote.items())
        if path not in local_by_remote
    ]
    return SyncPlan(
        unchanged=unchanged,
        upload=upload,
        replace=replace,
        delete_stale=delete_stale,
    )


def print_sync_plan(plan: SyncPlan) -> None:
    print("MODAL_MUTATION_PLAN")
    for label, items in (
        ("UNCHANGED", plan.unchanged),
        ("UPLOAD", plan.upload),
        ("REPLACE", plan.replace),
        ("DELETE_STALE", plan.delete_stale),
    ):
        print(f"{label}:")
        if not items:
            print("  (none)")
            continue
        for item in items:
            remote = item.remote if hasattr(item, "remote") else str(item)
            print(f"  {remote}")


def execute_sync_plan(
    *,
    volume: str,
    plan: SyncPlan,
    dry_run: bool,
    allow_replace_adapter_weights: bool,
) -> int:
    print_sync_plan(plan)
    if plan.has_adapter_weight_replacement() and not allow_replace_adapter_weights:
        print("STOP: adapter_model.safetensors replacement detected; rerun only with explicit --allow-replace-adapter-weights.")
        return 2
    commands: list[list[str]] = []
    for item in [*plan.upload, *plan.replace]:
        commands.append(["modal", "volume", "put", "--force", volume, str(item.source), item.remote])
    for item in plan.delete_stale:
        commands.append(["modal", "volume", "rm", volume, item.remote])
    for command in commands:
        print(subprocess.list2cmdline(command))
        if not dry_run:
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                return completed.returncode
    if not dry_run:
        print("Verifying managed remote hashes after mutation...")
        fetch_remote_inventory()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.exact_sync:
        if not args.local_dir:
            raise ValueError("--exact-sync requires --local-dir.")
        local_files = collect_local_files(args.local_dir, args.remote_dir)
        remote_files = (
            load_remote_inventory(args.remote_inventory_json)
            if args.remote_inventory_json
            else fetch_remote_inventory()
        )
        plan = build_sync_plan(local_files, remote_files, remote_dir=args.remote_dir)
        return execute_sync_plan(
            volume=args.volume,
            plan=plan,
            dry_run=args.dry_run,
            allow_replace_adapter_weights=args.allow_replace_adapter_weights,
        )
    uploads = collect_uploads(args)
    for upload in uploads:
        command = ["modal", "volume", "put", "--force", args.volume, str(upload.source), upload.remote]
        print(subprocess.list2cmdline(command))
        if not args.dry_run:
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
