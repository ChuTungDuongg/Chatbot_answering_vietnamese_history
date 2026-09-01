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

from app.artifact_contract import (
    INFERENCE_CONFIG_RELATIVE_PATH,
    LOCK_FILENAME,
    MODEL_REGISTRY_RELATIVE_PATH,
    load_json,
    sha256_file,
    validate_artifact_lock,
)


CANONICAL_REQUIRED_FILES = (
    LOCK_FILENAME,
    "manifest.json",
    INFERENCE_CONFIG_RELATIVE_PATH,
    MODEL_REGISTRY_RELATIVE_PATH,
)


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
    parser.add_argument("--history-model", help=argparse.SUPPRESS)
    parser.add_argument("--history-adapter", help=argparse.SUPPRESS)
    parser.add_argument("--research-agent", help=argparse.SUPPRESS)
    parser.add_argument("--evidence-agent", help=argparse.SUPPRESS)
    parser.add_argument("--central-agent", help=argparse.SUPPRESS)
    parser.add_argument("--retrieval-dir", help=argparse.SUPPRESS)
    parser.add_argument("--corpus", help=argparse.SUPPRESS)
    parser.add_argument("--config-dir", help=argparse.SUPPRESS)
    parser.add_argument("--manifest", help=argparse.SUPPRESS)
    parser.add_argument("--local-dir", help="Validated canonical deployment bundle from export_artifacts.py.")
    parser.add_argument("--remote-dir", default="/", help="Canonical Volume root; production requires '/'.")
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


def _component_values(args: argparse.Namespace) -> tuple[Any, ...]:
    return tuple(getattr(args, name, None) for name in (
        "history_model", "history_adapter", "research_agent", "evidence_agent",
        "central_agent", "retrieval_dir", "corpus", "config_dir", "manifest",
    ))


def validate_canonical_bundle(local_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    bundle = _validated(str(local_dir), label="canonical deployment bundle", directory=True)
    missing = [relative for relative in CANONICAL_REQUIRED_FILES if not (bundle / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            "Canonical deployment bundle is incomplete; missing:\n"
            + "\n".join(f"- {item}" for item in missing)
        )
    lock = validate_artifact_lock(bundle)
    manifest = load_json(bundle / "manifest.json")
    if manifest.get("deployment_id") != lock.get("deployment_id"):
        raise RuntimeError(
            "Canonical bundle deployment_id mismatch: "
            f"manifest={manifest.get('deployment_id')!r} lock={lock.get('deployment_id')!r}"
        )
    return bundle, lock


def _mutation_priority(remote: str) -> tuple[int, str]:
    normalized = "/" + remote.strip().lstrip("/")
    if normalized == f"/{LOCK_FILENAME}":
        return 2, normalized
    if normalized == "/manifest.json":
        return 1, normalized
    return 0, normalized


def _bundle_uploads(bundle: Path) -> list[Upload]:
    uploads = [Upload(child, f"/{child.name}") for child in bundle.iterdir()]
    return sorted(uploads, key=lambda item: _mutation_priority(item.remote))


def collect_uploads(args: argparse.Namespace) -> list[Upload]:
    if any(_component_values(args)):
        raise ValueError(
            "Unsafe component upload is disabled. Build one canonical bundle with "
            "training.scripts.export_artifacts, then use --local-dir and preferably --exact-sync."
        )
    if not getattr(args, "local_dir", None):
        raise ValueError("--local-dir is required and must point to a validated canonical deployment bundle.")
    if str(getattr(args, "remote_dir", "/")).strip() not in {"", "/"}:
        raise ValueError("Canonical production bundle must be uploaded to Volume root '/'.")
    bundle, _ = validate_canonical_bundle(args.local_dir)
    return _bundle_uploads(bundle)


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
    if any(item.remote == f"/{LOCK_FILENAME}" for item in plan.delete_stale):
        raise RuntimeError("Refusing to delete the existing artifact_lock.json before a coherent replacement exists.")
    changed = sorted([*plan.upload, *plan.replace], key=lambda item: _mutation_priority(item.remote))
    ordinary = [item for item in changed if _mutation_priority(item.remote)[0] == 0]
    manifest = [item for item in changed if _mutation_priority(item.remote)[0] == 1]
    lock = [item for item in changed if _mutation_priority(item.remote)[0] == 2]
    commands: list[tuple[str, list[str]]] = []
    for item in ordinary:
        commands.append((item.remote, ["modal", "volume", "put", "--force", volume, str(item.source), item.remote]))
    for item in sorted(plan.delete_stale, key=lambda value: value.remote):
        commands.append((item.remote, ["modal", "volume", "rm", volume, item.remote]))
    for item in [*manifest, *lock]:
        commands.append((item.remote, ["modal", "volume", "put", "--force", volume, str(item.source), item.remote]))
    for remote, command in commands:
        print(subprocess.list2cmdline(command))
        if not dry_run:
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                print(f"MODAL_MUTATION_FAILED remote={remote} exit_code={completed.returncode}")
                return completed.returncode
    if not dry_run:
        print("Verifying managed remote hashes after mutation...")
        remote_after = fetch_remote_inventory()
        expected = {item.remote: item for item in [*plan.unchanged, *plan.upload, *plan.replace]}
        mismatches = [
            remote
            for remote, local in expected.items()
            if remote not in remote_after or remote_after[remote].sha256 != local.sha256
        ]
        stale_remaining = [item.remote for item in plan.delete_stale if item.remote in remote_after]
        if mismatches or stale_remaining:
            print(f"MODAL_SYNC_VERIFICATION_FAILED mismatches={mismatches} stale={stale_remaining}")
            return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if any(_component_values(args)):
        raise ValueError(
            "Unsafe component upload is disabled. Export and validate one canonical bundle, "
            "then upload it with --local-dir."
        )
    if not args.local_dir:
        raise ValueError("--local-dir is required and must point to a validated canonical deployment bundle.")
    bundle, lock = validate_canonical_bundle(args.local_dir)
    print(f"LOCAL_ARTIFACT_VALID deployment_id={lock['deployment_id']} root={bundle}")
    if args.exact_sync:
        if str(args.remote_dir).strip() not in {"", "/"}:
            raise ValueError("Canonical production exact-sync requires --remote-dir /.")
        local_files = collect_local_files(bundle, "/")
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
    uploads = _bundle_uploads(bundle)
    for upload in uploads:
        command = ["modal", "volume", "put", "--force", args.volume, str(upload.source), upload.remote]
        print(subprocess.list2cmdline(command))
        if not args.dry_run:
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                print(f"MODAL_UPLOAD_FAILED remote={upload.remote} exit_code={completed.returncode}")
                return completed.returncode
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as exc:
        print(f"UPLOAD_ABORTED_BEFORE_MUTATION\n{exc}", file=sys.stderr)
        raise SystemExit(2) from None
