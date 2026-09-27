"""Atomic stage files and verifiable checkpoint metadata."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def stage(root: Path, name: str, fingerprint: str, resume: bool,
          produce: Callable[[Path], dict[str, Any]]) -> dict[str, Any]:
    path = root / "intermediate" / (name + ".jsonl")
    manifest_path = root / "intermediate" / (name + ".manifest.json")
    if path.exists() or manifest_path.exists():
        if not resume:
            raise FileExistsError(f"Stage {name} exists; use --resume or a fresh output directory")
        if not path.exists() or not manifest_path.exists():
            raise RuntimeError(f"Incomplete stage {name}; inspect/remove partial stage before resume")
        meta = json.loads(manifest_path.read_text(encoding="utf-8"))
        if meta.get("config_fingerprint") != fingerprint or meta.get("sha256") != digest_file(path):
            raise RuntimeError(f"Stage {name} config or SHA256 mismatch; use a new output directory")
        return meta
    temp = path.with_name(path.name + ".partial")
    if temp.exists():
        raise RuntimeError(f"Incomplete temporary stage {temp}; inspect/remove it before resume")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        counts = produce(temp)
        os.replace(temp, path)
        meta = {"stage": name, "config_fingerprint": fingerprint,
                "sha256": digest_file(path), "size_bytes": path.stat().st_size, **counts}
        atomic_json(manifest_path, meta)
        return meta
    except Exception:
        # Keep partial output for diagnosis; it can never be mistaken for a checkpoint.
        raise
