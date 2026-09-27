"""Durable per-source-block checkpoints with manifest-last commit semantics."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable

from scripts.corpus_v1.provenance import atomic_json, digest_file


PART = re.compile(r"part-(\d{6})\.manifest\.json\Z")
KINDS = ("records", "documents", "chunks")


def shard_directory(root: Path, split: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", split):
        raise ValueError(f"Unsafe source split name: {split!r}")
    return root / "intermediate" / "shards" / split


def shard_file(root: Path, split: str, part: int, kind: str) -> Path:
    return shard_directory(root, split) / f"part-{part:06d}.{kind}.jsonl"


def completed_shards(root: Path, split: str, fingerprint: str, *, resume: bool,
                     on_orphan_cleared: Callable[[Path], None] | None = None) -> list[dict[str, Any]]:
    directory = shard_directory(root, split)
    if not directory.exists():
        return []
    manifests = sorted(directory.glob("part-*.manifest.json"))
    result = []
    expected_start = 0
    for part, path in enumerate(manifests):
        match = PART.fullmatch(path.name)
        if not match or int(match.group(1)) != part:
            raise RuntimeError(f"Noncontiguous shard manifests in {directory}")
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Corrupt completed shard manifest: {path}") from exc
        if not isinstance(meta, dict) or not isinstance(meta.get("source_count"), int) or meta["source_count"] < 1:
            raise RuntimeError(f"Corrupt completed shard manifest: {path}")
        if meta.get("config_fingerprint") != fingerprint or meta.get("source_split") != split:
            raise RuntimeError(f"Shard configuration mismatch: {path}")
        if meta.get("start_index") != expected_start or meta.get("end_index") != expected_start + meta.get("source_count", -1):
            raise RuntimeError(f"Shard source row range mismatch: {path}")
        for kind in KINDS:
            file = shard_file(root, split, part, kind)
            if not file.is_file() or meta.get("hashes", {}).get(kind) != digest_file(file):
                raise RuntimeError(f"Corrupt completed shard: {file}")
        expected_start = meta["end_index"]
        result.append(meta)
    committed = {shard_file(root, split, part, kind) for part in range(len(result)) for kind in KINDS}
    orphaned = [path for path in directory.iterdir() if path.is_file() and
                (path.name.endswith(".partial") or
                 (path.name.startswith("part-") and path.name.endswith(".jsonl") and path not in committed))]
    if orphaned and not resume:
        raise RuntimeError(f"Incomplete shard files in {directory}; use --resume")
    for path in orphaned:
        path.unlink()  # no manifest means not committed; rebuild only this shard
        if on_orphan_cleared:
            on_orphan_cleared(path)
    return result


def commit_shard(root: Path, split: str, part: int, fingerprint: str,
                 scratch_files: dict[str, Path], counts: dict[str, Any], start_index: int) -> dict[str, Any]:
    directory = shard_directory(root, split)
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    sizes = {}
    for kind in KINDS:
        source = scratch_files[kind]
        destination = shard_file(root, split, part, kind)
        temporary = destination.with_name(destination.name + ".partial")
        with source.open("rb") as reader, temporary.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        expected = digest_file(source)
        if digest_file(temporary) != expected:
            raise RuntimeError(f"Shard copy checksum mismatch: {temporary}")
        os.replace(temporary, destination)
        hashes[kind] = expected
        sizes[kind] = destination.stat().st_size
    meta = {"source_split": split, "part": part, "start_index": start_index,
            "end_index": start_index + counts["source_count"],
            "config_fingerprint": fingerprint, "hashes": hashes, "size_bytes": sizes, **counts}
    atomic_json(directory / f"part-{part:06d}.manifest.json", meta)
    return meta
