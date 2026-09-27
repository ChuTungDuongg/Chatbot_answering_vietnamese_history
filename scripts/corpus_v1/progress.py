"""Flushed, line-oriented Corpus V1 build progress on stderr."""

from __future__ import annotations

from pathlib import Path
import math
import sys
import time
from typing import Any, TextIO


def duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class ProgressReporter:
    """Observe committed work without participating in corpus identity or writes."""

    def __init__(self, *, enabled: bool = True, stream: TextIO | None = None) -> None:
        self.enabled = enabled
        self.stream = stream if stream is not None else sys.stderr
        self.started = time.monotonic()
        self.session_rows = 0
        self.targets: dict[str, int | None] = {}
        self.context = ""

    def line(self, message: str) -> None:
        if self.enabled:
            try:
                print(message, file=self.stream, flush=True)
            except (OSError, UnicodeError, ValueError):
                self.enabled = False  # A closed/unsupported stream must not stop corpus construction.

    def phase(self, message: str) -> None:
        self.line(f"[Corpus V1] {message}")

    def startup(self, cfg: dict[str, Any], root: Path, scratch: Path, *, resume: bool) -> None:
        limit = cfg["record_limit_per_split"]
        self.targets = {}
        for split in cfg["source_splits"]:
            expected_rows = cfg["expected_split_sizes"].get(split)
            self.targets[split] = (min(limit, expected_rows) if limit is not None and expected_rows is not None
                                   else limit if limit is not None else expected_rows)
        expected = ",".join(f"{split}:{cfg['expected_split_sizes'].get(split, 'unknown')}"
                            for split in cfg["source_splits"])
        self.phase(f"dataset={cfg['dataset_id']} config={cfg['dataset_config']} "
                   f"requested_revision={cfg['requested_revision']} "
                   f"resolved_revision_sha={cfg['resolved_revision_sha']}")
        self.phase(f"scope={cfg['build_scope']} splits={','.join(cfg['source_splits'])} "
                   f"expected_split_sizes={expected} limit_per_split={limit if limit is not None else 'none'}")
        self.phase(f"shard_size={cfg['shard_size']} chunk_tokens={cfg['chunk_tokens']} "
                   f"chunk_overlap={cfg['chunk_overlap']} tokenizer_id={cfg['tokenizer_id']} "
                   f"tokenizer_revision_sha={cfg['tokenizer_revision_sha'] or 'unavailable'}")
        self.phase(f"output={root} scratch={scratch} cache={cfg['cache_dir'] or 'default'} "
                   f"resume={str(resume).lower()}")

    def resume_validating(self) -> None:
        self.line("[resume] validating completed shards...")

    def resume_split(self, split: str, shards: int, rows: int) -> None:
        target = self.targets.get(split)
        self.line(f"[resume] {split}: completed_shards={shards} rows_to_skip={rows} "
                  f"parts={'1-' + str(shards) if shards else 'none'} "
                  f"target_reached={str(target is not None and rows >= target).lower()}")

    def orphan_cleared(self, path: Path) -> None:
        self.line(f"[resume] cleared_uncommitted={path}")

    def dedup(self, *, restored: bool) -> None:
        self.line("[resume] global dedup state restored" if restored else
                  "[resume] rebuilding global dedup state...")

    def split_start(self, split: str, observed: int, expected: int | None, *, resume: bool) -> None:
        verb = "resuming" if resume and observed else "starting"
        self.line(f"[{split}] {verb} row={observed} expected={expected if expected is not None else 'unknown'}")

    def split_done(self, split: str, observed: int, expected: int | None,
                   limit: int | None, *, exhausted: bool) -> None:
        if limit is not None:
            status = "pilot_limit_reached" if observed >= limit else "pilot_source_exhausted"
            complete = False
        else:
            complete = observed == expected if expected is not None else exhausted
            status = "finished" if complete else "incomplete"
        self.line(f"[{split}] {status} observed={observed} "
                  f"expected={expected if expected is not None else 'unknown'} "
                  f"source_complete={str(complete).lower()}")

    def shard_start(self, split: str, number: int, start: int, planned: int) -> float:
        self.context = f"split={split} shard={number} rows={start}-{start + planned - 1}"
        self.line(f"[{split}] shard={number} starting rows={start}-{start + planned - 1} "
                  f"expected_rows={planned}")
        return time.monotonic()

    def shard_done(self, split: str, number: int, counts: dict[str, Any],
                   shard_started: float, observed: dict[str, int]) -> None:
        elapsed = max(time.monotonic() - shard_started, 0.000001)
        self.session_rows += counts["source_count"]
        decisions = counts["filter_counts"]
        self.line(f"[{split}] shard={number} completed processed={counts['source_count']} "
                  f"keep={decisions['KEEP']} review={decisions['REVIEW']} drop={decisions['DROP']} "
                  f"documents={counts['document_count']} chunks={counts['chunk_count']} "
                  f"duplicate_documents={counts['duplicate_document_count']} "
                  f"duplicate_chunks={counts['duplicate_chunk_count']} "
                  f"elapsed={duration(elapsed)} rate={counts['source_count'] / elapsed:.1f} rows/s")
        self.overall(observed, split=split)
        self.context = ""

    def overall(self, observed: dict[str, int], *, split: str | None = None) -> None:
        if split:
            target = self.targets[split]
            if target:
                self.line(f"[{split}] progress={observed[split]}/{target} "
                          f"percent={100 * observed[split] / target:.1f}%")
            else:
                self.line(f"[{split}] progress={observed[split]}/unknown percent=unknown")
        total = sum(observed.values())
        expected = sum(self.targets.values()) if all(value is not None for value in self.targets.values()) else None
        elapsed = max(time.monotonic() - self.started, 0.000001)
        rate = self.session_rows / elapsed
        rate_text = f"{rate:.1f}" if self.session_rows else "unknown"
        percent = f"{100 * total / expected:.1f}%" if expected else "unknown"
        eta = (duration(math.ceil((expected - total) / rate)) if expected is not None and rate > 0 and total < expected
               else "00:00:00" if expected is not None and total >= expected else "unknown")
        self.line(f"[overall] progress={total}/{expected if expected is not None else 'unknown'} "
                  f"percent={percent} elapsed={duration(elapsed)} rate={rate_text} rows/s eta={eta}")

    def finalize(self, action: str, *, completed: bool = False) -> None:
        self.line(f"[finalize] {action} {'completed' if completed else 'starting'}")

    def done(self, manifest: dict[str, Any], root: Path, *, already_complete: bool = False) -> None:
        elapsed = max(time.monotonic() - self.started, 0.000001)
        counts = manifest["filter_counts"]
        average_rate = f"{self.session_rows / elapsed:.1f}" if self.session_rows else "unknown"
        self.line(f"[done] Corpus V1 build complete scope={manifest['build_scope']} "
                  f"source_rows={sum(manifest['observed_split_sizes'].values())} "
                  f"documents={manifest['document_count']} chunks={manifest['chunk_count']} "
                  f"keep={counts['KEEP']} review={counts['REVIEW']} drop={counts['DROP']} "
                  f"duplicate_documents={manifest['duplicate_document_count']} "
                  f"duplicate_chunks={manifest['duplicate_chunk_count']} "
                  f"elapsed={duration(elapsed)} average_rate={average_rate} rows/s "
                  f"output={root} already_complete={str(already_complete).lower()}")

    def error(self, output: Path, exc: Exception) -> None:
        detail = f" {self.context}" if self.context else ""
        try:
            print(f"[error] output={output}{detail} {type(exc).__name__}: {exc}",
                  file=self.stream, flush=True)
        except (OSError, UnicodeError, ValueError):
            pass  # Preserve the original build exception.
