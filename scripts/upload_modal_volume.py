from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Upload:
    source: Path
    remote: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and upload three Qwen3 role adapters plus retrieval artifacts to Modal Volume."
    )
    parser.add_argument("--volume", required=True, help="Existing Modal Volume name.")
    parser.add_argument("--history-model", help="Optional legacy Qwen2.5 benchmark model directory.")
    parser.add_argument("--history-adapter", help="Fresh Qwen3 History Answerer adapter directory.")
    parser.add_argument("--research-agent", help="Research Agent LoRA adapter directory.")
    parser.add_argument("--evidence-agent", help="Evidence Agent LoRA adapter directory.")
    parser.add_argument("--retrieval-dir", help="Directory containing faiss/ and bm25s_index/.")
    parser.add_argument("--corpus", help="Enriched corpus JSONL file.")
    parser.add_argument("--config-dir", help="Directory containing inference/runtime config files.")
    parser.add_argument("--manifest", help="Deployment manifest JSON file.")
    parser.add_argument("--local-dir", help="Upload a prebuilt artifact tree as-is (legacy/full-bundle mode).")
    parser.add_argument("--remote-dir", default="/", help="Remote root for --local-dir mode.")
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
