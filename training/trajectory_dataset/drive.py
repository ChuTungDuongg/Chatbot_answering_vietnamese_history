from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


CORPUS_FILENAME = "vn_history_rag_chunks_enriched.jsonl"
DEFAULT_CORPUS_PATH = Path("artifacts/vn_history_deployment/corpus")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def mount_google_drive(mount_point: str | Path, *, module: Any | None = None) -> Path:
    point = Path(mount_point)
    if module is None:
        try:
            module = importlib.import_module("google.colab")
        except ImportError as exc:
            raise RuntimeError(
                "--mount-drive was requested outside Google Colab. Run without --mount-drive "
                "or execute this command in a Colab runtime with google.colab available."
            ) from exc
    drive = getattr(module, "drive", None)
    if drive is None or not callable(getattr(drive, "mount", None)):
        raise RuntimeError("google.colab.drive.mount is unavailable in this environment")
    drive.mount(str(point))
    return point


def resolve_corpus_path(
    corpus_path: str | Path | None = None,
    *,
    drive_corpus_path: str | Path | None = None,
    mount_drive: bool = False,
    drive_mount_point: str | Path = "/content/drive",
    colab_module: Any | None = None,
    must_exist: bool = True,
) -> Path:
    if mount_drive:
        mount_google_drive(drive_mount_point, module=colab_module)
    selected = drive_corpus_path if drive_corpus_path is not None else corpus_path
    if selected is None:
        selected = DEFAULT_CORPUS_PATH
    path = Path(selected).expanduser()
    if not path.is_absolute():
        path = repo_root() / path
    path = path.resolve()
    if path.is_dir():
        path = path / CORPUS_FILENAME
    if must_exist and not path.is_file():
        raise FileNotFoundError(
            f"Enriched corpus not found at {path}. Pass --corpus-path to the corpus directory/file; "
            "in Colab, mount Drive explicitly and pass --drive-corpus-path or --corpus-path."
        )
    return path
