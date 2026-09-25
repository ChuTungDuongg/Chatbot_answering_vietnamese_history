"""Read-only provenance and hardware metadata for benchmark runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from app.config import settings


def _hash_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_index(paths: list[Path]) -> str | None:
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(item for item in path.rglob("*") if item.is_file())
        elif path.is_file():
            files.append(path)
        else:
            return None
    digest = hashlib.sha256()
    for path in sorted(set(files), key=lambda item: item.as_posix()):
        digest.update(path.relative_to(settings.artifact_root).as_posix().encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _hardware() -> dict[str, Any]:
    try:
        import psutil

        ram_bytes = psutil.virtual_memory().total
    except ImportError:
        if os.name == "nt":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong),
                            ("total_physical", ctypes.c_ulonglong),
                            ("available_physical", ctypes.c_ulonglong),
                            ("total_page_file", ctypes.c_ulonglong),
                            ("available_page_file", ctypes.c_ulonglong),
                            ("total_virtual", ctypes.c_ulonglong),
                            ("available_virtual", ctypes.c_ulonglong),
                            ("available_extended_virtual", ctypes.c_ulonglong)]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            ram_bytes = status.total_physical if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) else None
        else:
            try:
                ram_bytes = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            except (AttributeError, ValueError, OSError):
                ram_bytes = None
    gpu_model = gpu_memory_bytes = cuda_version = None
    if settings.device == "cuda":
        import torch

        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            gpu_model = properties.name
            gpu_memory_bytes = properties.total_memory
            cuda_version = torch.version.cuda
    return {"os": platform.platform(), "python": platform.python_version(),
            "cpu": platform.processor() or None, "cpu_count": os.cpu_count(),
            "ram_bytes": ram_bytes, "gpu_model": gpu_model,
            "gpu_memory_bytes": gpu_memory_bytes, "cuda_version": cuda_version,
            "torch_version": _version("torch"), "transformers_version": _version("transformers")}


def _git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                capture_output=True, text=True, timeout=3, check=True)
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def build_baseline_metadata(mode: str, app: Any) -> dict[str, Any]:
    runtime = (getattr(app.state, "hybrid_runtime", None) if mode == "hybrid"
               else getattr(app.state, "central_runtime", None))
    retriever = getattr(app.state, "retriever", None)
    model = runtime.model if runtime else None
    generation = (model.generation_settings if model else
                  {"do_sample": settings.do_sample, "enable_thinking": settings.enable_thinking,
                   "adapter": None, "quantization": None, "dtype": settings.dtype})
    generation = {**generation, "max_new_tokens": settings.hybrid_max_new_tokens
                  if mode == "hybrid" else settings.central_final_max_new_tokens}
    retrieval = retriever.retrieval_config if retriever is not None else None
    return {"schema_version": 1, "git_commit": _git_commit(), "mode": mode,
            "model_ids": {"hybrid": settings.hybrid_model_id, "central": settings.central_model_id},
            "model_id": settings.hybrid_model_id if mode == "hybrid" else settings.central_model_id,
            "model_revisions": {"hybrid": getattr(getattr(app.state, "hybrid_runtime", None), "model", None)
                                and app.state.hybrid_runtime.model.resolved_revision or settings.hybrid_model_revision,
                                "central": getattr(getattr(app.state, "central_runtime", None), "model", None)
                                and app.state.central_runtime.model.resolved_revision or settings.central_model_revision},
            "model_revision": getattr(model, "resolved_revision", None) or
                              (settings.hybrid_model_revision if mode == "hybrid" else settings.central_model_revision),
            "corpus_hash": _hash_file(settings.corpus_path),
            "retrieval_index_hash": _hash_index([settings.faiss_path, settings.bm25_path]),
            "generation_settings": generation, "retrieval_settings": retrieval,
            "server_hardware": _hardware(),
            "server_environment": {"app_mode": settings.app_mode, "app_env": settings.app_env}}
