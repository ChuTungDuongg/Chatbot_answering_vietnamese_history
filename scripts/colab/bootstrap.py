"""Prepare a Drive or local workspace without importing Colab elsewhere."""

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
from importlib.metadata import PackageNotFoundError, version


def _versions() -> dict[str, str | None]:
    result = {}
    for name in ("datasets", "huggingface_hub", "transformers", "tokenizers"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def _ram_bytes() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def bootstrap(root: Path, mount_drive: bool = False, *, cache_dir: Path | None = None,
              scratch_dir: Path | None = None) -> dict[str, object]:
    is_colab = "COLAB_RELEASE_TAG" in os.environ or "google.colab" in __import__("sys").modules
    if mount_drive:
        if not is_colab:
            raise RuntimeError("--mount-drive requires a Google Colab runtime")
        from google.colab import drive
        drive.mount("/content/drive")
    root = root.expanduser().resolve()
    paths = {}
    for name in ("raw", "cache", "intermediate", "corpus_v1", "logs", "reports"):
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        paths[name] = str(path)
    cache = cache_dir.expanduser().resolve() if cache_dir else Path(paths["cache"])
    scratch = scratch_dir.expanduser().resolve() if scratch_dir else Path("/content/corpus_v1_scratch" if is_colab else paths["intermediate"])
    return {"is_colab": is_colab, "python": platform.python_version(),
            "platform": platform.platform(), "root": str(root),
            "drive_root": str(root) if is_colab else None,
            "free_disk_bytes": shutil.disk_usage(root).free,
            "available_ram_bytes": _ram_bytes(),
            "cache_path": str(cache), "scratch_path": str(scratch),
            "package_versions": _versions(), "paths": paths}


def main(argv: list[str] | None = None) -> int:
    command = argparse.ArgumentParser()
    command.add_argument("--mount-drive", action="store_true")
    command.add_argument("--drive-root", type=Path,
                         default=Path("/content/drive/MyDrive/vn_history_llm"))
    command.add_argument("--cache-dir", type=Path)
    command.add_argument("--scratch-dir", type=Path)
    args = command.parse_args(argv)
    print(json.dumps(bootstrap(args.drive_root, args.mount_drive,
                               cache_dir=args.cache_dir, scratch_dir=args.scratch_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
