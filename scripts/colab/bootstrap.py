"""Prepare a Drive or local workspace without importing Colab elsewhere."""

import argparse
import json
import os
from pathlib import Path
import platform


def bootstrap(root: Path, mount_drive: bool = False) -> dict[str, object]:
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
    return {"is_colab": is_colab, "python": platform.python_version(),
            "platform": platform.platform(), "root": str(root), "paths": paths}


def main(argv: list[str] | None = None) -> int:
    command = argparse.ArgumentParser()
    command.add_argument("--mount-drive", action="store_true")
    command.add_argument("--drive-root", type=Path,
                         default=Path("/content/drive/MyDrive/vn_history_llm"))
    args = command.parse_args(argv)
    print(json.dumps(bootstrap(args.drive_root, args.mount_drive), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
