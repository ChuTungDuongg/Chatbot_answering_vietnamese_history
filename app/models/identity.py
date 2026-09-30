"""Stable file-tree identity shared by serving and six-way evaluation."""

import hashlib
from pathlib import Path


def file_tree_sha(path: Path) -> str:
    if not path.is_dir():
        raise FileNotFoundError(f"Adapter directory missing: {path}")
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*")):
        if file.is_file():
            file_digest = hashlib.sha256()
            with file.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    file_digest.update(block)
            digest.update(file.relative_to(path).as_posix().encode())
            digest.update(file_digest.hexdigest().encode())
    return digest.hexdigest()
