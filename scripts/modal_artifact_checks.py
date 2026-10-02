"""CPU-only Modal audit of restored files; independent of GPU/model startup."""

import hashlib
from pathlib import Path

import modal

app = modal.App("vn-history-artifact-checks")
volume = modal.Volume.from_name("vn-history-artifacts", create_if_missing=False)


@app.function(image=modal.Image.debian_slim(python_version="3.11"),
              volumes={"/artifacts": volume}, cpu=2, memory=4096, timeout=900)
def audit_volume(records: list[dict]) -> list[dict]:
    volume.reload()
    result = []
    for record in records:
        path = Path("/artifacts") / record["remote_path"].lstrip("/")
        if not path.resolve().is_relative_to(Path("/artifacts").resolve()):
            raise ValueError("Audit path escapes volume")
        status = "MISSING"
        actual_bytes = actual_sha = None
        if path.is_file():
            actual_bytes = path.stat().st_size
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            actual_sha = digest.hexdigest()
            status = "PASS" if (actual_bytes == record["size_bytes"] and actual_sha == record["sha256"]) else "DIFFERENT"
        result.append({"remote_path": record["remote_path"], "status": status,
                       "bytes": actual_bytes, "sha256": actual_sha})
    return result
