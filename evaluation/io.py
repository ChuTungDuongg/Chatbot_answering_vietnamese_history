"""Small append-only JSONL utilities, independent of training and production."""
import hashlib
import json
from pathlib import Path

from evaluation.schema import EvaluationRecord, Question, RunMetadata


def read_jsonl(path, model):
    with Path(path).open(encoding="utf-8") as handle:
        rows = [model.model_validate_json(line) for line in handle if line.strip()]
    key = (lambda row: row.id) if model is Question else (lambda row: row.question_id)
    ids = [key(row) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question IDs")
    return rows


def load_run(directory):
    root = Path(directory)
    metadata = RunMetadata.model_validate_json((root / "metadata.json").read_text(encoding="utf-8"))
    records = read_jsonl(root / "records.jsonl", EvaluationRecord)
    if any(row.run_id != metadata.run_id or row.variant != metadata.variant for row in records):
        raise ValueError("record run/variant does not match metadata")
    return metadata, records


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_paths(paths):
    """Hash names AND bytes of explicit local artifact paths; no downloads."""
    digest = hashlib.sha256()
    for index, root in enumerate(map(Path, paths)):
        if not root.exists():
            raise ValueError(f"missing fingerprint input: {root}")
        files = sorted(root.rglob("*")) if root.is_dir() else [root]
        for path in files:
            if path.is_file():
                name = path.relative_to(root).as_posix() if root.is_dir() else root.name
                digest.update(f"{index}/{name}\0{file_sha256(path)}\n".encode())
    return digest.hexdigest()
