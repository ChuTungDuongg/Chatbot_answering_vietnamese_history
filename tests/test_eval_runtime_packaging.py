"""The production image must import Studio without the repository/test tree."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_modal_build_context_excludes_nested_caches_secrets_and_datasets():
    from modal.file_pattern_matcher import FilePatternMatcher

    excluded = FilePatternMatcher.from_file(ROOT / ".dockerignore")
    for name in (
        "app/eval_studio/__pycache__/api.cpython-311.pyc",
        "evaluation/metrics/__pycache__/answer.cpython-311.pyc",
        ".env",
        ".git/modal-prod-qa/baseline.sqlite3",
        "evaluation/datasets/private.jsonl",
        "frontend/node_modules/example/index.js",
    ):
        assert excluded(Path(name)), name
    for name in ("app/eval_studio/behavior.py", "evaluation/__init__.py", "evaluation/metrics/answer.py"):
        assert not excluded(Path(name)), name


@pytest.mark.parametrize("dockerfile", ["Dockerfile", "Dockerfile.vllm"])
def test_backend_image_copies_every_studio_runtime_dependency(tmp_path, dockerfile):
    # Materialize the Python COPY inputs, excluding multi-stage frontend assets.
    # -I removes the checkout, PYTHONPATH and user site from the import path.
    for line in (ROOT / dockerfile).read_text().splitlines():
        fields = line.split()
        if len(fields) != 3 or fields[0] != "COPY" or fields[1].startswith("--"):
            continue
        source, destination = ROOT / fields[1], fields[2]
        if not destination.startswith("/workspace/"):
            continue
        target = tmp_path / destination.removeprefix("/workspace/")
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copyfile(source, target)
    probe = "import sys; sys.path.insert(0, sys.argv[1]); import app.main; print('runtime import: OK')"
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert "runtime import: OK" in result.stdout
