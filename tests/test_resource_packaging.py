"""Exercise resources from an isolated COPY-app-style runtime filesystem."""

import os
from pathlib import Path
import shutil
import subprocess
import sys


def test_resources_survive_runtime_copy_and_arbitrary_working_directory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    runtime = tmp_path / "runtime"
    shutil.copytree(root / "app", runtime / "app", ignore=shutil.ignore_patterns("__pycache__"))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    script = (
        "from pathlib import Path; from app.prompts.loader import validate_prompt_resources, PROMPT_ROOT; "
        "from app.tools.declarations import validate_builtin_contracts, DEFINITIONS_PATH; "
        "validate_prompt_resources(); validate_builtin_contracts(); "
        "assert Path.cwd() not in PROMPT_ROOT.parents; "
        "assert str(PROMPT_ROOT).startswith(__import__('os').environ['PYTHONPATH']); "
        "assert DEFINITIONS_PATH.is_file(); print('resources validated')"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=elsewhere,
                            env={**os.environ, "PYTHONPATH": str(runtime)},
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "resources validated" in result.stdout
    for name in ("Dockerfile", "Dockerfile.vllm"):
        assert "COPY app /workspace/app" in (root / name).read_text(encoding="utf-8")
