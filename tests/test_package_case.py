from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def test_training_package_is_lowercase_and_linux_resolvable():
    root = Path(__file__).parents[1]
    assert (root / "training" / "__init__.py").is_file()
    root_names = {entry.name for entry in os.scandir(root)}
    assert "training" in root_names
    assert "Training" not in root_names
    assert not any("Training" in directory_names for _, directory_names, _ in os.walk(root / "training"))
    assert importlib.util.find_spec("training.research_agent.train") is not None
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    assert not any(part == "Training" for path in tracked for part in Path(path).parts)


def test_required_train_modules_resolve_help():
    for module in (
        "training.research_agent.train",
        "training.research_agent.prepare_dataset",
        "training.research_agent.build_history_trajectories",
        "training.evidence_agent.train",
        "training.history_answerer.train",
    ):
        result = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
