"""The single V2 mix config, deterministic sampling, and assistant signal audit."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from training.central.normalization.validation import SOURCES, require_v2_trajectory
from training.trajectory_dataset.mix import mix_sources
from training.trajectory_dataset.preprocess import assistant_labeled_token_counts

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/mix.json"


def load_mix_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    ratios = config.get("ratios", {})
    if config.get("profile") != "central_v2" or set(config.get("allowed_sources", [])) != SOURCES or set(ratios) != SOURCES:
        raise ValueError("V2 mix must contain exactly the two intended sources")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in ratios.values()) or not math.isclose(sum(ratios.values()), 1):
        raise ValueError("V2 ratios must be positive finite values summing to one")
    if config.get("duplication_allowed") is not False or config.get("enable_thinking") is not False:
        raise ValueError("V2 requires no duplicated samples and enable_thinking=False")
    return config


def mix_v2(sources, *, config_path=DEFAULT_CONFIG, seed=42, max_total=None):
    config = load_mix_config(config_path)
    if set(sources) != SOURCES:
        raise ValueError("V2 source pools must match the configured sources")
    seen = set()
    for source, rows in sources.items():
        for row in rows:
            require_v2_trajectory(row)
            if row["source_dataset"] != source or row["id"] in seen:
                raise ValueError("wrong source pool or duplicate trajectory ID")
            seen.add(row["id"])
    return mix_sources(sources, config["ratios"], seed=seed, max_total=max_total)


def assistant_token_share(rows, tokenizer):
    """Actual supervised token shares, not sample proportions or character estimates."""
    totals = Counter()
    for row in rows:
        counts = assistant_labeled_token_counts(tokenizer, row)
        totals[row["source_dataset"]] += counts["total_assistant_labeled_tokens"]
    denominator = sum(totals.values())
    return {source: {"assistant_tokens": count, "share": count / denominator if denominator else None}
            for source, count in sorted(totals.items())}
