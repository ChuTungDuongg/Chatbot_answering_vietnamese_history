"""Freeze a deterministic workload subset of an existing evaluation split."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from evaluation.schema import load_questions

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "evaluation/datasets/v1_silver/splits/v1_seed42/test.jsonl"
FIELDS = ("category", "difficulty", "question_type", "answerable", "in_domain")


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rank(seed: int, value: str) -> str:
    return _hash(f"latency-v1:{seed}:{value}".encode("utf-8"))


def sample(source: Path, *, seed: int = 2026, count: int = 100) -> tuple[bytes, dict]:
    load_questions(source)  # Validate source schema and unique IDs without altering it.
    source_bytes = source.read_bytes()
    rows = [json.loads(line) for line in source_bytes.decode("utf-8").splitlines() if line.strip()]
    eligible = [row for row in rows if 2 <= len(row["question"].strip()) <= 1000]
    accepted = [row for row in eligible if row.get("review_status") == "accepted"
                or row.get("annotation_status") == "human_accepted"]
    reviewed = [row for row in eligible if row.get("annotation_status") == "auto_reviewed"]
    pool, tier = (accepted, "human_accepted") if len(accepted) >= count else (
        (reviewed, "auto_reviewed") if len(reviewed) >= count else (eligible, "available_valid_questions"))
    if count < 1 or len(pool) < count:
        raise ValueError(f"Need {count} eligible source questions; found {len(pool)}")
    strata = defaultdict(list)
    for row in pool:
        key = json.dumps([row.get(field) for field in FIELDS], ensure_ascii=False, separators=(",", ":"))
        strata[key].append(row)
    keys = sorted(strata, key=lambda key: (_rank(seed, key), key))
    # Guarantee rare joint strata when there is room, then allocate by largest
    # proportional deficit. Within each stratum, SHA ranking is order-independent.
    quotas = {key: int(len(keys) <= count) for key in keys}
    for _ in range(count - sum(quotas.values())):
        key = max((key for key in keys if quotas[key] < len(strata[key])),
                  key=lambda key: (count * len(strata[key]) / len(pool) - quotas[key], _rank(seed, key)))
        quotas[key] += 1
    selected = []
    for key in keys:
        selected.extend(sorted(strata[key], key=lambda row: (_rank(seed, row["id"]), row["id"]))[:quotas[key]])
    selected.sort(key=lambda row: row["id"])
    data = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                    .encode("utf-8") for row in selected)
    try:
        source_path = source.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        source_path = str(source.resolve())
    manifest = {
        "schema_version": 1, "dataset_id": "latency_v1", "source_dataset_path": source_path,
        "source_dataset_sha256": _hash(source_bytes), "source_count": len(rows),
        "sampling_seed": seed, "count": len(selected), "dataset_sha256": _hash(data),
        "selected_question_ids": [row["id"] for row in selected],
        "selection_policy": {"version": 1, "preferred_review_tier": tier,
                             "stratification_fields": list(FIELDS),
                             "quota": "one per joint stratum when possible, then largest proportional deficit",
                             "within_stratum": "SHA256(latency-v1:seed:id), then id; final order by id",
                             "api_eligible_question_length": [2, 1000]},
        "distributions": {field: dict(sorted(Counter(str(row.get(field)) for row in selected).items()))
                          for field in FIELDS},
        "limitations": ["Workload subset, not a new human gold quality benchmark.",
                        "Missing source categories or edge types are not fabricated."],
    }
    split_manifest = source.parent / "split_manifest.json"
    if split_manifest.is_file():
        manifest["source_split_manifest_sha256"] = _hash(split_manifest.read_bytes())
    return data, manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/datasets/latency_v1")
    parser.add_argument("--verify", action="store_true", help="Reproduce the frozen bytes; never overwrite a different dataset")
    args = parser.parse_args(argv)
    data, manifest = sample(args.source, seed=args.seed, count=args.count)
    args.output.mkdir(parents=True, exist_ok=True)
    assets = {f"questions_{args.count}.jsonl": data,
              "manifest.json": (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")}
    for name, content in assets.items():
        path = args.output / name
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError(f"Frozen asset differs: {path}; create a new dataset version")
        elif args.verify:
            raise FileNotFoundError(path)
        else:
            path.write_bytes(content)
    print(f"{manifest['count']} questions; SHA256={manifest['dataset_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
