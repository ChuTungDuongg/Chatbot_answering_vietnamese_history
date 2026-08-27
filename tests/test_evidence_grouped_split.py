from __future__ import annotations

from copy import deepcopy

from tests.evidence_v2_fixtures import sanity_rows
from training.common.datasets import split_rows


def test_variants_from_one_original_group_never_cross_splits():
    rows = sanity_rows()
    variants = []
    for suffix in ("duplicate", "conflict"):
        variant = deepcopy(rows[0])
        variant["id"] = f"case-a-{suffix}"
        variant["group_id"] = rows[0]["group_id"]
        variants.append(variant)
    rows.extend(variants)

    splits = split_rows(rows, seed=7, group_key="group_id")
    group_sets = [
        {row["group_id"] for row in split}
        for split in (splits.train, splits.eval, splits.test)
    ]
    assert group_sets[0].isdisjoint(group_sets[1])
    assert group_sets[0].isdisjoint(group_sets[2])
    assert group_sets[1].isdisjoint(group_sets[2])
    locations = [name for name, split in zip(("train", "eval", "test"), (splits.train, splits.eval, splits.test))
                 if any(row["group_id"] == rows[0]["group_id"] for row in split)]
    assert len(locations) == 1

