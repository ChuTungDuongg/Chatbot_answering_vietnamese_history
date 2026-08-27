from training.common.datasets import split_rows


def test_grouped_split_keeps_trajectory_and_question_variants_together():
    rows = [
        {"id": f"{group}-{step}", "group_id": group, "trajectory_id": group, "step": step}
        for group in ("a", "b", "c", "d", "e", "f")
        for step in (1, 2)
    ]
    splits = split_rows(rows, train_ratio=0.5, eval_ratio=0.25, seed=7, group_key="group_id")
    group_sets = [{row["group_id"] for row in getattr(splits, name)} for name in ("train", "eval", "test")]
    assert group_sets[0].isdisjoint(group_sets[1])
    assert group_sets[0].isdisjoint(group_sets[2])
    assert group_sets[1].isdisjoint(group_sets[2])


def test_max_samples_shuffles_groups_before_sampling():
    rows = [{"id": str(i), "group_id": str(i)} for i in range(20)]
    first = split_rows(rows, seed=9, max_samples=5)
    second = split_rows(rows, seed=9, max_samples=5)
    selected = [row["id"] for split in (first.train, first.eval, first.test) for row in split]
    assert selected == [row["id"] for split in (second.train, second.eval, second.test) for row in split]
    assert selected != [str(i) for i in range(5)]


def test_grouped_max_samples_keeps_three_indivisible_groups_for_smoke_split():
    rows = [
        {"id": f"{group}-{index}", "group_id": group}
        for group in ("a", "b", "c", "d")
        for index in range(60)
    ]
    splits = split_rows(rows, seed=3, max_samples=100)
    group_sets = [{row["group_id"] for row in getattr(splits, name)} for name in ("train", "eval", "test")]
    assert all(group_sets)
    assert not (group_sets[0] & group_sets[1] | group_sets[0] & group_sets[2] | group_sets[1] & group_sets[2])
