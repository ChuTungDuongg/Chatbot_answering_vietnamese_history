"""Duplicate validation should normalize each question once per run."""

from evaluation.annotation import policy


def _reference_pairs(rows):
    exact, near = [], []
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            pair = (left["id"], right["id"])
            a, b = left["question"], right["question"]
            if policy.question_key(a) == policy.question_key(b):
                exact.append(pair)
            elif policy.similar(a, b):
                near.append(pair)
    return exact, near


def test_duplicate_pairs_caches_normalized_questions(monkeypatch):
    rows = [
        {"id": "a", "question": "Ai lập ra nhà Lý?"},
        {"id": "b", "question": "Ai lập ra nhà Lý!"},
        {"id": "c", "question": "Ai là người lập nên nhà Lý?"},
        {"id": "d", "question": "Thành Cát Tư Hãn lên ngôi năm nào?"},
    ]
    expected = _reference_pairs(rows)
    original = policy.question_key
    calls = 0

    def counting_key(value):
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(policy, "question_key", counting_key)
    assert policy.duplicate_pairs(rows) == expected
    assert calls == len(rows)
