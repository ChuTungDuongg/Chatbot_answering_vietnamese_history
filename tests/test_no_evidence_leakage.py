import json

from training.research_agent.build_history_trajectories import build_trajectory_samples


def phase6_row():
    return {
        "id": "sample-1",
        "type": "false_premise",
        "messages": [
            {"role": "user", "content": "Câu hỏi:\nQuang Trung ký Genève?\n\nTài liệu tham khảo:\n[gold-1] RAW GOLD TEXT"},
            {"role": "assistant", "content": "Nguồn được dùng: [gold-1]\n\nTrả lời: Sai tiền đề."},
        ],
    }


def test_initial_state_has_no_phase6_evidence_or_gold_ids():
    first = build_trajectory_samples(phase6_row())[0]
    state = first["training_prompt"]
    rendered = json.dumps(first, ensure_ascii=False)
    assert state["question"] == "Quang Trung ký Genève?"
    assert state["observations"] == []
    assert state["evidence_ids"] == []
    assert "Tài liệu tham khảo:" not in rendered
    assert "RAW GOLD TEXT" not in rendered
    assert "gold-1" not in rendered
    assert first["trajectory_class"] == "false_premise"
