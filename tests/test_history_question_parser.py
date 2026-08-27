from training.research_agent.build_history_trajectories import extract_question_only, extract_reference_ids


def test_extract_question_only_removes_phase6_context():
    text = "Câu hỏi:\nAi thắng Bạch Đằng?\n\nTài liệu tham khảo:\n[c1] raw gold chunk"
    assert extract_question_only(text) == "Ai thắng Bạch Đằng?"
    assert extract_reference_ids(text) == ["c1"]
