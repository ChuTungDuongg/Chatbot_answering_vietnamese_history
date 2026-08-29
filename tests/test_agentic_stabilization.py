from app.agents.comparison import comparison_target_relevance
from app.agents.evidence_agent import (
    BROAD_SUMMARY_FACETS,
    EvidenceCriticAgent,
    _broad_summary_facets_for_text,
    _candidate_affiliation_constraint_pass,
    _candidate_quality,
    _evidence_question_type,
)
from app.agents.history_answerer import HistoryAnswererAgent
from app.rag.retrieval import balance_comparison_candidates


def test_comparison_role_propagation_preserves_target_b_for_target_specific_candidate():
    candidates = [{
        "chunk_id": "b-only",
        "title": "Đối tượng B",
        "text": "Thông tin riêng về đối tượng B.",
        "retrieval_query_roles": ["target_b"],
        "final_retrieval_score": 0.9,
    }, {
        "chunk_id": "a",
        "title": "Đối tượng A",
        "text": "Thông tin riêng về đối tượng A.",
        "final_retrieval_score": 0.8,
    }]

    selected, _ = balance_comparison_candidates("So sánh Đối tượng A và Đối tượng B", candidates, 2)

    assert next(item for item in selected if item["chunk_id"] == "b-only")["comparison_target"] == "target_b"


def test_direct_event_subject_beats_incidental_monument_title():
    direct = comparison_target_relevance(
        "Chiến dịch Điện Biên Phủ",
        {"title": "Chiến dịch Điện Biên Phủ", "text": "Chiến dịch diễn ra năm 1954."},
    )
    incidental = comparison_target_relevance(
        "Chiến dịch Điện Biên Phủ",
        {"title": "Tượng đài Chiến thắng Điện Biên Phủ", "text": "Tượng đài ghi dấu chiến thắng."},
    )

    assert direct["direct"] is True
    assert incidental["direct"] is False
    assert direct["direct_subject_score"] > incidental["direct_subject_score"]


def test_affiliation_constraint_rejects_opposite_faction_evidence():
    question = "Ai là vị tướng giỏi nhất phe VNCH?"
    vnch = {"title": "Một tướng VNCH", "text": "Ông là tướng VNCH và chỉ huy lực lượng VNCH."}
    opposite = {"title": "Chiến thắng", "text": "Quân đội đã đánh bại VNCH trong trận chiến."}

    assert _candidate_affiliation_constraint_pass(question, vnch) is True
    assert _candidate_affiliation_constraint_pass(question, opposite) is False


def test_superlative_and_broad_summary_question_policies_are_explicit():
    assert _evidence_question_type("Ai là vị tướng giỏi nhất?") == "evaluative"
    assert _evidence_question_type("Tóm tắt diễn biến, lực lượng, kết quả và ý nghĩa sự kiện") == "broad_summary"
    facets = _broad_summary_facets_for_text(
        "Năm 1954, tướng Võ Nguyên Giáp chỉ huy chiến dịch; chiến thắng tạo bước ngoặt lịch sử."
    )
    assert {"timeframe_context", "actors", "course", "result", "significance"}.issubset(facets)
    assert BROAD_SUMMARY_FACETS == ["timeframe_context", "actors", "course", "result", "significance"]


def test_navigation_noise_is_ranked_below_historical_evidence():
    question = "Chiến thắng Bạch Đằng năm 938 có ý nghĩa gì?"
    clean = {"title": "Chiến thắng Bạch Đằng", "text": "Năm 938, chiến thắng chấm dứt Bắc thuộc và mở ra độc lập."}
    noisy = {"title": "Video game Chiến thắng Bạch Đằng", "text": "Danh sách trò chơi điện tử và video game."}

    assert _candidate_quality(question, clean) > _candidate_quality(question, noisy)


def test_evidence_request_uses_compact_model_excerpts_but_keeps_full_source_available():
    full_text = "Năm 938, Ngô Quyền đánh bại quân Nam Hán trên sông Bạch Đằng. " * 120
    agent = EvidenceCriticAgent(max_contexts=4)
    request, available, report = agent._build_evidence_request(
        "Chiến thắng Bạch Đằng năm 938 có ý nghĩa gì?",
        [{"chunk_id": "ev1", "title": "Bạch Đằng", "text": full_text, "source_kind": "history"}],
        final_k=2,
    )

    assert len(request.evidence[0].text) < len(full_text)
    assert available["ev1"]["text"] == full_text
    assert report["model_input_chars"] > 0
    assert report["model_input_tokens_estimate"] > 0


def test_history_quality_only_retry_requires_material_two_sided_evidence():
    assert HistoryAnswererAgent._should_retry_history(
        inference_mode="agentic_rag",
        answer_depth="deep",
        question_type="cause",
        contexts=[{"chunk_id": "only", "text": "Một nguyên nhân được nêu."}],
        first_quality_issues=["deep_answer_collapse"],
        question="Vì sao sự kiện xảy ra?",
    ) is False
    assert HistoryAnswererAgent._should_retry_history(
        inference_mode="agentic_rag",
        answer_depth="deep",
        question_type="compare",
        contexts=[
            {"chunk_id": "a", "comparison_target": "target_a", "text": "Đối tượng A có kết quả khác."},
            {"chunk_id": "b", "comparison_target": "target_b", "text": "Đối tượng B có kết quả khác."},
        ],
        first_quality_issues=["shallow_comparison"],
        question="So sánh A và B",
    ) is True
