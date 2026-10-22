"""Frozen prompt provenance and advisory checks for real live-answer captures."""
import hashlib
import json
from pathlib import Path
import pytest

from app.rag.prompting import build_messages, build_no_rag_messages


@pytest.mark.parametrize("mode", ["concise", "standard", "detailed"])
@pytest.mark.parametrize("method", ["build_messages", "build_no_rag_messages"])
def test_offline_prompts_match_main_bytes(method, mode):
    golden = json.loads((Path(__file__).parent / "fixtures/offline_prompt_sha256.json").read_text())
    function = {"build_messages": build_messages, "build_no_rag_messages": build_no_rag_messages}[method]
    args = ["Waterloo?"] + ([[]] if method == "build_messages" else [])
    messages = function(*args, response_mode=mode)
    data = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(data).hexdigest() == golden[f"{method}:{mode}"]


def test_waterloo_checks_reject_strength_as_casualties_and_certain_counterfactual():
    from scripts.check_live_answer_regressions import review_waterloo
    sources = [{"chunk_id": "nam", "text": "Grouchy led 33,000 troops."}]
    flags = review_waterloo("Grouchy có 33.000 quân thiệt mạng. [nam] Nếu đến thì chắc chắn thắng.", sources)
    assert "strength_as_casualties" in flags
    assert "certain_counterfactual" in flags
    assert "invalid_citations" not in flags


def test_waterloo_checks_accept_qualified_strength_and_refusal_without_sources():
    from scripts.check_live_answer_regressions import review_waterloo
    sources = [{"chunk_id": "nam", "text": "Grouchy led 33,000 troops."}]
    answer = "Grouchy chỉ huy khoảng 33.000 quân; đây là quân số, không phải số thương vong. [nam] Việc tiếp viện có thể thay đổi tương quan, nhưng chưa thể khẳng định kết quả."
    assert review_waterloo(answer, sources) == []
    assert review_waterloo("Chưa có nguồn để xác minh câu hỏi này.", []) == []
    assert "invalid_citations" in review_waterloo("Có thể thay đổi. [invented]", sources)


def test_waterloo_flags_observed_unsupported_causal_summary_and_body_boilerplate():
    from scripts.check_live_answer_regressions import review_waterloo
    sources = [{"chunk_id": "nam", "text": "Prussian reinforcements supported Wellington."}]
    answer = "Napoleon thua do sự phối hợp không hiệu quả giữa các lực lượng liên minh. [nam] Về Grouchy, theo thông tin được cung cấp, quân số là 33.000."
    flags = review_waterloo(answer, sources)
    assert "coalition_coordination_claim_requires_review" in flags
    assert "formulaic_phrase" in flags


def test_waterloo_flags_prussian_absence_claim_from_real_final_capture():
    from scripts.check_live_answer_regressions import review_waterloo
    sources = [{"chunk_id": "wiki_vi_83457", "text": "Quân Phổ tới hỗ trợ Wellington và tấn công sườn quân Pháp."}]
    answer = "Kết quả là quân Phổ không thể tham gia trận Waterloo do bị phân tán và mất nhiều binh sĩ [wiki_vi_83457]."
    flags = review_waterloo(answer, sources)
    assert "prussian_participation_claim_requires_review" in flags
    assert "invalid_citations" not in flags  # Valid IDs alone do not prove entailment.
