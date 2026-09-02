from dataclasses import replace

import pytest

from app.agents.central_citations import check_citations
from app.agents.central_evidence import build_evidence_packet, render_evidence_packet
from app.agents.central_model_runtime import CentralGeneration
from app.agents.central_viewpoints import annotate_viewpoints
from app.agents.config import CentralAgentConfig
from tests.test_central_agent import FakeCentralRuntime, FakeTool, build_agent
from tests.test_central_consolidated import WAR, row


def production_sources():
    return [
        row("war_overview", "Chiến tranh Việt Nam", 'Thất bại của Mỹ và Việt Nam Cộng hòa liên quan đến nhiều yếu tố quân sự, chính trị và chiến lược. Phong trào phản chiến gây áp lực trong nước. Một tài liệu dùng khẩu hiệu “lũ tay sai gây chiến tranh phi nghĩa”.'),
        row("vietnamization", "Chiến tranh Việt Nam", 'Chính sách "Việt Nam hóa chiến tranh" làm giảm dần mức độ tham chiến trực tiếp của quân đội Mỹ. Chính phủ chịu áp lực chính trị trong nước để giảm can dự. Trong bản ghi âm, Nixon nói: “Chúng tôi phải tìm một cách để rút khỏi cuộc chiến này”.'),
        row("chomsky", "Chiến tranh Việt Nam", 'Khó khăn về viện trợ kinh tế và tổ chức chính quyền góp phần vào thất bại. Noam Chomsky nói: “VNCH không có cơ sở thành trì trong nhân dân”. Đây là nhận định của một nhà phê bình đương thời.'),
        row("diplomacy", "Chiến tranh Việt Nam", 'Đấu tranh ngoại giao quốc tế và khả năng huy động lực lượng của đối phương tác động đến chiến lược quân sự. Một diễn văn tuyên bố: “Chúng ta nhất định thắng”. Nguồn này còn dùng cụm từ “bọn đế quốc”.'),
    ]


NEUTRAL = (
    'Thất bại của Mỹ và VNCH liên quan đến nhiều yếu tố quân sự, chính trị và chiến lược; phong trào phản chiến gây áp lực trong nước. [S1]\n\n'
    'Chính sách "Việt Nam hóa chiến tranh" làm giảm dần mức độ tham chiến trực tiếp của quân đội Mỹ. [S2]\n\n'
    'Khó khăn về viện trợ kinh tế và tổ chức chính quyền góp phần vào thất bại. [S3]\n\n'
    'Đấu tranh ngoại giao và khả năng huy động lực lượng của đối phương tác động đến chiến lược quân sự. [S4]'
)


def packet():
    return build_evidence_packet(production_sources())


def test_annotations_are_bounded_spans_and_sensitivity_is_only_telemetry():
    text = production_sources()[1]["text"]
    annotations = annotate_viewpoints(text)
    assert annotations
    assert all(text[a["start"]:a["end"]] == a["text"] for a in annotations)
    policy = next(a for a in annotations if a["text"] == "Việt Nam hóa chiến tranh")
    assert policy["type"] == "direct_quote" and not policy["requires_attribution"]
    speech = next(a for a in annotations if a["type"] == "first_person")
    assert speech["attribution_hint"] == "Nixon"
    assert all(item.viewpoint_sensitive for item in packet())
    assert not check_citations(NEUTRAL, packet()).unattributed_viewpoints
    # Even a legacy true flag on evidence containing only neutral facts confers no rule.
    neutral_source = replace(packet()[0], text="Quân đội Mỹ giảm mức độ tham chiến trực tiếp.", viewpoint_sensitive=True)
    assert not check_citations("Quân đội Mỹ giảm mức độ tham chiến trực tiếp. [S1]", [neutral_source]).unattributed_viewpoints


@pytest.mark.parametrize("text", [
    "Thất bại, chiến thắng, đối phương, quân đội và chính quyền.",
    "Chúng ta có thể xem xét các yếu tố sau đây.",
    "Quân đội đối phương giành chiến thắng và chính quyền thay đổi.",
])
def test_generic_history_words_and_assistant_voice_are_not_sensitive(text):
    assert annotate_viewpoints(text) == []


def test_us_vnch_adaptive_candidates_four_sensitive_sources_one_valid_synthesis():
    rows = production_sources() + [row(f"noise_{i}", "Địa danh không liên quan", "Không phải sự kiện cần hỏi.", 0.01) for i in range(6)]
    runtime = FakeCentralRuntime([CentralGeneration(content=NEUTRAL)])
    result = build_agent(runtime, FakeTool("search_history", rows), config=CentralAgentConfig()).chat(WAR)
    assert result["status"] == "ok"
    debug = result["central_debug"]
    assert debug["event"] == "Chiến tranh Việt Nam" and debug["actors"] == ["Mỹ", "Việt Nam Cộng hòa"]
    assert debug["outcome"] == "thất bại"
    assert debug["retrieval_candidates_before_filter"] == 10
    assert len(debug["retrieval_queries_skipped"]) == 1
    assert debug["retrieval_candidates_after_filter"] == debug["viewpoint_sensitive_evidence_count"] == 4
    assert debug["evidence_sufficient"]
    assert debug["evidence_sufficiency_reason"] == "causal_target_and_dimension_coverage"
    assert not debug["repair_used"] and len(runtime.calls) == 1
    assert debug["viewpoint_attribution_issues"] == []
    assert "[1]" in result["answer"] and "[2]" in result["answer"]


@pytest.mark.parametrize("answer,fails", [
    ("Lũ tay sai gây chiến tranh phi nghĩa. [S1]", True),
    ('Theo nguồn, khẩu hiệu “lũ tay sai gây chiến tranh phi nghĩa” là ngôn từ của tài liệu. [S1]', False),
    ("VNCH không có cơ sở trong nhân dân. [S3]", True),
    ("Theo nhận định của Noam Chomsky, VNCH không có cơ sở trong nhân dân. [S3]", False),
    ("Theo Noam Chomsky, VNCH không có cơ sở trong nhân dân. [S3]", False),
    ("Một số nhà phê bình đương thời cho rằng VNCH không có cơ sở trong nhân dân. [S3]", True),
    ('“Chúng tôi phải tìm một cách để rút khỏi cuộc chiến này”. [S2]', True),
    ('Nixon nói: “Chúng tôi phải tìm một cách để rút khỏi cuộc chiến này”. [S2]', False),
    ('Chính sách "Việt Nam hóa chiến tranh" làm giảm mức độ tham chiến trực tiếp. [S2]', False),
    ("Mỹ chịu áp lực chính trị trong nước để giảm can dự. [S2]", False),
    ("Theo nguồn, Mỹ chịu áp lực trong nước. Lũ tay sai gây chiến tranh phi nghĩa. [S1]", True),
])
def test_attribution_applies_to_the_actual_claim(answer, fails):
    checked = check_citations(answer, packet())
    assert bool(checked.unattributed_viewpoints) is fails
    assert checked.source_ids and not checked.invalid


@pytest.mark.parametrize("repaired", [
    "Thất bại liên quan đến nhiều yếu tố quân sự, chính trị và chiến lược. [S1]",
    'Theo nguồn, “lũ tay sai gây chiến tranh phi nghĩa” là khẩu hiệu trong tài liệu. [S1]',
])
def test_real_viewpoint_failure_can_be_neutralized_or_attributed_once(repaired):
    # A repaired viewpoint must also retain the supported causal depth.
    runtime = FakeCentralRuntime([CentralGeneration(content="Lũ tay sai gây chiến tranh phi nghĩa. [S1]"), CentralGeneration(content=repaired + "\n\n" + NEUTRAL)])
    result = build_agent(runtime, FakeTool("search_history", production_sources()), config=CentralAgentConfig()).chat(WAR)
    assert result["status"] == "ok" and len(runtime.calls) == 2
    assert result["central_debug"]["repair_reason"] == "unattributed_viewpoint"
    assert result["central_debug"]["viewpoint_attribution_issues"] == []
    assert "trung lập" in runtime.calls[1]["messages"][-1]["content"]


def test_unresolved_viewpoint_remains_fail_closed():
    bad = "Lũ tay sai gây chiến tranh phi nghĩa. [S1]"
    runtime = FakeCentralRuntime([CentralGeneration(content=bad), CentralGeneration(content=bad)])
    result = build_agent(runtime, FakeTool("search_history", production_sources()), config=CentralAgentConfig()).chat(WAR)
    assert result["status"] == "answer_validation_failed" and len(runtime.calls) == 2
    assert result["central_debug"]["viewpoint_attribution_issues"]


def test_source_prompt_marks_excerpts_instead_of_requiring_source_wide_attribution():
    rendered = render_evidence_packet(packet())
    assert "viewpoint_annotations" in rendered and "Noam Chomsky" in rendered
    assert "viewpoint_sensitive=true" not in rendered


def test_unquoted_attributed_opinion_and_recomputed_truncated_annotations():
    text = "Noam Chomsky cho rằng VNCH không có cơ sở trong nhân dân."
    annotations = annotate_viewpoints(text)
    assert annotations[0]["type"] == "attributed_opinion"
    source = build_evidence_packet([row("opinion", "Lịch sử", text)])[0]
    assert check_citations("VNCH không có cơ sở trong nhân dân. [S1]", [source]).unattributed_viewpoints
    trimmed = replace(source, text="Nguồn này có một nhận định.")
    assert trimmed.viewpoint_annotations == ()
