"""Focused host regressions: invented fixtures, fake generation, no model/network."""
from dataclasses import replace

import pytest

from app.agents.central.analytical import annotate_evidence, coverage_report, coverage_select
from app.agents.central.citation_recovery import align_citations, apply_citation_mapping
from app.agents.central.citation_support import sentence_support
from app.agents.central.citations import check_citations
from app.agents.central.compaction import excerpt_evidence
from app.agents.central.depth import answer_coverage, depth_contract, dimension_spans
from app.agents.central.evidence import evidence_plan
from app.agents.central.evidence import build_evidence_packet, select_evidence, select_synthesis_evidence
from app.agents.central.model_runtime import CentralGeneration
from app.agents.central.question import analyze_central_question, plan_analytical_queries
from app.agents.central.config import CentralAgentConfig
from tests.test_central_agent import FakeCentralRuntime, FakeTool, build_agent


CONFIG = CentralAgentConfig()
QUESTION = "Vì sao Alpha và Beta đồng ý bình thường hóa quan hệ?"
TITLE = "Bình thường hóa quan hệ ngoại giao Alpha – Beta"
ECONOMIC = "Chi phí kinh tế tăng gây áp lực ngân sách khiến hai bên tìm cách mở rộng hợp tác."
POLITICAL = "Chính phủ cần ổn định thể chế nhằm duy trì khả năng kiểm soát và tổ chức chính quyền."
INTERNATIONAL = "Sức ép quốc tế thúc đẩy đàm phán ngoại giao nhằm giảm tình trạng cô lập trong khu vực."
STRATEGY = "Lợi ích chiến lược thúc đẩy thay đổi chính sách nhằm giảm nguy cơ đối đầu lâu dài."
JOINT = "Alpha và Beta bình thường hóa quan hệ nhằm mở rộng hợp tác kinh tế và giảm sức ép ngân sách."


def row(key, title, text, score=.9):
    return {"chunk_id": key, "title": title, "text": text, "reranker_score": score}


def pool():
    return [row("S1", TITLE, JOINT + " " + ECONOMIC + " " + POLITICAL + " " + INTERNATIONAL, .99),
            row("S2", TITLE, STRATEGY + " " + ECONOMIC, .96),
            row("T1", "Bình thường hóa quan hệ Alpha – Gamma", ECONOMIC + POLITICAL + INTERNATIONAL + STRATEGY, .8),
            row("T2", "Lịch sử khu vực", ECONOMIC + POLITICAL + INTERNATIONAL + STRATEGY, .7)]


@pytest.mark.parametrize("a,b,conjunction", [
    ("Alpha", "Beta", "{a} và {b}"),
    ("Vương quốc Delta", "Liên minh Epsilon", "{a} với {b}"),
    ("Charles de Gaulle", "Konrad Adenauer", "cả {a} lẫn {b}"),
    ("Mỹ", "Việt Nam", "{a} và {b}"),
])
@pytest.mark.parametrize("predicate,event", [
    ("đồng ý bình thường hoá quan hệ", "bình thường hóa quan hệ"),
    ("thiết lập quan hệ ngoại giao", "thiết lập quan hệ ngoại giao"),
    ("cắt đứt quan hệ", "cắt đứt quan hệ"),
    ("ký hiệp định Geneva", "ký hiệp định Geneva"),
    ("đồng ý chấm dứt xung đột", "chấm dứt xung đột"),
])
def test_coordinated_named_actors_and_action_are_canonical(a, b, conjunction, predicate, event):
    question = f"Vì sao {conjunction.format(a=a, b=b)} {predicate}?"
    analysis = analyze_central_question(question)
    assert analysis.actors == (a, b)
    assert analysis.event == event
    assert analysis.event_type
    assert all(part in analysis.canonical_target for part in (event, a, b))
    variants = next(iter(plan_analytical_queries(analysis).values()))
    assert all(part in variants[0] for part in (event, a, b, "nguyên nhân"))
    assert variants[-1] == question


@pytest.mark.parametrize("question,actors,event", [
    ("Vì sao bình thường hóa quan hệ giữa Alpha và Beta?", ("Alpha", "Beta"), "bình thường hóa quan hệ"),
    ("Vì sao Alpha ký hiệp định Geneva với Beta?", ("Alpha", "Beta"), "ký hiệp định Geneva"),
    ("Vì sao Delta và Epsilon đàm phán?", ("Delta", "Epsilon"), "đàm phán"),
    ("Vì sao Delta và Epsilon rút quân?", ("Delta", "Epsilon"), "rút quân"),
    ("Vì sao Delta tuyên bố độc lập?", ("Delta",), "tuyên bố độc lập"),
    ("Vì sao Alpha và Beta đồng ý hợp tác?", ("Alpha", "Beta"), "đồng ý hợp tác"),
])
def test_relational_predicate_and_nominal_forms(question, actors, event):
    analysis = analyze_central_question(question)
    assert analysis.actors == actors
    assert analysis.event == event


@pytest.mark.parametrize("question", [
    "Vì sao kinh tế và chính trị thúc đẩy cải cách?",
    "Vì sao chi phí và lợi ích dẫn đến bình thường hóa quan hệ?",
    "Vì sao Alpha và chi phí đồng ý cải cách?",
])
def test_ordinary_or_mixed_conjuncts_do_not_become_actors(question):
    assert analyze_central_question(question).actors == ()


def test_direct_joint_sources_dominate_more_diverse_wrong_relation():
    analysis = analyze_central_question(QUESTION)
    kept, _ = select_evidence(list(reversed(pool())), analysis, CONFIG)
    ranked = coverage_select(kept, analysis, 4)
    assert [r["chunk_id"] for r in ranked[:2]] == ["S1", "S2"]
    selected = select_synthesis_evidence(kept, analysis, CONFIG)
    assert [r["chunk_id"] for r in selected[:2]] == ["S1", "S2"]
    assert ranked[0]["target_consistency_score"] > ranked[2]["target_consistency_score"]
    assert ranked[0]["direct_target_coverage"]
    sufficient, debug = coverage_report(selected, pool(), analysis, CONFIG)
    assert sufficient and debug["direct_target_coverage"]
    assert debug["required_actor_coverage"] == ["Alpha", "Beta"]
    assert all(debug["selected_actor_coverage"].values())
    assert debug["unresolved_actor_scopes"] == []


def test_near_exact_alias_title_is_a_joint_anchor():
    analysis = analyze_central_question("Vì sao Mỹ và Việt Nam đồng ý bình thường hoá quan hệ?")
    source = annotate_evidence(row("alias", "Bình thường hóa quan hệ ngoại giao Hoa Kỳ – Việt Nam", ECONOMIC), analysis)
    assert source["overview_anchor"] and source["direct_target_coverage"]
    assert source["actor_scope"] == ["Mỹ", "Việt Nam"]


def test_canonical_metadata_scope_survives_the_synthesis_packet():
    analysis = analyze_central_question(QUESTION)
    source = {**row("metadata", "Bối cảnh", ECONOMIC + " " + POLITICAL),
              "metadata": {"canonical_title": TITLE}}
    selected = select_synthesis_evidence([source], analysis, CONFIG)
    packet = build_evidence_packet(selected)
    assert selected[0]["direct_target_coverage"]
    assert TITLE in packet[0].title
    assert evidence_plan(packet, analysis, CONFIG)["unresolved_actor_scopes"] == []
    assert set(evidence_plan(packet, analysis, CONFIG)["strong_evidence_dimensions"]) == {"economic", "political"}


def test_separate_actor_or_wrong_action_evidence_cannot_claim_complete_joint_support():
    analysis = analyze_central_question(QUESTION)
    a = row("A", "Alpha", "Alpha bình thường hóa quan hệ nhằm mở rộng hợp tác kinh tế và giảm sức ép ngân sách.")
    b = row("B", "Beta", "Beta bình thường hóa quan hệ nhằm giảm sức ép quốc tế và thay đổi chiến lược.")
    for sources, missing in [([a], ["Beta"]), ([a, b], [])]:
        selected = [annotate_evidence(s, analysis) for s in sources]
        sufficient, debug = coverage_report(selected, sources, analysis, CONFIG)
        assert not sufficient and not debug["direct_target_coverage"]
        assert debug["unresolved_actor_scopes"] == missing
    wrong = annotate_evidence(row("wrong", "Alpha và Beta cắt đứt quan hệ", ECONOMIC + POLITICAL), analysis)
    assert not wrong["direct_target_coverage"]


def test_chronology_is_not_cause_and_distant_mentions_are_not_joint_evidence():
    analysis = analyze_central_question(QUESTION)
    timeline = row("timeline", TITLE, "Alpha và Beta bình thường hóa quan hệ vào năm 1900. Sau đó hai bên trao đổi đại sứ và tổ chức các cuộc gặp trong năm tiếp theo.")
    selected = [annotate_evidence(timeline, analysis)]
    assert selected[0]["direct_target_coverage"]
    assert not selected[0]["causal_relevance"]
    assert not coverage_report(selected, [timeline], analysis, CONFIG)[0]
    unrelated = row("scattered", "Tổng quan", "Alpha và Gamma bình thường hóa quan hệ nhằm phát triển kinh tế. " + ECONOMIC * 5 + " Beta xuất hiện trong một cuộc xung đột khác.")
    assert not annotate_evidence(unrelated, analysis)["direct_target_coverage"]


@pytest.mark.parametrize("depth,minimum", [("focused_explanation", 2), ("broad_analysis", 3)])
def test_supported_breadth_must_appear_in_explanation(depth, minimum):
    analysis = replace(analyze_central_question(QUESTION), answer_depth=depth)
    packet = build_evidence_packet(pool()[:2])
    plan = evidence_plan(packet, analysis, CONFIG)
    assert set(plan["strong_evidence_dimensions"]) >= {"economic", "political", "international", "strategy"}
    assert answer_coverage(ECONOMIC + " [S1]", plan, analysis, CONFIG)["analytical_coverage_too_shallow"]
    developed = " ".join([ECONOMIC, POLITICAL, INTERNATIONAL][:minimum])
    assert not answer_coverage(developed, plan, analysis, CONFIG)["analytical_coverage_too_shallow"]
    assert "nguyên nhân" in depth_contract(analysis, plan)


def test_simple_fact_and_small_evidence_breadth_are_not_forced_into_essays():
    analysis = analyze_central_question(QUESTION)
    plan = {"strong_evidence_dimensions": ["economic", "political"]}
    assert not answer_coverage(ECONOMIC, plan, analysis, CONFIG)["analytical_coverage_too_shallow"]
    assert not answer_coverage(ECONOMIC, {"strong_evidence_dimensions": ["economic", "political", "international", "strategy"]},
                               replace(analysis, answer_depth="simple_fact"), CONFIG)["analytical_coverage_too_shallow"]
    config = replace(CONFIG, focused_coverage_support_threshold=2, focused_coverage_min_dimensions=2)
    assert answer_coverage(ECONOMIC, plan, analysis, config)["analytical_coverage_too_shallow"]


def test_only_selected_excerpts_supply_dimensions_and_event_label_is_not_a_factor():
    analysis = analyze_central_question(QUESTION)
    oversized = " ".join(("Bối cảnh của khu vực được mô tả trong tài liệu " * 25) + factor
                         for factor in (POLITICAL, INTERNATIONAL, STRATEGY))
    source = row("long", TITLE, JOINT + " " + ECONOMIC + " " + oversized)
    assert set(dimension_spans(source["text"])) >= {"economic", "political", "international", "strategy"}
    selected = select_synthesis_evidence([source], analysis, replace(CONFIG, evidence_excerpt_chars=600))
    plan = evidence_plan(build_evidence_packet(selected), analysis, CONFIG)
    assert plan["strong_evidence_dimensions"] == ["economic"]
    assert len(selected[0]["text"]) <= 600


def test_excerpts_preserve_joint_event_before_diverse_background():
    analysis = analyze_central_question(QUESTION)
    text = POLITICAL + " " + INTERNATIONAL + " " + STRATEGY + " " + JOINT
    excerpt = excerpt_evidence(text, analysis, 180)
    assert JOINT in excerpt


def citation_packet():
    return build_evidence_packet([row("econ", TITLE, ECONOMIC), row("pol", TITLE, POLITICAL)])


def test_supported_summary_inherits_only_immediately_preceding_supported_claims():
    packet = citation_packet()
    answer = ECONOMIC + " [S1]\n\n" + POLITICAL + " [S2]\n\nTóm lại, chi phí kinh tế gây áp lực ngân sách và chính phủ cần ổn định thể chế."
    checked = check_citations(answer, packet)
    assert checked.uncited_paragraphs == 0
    assert checked.paragraph_classifications[-1]["kind"] == "supported_synthesis_summary"
    aligned, confidence = align_citations(answer, packet, CONFIG)
    assert aligned == answer and confidence == {}


@pytest.mark.parametrize("new_fact", [
    "Năm 1945, Gamma tuyên bố độc lập.",
    "Chính phủ không cần ổn định thể chế nhằm duy trì khả năng kiểm soát và tổ chức chính quyền.",
    "Chính phủ chấm dứt hợp tác kinh tế để kiểm soát cả hai bên.",
])
def test_new_year_entity_negation_or_predicate_is_still_uncited(new_fact):
    answer = ECONOMIC + " [S1]\n\n" + POLITICAL + " [S2]\n\nTóm lại, " + new_fact
    checked = check_citations(answer, citation_packet())
    assert checked.uncited_paragraphs == 1
    assert checked.paragraph_classifications[-1]["kind"] == "new_factual_claim"


def test_summary_cannot_inherit_a_prior_claim_that_is_only_decorated_with_citation():
    answer = "Gamma tuyên bố độc lập vào năm 1945. [S1]\n\nTóm lại, Gamma tuyên bố độc lập vào năm 1945."
    assert check_citations(answer, citation_packet()).uncited_paragraphs == 1


def test_target_aware_alignment_and_mapping_do_not_borrow_wrong_relation():
    analysis = analyze_central_question(QUESTION)
    wrong = row("wrong", "Bình thường hóa quan hệ Alpha – Gamma", JOINT.replace("Beta", "Gamma"))
    correct = row("correct", TITLE, JOINT)
    packet = build_evidence_packet([wrong, correct])
    aligned, _ = align_citations(JOINT, packet, CONFIG, analysis)
    assert aligned.endswith("[S2]")
    assert check_citations(aligned, packet, analysis).uncited_paragraphs == 0
    assert apply_citation_mapping(JOINT, '{"P1":["S1"]}', packet, CONFIG, analysis) == JOINT
    assert check_citations(JOINT + " [S1]", packet, analysis).target_mismatches


def test_target_alignment_prefers_joint_source_for_an_implicit_causal_claim():
    analysis = analyze_central_question(QUESTION)
    packet = build_evidence_packet([row("correct", TITLE, ECONOMIC),
                                   row("wrong", "Bình thường hóa quan hệ Alpha – Gamma", ECONOMIC)])
    aligned, _ = align_citations(ECONOMIC, packet, CONFIG, analysis)
    assert aligned.endswith("[S1]")


def test_new_arbitrary_entity_cannot_hide_in_a_high_overlap_claim():
    packet = citation_packet()
    claim = ECONOMIC.replace("hai bên", "Gamma")
    assert sentence_support(claim, packet[0]) == 0
    assert align_citations(claim, packet, CONFIG)[0] == claim


def test_title_cannot_confer_joint_coverage_on_an_explicit_other_relation():
    analysis = analyze_central_question(QUESTION)
    conflicting = row("conflicting", TITLE, JOINT.replace("Beta", "Gamma"))
    assert not annotate_evidence(conflicting, analysis)["direct_target_coverage"]


def test_treaty_title_matches_the_named_agreement_without_signing_verb():
    analysis = analyze_central_question("Vì sao Alpha và Beta ký hiệp ước Delta?")
    assert analysis.event == "ký hiệp ước Delta"
    evidence = annotate_evidence(row("treaty", "Hiệp ước Delta giữa Alpha và Beta", ECONOMIC), analysis)
    assert evidence["overview_anchor"] and evidence["direct_target_coverage"]


def test_runtime_uses_canonical_query_and_fixes_missing_citation_without_generation():
    answer = ECONOMIC + "\n\n" + POLITICAL + " [S1]\n\n" + INTERNATIONAL + " [S1]"
    runtime = FakeCentralRuntime([CentralGeneration(content=answer)])
    tool = FakeTool("search_history", pool())
    result = build_agent(runtime, tool, config=CONFIG).chat(QUESTION)
    debug = result["central_debug"]
    assert result["status"] == "ok", debug
    assert all(word in tool.calls[0]["query"] for word in ("Alpha", "Beta", "bình thường hóa quan hệ", "nguyên nhân"))
    assert len(runtime.calls) == 1
    assert debug["citation_alignment_success"] and debug["citation_repair_progress"]
    assert debug["unresolved_actor_scopes"] == []


def test_minor_citation_failure_gets_one_bounded_rewrite_and_revalidates():
    initial = ECONOMIC + " [S1]\n\n" + POLITICAL + " [S1]\n\nChính sách được thay đổi hoàn toàn mà không còn trở ngại."
    fixed = ECONOMIC + " [S1]\n\n" + POLITICAL + " [S1]"
    runtime = FakeCentralRuntime([CentralGeneration(content=initial), CentralGeneration(content='{"P3":[]}'), CentralGeneration(content=fixed)])
    result = build_agent(runtime, FakeTool("search_history", pool()), config=CONFIG).chat(QUESTION)
    debug = result["central_debug"]
    assert result["status"] == "ok", debug
    assert len(runtime.calls) == 3
    assert debug["citation_repair_progress"] is False
    assert debug["full_quality_repair_used"] and debug["repair_progress"]
    assert debug["uncited_factual_paragraphs"] == 0
