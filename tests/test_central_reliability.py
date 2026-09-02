from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.agents.central_agent import FAILURE_ANSWERS
from app.agents.central_citation_recovery import align_citations, apply_citation_mapping, sentence_support
from app.agents.central_citations import check_citations
from app.agents.central_compaction import compact_history, excerpt_evidence
from app.agents.central_evidence import build_evidence_packet, select_evidence
from app.agents.central_model_runtime import CentralGeneration, choose_attention_backend
from app.agents.central_question import analyze_central_question
from app.agents.config import CentralAgentConfig
from app.agents.lazy_runtime import LazyRuntime
from app.tools.local_search import SearchHistoryTool
from tests.test_central_agent import FakeCentralRuntime, FakeTool, SleepingRuntime, build_agent


QUESTION = "Vì sao Cách mạng Tháng Tám thành công?"
FACTS = [
    "Cách mạng Tháng Tám thành công nhờ sự chuẩn bị lực lượng và tổ chức chính trị. Quần chúng nhân dân tham gia khởi nghĩa giành chính quyền trên phạm vi rộng.",
    "Cách mạng Tháng Tám thành công trong bối cảnh quốc tế thuận lợi và chính quyền cũ suy yếu. Thời cơ khởi nghĩa gắn với kết quả của chiến tranh và sự thay đổi tương quan lực lượng.",
    "Cách mạng Tháng Tám thành công nhờ sự phối hợp giữa lực lượng tại các địa phương. Khả năng huy động nhân dân và tổ chức đấu tranh chính trị góp phần vào thắng lợi của cuộc khởi nghĩa.",
]
ANSWER = "\n\n".join(text.split(". ")[0] + "." for text in FACTS)


def sources():
    return [{"chunk_id": f"cmt8_{index}", "title": "Cách mạng Tháng Tám", "text": text, "reranker_score": 0.98 - index * .01} for index, text in enumerate(FACTS, 1)]


class ScheduledTool(FakeTool):
    def __init__(self, events, safe):
        super().__init__("search_history", sources())
        self.events, self.safe = events, safe

    def can_overlap_model_load_and_retrieval(self):
        return self.safe

    async def run(self, arguments):
        self.events.append(("retrieval_start", time.perf_counter()))
        await asyncio.sleep(.08)
        self.calls.append(arguments.model_dump())
        self.events.append(("retrieval_end", time.perf_counter()))
        return self.result


@pytest.mark.parametrize("safe", [True, False])
def test_cold_first_turn_overlap_is_gated_and_synthesis_waits_for_readiness(safe):
    events = []
    runtime = FakeCentralRuntime([CentralGeneration(content=ANSWER)])
    def factory():
        events.append(("load_start", time.perf_counter()))
        time.sleep(.2)
        events.append(("load_end", time.perf_counter()))
        return runtime
    lazy = LazyRuntime(factory, name="fake-central")
    tool = ScheduledTool(events, safe)
    result = build_agent(lazy, tool, config=CentralAgentConfig()).chat(QUESTION, history=[])
    timings = dict(events)
    assert result["status"] == "ok", result["central_debug"]
    assert len(runtime.calls) == len(tool.calls) == 1
    debug = result["central_debug"]
    assert debug["model_was_cold"] and debug["model_load_overlap_enabled"] is safe
    assert debug["citation_alignment_success"] and not debug["repair_used"]
    assert debug["history_input_turns"] == 0
    if safe:
        assert timings["retrieval_start"] < timings["load_end"]
        assert debug["model_load_overlap_ms_saved_estimate"] > 0
    else:
        assert timings["retrieval_start"] >= timings["load_end"]
        assert debug["model_load_overlap_ms_saved_estimate"] == 0
    # Timeline proof rather than a brittle A100/CI wall-clock threshold.
    assert lazy.is_ready and len([event for event, _ in events if event == "load_start"]) == 1


def test_warm_fast_path_and_cold_warm_validation_parity():
    calls = []
    runtime = FakeCentralRuntime([CentralGeneration(content=ANSWER)] * 2)
    def factory():
        calls.append("load")
        return runtime
    lazy = LazyRuntime(factory, name="fake")
    agent = build_agent(lazy, ScheduledTool([], True), config=CentralAgentConfig())
    cold = agent.chat(QUESTION, history=[])
    warm = agent.chat(QUESTION, history=[])
    assert cold["answer"] == warm["answer"] and cold["status"] == warm["status"] == "ok"
    assert cold["central_debug"]["answer_quality_issues"] == warm["central_debug"]["answer_quality_issues"] == []
    assert len(calls) == 1
    assert warm["central_debug"]["model_load_wait_ms"] == 0
    assert not warm["central_debug"]["model_was_cold"]
    assert len(agent._schema_cache) == 1


def test_overlap_capability_is_read_only_and_requires_both_retrieval_models_on_cpu():
    for devices, expected in [(("cpu", "cpu"), True), (("cuda:0", "cuda:0"), False), (("cpu", "cuda:0"), False), (("unknown", "cpu"), False)]:
        service = SimpleNamespace(embedder=SimpleNamespace(device=devices[0]), reranker=SimpleNamespace(device=devices[1]))
        assert SearchHistoryTool(SimpleNamespace(service=service)).can_overlap_model_load_and_retrieval() is expected


def test_adaptive_cmt8_filters_incidental_pages_and_skips_only_sufficient_secondary_query():
    noise = [
        ("Tháng tám", "Tháng tám là một tháng trong lịch. Cách mạng Tháng Tám diễn ra trong tháng này."),
        ("Mười chín tháng Tám", "Mười chín tháng Tám là một bài hát do Xuân Oanh sáng tác, nói về Cách mạng Tháng Tám."),
        ("Đường Cách Mạng Tháng Tám", "Con đường được đặt tên theo Cách mạng Tháng Tám."),
        ("Quảng trường Cách mạng Tháng Tám", "Quảng trường này gắn với Cách mạng Tháng Tám."),
        ("30 tháng 4 năm 1975", "Một sự kiện của Chiến tranh Việt Nam."),
    ]
    tool = FakeTool("search_history", sources() + [{"chunk_id": f"noise{i}", "title": title, "text": text} for i, (title, text) in enumerate(noise)])
    runtime = FakeCentralRuntime([CentralGeneration(content=ANSWER)])
    result = build_agent(runtime, tool, config=CentralAgentConfig()).chat(QUESTION)
    assert result["status"] == "ok" and len(tool.calls) == 1
    debug = result["central_debug"]
    assert len(debug["retrieval_queries_planned"]) == 2
    assert len(debug["retrieval_queries_skipped"]) == 1
    assert debug["retrieval_filter_reasons"]["event_incidental_artifact"] >= 1
    assert {row["chunk_id"] for row in result["source_chunks"]} == {"cmt8_1", "cmt8_2", "cmt8_3"}
    song_question = analyze_central_question('Bài hát Mười chín tháng Tám là gì?')
    kept, _ = select_evidence([{"title": noise[1][0], "text": noise[1][1]}], song_question, CentralAgentConfig())
    assert kept


def test_secondary_query_executes_when_primary_lacks_diversity_or_coverage():
    tool = FakeTool("search_history", lambda args: sources()[:1] if "nguyên nhân" in args["query"] else sources())
    result = build_agent(FakeCentralRuntime([CentralGeneration(content=ANSWER)]), tool, config=CentralAgentConfig()).chat(QUESTION)
    assert result["status"] == "ok" and len(tool.calls) == 2
    assert not result["central_debug"]["retrieval_queries_skipped"]


def test_compaction_finds_late_relevant_windows_and_preserves_whole_speech():
    noise = "Trang này mô tả danh mục tài liệu và cách sắp xếp các mục. " * 45
    quote = 'Nixon nói: “Chúng tôi phải xem xét chiến lược. Chúng tôi cần thay đổi chính sách”.'
    text = noise + FACTS[0] + " " + quote + " " + noise
    excerpt = excerpt_evidence(text, analyze_central_question(QUESTION), 650)
    assert FACTS[0].split(". ")[0] in excerpt
    assert len(excerpt) <= 650 < len(text)
    assert excerpt.endswith(".")
    if "Nixon" in excerpt:
        assert quote in excerpt
    assert excerpt.count("“") == excerpt.count("”")


def test_compaction_telemetry_uses_selected_original_text():
    rows = [{**row, "text": row["text"] + " Trang này mô tả danh mục tài liệu." * 75} for row in sources()]
    result = build_agent(FakeCentralRuntime([CentralGeneration(content=ANSWER)]), FakeTool("search_history", rows), config=CentralAgentConfig()).chat(QUESTION)
    debug = result["central_debug"]
    assert result["status"] == "ok"
    assert debug["evidence_chars_after_compaction"] < debug["evidence_chars_before_compaction"]
    assert debug["evidence_tokens_estimated_after"] < debug["evidence_tokens_estimated_before"]


@pytest.mark.parametrize("syntax", ["[S1, S2]", "[S1,S2]", "[S1] [S2]", "[S1][S2]"])
def test_citation_normalization_preserves_dates_and_supplied_aliases(syntax):
    checked = check_citations("Dữ kiện [938], [1945]. " + syntax, build_evidence_packet(sources()))
    assert checked.answer.endswith("[S1] [S2]") and "[938], [1945]" in checked.answer
    assert len(checked.source_ids) == 2 and not checked.invalid


def test_host_alignment_recovers_missing_citations_with_one_generation():
    runtime = FakeCentralRuntime([CentralGeneration(content=ANSWER)])
    result = build_agent(runtime, FakeTool("search_history", sources()), config=CentralAgentConfig()).chat(QUESTION, history=[])
    assert result["status"] == "ok" and len(runtime.calls) == 1
    assert all(f"[{i}]" in result["answer"] for i in (1, 2, 3))
    debug = result["central_debug"]
    assert debug["citation_alignment_used"] and debug["citation_alignment_success"]
    assert not debug["full_quality_repair_used"] and not debug["repair_used"]


def test_low_confidence_alignment_does_not_guess_and_tiny_repair_cannot_fabricate_support():
    weak = "Tình hình diễn biến theo chiều hướng phức tạp và liên quan đến nhiều yếu tố khác nhau."
    packet = build_evidence_packet(sources())
    aligned, confidence = align_citations(weak, packet, CentralAgentConfig())
    assert aligned == weak and max(confidence.values()) < .88
    runtime = FakeCentralRuntime([CentralGeneration(content=weak), CentralGeneration(content='{"P1":["S1"]}', output_tokens=12)])
    result = build_agent(runtime, FakeTool("search_history", sources()), config=CentralAgentConfig()).chat(QUESTION)
    assert [call["stage"] for call in runtime.calls] == ["synthesis", "citation_repair"]
    assert runtime.calls[1]["max_new_tokens"] == 128
    assert result["status"] == "answer_validation_failed"
    assert result["central_debug"]["evidence_sufficient"]
    assert result["central_debug"]["citation_repair_tokens"] == 12
    assert not result["central_debug"]["full_quality_repair_used"]
    assert "Chưa đủ tư liệu" not in result["answer"]


def test_mapping_validation_rejects_unknown_aliases_duplicates_and_changed_text():
    packet = build_evidence_packet(sources())
    for invalid in ['{"P1":["S99"]}', '{"P99":["S1"]}', '{"P1":["S1"],"P1":["S2"]}', 'A new answer [S1]', '{"P1":"S1"}']:
        assert apply_citation_mapping(ANSWER, invalid, packet, CentralAgentConfig()) == ANSWER
    valid = apply_citation_mapping(ANSWER, '{"P1":["S1"],"P2":["S2"],"P3":["S3"]}', packet, CentralAgentConfig())
    assert not check_citations(valid, packet).uncited_paragraphs
    fenced = apply_citation_mapping(ANSWER, '```json\n{"P1":["S1"],"P2":["S2"],"P3":["S3"]}\n```', packet, CentralAgentConfig())
    assert fenced == valid


def test_tiny_mapping_resolves_conservative_ambiguity_without_rewriting():
    # Both are plausible; the configured strict threshold admits only the exact
    # wording. The host abstains on the close runner-up; the mapping selects S1.
    rows = sources()[:1]
    rows[0]["text"] += " Bối cảnh quốc tế tạo điều kiện thuận lợi cho thắng lợi."
    rows.append({**rows[0], "chunk_id": "similar", "reranker_score": .9, "text": rows[0]["text"].replace("nhờ sự", "nhờ")})
    answer = FACTS[0].split(". ")[0] + "."
    config = CentralAgentConfig(citation_alignment_threshold=.99)
    runtime = FakeCentralRuntime([CentralGeneration(content=answer), CentralGeneration(content='{"P1":["S1"]}', output_tokens=10)])
    result = build_agent(runtime, FakeTool("search_history", rows), config=config).chat(QUESTION)
    assert result["status"] == "ok" and result["answer"] == answer + " [1]"
    assert [call["stage"] for call in runtime.calls] == ["synthesis", "citation_repair"]
    assert result["central_debug"]["citation_repair_used"]
    assert not result["central_debug"]["full_quality_repair_used"]


def test_opt_in_full_rewrite_is_last_resort_after_citation_mapping_fails():
    runtime = FakeCentralRuntime([
        CentralGeneration(content="Tình hình diễn biến phức tạp theo nhiều chiều hướng khác nhau."),
        CentralGeneration(content='{"P1":[]}'),
        CentralGeneration(content=FACTS[0] + " [S1]"),
    ])
    result = build_agent(runtime, FakeTool("search_history", sources()), config=CentralAgentConfig(citation_full_rewrite_fallback=True)).chat(QUESTION)
    assert result["status"] == "ok"
    assert [call["stage"] for call in runtime.calls] == ["synthesis", "citation_repair", "quality_repair"]
    assert result["central_debug"]["full_quality_repair_used"]


def test_alignment_rejects_negation_and_new_numbers():
    packet = build_evidence_packet(sources())
    assert sentence_support(FACTS[0].replace("thành công", "không thành công"), packet[0]) == 0
    assert sentence_support(FACTS[0] + " Có 77 đơn vị.", packet[0]) == 0


def test_alignment_preserves_bracketed_year_support_and_ignores_list_ordinals():
    packet = build_evidence_packet(sources())
    claim = FACTS[0].split(". ")[0]
    numbered, _ = align_citations("1. " + claim + ".", packet, CentralAgentConfig())
    assert numbered.endswith("[S1]")
    packet[1] = replace(packet[1], text=packet[1].text + " Tài liệu được xuất bản năm 1945.")
    dated = claim + " năm [1945]."
    aligned, _ = align_citations(dated, packet, CentralAgentConfig())
    assert aligned == dated  # A date elsewhere in the packet cannot confer support.


def test_comparison_alignment_never_borrows_the_other_targets_source():
    packet = build_evidence_packet([{**sources()[0], "comparison_target": "Cách mạng Tháng Tám"}])
    answer = "Điện Biên Phủ thành công nhờ sự chuẩn bị lực lượng và tổ chức chính trị."
    aligned, _ = align_citations(answer, packet, CentralAgentConfig())
    assert aligned == answer


def test_genuine_evidence_failure_and_load_failure_have_distinct_reasons():
    result = build_agent(FakeCentralRuntime([]), FakeTool("search_history", []), config=CentralAgentConfig(max_action_rounds=0)).chat(QUESTION)
    assert result["status"] == "insufficient_evidence"
    assert result["final_failure_reason"] == "evidence_insufficient"
    def broken():
        raise RuntimeError("fake load failure")
    failure = build_agent(LazyRuntime(broken, name="fake"), FakeTool("search_history", sources())).chat(QUESTION)
    assert failure["final_failure_reason"] == "model_load_failed"
    assert failure["answer"] == FAILURE_ANSWERS["model_load_failed"]


@pytest.mark.parametrize("cold", [False, True])
def test_generation_deadline_does_not_return_evidence_failure(monkeypatch, cold):
    runtime = SleepingRuntime([CentralGeneration(content=ANSWER)], .15)
    agent = build_agent(LazyRuntime(lambda: runtime, name="fake") if cold else runtime,
                        ScheduledTool([], True), config=CentralAgentConfig(timeout_seconds=10))
    generate = agent._generate
    async def almost_expired(state, **kwargs):
        # Exhaust the deadline only after deterministic grounding, independent
        # of how long a busy CI worker takes to run retrieval/validation.
        state.deadline_monotonic = time.monotonic() + .02
        return await generate(state, **kwargs)
    monkeypatch.setattr(agent, "_generate", almost_expired)
    result = agent.chat(QUESTION)
    assert result["status"] == "generation_timeout"
    assert result["central_debug"]["evidence_sufficient"]


def test_history_budget_removes_duplicate_failed_attempts_and_keeps_real_followup():
    history = [{"role": "user", "content": QUESTION}, {"role": "assistant", "content": FAILURE_ANSWERS["answer_validation_failed"], "debug_trace": {"large": "x" * 5000}}]
    assert compact_history(QUESTION, history, max_messages=4, char_budget=500) == []
    compact = compact_history("Còn yếu tố quốc tế thì sao?", history + [{"role": "assistant", "content": FACTS[0]}], max_messages=4, char_budget=500)
    assert any(row["content"] == QUESTION for row in compact)
    assert sum(len(row["content"]) for row in compact) <= 500
    assert all(set(row) == {"role", "content"} for row in compact)


@pytest.mark.parametrize("device,dtype,flash,major,sdpa,expected", [
    ("cuda", "bfloat16", True, 8, True, "flash_attention_2"),
    ("cuda", "bfloat16", False, 8, True, "sdpa"),
    ("cuda", "float32", True, 8, True, "sdpa"),
    ("cpu", "float32", False, 0, False, "eager"),
])
def test_attention_backend_selection_is_capability_gated(device, dtype, flash, major, sdpa, expected):
    assert choose_attention_backend(device=device, dtype=dtype, flash_available=flash, cuda_major=major, sdpa_available=sdpa) == expected


def test_runtime_loading_and_generation_flags_without_loading_weights(monkeypatch, tmp_path):
    import torch
    from app.agents.central_model_runtime import CentralModelRuntime
    import app.agents.hf_cache as cache
    load_kwargs, generation_kwargs = {}, {}
    class Inputs(dict):
        def to(self, device):
            assert str(device) == "cpu"
            return self
    class Tokenizer:
        pad_token_id, eos_token_id = 0, 2
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return "fake prompt"
        def __call__(self, text, **kwargs):
            return Inputs(input_ids=torch.tensor([[1, 2]]))
        def decode(self, tokens, **kwargs):
            return "Fake answer [S1]"
    class Model:
        generation_config = SimpleNamespace(do_sample=True, temperature=.7, top_p=.9, top_k=20)
        def eval(self):
            self.evaluated = True
        def get_input_embeddings(self):
            return SimpleNamespace(weight=SimpleNamespace(device=torch.device("cpu")))
        def generate(self, **kwargs):
            assert self.evaluated and torch.is_inference_mode_enabled()
            generation_kwargs.update(kwargs)
            return torch.tensor([[1, 2, 42]])
    def load_model(*args, **kwargs):
        load_kwargs.update(kwargs)
        return Model()
    fake = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model), StoppingCriteria=object, StoppingCriteriaList=list)
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setitem(sys.modules, "transformers.utils", SimpleNamespace(is_flash_attn_2_available=lambda: False))
    monkeypatch.setattr(cache, "hf_cache_status", lambda *a, **k: {"cache_hit": True})
    monkeypatch.setattr(cache, "resolve_hf_hub_cache_dir", lambda *a: tmp_path)
    runtime = CentralModelRuntime(model_id="Qwen/Qwen3-8B", device="cpu", dtype="float32")
    assert load_kwargs["low_cpu_mem_usage"] and load_kwargs["use_safetensors"]
    assert load_kwargs["attn_implementation"] == "sdpa"
    assert runtime.adapter_loaded is False
    runtime.generate(messages=[], tools=[], max_new_tokens=128)
    assert generation_kwargs["use_cache"] and generation_kwargs["do_sample"] is False
    for field in ("output_scores", "output_hidden_states", "output_attentions", "return_dict_in_generate"):
        assert generation_kwargs[field] is False
