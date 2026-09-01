from __future__ import annotations

import copy
from collections import deque

from pydantic import BaseModel, Field

from app.agents.central_agent import CentralAgent, INSUFFICIENT_EVIDENCE_ANSWER
from app.agents.central_model_runtime import CentralGeneration, CentralToolCall, parse_central_generation
from app.agents.central_model_runtime import parse_central_generation_detailed
from app.agents.central_question import analyze_central_question
from app.agents.config import CentralAgentConfig
from app.tools.registry import ToolRegistry
from app.telemetry import GenerationMetric, RequestTelemetry


class QueryInput(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


class FetchInput(BaseModel):
    page_id_or_title: str = Field(min_length=1)
    max_chars: int = 6000


class FakeTool:
    def __init__(self, name, result=None, error: Exception | None = None):
        self.name = name
        self.description = f"fixture {name}"
        self.input_schema = FetchInput if name == "fetch_wikipedia_page" else QueryInput
        self.result = result or []
        self.error = error
        self.calls = 0

    def run(self, _arguments):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


class FakeCentralRuntime:
    model_id = "Qwen/Qwen3-8B"
    adapter_loaded = True

    def __init__(self, outputs):
        self.outputs = deque(outputs)
        self.calls = []
        self.cache_info = {
            "central_cache_root": "/hf-cache/hub",
            "central_cache_hit": True,
            "central_cache_miss": False,
        }

    def generate(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        return self.outputs.popleft()


def call(name: str, arguments: dict, call_id: str = "call_1") -> CentralGeneration:
    return CentralGeneration(tool_calls=(CentralToolCall(call_id, name, arguments),))


def agent(runtime, *tools, config=None, has_documents=None):
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return CentralAgent(
        model_runtime=runtime,
        tool_registry=registry,
        config=config or CentralAgentConfig(max_steps=3),
        has_uploaded_documents=has_documents,
    )


def test_central_tool_loop_calls_history_then_same_model_writes_final():
    history = FakeTool("search_history", [{
        "chunk_id": "hist_938", "title": "Bạch Đằng", "source_kind": "history",
        "text": "Ngô Quyền đánh bại quân Nam Hán trên sông Bạch Đằng năm 938.",
    }])
    runtime = FakeCentralRuntime([
        call("search_history", {"query": "Bạch Đằng 938", "top_k": 5}),
        CentralGeneration(content="Chiến thắng đánh bại quân Nam Hán. [hist_938]"),
    ])

    result = agent(runtime, history).chat("Bạch Đằng năm 938 diễn ra thế nào?")

    assert len(runtime.calls) == 2
    assert history.calls == 1
    assert runtime.calls[1]["messages"][-1]["role"] == "tool"
    assert "hist_938" in runtime.calls[1]["messages"][-1]["content"]
    assert result["source_ids"] == ["hist_938"]
    assert result["answer_provenance"]["central_model_calls"] == 2
    assert result["answer_provenance"]["research_generation_calls"] == 0
    assert result["answer_provenance"]["evidence_generation_calls"] == 0
    assert result["answer_provenance"]["history_generation_calls"] == 0
    assert result["central_debug"]["tool_schema_count"] == 1
    assert result["central_debug"]["tools_exposed_to_model"] == ["search_history"]


def test_vietnamese_comparison_parser_extracts_both_targets():
    analysis = analyze_central_question("So sánh Cách mạng Tháng Tám và chiến thắng Điện Biên Phủ.")

    assert analysis.question_type == "comparison"
    assert analysis.analytical is True
    assert analysis.comparison_targets == ("Cách mạng Tháng Tám", "chiến thắng Điện Biên Phủ")


def test_vietnamese_question_parser_recognizes_major_analytical_facets():
    assert analyze_central_question("Vì sao nhà Trần thắng quân Nguyên?").question_type == "cause"
    assert analyze_central_question("Sự kiện này có ý nghĩa lịch sử gì?").question_type == "significance"
    assert analyze_central_question("Hệ quả của Hiệp định Genève là gì?").question_type == "consequence"
    assert analyze_central_question("Đánh giá vai trò của Ngô Quyền.").question_type == "significance"


def test_analytical_no_tool_final_gets_one_research_intervention():
    history = FakeTool("search_history", [{
        "chunk_id": "cmt8", "title": "Cách mạng Tháng Tám",
        "text": "Cách mạng Tháng Tám giành chính quyền trên cả nước.",
    }])
    runtime = FakeCentralRuntime([
        CentralGeneration(content="Hai sự kiện đều rất quan trọng."),
        call("search_history", {"query": "Cách mạng Tháng Tám Điện Biên Phủ", "top_k": 6}),
        CentralGeneration(content=(
            "Cách mạng Tháng Tám và chiến thắng Điện Biên Phủ đều là mốc lớn. "
            "Điểm giống nhau là đều làm thay đổi cục diện chính trị. "
            "Điểm khác biệt là một bên là cách mạng giành chính quyền, một bên là thắng lợi quân sự. "
            "Vì vậy ý nghĩa lịch sử của chúng nằm ở hai tầng khác nhau. [cmt8]"
        )),
    ])

    result = agent(runtime, history).chat("So sánh Cách mạng Tháng Tám và chiến thắng Điện Biên Phủ.")

    assert len(runtime.calls) == 3
    assert history.calls == 1
    assert result["analysis"]["question_type"] == "comparison"
    assert result["answer_provenance"]["analytical_no_tool_intervention_attempted"] is True
    assert result["answer_provenance"]["analytical_no_tool_intervention_used"] is True
    assert "search_history" in runtime.calls[1]["messages"][-1]["content"]


def test_central_wikipedia_search_fetch_and_final_are_bounded_to_three_calls():
    search = FakeTool("search_wikipedia", [{
        "chunk_id": "wiki_vi_1", "title": "Trận Bạch Đằng (938)", "source_kind": "wikipedia",
        "text": "Kết quả tìm kiếm.",
    }])
    fetch = FakeTool("fetch_wikipedia_page", {
        "chunk_id": "wiki_vi_1", "title": "Trận Bạch Đằng (938)", "source_kind": "wikipedia",
        "text": "Bài viết đầy đủ về chiến thắng và nền tự chủ.",
    })
    runtime = FakeCentralRuntime([
        call("search_wikipedia", {"query": "Trận Bạch Đằng 938"}, "wiki_search"),
        call("fetch_wikipedia_page", {"page_id_or_title": "Trận Bạch Đằng (938)"}, "wiki_fetch"),
        CentralGeneration(content="Nguồn đối chiếu xác nhận ý nghĩa của chiến thắng. [wiki_vi_1]"),
    ])

    result = agent(runtime, search, fetch).chat("Hãy kiểm chứng Bạch Đằng 938")

    assert len(runtime.calls) == 3
    assert search.calls == fetch.calls == 1
    assert result["source_ids"] == ["wiki_vi_1"]
    assert "Bài viết đầy đủ" in result["source_chunks"][0]["text"]
    assert result["answer_provenance"]["central_external_results_count"] == 2


def test_web_tool_failure_becomes_observation_and_does_not_crash():
    web = FakeTool("search_web", error=RuntimeError("provider unavailable"))
    runtime = FakeCentralRuntime([
        call("search_web", {"query": "tin hiện tại"}),
        CentralGeneration(content="Không thể xác minh nguồn trực tuyến lúc này."),
    ])

    result = agent(runtime, web).chat("Hãy xác minh thông tin hiện tại")

    assert result["status"] == "ok"
    assert "provider unavailable" in runtime.calls[1]["messages"][-1]["content"]


def test_duplicate_tool_call_is_prevented_without_second_execution():
    history = FakeTool("search_history", [{"chunk_id": "h1", "text": "evidence"}])
    runtime = FakeCentralRuntime([
        call("search_history", {"query": "q"}, "one"),
        call("search_history", {"query": "q"}, "two"),
        CentralGeneration(content="Kết luận. [h1]"),
    ])

    result = agent(runtime, history).chat("q lịch sử")

    assert history.calls == 1
    assert result["central_debug"]["tools"][1]["error"] == "duplicate_tool_call_prevented"


def test_invalid_tool_and_invalid_citation_are_rejected_cleanly():
    runtime = FakeCentralRuntime([
        call("invented_tool", {"query": "q"}),
        CentralGeneration(content="Không có nguồn hợp lệ [fake_123]."),
    ])

    result = agent(runtime).chat("q lịch sử")

    assert result["source_ids"] == []
    assert result["invalid_source_ids"] == ["fake_123"]
    assert "[fake_123]" not in result["answer"]
    assert result["central_debug"]["tools"][0]["error"] == "tool_not_available"


def test_malformed_numeric_tool_limits_are_bounded_without_crashing_request():
    history = FakeTool("search_history", [{"chunk_id": "h1", "text": "evidence"}])
    runtime = FakeCentralRuntime([
        call("search_history", {"query": "q", "top_k": "not-a-number"}),
        CentralGeneration(content="Kết luận. [h1]"),
    ])

    result = agent(runtime, history).chat("q lịch sử")

    assert result["status"] == "ok"
    assert result["central_debug"]["tools"][0]["arguments"]["top_k"] == 6


def test_max_step_enforcement_returns_bounded_fallback():
    history = FakeTool("search_history", [{"chunk_id": "h1", "text": "evidence"}])
    runtime = FakeCentralRuntime([
        call("search_history", {"query": "q1"}, "one"),
        call("search_history", {"query": "q2"}, "two"),
    ])
    bounded = CentralAgentConfig(max_steps=2, hard_max_steps=3)

    result = agent(runtime, history, config=bounded).chat("q lịch sử")

    assert len(runtime.calls) == 2
    assert result["answer"] == INSUFFICIENT_EVIDENCE_ANSWER
    assert result["status"] == "insufficient_evidence"


def test_empty_output_gets_at_most_one_repair_generation():
    runtime = FakeCentralRuntime([
        CentralGeneration(content=""),
        CentralGeneration(content="Câu trả lời sau sửa protocol."),
    ])

    result = agent(runtime).chat("Một câu hỏi lịch sử đơn giản")

    assert len(runtime.calls) == 2
    assert result["answer"] == "Câu trả lời sau sửa protocol."
    assert result["answer_provenance"]["repair_generation_attempted"] is True


def test_uploaded_document_tool_schema_is_hidden_when_conversation_has_no_documents():
    documents = FakeTool("search_uploaded_documents", [])
    runtime = FakeCentralRuntime([CentralGeneration(content="Trả lời trực tiếp.")])

    agent(runtime, documents, has_documents=lambda _owner, _conversation: False).chat(
        "Hỏi về file", owner_id="owner", conversation_id="conversation",
    )

    tool_names = {item["function"]["name"] for item in runtime.calls[0]["tools"]}
    assert "search_uploaded_documents" not in tool_names


def test_uploaded_document_tool_schema_is_available_only_when_documents_exist():
    documents = FakeTool("search_uploaded_documents", [])
    runtime = FakeCentralRuntime([CentralGeneration(content="Trả lời trực tiếp.")])

    agent(runtime, documents, has_documents=lambda _owner, _conversation: True).chat(
        "Hỏi về file", owner_id="owner", conversation_id="conversation",
    )

    tool_names = {item["function"]["name"] for item in runtime.calls[0]["tools"]}
    assert "search_uploaded_documents" in tool_names


def test_bach_dang_analytical_prompt_supports_deep_grounded_answer_contract():
    evidence = FakeTool("search_history", [{
        "chunk_id": "bd938", "title": "Chiến thắng Bạch Đằng năm 938",
        "text": "Chiến thắng đánh bại Nam Hán, củng cố nền tự chủ và vị thế Ngô Quyền.",
    }])
    deep_answer = " ".join(["Phân tích lịch sử nhiều chiều"] * 90) + " [bd938]"
    runtime = FakeCentralRuntime([
        call("search_history", {"query": "ý nghĩa Bạch Đằng 938"}),
        CentralGeneration(content=deep_answer),
    ])

    result = agent(runtime, evidence).chat("Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?")

    system_prompt = runtime.calls[0]["messages"][0]["content"]
    assert "300-700" in system_prompt
    assert "ý nghĩa chính trị" in system_prompt
    assert len(result["answer"].split()) >= 250
    assert result["source_ids"] == ["bd938"]


def test_extremely_short_analytical_answer_gets_only_one_repair():
    evidence = FakeTool("search_history", [{"chunk_id": "bd938", "text": "Bằng chứng lịch sử."}])
    repaired = " ".join(["Phân tích có dẫn chứng"] * 70) + " [bd938]"
    runtime = FakeCentralRuntime([
        call("search_history", {"query": "Bạch Đằng 938"}),
        CentralGeneration(content="Chiến thắng rất có ý nghĩa. [bd938]"),
        CentralGeneration(content=repaired),
    ])

    result = agent(runtime, evidence).chat("Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?")

    assert len(runtime.calls) == 3
    assert result["answer_provenance"]["repair_generation_used"] is True
    assert result["answer_provenance"]["repair_reason"] == "analytical_answer_too_shallow"
    assert len(result["answer"].split()) > 200


def test_deep_comparison_guard_repairs_missing_similarity_difference_and_citation():
    evidence = FakeTool("search_history", [{"chunk_id": "cmp1", "text": "Bằng chứng về hai sự kiện."}])
    repaired = (
        "Cách mạng Tháng Tám và chiến thắng Điện Biên Phủ đều có ý nghĩa lịch sử lớn. "
        "Điểm giống nhau là chúng cùng tạo bước ngoặt chính trị cho Việt Nam. "
        "Điểm khác biệt là Cách mạng Tháng Tám thuộc quá trình giành chính quyền, "
        "trong khi chiến thắng Điện Biên Phủ là thắng lợi quân sự quyết định. "
        "Vì vậy, tác động của mỗi sự kiện cần được nhìn trong bối cảnh riêng. [cmp1]"
    )
    runtime = FakeCentralRuntime([
        call("search_history", {"query": "so sánh"}),
        CentralGeneration(content="Hai sự kiện quan trọng."),
        CentralGeneration(content=repaired),
    ])

    result = agent(runtime, evidence).chat("So sánh Cách mạng Tháng Tám và chiến thắng Điện Biên Phủ.")

    assert len(runtime.calls) == 3
    assert result["answer_provenance"]["repair_generation_used"] is True
    assert result["source_ids"] == ["cmp1"]


def test_simple_factual_question_can_stay_direct_and_concise():
    runtime = FakeCentralRuntime([CentralGeneration(content="Ngô Quyền thắng quân Nam Hán năm 938.")])

    result = agent(runtime).chat("Ngô Quyền thắng quân Nam Hán năm nào?")

    assert len(runtime.calls) == 1
    assert result["analysis"]["analytical"] is False
    assert result["answer"] == "Ngô Quyền thắng quân Nam Hán năm 938."


def test_central_request_telemetry_exposes_model_tokens_and_zero_role_calls():
    telemetry = RequestTelemetry(request_id="central-test", inference_mode="central")
    telemetry.add_generation(GenerationMetric(
        adapter="central", input_tokens=321, output_tokens=222, max_new_tokens=1536, generation_ms=50,
    ))

    summary = telemetry.summary(result="success")

    assert summary["central_model_calls"] == 1
    assert summary["central_input_tokens"] == 321
    assert summary["central_output_tokens"] == 222
    assert summary["research_generation_calls"] == 0
    assert summary["evidence_generation_calls"] == 0
    assert summary["history_generation_calls"] == 0


def test_qwen_native_tool_call_tags_parse_into_canonical_call():
    content, calls = parse_central_generation(
        '<tool_call>{"name":"search_history","arguments":{"query":"Bạch Đằng","top_k":5}}</tool_call><|im_end|>'
    )

    assert content == ""
    assert calls[0].name == "search_history"
    assert calls[0].arguments == {"query": "Bạch Đằng", "top_k": 5}


def test_malformed_qwen_tool_call_is_observable_without_exposing_thinking():
    content, calls, failures, malformed = parse_central_generation_detailed(
        '<tool_call>{"name":"search_history","arguments":</tool_call><think>hidden</think>'
    )

    assert content == ""
    assert calls == ()
    assert failures == 1
    assert malformed


def test_hidden_thinking_tags_are_never_exposed_as_final_content():
    content, calls = parse_central_generation(
        "<think>Lập luận nội bộ không được trả về.</think>Câu trả lời công khai."
    )

    assert content == "Câu trả lời công khai."
    assert calls == ()
