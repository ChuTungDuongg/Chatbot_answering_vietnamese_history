from __future__ import annotations

import copy
import threading
import time
from collections import deque

from pydantic import BaseModel, Field

from app.agents.central.agent import CentralAgent, INSUFFICIENT_EVIDENCE_ANSWER, FAILURE_ANSWERS
from app.agents.central.model_runtime import CentralGeneration, CentralToolCall, parse_central_generation_detailed
from app.agents.central.question import analyze_central_question
from app.agents.central.config import CentralAgentConfig
from app.agents.common.lazy_runtime import LazyRuntime
from app.tools.registry import ToolRegistry


class QueryInput(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


class FetchInput(BaseModel):
    page_id_or_title: str = Field(min_length=1)
    max_chars: int = 6000


class FakeTool:
    def __init__(self, name, result=None, *, delay=0.0, error: Exception | None = None):
        self.name = name
        self.description = f"fixture {name}"
        self.input_schema = FetchInput if name == "fetch_wikipedia_page" else QueryInput
        self.result = result if result is not None else []
        self.delay = delay
        self.error = error
        self.calls: list[dict] = []

    def run(self, arguments):
        self.calls.append(arguments.model_dump())
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        return self.result(arguments.model_dump()) if callable(self.result) else self.result


class FakeCentralRuntime:
    model_id = "Qwen/Qwen3-8B"
    adapter_configured = False
    adapter_loaded = False
    adapter_path = None
    adapter_source = "none"

    def __init__(self, outputs):
        self.outputs = deque(outputs)
        self.calls: list[dict] = []
        self.cache_info = {"central_cache_root": "/hf-cache/hub", "central_cache_hit": True}

    def generate(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        return self.outputs.popleft()


class SleepingRuntime(FakeCentralRuntime):
    def __init__(self, outputs, delay):
        super().__init__(outputs)
        self.delay = delay

    def generate(self, **kwargs):
        time.sleep(self.delay)
        return super().generate(**kwargs)


def generation_call(name: str, arguments: dict, call_id: str = "call_1") -> CentralGeneration:
    return CentralGeneration(tool_calls=(CentralToolCall(call_id, name, arguments),), generation_stage="action")


def build_agent(runtime, *tools, config: CentralAgentConfig | None = None, has_documents=None):
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return CentralAgent(
        model_runtime=runtime,
        tool_registry=registry,
        config=config or CentralAgentConfig(repair_max_generations=0),
        has_uploaded_documents=has_documents,
    )


def deep_comparison(source_id: str) -> str:
    sentence = (
        "Hiệp định Genève năm 1954 và Hiệp định Paris năm 1973 có điểm giống nhau là đều gắn với "
        "đấu tranh ngoại giao, nhưng điểm khác biệt nằm ở bối cảnh và nội dung chính; vì vậy kết quả, "
        "tác động và ý nghĩa lịch sử của mỗi hiệp định phải được đặt trong tiến trình riêng. "
    )
    return sentence * 8 + f"[{source_id}]"


def test_factual_biography_and_cause_questions_are_grounded_before_synthesis():
    questions = (
        "Ngô Quyền lên ngôi vào năm nào?",
        "Trương Định là ai và từng giữ vai trò gì?",
        "Vì sao Cách mạng Tháng Tám năm 1945 thành công?",
    )
    for question in questions:
        evidence = "Trương Định là một nhân vật lịch sử."
        if "Vì sao" in question:
            evidence = "Cách mạng Tháng Tám thành công do bối cảnh thuận lợi, lực lượng nhân dân đấu tranh giành chính quyền. Các nguyên nhân chính trị và xã hội góp phần vào kết quả thành công."
        history = FakeTool("search_history", [{"chunk_id": "hist_1", "text": evidence}])
        answer = ("Phân tích nguyên nhân từ bằng chứng. " * 130 if "Vì sao" in question else "Câu trả lời có căn cứ. ") + "[hist_1]"
        runtime = FakeCentralRuntime([CentralGeneration(content=answer, generation_stage="synthesis")])

        result = build_agent(runtime, history).chat(question)

        assert len(history.calls) == (2 if "Vì sao" in question else 1)
        assert len(runtime.calls) == 1
        assert runtime.calls[0]["stage"] == "synthesis"
        assert "Gói bằng chứng:" in runtime.calls[0]["messages"][-1]["content"]
        assert "[S1]" in runtime.calls[0]["messages"][-1]["content"]
        assert result["source_ids"] == ["hist_1"]
        assert result["answer_provenance"]["research_generation_calls"] == 0
        assert result["answer_provenance"]["evidence_generation_calls"] == 0
        assert result["answer_provenance"]["history_generation_calls"] == 0


def test_comparison_targets_are_clean_and_both_are_retrieved_concurrently():
    question = (
        "So sánh Hiệp định Genève năm 1954 và Hiệp định Paris năm 1973 "
        "về bối cảnh, nội dung chính, kết quả và ý nghĩa lịch sử."
    )
    analysis = analyze_central_question(question)
    assert analysis.comparison_targets == ("Hiệp định Genève năm 1954", "Hiệp định Paris năm 1973")
    history = FakeTool(
        "search_history",
        lambda args: [{"chunk_id": "src_a" if "Genève" in args["query"] else "src_b", "text":
                       (analysis.comparison_targets[0] if "Genève" in args["query"] else analysis.comparison_targets[1])
                       + " có bối cảnh đấu tranh ngoại giao, mục tiêu chính trị và kết quả có ý nghĩa lịch sử. Các lực lượng tham gia có phương pháp đấu tranh riêng."}],
        delay=0.06,
    )
    runtime = FakeCentralRuntime([CentralGeneration(content=deep_comparison("src_a") + " [src_b]")])

    result = build_agent(runtime, history).chat(question)
    # Each target's primary is independent; secondary variants form a later stage.
    assert history.calls[0]["query"] == analysis.comparison_targets[0]
    assert history.calls[1]["query"] == analysis.comparison_targets[1]
    assert set(analysis.comparison_targets) <= {call["query"] for call in history.calls}
    assert len(history.calls) == 4
    assert result["status"] == "ok"
    assert result["central_debug"]["initial_grounding_coverage"] == {
        analysis.comparison_targets[0]: 1,
        analysis.comparison_targets[1]: 1,
    }


def test_insufficient_local_evidence_enters_structured_action_then_synthesis():
    history = FakeTool("search_history", [])
    wikipedia = FakeTool("search_wikipedia", [{"chunk_id": "wiki_1", "title": "Trương Định", "text": "Đối chiếu."}])
    fetch = FakeTool("fetch_wikipedia_page", {"chunk_id": "wiki_1", "title": "Trương Định", "text": "Trương Định là một nhân vật lịch sử."})
    runtime = FakeCentralRuntime([
        generation_call("search_wikipedia", {"query": "Trương Định", "top_k": 4}),
        CentralGeneration(content="Thông tin đã được đối chiếu. [wiki_1]", generation_stage="synthesis"),
    ])

    result = build_agent(runtime, history, wikipedia, fetch).chat("Trương Định là ai?")

    assert [call["stage"] for call in runtime.calls] == ["action", "synthesis"]
    assert runtime.calls[0]["max_new_tokens"] == 256
    assert runtime.calls[1]["max_new_tokens"] == 1536
    assert runtime.calls[0]["tools"][0]["type"] == "function"
    assert wikipedia.calls[0]["query"] == "Trương Định"
    assert len(fetch.calls) == 1
    assert result["source_ids"] == ["wiki_1"]
    assert result["central_debug"]["phase_trace"] == [
        "prepare", "initial_grounding", "action", "tool_execution", "synthesis", "final",
    ]


def test_multiple_independent_action_calls_execute_in_parallel():
    history = FakeTool("search_history", [])
    rendezvous = threading.Barrier(2)
    def search(args):
        # Both calls must enter before either completes; total test wall time
        # includes unrelated scheduler/import overhead and cannot prove this.
        rendezvous.wait(timeout=2)
        return [{"chunk_id": "wiki_" + args["query"], "text": args["query"]}]
    wikipedia = FakeTool("search_wikipedia", search)
    runtime = FakeCentralRuntime([
        CentralGeneration(tool_calls=(
            CentralToolCall("a", "search_wikipedia", {"query": "A"}),
            CentralToolCall("b", "search_wikipedia", {"query": "B"}),
        )),
        CentralGeneration(content="Đối chiếu A và B. [wiki_A] [wiki_B]"),
    ])

    result = build_agent(runtime, history, wikipedia).chat("Đối chiếu sự kiện lịch sử này")
    assert not rendezvous.broken
    assert len(wikipedia.calls) == 2
    assert result["answer_provenance"]["central_tool_calls_by_type"]["search_wikipedia"] == 2


def test_duplicate_and_invalid_calls_become_bounded_observations_without_execution():
    history = FakeTool("search_history", [])
    duplicate_runtime = FakeCentralRuntime([
        generation_call("search_history", {"query": "q lịch sử", "top_k": 6}),
    ])
    duplicate = build_agent(
        duplicate_runtime,
        history,
        config=CentralAgentConfig(max_action_rounds=1, repair_max_generations=0),
    ).chat("q lịch sử")
    assert len(history.calls) == 1
    assert duplicate["central_debug"]["tools"][1]["error"] == "duplicate_tool_call_prevented"

    invalid_runtime = FakeCentralRuntime([generation_call("invented_tool", {"query": "q"})])
    invalid_history = FakeTool("search_history", [])
    invalid = build_agent(
        invalid_runtime,
        invalid_history,
        config=CentralAgentConfig(max_action_rounds=1, repair_max_generations=0),
    ).chat("q lịch sử")
    assert invalid["status"] == "tool_failed"
    invalid_events = invalid["central_debug"]["tools"]
    assert invalid_events[0]["name"] == "search_history"
    assert invalid_events[1]["error"] == "tool_not_available"


def test_tool_error_is_recoverable_and_action_rounds_are_bounded():
    history = FakeTool("search_history", [])
    wikipedia = FakeTool("search_wikipedia", error=RuntimeError("network timeout"))
    runtime = FakeCentralRuntime([
        generation_call("search_wikipedia", {"query": "q"}),
        CentralGeneration(content="No more calls"),
    ])
    result = build_agent(runtime, history, wikipedia).chat("q lịch sử")

    assert len(runtime.calls) == 2
    assert result["status"] == "tool_failed"
    assert result["central_debug"]["tools"][1]["error"] == "network timeout"
    assert any("network timeout" in str(message.get("content")) for message in runtime.calls[1]["messages"])


def test_local_only_web_provider_hides_web_tools_but_keeps_wikipedia():
    history = FakeTool("search_history", [])
    web = FakeTool("search_web", [])
    wikipedia = FakeTool("search_wikipedia", [])
    runtime = FakeCentralRuntime([CentralGeneration(content="No call")])
    result = build_agent(runtime, history, web, wikipedia).chat("q lịch sử")

    exposed = {item["function"]["name"] for item in runtime.calls[0]["tools"]}
    assert "search_web" not in exposed
    assert "search_wikipedia" in exposed
    assert "search_web" not in result["central_debug"]["allowed_tools"]


def test_quality_repair_is_single_and_uses_repair_budget():
    history = FakeTool("search_history", [{"chunk_id": "h1", "text": "evidence"}])
    runtime = FakeCentralRuntime([
        CentralGeneration(content="Nguyễn Huệ sinh năm 999. [h1]"),
        CentralGeneration(content=("Phân tích có bằng chứng. " * 140) + "[h1]"),
    ])
    config = CentralAgentConfig(repair_max_generations=1)
    result = build_agent(runtime, history, config=config).chat("Vì sao sự kiện này thành công?")

    assert [call["stage"] for call in runtime.calls] == ["synthesis", "quality_repair"]
    assert runtime.calls[1]["max_new_tokens"] == 192
    assert result["answer_provenance"]["repair_generation_used"] is True


def test_generation_stop_telemetry_and_numeric_brackets_are_preserved():
    history = FakeTool("search_history", [{"chunk_id": "h1", "text": "Năm 1945 là một mốc lịch sử."}])
    runtime = FakeCentralRuntime([CentralGeneration(
        content="Năm [1945] là một mốc; xem nguồn [h1].",
        generation_stage="synthesis",
        generation_stop_reason="token_limit",
        generation_hit_token_limit=True,
    )])
    result = build_agent(runtime, history).chat("Mốc lịch sử nào?")

    assert "[1945]" in result["answer"]
    assert result["source_ids"] == ["h1"]
    metric = result["performance_debug"]["generation_metrics"][0]
    assert metric["generation_stage"] == "synthesis"
    assert metric["generation_stop_reason"] == "token_limit"
    assert metric["generation_hit_token_limit"] is True


def test_model_load_timeout_is_separate_and_lazy_initialization_is_single_flight():
    loads: list[str] = []
    history = FakeTool("search_history", [{"chunk_id": "h1", "text": "evidence"}])

    def build():
        time.sleep(0.03)
        loads.append("central")
        return FakeCentralRuntime([
            CentralGeneration(content="Một. [h1]"),
            CentralGeneration(content="Hai. [h1]"),
        ])

    lazy = LazyRuntime(build, name="central")
    central = build_agent(lazy, history)
    answers: list[str] = []
    threads = [threading.Thread(target=lambda q=q: answers.append(central.chat(q)["answer"])) for q in ("q1", "q2")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
    assert loads == ["central"]
    assert len(answers) == 2

    slow = LazyRuntime(lambda: (time.sleep(0.05), FakeCentralRuntime([]))[1], name="slow")
    timeout = build_agent(
        slow,
        history,
        config=CentralAgentConfig(model_load_timeout_seconds=0.01, repair_max_generations=0),
    ).chat("q")
    assert timeout["answer_provenance"]["timeout_stage"] == "model_initialization"


def test_agent_timeout_reports_generation_stage():
    history = FakeTool("search_history", [{"chunk_id": "h1", "text": "evidence"}])
    runtime = SleepingRuntime([CentralGeneration(content="late [h1]")], delay=0.4)
    result = build_agent(
        runtime,
        history,
        config=CentralAgentConfig(timeout_seconds=0.2, repair_max_generations=0),
    ).chat("q")
    assert result["answer"] == FAILURE_ANSWERS["generation_timeout"]
    assert result["answer_provenance"]["source"] == "central_timeout"
    assert result["answer_provenance"]["timeout_stage"] == "generation_synthesis"


def test_hermes_codec_handles_parallel_calls_and_reports_malformed_frames():
    content, calls, failures, malformed = parse_central_generation_detailed(
        '<tool_call>{"name":"a","arguments":{"x":1}}</tool_call>'
        '<tool_call>{"name":"b","arguments":{"y":2}}</tool_call>'
        '<tool_call>{bad}</tool_call>'
    )
    assert content == ""
    assert [(call.name, call.arguments) for call in calls] == [("a", {"x": 1}), ("b", {"y": 2})]
    assert failures == 1
    assert malformed == ("{bad}",)
