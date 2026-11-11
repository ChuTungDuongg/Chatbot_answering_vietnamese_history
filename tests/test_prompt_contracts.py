"""Baseline captured from aba83fd before extracting any prompt literals."""

import json
from pathlib import Path

import pytest

from app.central.planning import PLANNER_INSTRUCTION
from app.rag import prompting, response_modes


BASELINE = json.loads((Path(__file__).parent / "fixtures/prompt_contracts.json").read_text(encoding="utf-8"))


def test_public_prompt_constants_preserve_utf8_bytes():
    for name, value in BASELINE["constants"].items():
        actual = (PLANNER_INSTRUCTION if name == "PLANNER_INSTRUCTION" else
                  getattr(prompting, name, getattr(response_modes, name, None)))
        assert actual.encode("utf-8") == value.encode("utf-8")
    assert response_modes.MODE_INSTRUCTIONS == BASELINE["mode_instructions"]


@pytest.mark.parametrize("mode", [None, "concise", "standard", "detailed"])
def test_grounded_messages_match_pre_refactor_baseline(mode):
    assert prompting.build_messages(BASELINE["question"], BASELINE["contexts"], BASELINE["history"],
                                    response_mode=mode) == BASELINE["messages"][str(mode)]


@pytest.mark.parametrize("mode", ["concise", "standard", "detailed"])
def test_offline_and_sft_messages_match_pre_refactor_baseline(mode):
    assert prompting.build_no_rag_messages(BASELINE["question"], response_mode=mode) == BASELINE["no_rag_messages"][mode]
    assert response_modes.sft_system_instruction(mode) == BASELINE["sft"][mode]


def test_all_additional_instructions_preserve_composition_boundaries():
    from app.prompts.loader import load_prompt
    from app.central.runtime import EXTERNAL_DOMAIN_INSTRUCTION, GROUNDED_FINAL_INSTRUCTION
    from app.central.planning import COMPLETION_DESCRIPTION
    from evaluation.annotation.silver_logic import SEMANTIC_INSTRUCTIONS, JSON_SUFFIX

    for name, expected in BASELINE["extra_instructions"].items():
        assert load_prompt(name).encode() == expected.encode()
    assert EXTERNAL_DOMAIN_INSTRUCTION == BASELINE["extra_instructions"]["central/external_domain"]
    assert GROUNDED_FINAL_INSTRUCTION == BASELINE["extra_instructions"]["central/grounded_final"]
    assert COMPLETION_DESCRIPTION == BASELINE["extra_instructions"]["central/completion_parameter"]
    assert SEMANTIC_INSTRUCTIONS == BASELINE["semantic_instructions"]
    assert JSON_SUFFIX == BASELINE["extra_instructions"]["offline/semantic/json_suffix"]


@pytest.mark.parametrize("stage", list(BASELINE["semantic_instructions"]))
def test_semantic_provider_sends_exact_pre_refactor_instruction(monkeypatch, stage):
    import io
    from evaluation.annotation import silver_logic

    captured = []
    def respond(request, timeout):
        captured.append(json.loads(request.data))
        return io.StringIO(json.dumps({"choices": [{"message": {"content": "{}"}}]}))
    monkeypatch.setattr(silver_logic, "urlopen", respond)
    provider = silver_logic.JsonChatProvider(kind="local", model="mock-model", revision=None,
                                             base_url="http://localhost:8000/v1")
    assert provider.complete(stage, {"question": BASELINE["question"]}) == {}
    assert captured[0]["messages"][0] == {
        "role": "system", "content": BASELINE["semantic_instructions"][stage]
        + BASELINE["extra_instructions"]["offline/semantic/json_suffix"] + silver_logic.PROMPT_VERSION}
