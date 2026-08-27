from training.research_agent.converters.agentinstruct import convert_agentinstruct


def test_agentinstruct_unrolls_official_conversations_shape():
    row = {
        "id": "os_1",
        "conversations": [
            {"from": "human", "value": "Count files"},
            {"from": "gpt", "value": "Think: list\n\nAct: bash\n\n```bash\nls -1 | wc -l\n```"},
            {"from": "human", "value": "The output of the OS:\n12"},
            {"from": "gpt", "value": "Think: done\n\nAct: answer(12)"},
        ],
    }
    rows = convert_agentinstruct(row, source_split="os")
    assert [item["training_target"]["action"] for item in rows] == ["tool", "finish"]
    assert rows[0]["training_target"]["arguments"]["command"] == "ls -1 | wc -l"
    assert rows[1]["training_prompt"]["observations"][0]["result"].endswith("12")
