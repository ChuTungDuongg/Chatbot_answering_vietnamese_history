"""Reproducible old/new harness comparison using identical scripted decisions.

This measures orchestration, not LLM intelligence or historical answer quality.
The baseline is loaded read-only from the specified reviewed Git commit.
"""

import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path
import statistics
import subprocess
import time
from types import ModuleType, SimpleNamespace

from app.central.runtime import CentralRuntime
from app.config import CENTRAL_MODEL_ID
from app.models.base import ModelDone
from app.models.openai_responses import OpenAIOptions, OpenAIResponsesRuntime
from app.tools.local_search import SearchHistoryInput
from app.tools.wikipedia import SearchWikipediaInput, FetchWikipediaPageInput
from app.tools.registry import ToolExecutionPolicy, ToolRegistry


ROOT = Path(__file__).resolve().parents[2]
BASELINE = "d8bc36a51ee5d158b95c1eb2cb863dcd431deb16"
CASES_PATH = Path(__file__).with_name("cases.json")


def load_baseline():
    def module(path):
        try:
            source = subprocess.check_output(["git", "show", f"{BASELINE}:{path}"], cwd=ROOT, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            raise RuntimeError(f"Fetch baseline first: git fetch origin {BASELINE}") from None
        result = ModuleType("reviewed_agent_baseline")
        exec(compile(source, f"{BASELINE}:{path}", "exec"), result.__dict__)
        return result
    planning, runtime = module("app/central/planning.py"), module("app/central/runtime.py")
    for name in ("PLANNER_INSTRUCTION", "evidence_chunks", "final_after_tools", "question_domain",
                 "terminal_evidence", "planner_tool_schemas", "COMPLETION_FIELD"):
        setattr(runtime, name, getattr(planning, name))
    return runtime.CentralRuntime


def load_cases():
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]


def arguments(plan, schemas):
    value = ({"page_id_or_title": plan["page"]} if "page" in plan else {"query": plan.get("query", "")})
    functions = [schema.get("function", schema) for schema in schemas]
    properties = next((f.get("parameters", {}).get("properties", {}) for f in functions if f["name"] == plan["name"]), {})
    if plan.get("finish"):
        value["__central_final_after_success"] = True
    if plan.get("independent") and "__central_independent" in properties:
        value["__central_independent"] = True
    return value


class ScriptedModel:
    model_id = CENTRAL_MODEL_ID

    def __init__(self, case):
        self.case, self.calls, self.prompt_chars = case, 0, 0

    async def generate(self, messages, *, tools, max_new_tokens, cancel):
        started = time.perf_counter_ns()
        self.prompt_chars += len(json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False))
        plans = self.case["plans"][self.calls] if self.calls < len(self.case["plans"]) else []
        self.calls += 1
        await asyncio.sleep(.001)
        text = "".join("<tool_call>" + json.dumps({"name": p["name"], "arguments": arguments(p, tools)}) + "</tool_call>" for p in plans)
        # Simulated provider usage is deliberately labeled in the output report.
        return text, ModelDone(self.model_id, "fixture", started, started + 1, time.perf_counter_ns(),
                               100, 20, max_new_tokens, "stop")


class FakeStream:
    def __init__(self, output):
        self.event = SimpleNamespace(type="response.completed", response=SimpleNamespace(status="completed",
            output=output, usage=SimpleNamespace(input_tokens=100, output_tokens=20,
            input_tokens_details=SimpleNamespace(cached_tokens=0), output_tokens_details=SimpleNamespace(reasoning_tokens=0))))
    def __aiter__(self): return self
    async def __anext__(self):
        if self.event is None: raise StopAsyncIteration
        event, self.event = self.event, None
        return event
    async def close(self): pass


class ScriptedClient:
    def __init__(self, case):
        self.case, self.requests = case, []
        self.responses = self
    async def create(self, **kwargs):
        index = len(self.requests)
        self.requests.append(kwargs)
        plans = self.case["plans"][index] if index < len(self.case["plans"]) else []
        output = [{"type": "reasoning", "encrypted_content": "fixture-only"}]
        output += [{"type": "function_call", "name": plan["name"], "call_id": f"r{index}_c{ordinal}",
                    "arguments": "invalid" if plan.get("malformed") else json.dumps(arguments(plan, kwargs.get("tools", [])))}
                   for ordinal, plan in enumerate(plans)]
        await asyncio.sleep(.001)
        return FakeStream(output)


class FixtureTool:
    execution_policy = ToolExecutionPolicy(read_only=True, parallel_safe=True)
    description = "Offline evidence fixture"

    def __init__(self, name, case, log):
        self.name, self.case, self.log = name, case, log
        self.input_schema = {"search_history": SearchHistoryInput, "search_wikipedia": SearchWikipediaInput,
                             "fetch_wikipedia_page": FetchWikipediaPageInput}[name]
    async def run(self, args):
        key = getattr(args, "query", getattr(args, "page_id_or_title", ""))
        self.log.append({"name": self.name, "key": key, "started_ns": time.perf_counter_ns()})
        row = self.log[-1]
        await asyncio.sleep(.012)
        row["finished_ns"] = time.perf_counter_ns()
        return [{"chunk_id": key_id, "text": f"Verified fixture evidence: {key_id}", "title": key_id,
                 "source_kind": "wikipedia" if self.name != "search_history" else "history",
                 "metadata": {"page_id": 77, "language": "vi"} if key_id == "wiki_vi_77" else {}}
                for key_id in self.case["results"].get(key, [])]


async def evaluate_case(case, runtime_class=CentralRuntime):
    registry, log = ToolRegistry(), []
    for name in case["tools"]:
        registry.register(FixtureTool(name, case, log))
    native = case.get("provider") == "openai"
    client = ScriptedClient(case) if native else None
    model = OpenAIResponsesRuntime(client, OpenAIOptions()) if native else ScriptedModel(case)
    started = time.perf_counter_ns()
    error, result = None, None
    try:
        result = await runtime_class(model=model, tools=registry).prepare(case["question"], 6, [])
    except RuntimeError as exc:
        error = type(exc).__name__  # No raw model/tool content in the report.
    ids = {c["chunk_id"] for c in result.contexts} if result else set()
    coverage = len(ids.intersection(case["required"])) / len(case["required"])
    confirmations = result and result.planning["exit_reason"] in {"evidence_sufficient", "planner_no_tools"}
    dependencies_ok = all(row["name"] != "fetch_wikipedia_page" or any(
        prior["name"] == "search_wikipedia" and prior["finished_ns"] <= row["started_ns"] and row["key"] == "77"
        for prior in log if "finished_ns" in prior) for row in log)
    closure_ok = None if native and result is None else True
    if native and result:
        transcript = result.messages
        requested = [m["call_id"] for m in transcript if m.get("type") == "function_call"]
        returned = [m["call_id"] for m in transcript if m.get("type") == "function_call_output"]
        closure_ok = requested == returned and len(set(requested)) == len(requested)
    repeated = len(log) - len({(row["name"], row["key"]) for row in log})
    valid_args = sum(row["key"] in case["results"] and row["key"] not in case.get("invalid_arguments", []) for row in log)
    # Feed a fixed candidate (including a deliberately invalid reference) through
    # the production citation filter. This checks provenance, not answer semantics.
    from app.api.routes import CitationFilter
    candidate = " ".join(f"Fixture claim [{key}]." for key in [*case["required"], "invented_source"])
    filtered = CitationFilter(result.contexts if result else []).feed(candidate, final=True)
    cited = re.findall(r"\[([^\]]+)\]", filtered)
    return {"id": case["id"], "error": error, "task_complete": bool(coverage == 1 and confirmations and dependencies_ok and closure_ok),
        "evidence_coverage": coverage, "dependency_order_valid": dependencies_ok, "native_output_correlation_valid": closure_ok,
        "tool_selection_valid": all(row["name"] in case["tools"] for row in log),
        "argument_contract_valid_fraction": valid_args / len(log) if log else None,
        "fixture_citation_precision": sum(key in ids for key in cited) / len(cited) if cited else None,
        "fixture_citation_recall": len(cited) / len(case["required"]),
        "tool_calls": len(log), "duplicate_calls": repeated,
        "excess_calls": max(0, len(log) - case["maximum_useful_calls"]),
        "parallel_batches": result.planning.get("parallel_batches", 0) if result else 0,
        "model_calls": len(client.requests) if native else model.calls,
        "planner_input_chars": sum(len(json.dumps({"messages": r["input"], "tools": r.get("tools")}, ensure_ascii=False)) for r in client.requests) if native else model.prompt_chars,
        "simulated_input_tokens": (len(client.requests) if native else model.calls) * 100,
        "simulated_output_tokens": (len(client.requests) if native else model.calls) * 20,
        "planning_wall_ms": result.planning["total_wall_ms"] if result else None,
        "wall_ms": (time.perf_counter_ns() - started) / 1e6}


def source_digest():
    paths = sorted([*ROOT.glob("app/central/*.py"), *ROOT.glob("app/prompts/central/*.md"),
                    *ROOT.glob("app/tools/builtin/*/tool.py"), ROOT / "app/tools/definitions/tools.yaml",
                    ROOT / "app/tools/registry.py", ROOT / "app/models/openai_responses.py",
                    ROOT / "app/api/routes.py", ROOT / "app/mcp/manager.py",
                    ROOT / "app/models/qwen.py",
                    ROOT / "app/models/tool_calls.py", ROOT / "app/security/guardrails.py", Path(__file__), CASES_PATH])
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


async def compare(repeats):
    baseline = load_baseline()
    variants = {}
    for label, runtime in (("baseline", baseline), ("agentic_v2", CentralRuntime)):
        rows = []
        for case in load_cases():
            runs = [await evaluate_case(case, runtime) for _ in range(repeats)]
            row = dict(runs[-1])
            for key in ("wall_ms", "planning_wall_ms"):
                values = [r[key] for r in runs if r[key] is not None]
                row[key] = round(statistics.median(values), 3) if values else None
            rows.append(row)
        variants[label] = {"cases": rows, "summary": {
            "completed": sum(row["task_complete"] for row in rows), "cases": len(rows),
            "dependency_valid": sum(row["dependency_order_valid"] for row in rows),
            "excess_calls": sum(row["excess_calls"] for row in rows),
            "duplicate_calls": sum(row["duplicate_calls"] for row in rows),
            "model_calls": sum(row["model_calls"] for row in rows),
            "simulated_input_tokens": sum(row["simulated_input_tokens"] for row in rows),
            "simulated_output_tokens": sum(row["simulated_output_tokens"] for row in rows)}}
    return {"baseline_commit": BASELINE, "source_sha256": source_digest(), "repeats": repeats,
            "measurement": "Scripted decisions, simulated usage, measured mock wall time; baseline Central planning/runtime only, shared current adapters/registry/citation filter; no live LLM quality or USD estimate.",
            "usd_cost": None, "variants": variants}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("evaluation/agentic/results.json"))
    args = parser.parse_args()
    if not 1 <= args.repeats <= 30:
        parser.error("repeats must be between 1 and 30")
    result = asyncio.run(compare(args.repeats))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: variant["summary"] for name, variant in result["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()
