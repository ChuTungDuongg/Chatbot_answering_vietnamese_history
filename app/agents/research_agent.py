from __future__ import annotations

import hashlib
import json
from typing import Any

from app.agents.model_runtime import SharedAgentModelRuntime
from app.agents.schemas import ResearchResult
from app.rag.generation import RAGGenerator
from app.tools.evidence_tools import SessionEvidenceStore
from app.tools.registry import ToolRegistry


class ResearchAgent:
    """Bounded PLAN/ACTION/OBSERVATION agent with a deterministic fallback."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        evidence_store: SessionEvidenceStore,
        generator: RAGGenerator,
        model_runtime: SharedAgentModelRuntime | None = None,
        max_steps: int = 6,
        max_web_searches: int = 3,
        max_page_fetches: int = 5,
    ):
        self.registry = registry
        self.evidence_store = evidence_store
        self.generator = generator
        self.model_runtime = model_runtime
        self.max_steps = max_steps
        self.max_web_searches = max_web_searches
        self.max_page_fetches = max_page_fetches

    async def run(
        self,
        question: str,
        *,
        final_k: int,
        history: list[dict[str, str]] | None = None,
        session_id: str = "default",
    ) -> ResearchResult:
        normalized_history = self.generator.normalize_history(history, current_question=question)
        retrieval_question, history_used = self.generator.build_retrieval_question(question, normalized_history)
        if self.model_runtime is not None:
            chunks, tool_trace = await self._run_model_policy(
                question=question,
                retrieval_question=retrieval_question,
                session_id=session_id,
                final_k=final_k,
            )
        else:
            chunks, tool_trace = await self._run_fallback(
                question=question,
                retrieval_question=retrieval_question,
                session_id=session_id,
                final_k=final_k,
            )
        retrieval = self.generator.retriever.retrieve(retrieval_question, final_k=final_k)
        analysis = self.generator.retriever.analyze_question(question)
        retrieval["analysis"] = analysis
        retrieval["question"] = question
        retrieval["retrieval_question"] = retrieval_question
        retrieval["history_used_for_retrieval"] = history_used
        retrieval["tool_trace"] = tool_trace
        return ResearchResult(
            question=question,
            evidence=self.evidence_store.all(session_id),
            tool_trace=tool_trace,
            is_ood=bool(retrieval.get("is_ood")),
            ood_reason=str(retrieval.get("ood_reason", "")),
            analysis=analysis,
        )

    async def _run_fallback(
        self,
        *,
        question: str,
        retrieval_question: str,
        session_id: str,
        final_k: int,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        chunks, record = await self.registry.call(
            "search_history",
            {"query": retrieval_question, "top_k": max(final_k, 6)},
        )
        tool_trace = [f"{record.name}:{record.result_count}" if not record.error else f"{record.name}:error"]
        chunks = chunks or []
        self.evidence_store.add_documents(session_id, chunks)
        if not chunks and self.max_steps > 1 and "search_web" in self.registry.names():
            web_rows, web_record = await self.registry.call("search_web", {"query": question, "top_k": 5})
            tool_trace.append(
                f"{web_record.name}:{web_record.result_count}" if not web_record.error else f"{web_record.name}:error"
            )
            self.evidence_store.add_documents(session_id, web_rows or [])
        return chunks, tool_trace

    async def _run_model_policy(
        self,
        *,
        question: str,
        retrieval_question: str,
        session_id: str,
        final_k: int,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        observations: list[dict[str, Any]] = []
        trace: list[str] = []
        web_searches = 0
        page_fetches = 0
        tools = self.registry.describe()
        for step in range(1, self.max_steps + 1):
            state = {
                "question": question,
                "retrieval_question": retrieval_question,
                "step": step,
                "limits": {
                    "max_steps": self.max_steps,
                    "web_searches_left": self.max_web_searches - web_searches,
                    "page_fetches_left": self.max_page_fetches - page_fetches,
                },
                "tools": tools,
                "observations": observations[-4:],
                "evidence_ids": [row.get("chunk_id") for row in self.evidence_store.all(session_id)],
            }
            decision = self.model_runtime.generate_json(
                adapter="research",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a Vietnamese-history research tool policy. Do not answer the history question. "
                            "Return JSON only: either action=tool with tool_name and arguments, or action=finish "
                            "with sufficient and missing_information. Never reveal hidden reasoning."
                        ),
                    },
                    {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
                ],
                max_new_tokens=384,
            )
            if decision.get("action") == "finish":
                trace.append(f"agent:finish:{step}")
                break
            tool_name = str(decision.get("tool_name", ""))
            arguments = dict(decision.get("arguments") or {})
            if tool_name not in self.registry.names():
                observations.append({"tool": tool_name, "error": "unknown_tool"})
                trace.append("agent:unknown_tool")
                continue
            if tool_name == "search_web":
                if web_searches >= self.max_web_searches:
                    observations.append({"tool": tool_name, "error": "web_search_budget_exhausted"})
                    continue
                web_searches += 1
            if tool_name == "fetch_web_page":
                if page_fetches >= self.max_page_fetches:
                    observations.append({"tool": tool_name, "error": "page_fetch_budget_exhausted"})
                    continue
                page_fetches += 1
            if tool_name in {"retrieve_evidence", "inspect_evidence"}:
                arguments["session_id"] = session_id
            result, record = await self.registry.call(tool_name, arguments)
            trace.append(f"{record.name}:{record.result_count}" if not record.error else f"{record.name}:error")
            if record.error:
                observations.append({"tool": tool_name, "error": record.error})
                continue
            rows = self._evidence_rows(tool_name, result)
            self.evidence_store.add_documents(session_id, rows)
            observations.append(
                {
                    "tool": tool_name,
                    "result_count": len(rows),
                    "evidence_ids": [row.get("chunk_id") for row in rows[:10]],
                }
            )
        return self.evidence_store.all(session_id), trace

    @staticmethod
    def _evidence_rows(tool_name: str, result: Any) -> list[dict[str, Any]]:
        rows = result if isinstance(result, list) else [result] if isinstance(result, dict) else []
        normalized: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            item = dict(row)
            if not item.get("chunk_id"):
                identity = str(item.get("url") or item.get("text") or json.dumps(item, sort_keys=True))
                item["chunk_id"] = f"web_{hashlib.sha1(identity.encode('utf-8')).hexdigest()[:12]}"
            item.setdefault("source_kind", "web" if tool_name != "search_history" else "history")
            normalized.append(item)
        return normalized
