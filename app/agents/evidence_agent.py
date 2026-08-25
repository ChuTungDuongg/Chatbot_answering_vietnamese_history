from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from app.agents.model_runtime import SharedAgentModelRuntime
from app.agents.schemas import EvidenceCritique, SelectedEvidence


class EvidenceCriticAgent:
    def __init__(
        self,
        *,
        max_contexts: int = 8,
        model_runtime: SharedAgentModelRuntime | None = None,
    ):
        self.max_contexts = max_contexts
        self.model_runtime = model_runtime

    def compress(self, question: str, evidence: list[dict[str, Any]], *, final_k: int) -> tuple[EvidenceCritique, list[dict[str, Any]]]:
        if self.model_runtime is not None:
            try:
                return self._model_compress(question, evidence, final_k=final_k)
            except (ValueError, ValidationError, KeyError, TypeError):
                critique, contexts = self._deterministic_compress(evidence, final_k=final_k)
                critique.warnings.append("model_output_invalid_fallback_used")
                return critique, contexts
        return self._deterministic_compress(evidence, final_k=final_k)

    def _deterministic_compress(
        self,
        evidence: list[dict[str, Any]],
        *,
        final_k: int,
    ) -> tuple[EvidenceCritique, list[dict[str, Any]]]:
        seen: set[str] = set()
        selected: list[dict[str, Any]] = []
        rejected_ids: list[str] = []
        for chunk in evidence:
            chunk_id = str(chunk.get("chunk_id", "")).strip()
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            text = str(chunk.get("text", "")).strip()
            if not text:
                rejected_ids.append(chunk_id)
                continue
            selected.append(chunk)
            if len(selected) >= min(max(final_k, 1), self.max_contexts):
                break
        selected_ids = [str(chunk.get("chunk_id")) for chunk in selected]
        compressed_context = "\n\n".join(
            f"[{chunk.get('chunk_id')}] {chunk.get('title') or ''}\n{str(chunk.get('text', ''))[:900]}"
            for chunk in selected
        )
        critique = EvidenceCritique(
            status="sufficient" if selected else "insufficient",
            selected_evidence=[
                SelectedEvidence(
                    evidence_id=str(chunk.get("chunk_id")),
                    relevance=max(
                        0.0,
                        min(1.0, float(chunk.get("score") or chunk.get("reranker_score") or 0.0)),
                    ),
                    compressed_text=str(chunk.get("text", ""))[:900],
                )
                for chunk in selected
            ],
            selected_ids=selected_ids,
            rejected_ids=rejected_ids,
            compressed_context=compressed_context,
            sufficient=bool(selected),
            warnings=[] if selected else ["no_supported_evidence"],
        )
        return critique, selected

    def _model_compress(
        self,
        question: str,
        evidence: list[dict[str, Any]],
        *,
        final_k: int,
    ) -> tuple[EvidenceCritique, list[dict[str, Any]]]:
        available = {str(item.get("chunk_id")): item for item in evidence if item.get("chunk_id")}
        payload = {
            "question": question,
            "max_selected": min(max(final_k, 1), self.max_contexts),
            "evidence": [
                {
                    "evidence_id": chunk_id,
                    "source_type": item.get("source_kind", "local"),
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "chunk_id": chunk_id,
                    "text": str(item.get("text", ""))[:1800],
                    "retrieval_score": item.get("score") or item.get("reranker_score"),
                }
                for chunk_id, item in available.items()
            ],
        }
        output = self.model_runtime.generate_json(
            adapter="evidence",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Select, verify, deduplicate and compress only the supplied evidence. Return JSON with "
                        "status (sufficient|insufficient|conflicting), selected_evidence, conflicts, "
                        "missing_information and summary. Never invent an evidence_id or external fact."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            max_new_tokens=768,
        )
        selected = [SelectedEvidence.model_validate(item) for item in output.get("selected_evidence", [])]
        selected_ids = [item.evidence_id for item in selected]
        unknown = [item for item in selected_ids if item not in available]
        if unknown:
            raise ValueError(f"Evidence model invented IDs: {unknown}")
        contexts: list[dict[str, Any]] = []
        for item in selected[: self.max_contexts]:
            context = dict(available[item.evidence_id])
            context["text"] = item.compressed_text or str(context.get("text", ""))
            contexts.append(context)
        critique = EvidenceCritique(
            status=output.get("status", "insufficient"),
            selected_evidence=selected[: self.max_contexts],
            selected_ids=[item.evidence_id for item in selected[: self.max_contexts]],
            rejected_ids=[chunk_id for chunk_id in available if chunk_id not in selected_ids],
            compressed_context="\n\n".join(str(item.get("text", "")) for item in contexts),
            conflicts=[str(item) for item in output.get("conflicts", [])],
            sufficient=output.get("status") == "sufficient" and bool(contexts),
            missing_information=[str(item) for item in output.get("missing_information", [])],
            summary=str(output.get("summary", "")),
        )
        return critique, contexts
