from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from app.agents.model_runtime import SharedAgentModelRuntime
from app.agents.prompts import EVIDENCE_AGENT_SYSTEM
from app.agents.schemas import EvidenceAgentRequest, EvidenceCritique, EvidenceModelOutput, SelectedEvidence
from app.agents.evidence_validation import (
    compressed_derived_from_own_claims,
    grounded_in_source,
    referenced_evidence_ids,
)


class EvidenceModelContractError(ValueError):
    """The Evidence adapter returned output that cannot satisfy the production contract."""


class EvidenceCriticAgent:
    def __init__(
        self,
        *,
        max_contexts: int = 8,
        model_runtime: SharedAgentModelRuntime | None = None,
        allow_model_fallback: bool = False,
    ):
        self.max_contexts = max_contexts
        self.model_runtime = model_runtime
        self.allow_model_fallback = allow_model_fallback

    def compress(self, question: str, evidence: list[dict[str, Any]], *, final_k: int) -> tuple[EvidenceCritique, list[dict[str, Any]]]:
        if self.model_runtime is not None:
            try:
                return self._model_compress(question, evidence, final_k=final_k)
            except (ValueError, ValidationError, KeyError, TypeError) as exc:
                if not self.allow_model_fallback:
                    if isinstance(exc, EvidenceModelContractError):
                        raise
                    raise EvidenceModelContractError(f"Evidence model output failed canonical schema validation: {exc}") from exc
                critique, contexts = self._deterministic_compress(evidence, final_k=final_k)
                critique.warnings.append(f"model_output_invalid_debug_fallback_used:{type(exc).__name__}")
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
        request = EvidenceAgentRequest.model_validate({
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
        })
        output = self.model_runtime.generate_json(
            adapter="evidence",
            messages=[
                {
                    "role": "system",
                    "content": EVIDENCE_AGENT_SYSTEM,
                },
                {"role": "user", "content": json.dumps(request.model_dump(), ensure_ascii=False, sort_keys=True)},
            ],
            max_new_tokens=768,
        )
        raw_selected = output.get("selected_evidence", []) if isinstance(output, dict) else []
        if any(isinstance(item, str) for item in raw_selected):
            raise EvidenceModelContractError(
                "Evidence model returned legacy selected_evidence format list[str]. Retrain or migrate the Evidence Agent."
            )
        try:
            model_output = EvidenceModelOutput.model_validate(output)
        except ValidationError as exc:
            raise EvidenceModelContractError(f"Evidence model returned invalid canonical output: {exc}") from exc
        selected = model_output.selected_evidence
        selected_ids = [item.evidence_id for item in selected]
        visible_sources = {item.evidence_id: item.text for item in request.evidence}
        unknown = [item for item in selected_ids if item not in visible_sources]
        if unknown:
            raise EvidenceModelContractError(f"Evidence model invented IDs: {unknown}")
        for item in selected:
            source_text = visible_sources[item.evidence_id]
            for claim in item.claims:
                if not grounded_in_source(claim, source_text):
                    raise EvidenceModelContractError(
                        f"claim under {item.evidence_id!r} is not grounded in that same evidence source"
                    )
            if not compressed_derived_from_own_claims(item, source_text):
                raise EvidenceModelContractError(
                    f"compressed_text under {item.evidence_id!r} is not derivable from its own grounded claims"
                )
        if model_output.status == "conflicting":
            for conflict in model_output.conflicts:
                mentioned = referenced_evidence_ids(conflict, visible_sources)
                if len(mentioned) < 2:
                    raise EvidenceModelContractError(
                        "each conflict must reference at least two supplied evidence IDs"
                    )
        contexts: list[dict[str, Any]] = []
        for item in selected[: self.max_contexts]:
            context = dict(available[item.evidence_id])
            context["text"] = item.compressed_text or str(context.get("text", ""))
            contexts.append(context)
        selected = selected[: self.max_contexts]
        selected_ids = [item.evidence_id for item in selected]
        critique = EvidenceCritique(
            status=model_output.status,
            selected_evidence=selected,
            selected_ids=selected_ids,
            rejected_ids=[chunk_id for chunk_id in available if chunk_id not in selected_ids],
            compressed_context="\n\n".join(
                f"[{item.evidence_id}] {item.compressed_text}" for item in selected
            ),
            conflicts=model_output.conflicts,
            sufficient=model_output.status == "sufficient" and bool(contexts),
            warnings=[],
            missing_information=model_output.missing_information,
            summary=model_output.summary,
        )
        return critique, contexts
