import threading
import time
from typing import Any

import torch

from app.rag.guards import AnswerGuards
from app.rag.prompting import IM_END, PromptBuilder
from app.rag.retrieval import HybridRetriever, clean_text
from app.services.rag_service import RAGService


class RAGGenerator:
    def __init__(self, service: RAGService, retriever: HybridRetriever):
        self.service = service
        self.retriever = retriever

        self.prompt_builder = PromptBuilder(service)
        self.guards = AnswerGuards(service)

        # Transformers generate() trên một GPU sẽ được serialize trước.
        # Sau này nếu chuyển sang vLLM thì bỏ lock này.
        self._generation_lock = threading.Lock()

    # ========================================================
    # Configuration
    # ========================================================

    @property
    def generation_config(self) -> dict[str, Any]:
        if not self.service.config:
            return {}
        return self.service.config.get("generation", {}) or {}

    def _cfg(self, name: str, default: Any) -> Any:
        if name in self.generation_config:
            return self.generation_config[name]

        if self.service.config and name in self.service.config:
            return self.service.config[name]

        return default

    @property
    def max_new_tokens(self) -> int:
        return int(self._cfg("max_new_tokens", 300))

    @property
    def temperature(self) -> float:
        return float(self._cfg("temperature", 0.0))

    @property
    def top_p(self) -> float:
        return float(self._cfg("top_p", 1.0))

    @property
    def repetition_penalty(self) -> float:
        return float(self._cfg("repetition_penalty", 1.05))

    # ========================================================
    # Runtime validation
    # ========================================================

    def _ensure_ready(self) -> None:
        if not self.service.loaded:
            raise RuntimeError("RAGService has not been loaded.")

        if self.service.model is None:
            raise RuntimeError(
                "Generation model is not loaded. "
                "Use APP_MODE=full to enable /api/v1/chat."
            )

        if self.service.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded.")

        if self.service.embedder is None:
            raise RuntimeError("Embedding model is not loaded.")

    def _model_device(self) -> torch.device:
        try:
            return self.service.model.get_input_embeddings().weight.device
        except Exception:
            return next(self.service.model.parameters()).device

    # ========================================================
    # Raw Qwen generation
    # ========================================================

    @torch.inference_mode()
    def generate_raw(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
    ) -> str:
        self._ensure_ready()

        tokenizer = self.service.tokenizer
        model = self.service.model
        device = self._model_device()

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device)

        im_end_id = tokenizer.convert_tokens_to_ids(IM_END)

        eos_ids = []

        if tokenizer.eos_token_id is not None:
            eos_ids.append(tokenizer.eos_token_id)

        if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_ids:
            eos_ids.append(im_end_id)

        temperature = self.temperature

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "do_sample": temperature > 0,
            "repetition_penalty": self.repetition_penalty,
            "pad_token_id": tokenizer.pad_token_id,
            "use_cache": True,
        }

        if eos_ids:
            generation_kwargs["eos_token_id"] = eos_ids

        if temperature > 0:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = self.top_p

        with self._generation_lock:
            output = model.generate(**generation_kwargs)

        generated_tokens = output[0][inputs["input_ids"].shape[-1]:]

        return tokenizer.decode(
            generated_tokens,
            skip_special_tokens=False,
        )

    # ========================================================
    # Initial generation pass
    # ========================================================

    def run_generation_pass(
        self,
        question: str,
        contexts: list[dict[str, Any]],
        analysis: dict[str, Any],
    ) -> dict[str, Any]:
        prompt, used_context, budget = self.prompt_builder.fit_rag_prompt(
            question,
            contexts,
            analysis,
        )

        raw = self.generate_raw(
            prompt,
            max_new_tokens=self.max_new_tokens,
        )

        parsed = self.prompt_builder.parse_rag_output(raw)

        validated = self.guards.validate_parsed_answer(
            parsed,
            used_context,
        )

        critique = self.guards.critique_answer(
            question,
            validated["answer"],
            analysis,
        )

        return {
            "prompt": prompt,
            "budget": budget,
            "used_context": used_context,
            "raw": raw,
            "parsed": parsed,
            "validated": validated,
            "critique": critique,
        }

    # ========================================================
    # Evidence-only repair pass
    # ========================================================

    def run_repair_pass(
        self,
        question: str,
        contexts: list[dict[str, Any]],
        draft: str,
        reasons: list[str],
        analysis: dict[str, Any],
    ) -> dict[str, Any]:
        prompt, used_context, budget = self.prompt_builder.fit_rewrite_prompt(
            question,
            contexts,
            draft,
            reasons,
            analysis,
        )

        raw = self.generate_raw(
            prompt,
            max_new_tokens=self.max_new_tokens,
        )

        parsed = self.prompt_builder.parse_rag_output(raw)

        validated = self.guards.validate_parsed_answer(
            parsed,
            used_context,
        )

        critique = self.guards.critique_answer(
            question,
            validated["answer"],
            analysis,
        )

        return {
            "prompt": prompt,
            "budget": budget,
            "used_context": used_context,
            "raw": raw,
            "parsed": parsed,
            "validated": validated,
            "critique": critique,
        }

    # ========================================================
    # Final answer from an existing retrieval result
    # ========================================================

    def answer_from_retrieval(
        self,
        question: str,
        retrieval: dict[str, Any],
    ) -> dict[str, Any]:
        self._ensure_ready()

        started = time.perf_counter()

        question = clean_text(question)

        analysis = retrieval.get("analysis") or self.retriever.analyze_question(
            question
        )

        # ----------------------------------------------------
        # Off-topic guard
        # ----------------------------------------------------

        if retrieval.get("is_ood"):
            return {
                "question": question,
                "answer": self.guards.safe_ood_answer,
                "status": "blocked_off_topic",
                "source_ids": [],
                "model_source_ids": [],
                "invalid_source_ids": [],
                "unsupported_years": [],
                "format_ok": True,
                "retrieval": retrieval,
                "analysis": analysis,
                "prompt_budget": None,
                "support_score": None,
                "quality_warnings": [],
                "rewrite_used": False,
                "repair_attempted": False,
                "initial_quality_issues": [],
                "raw_output": "",
                "tool_trace": retrieval.get("tool_trace", []) + ["final_ood_guard"],
                "latency_sec": time.perf_counter() - started,
            }

        # ----------------------------------------------------
        # No-context guard
        # ----------------------------------------------------

        contexts = retrieval.get("final_context", [])

        if not contexts:
            return {
                "question": question,
                "answer": self.guards.safe_insufficient_answer,
                "status": "blocked_no_context",
                "source_ids": [],
                "model_source_ids": [],
                "invalid_source_ids": [],
                "unsupported_years": [],
                "format_ok": True,
                "retrieval": retrieval,
                "analysis": analysis,
                "prompt_budget": None,
                "support_score": None,
                "quality_warnings": [],
                "rewrite_used": False,
                "repair_attempted": False,
                "initial_quality_issues": [],
                "raw_output": "",
                "tool_trace": retrieval.get("tool_trace", []) + ["no_context_guard"],
                "latency_sec": time.perf_counter() - started,
            }

        # ----------------------------------------------------
        # Initial Qwen generation
        # ----------------------------------------------------

        first = self.run_generation_pass(
            question,
            contexts,
            analysis,
        )

        first_valid = first["validated"]
        first_critique = first["critique"]

        reasons = list(
            dict.fromkeys(
                first_valid["guard_issues"] + first_critique.get("issues", [])
            )
        )

        chosen = first
        rewrite_used = False
        repair_attempted = False

        # ----------------------------------------------------
        # Evidence-only repair
        # ----------------------------------------------------

        should_repair = (
            self.guards.enable_completeness_rewrite
            and bool(reasons)
            and self.guards.max_rewrite_attempts > 0
            and not self.guards.is_refusal(first_valid["answer"])
        )

        if should_repair:
            repair_attempted = True

            repaired = self.run_repair_pass(
                question,
                contexts,
                first_valid["answer"],
                reasons,
                analysis,
            )

            repaired_valid = repaired["validated"]
            repaired_critique = repaired["critique"]

            structurally_better = not repaired_valid["guard_issues"]

            quality_better = (
                len(repaired_critique.get("issues", []))
                < len(first_critique.get("issues", []))
            )

            if structurally_better and (
                quality_better or bool(first_valid["guard_issues"])
            ):
                chosen = repaired
                rewrite_used = True

        # ----------------------------------------------------
        # Final validation decision
        # ----------------------------------------------------

        validated = chosen["validated"]
        critique = chosen["critique"]

        answer = validated["answer"]
        source_ids = validated["valid_ids"]
        status = "ok"

        if validated["guard_issues"]:
            if "invalid_source_id" in validated["guard_issues"]:
                status = "blocked_invalid_source_id"
            elif "missing_source" in validated["guard_issues"]:
                status = "blocked_missing_source"
            elif "unsupported_year" in validated["guard_issues"]:
                status = "blocked_unsupported_year"
            else:
                status = "blocked_guard"

            answer = self.guards.safe_insufficient_answer
            source_ids = []

        elif critique.get("issues"):
            status = "ok_with_quality_warning"

        # ----------------------------------------------------
        # Grounding support score
        # ----------------------------------------------------

        evidence = [
            self.service.chunk_by_id[source_id]
            for source_id in source_ids
            if source_id in self.service.chunk_by_id
        ]

        support_score = (
            self.guards.answer_support_score(answer, evidence)
            if evidence
            else None
        )

        # ----------------------------------------------------
        # Tool trace
        # ----------------------------------------------------

        tool_trace = retrieval.get("tool_trace", []) + [
            "qwen_generation",
            "style_polisher",
            "source_year_guard",
            "answer_quality_critic",
        ]

        if repair_attempted:
            tool_trace.append("evidence_only_repair")

        if rewrite_used:
            tool_trace.append("repair_accepted")
        elif repair_attempted:
            tool_trace.append("repair_rejected_keep_first")

        return {
            "question": question,
            "answer": answer,
            "status": status,
            "source_ids": source_ids,
            "model_source_ids": chosen["parsed"].get("source_ids", []),
            "invalid_source_ids": validated["invalid_ids"],
            "unsupported_years": validated["unsupported_years"],
            "format_ok": chosen["parsed"].get("format_ok", False),
            "raw_output": chosen["parsed"].get("raw_output", ""),
            "retrieval": retrieval,
            "analysis": analysis,
            "prompt_budget": chosen["budget"],
            "support_score": support_score,
            "quality_warnings": critique.get("issues", []),
            "rewrite_used": rewrite_used,
            "repair_attempted": repair_attempted,
            "initial_quality_issues": first_critique.get("issues", []),
            "tool_trace": tool_trace,
            "latency_sec": time.perf_counter() - started,
        }

    # ========================================================
    # Complete chat pipeline
    # ========================================================

    def chat(
        self,
        question: str,
        final_k: int | None = None,
    ) -> dict[str, Any]:
        self._ensure_ready()

        question = clean_text(question)

        if not question:
            raise ValueError("Question must not be empty.")

        total_started = time.perf_counter()

        retrieval_started = time.perf_counter()

        retrieval = self.retriever.retrieve(
            question,
            final_k=final_k,
        )

        retrieval_latency = time.perf_counter() - retrieval_started

        result = self.answer_from_retrieval(
            question,
            retrieval,
        )

        result["retrieval_latency_sec"] = retrieval_latency
        result["total_latency_sec"] = time.perf_counter() - total_started

        return result