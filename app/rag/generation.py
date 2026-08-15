import threading
import time
from typing import Any

import torch

from app.chat.attachments import (
    TemporaryCorpusRetriever,
    is_attachment_question,
    merge_global_and_temporary_contexts,
)
from app.rag.guards import AnswerGuards
from app.rag.prompting import IM_END, PromptBuilder
from app.rag.retrieval import (
    HybridRetriever,
    clean_text,
    match_norm,
)
from app.services.rag_service import RAGService


class RAGGenerator:
    def __init__(
        self,
        service: RAGService,
        retriever: HybridRetriever,
        temporary_retriever: TemporaryCorpusRetriever | None = None,
    ):
        self.service = service
        self.retriever = retriever
        self.temporary_retriever = temporary_retriever

        self.prompt_builder = PromptBuilder(service)
        self.guards = AnswerGuards(service)

        # Transformers generate() trên một GPU cần được chạy tuần tự.
        self._generation_lock = threading.Lock()

    # ========================================================
    # Configuration
    # ========================================================

    @property
    def generation_config(self) -> dict[str, Any]:
        if not self.service.config:
            return {}

        return self.service.config.get(
            "generation",
            {},
        ) or {}

    def _cfg(
        self,
        name: str,
        default: Any,
    ) -> Any:
        if name in self.generation_config:
            return self.generation_config[name]

        if (
            self.service.config
            and name in self.service.config
        ):
            return self.service.config[name]

        return default

    @property
    def max_new_tokens(self) -> int:
        return int(
            self._cfg(
                "max_new_tokens",
                300,
            )
        )

    @property
    def temperature(self) -> float:
        return float(
            self._cfg(
                "temperature",
                0.0,
            )
        )

    @property
    def top_p(self) -> float:
        return float(
            self._cfg(
                "top_p",
                1.0,
            )
        )

    @property
    def repetition_penalty(self) -> float:
        return float(
            self._cfg(
                "repetition_penalty",
                1.05,
            )
        )

    @property
    def retrieval_history_messages(self) -> int:
        return max(
            0,
            int(
                self._cfg(
                    "retrieval_history_messages",
                    4,
                )
            ),
        )

    @property
    def temporary_fetch_k(self) -> int:
        return max(
            1,
            int(
                self._cfg(
                    "temporary_fetch_k",
                    8,
                )
            ),
        )

    @property
    def temporary_min_dense_score(self) -> float:
        return float(
            self._cfg(
                "temporary_min_dense_score",
                0.72,
            )
        )

    # ========================================================
    # Runtime validation
    # ========================================================

    def _ensure_ready(self) -> None:
        if not self.service.loaded:
            raise RuntimeError(
                "RAGService has not been loaded."
            )

        if self.service.model is None:
            raise RuntimeError(
                "Generation model is not loaded. "
                "Use APP_MODE=full to enable /api/v1/chat."
            )

        if self.service.tokenizer is None:
            raise RuntimeError(
                "Tokenizer is not loaded."
            )

        if self.service.embedder is None:
            raise RuntimeError(
                "Embedding model is not loaded."
            )

    def _model_device(self) -> torch.device:
        try:
            embeddings = (
                self.service.model
                .get_input_embeddings()
            )

            return embeddings.weight.device
        except Exception:
            return next(
                self.service.model.parameters()
            ).device

    # ========================================================
    # Conversation history
    # ========================================================

    @staticmethod
    def normalize_history(
        history: list[dict[str, str]] | None,
        current_question: str | None = None,
    ) -> list[dict[str, str]]:
        normalized_history: list[
            dict[str, str]
        ] = []

        for message in history or []:
            role = str(
                message.get(
                    "role",
                    "",
                )
            ).strip().lower()

            content = clean_text(
                message.get(
                    "content",
                    "",
                )
            )

            if (
                role not in {"user", "assistant"}
                or not content
            ):
                continue

            normalized_history.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        # Tránh lặp câu hỏi hiện tại nếu route đã lưu user message
        # trước khi lấy history.
        if (
            current_question
            and normalized_history
            and normalized_history[-1]["role"] == "user"
            and match_norm(
                normalized_history[-1]["content"]
            )
            == match_norm(current_question)
        ):
            normalized_history.pop()

        return normalized_history

    @staticmethod
    def needs_history_for_retrieval(
        question: str,
        history: list[dict[str, str]],
    ) -> bool:
        if not history:
            return False

        normalized = match_norm(question)

        reference_terms = {
            "ong ay",
            "ba ay",
            "nguoi ay",
            "nguoi nay",
            "nhan vat nay",
            "nhan vat do",
            "su kien nay",
            "su kien do",
            "tran nay",
            "tran do",
            "trieu dai nay",
            "trieu dai do",
            "thoi ky nay",
            "thoi ky do",
            "dieu nay",
            "dieu do",
            "viec nay",
            "viec do",
            "tai lieu nay",
            "file nay",
            "pdf nay",
            "hinh nay",
            "anh nay",
            "noi dung tren",
            "nhan vat tren",
            "su kien tren",
            "cau tra loi tren",
            "nhu vua noi",
            "nhu tren",
            "truoc do",
            "sau do",
            "khi do",
            "luc do",
        }

        if any(
            term in normalized
            for term in reference_terms
        ):
            return True

        contextual_prefixes = (
            "con ",
            "vay ",
            "the ",
            "tiep theo",
            "sau nay",
            "sau do",
            "truoc do",
            "tai sao lai",
            "vi sao lai",
            "sinh nam nao",
            "mat nam nao",
            "dien ra khi nao",
            "xay ra khi nao",
            "ket thuc khi nao",
            "bat dau khi nao",
        )

        return normalized.startswith(
            contextual_prefixes
        )

    def build_retrieval_question(
        self,
        question: str,
        history: list[dict[str, str]],
    ) -> tuple[str, bool]:
        if not self.needs_history_for_retrieval(
            question,
            history,
        ):
            return question, False

        recent_history = history[
            -self.retrieval_history_messages:
        ]

        history_lines: list[str] = []

        for message in recent_history:
            role_label = (
                "Người dùng"
                if message["role"] == "user"
                else "Trợ lý"
            )

            content = clean_text(
                message["content"]
            )[:600]

            history_lines.append(
                f"{role_label}: {content}"
            )

        if not history_lines:
            return question, False

        retrieval_question = (
            "Ngữ cảnh hội thoại trước đó:\n"
            + "\n".join(history_lines)
            + "\n\nCâu hỏi hiện tại:\n"
            + question
        )

        return retrieval_question, True

    def temporary_context_is_relevant(
        self,
        question: str,
        contexts: list[dict[str, Any]],
    ) -> bool:
        if not contexts:
            return False

        if is_attachment_question(question):
            return True

        best_dense_score = max(
            (
                float(
                    context.get(
                        "temporary_dense_score",
                        -1.0,
                    )
                    or -1.0
                )
                for context in contexts
            ),
            default=-1.0,
        )

        return (
            best_dense_score
            >= self.temporary_min_dense_score
        )

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

        im_end_id = (
            tokenizer.convert_tokens_to_ids(
                IM_END
            )
        )

        eos_ids: list[int] = []

        if tokenizer.eos_token_id is not None:
            eos_ids.append(
                tokenizer.eos_token_id
            )

        if (
            isinstance(im_end_id, int)
            and im_end_id >= 0
            and im_end_id not in eos_ids
        ):
            eos_ids.append(im_end_id)

        temperature = self.temperature

        generation_kwargs: dict[
            str,
            Any,
        ] = {
            **inputs,
            "max_new_tokens": (
                max_new_tokens
                if max_new_tokens is not None
                else self.max_new_tokens
            ),
            "do_sample": temperature > 0,
            "repetition_penalty": (
                self.repetition_penalty
            ),
            "pad_token_id": (
                tokenizer.pad_token_id
            ),
            "use_cache": True,
        }

        if eos_ids:
            generation_kwargs[
                "eos_token_id"
            ] = eos_ids

        if temperature > 0:
            generation_kwargs[
                "temperature"
            ] = temperature

            generation_kwargs[
                "top_p"
            ] = self.top_p

        with self._generation_lock:
            output = model.generate(
                **generation_kwargs
            )

        prompt_length = (
            inputs["input_ids"].shape[-1]
        )

        generated_tokens = output[
            0,
            prompt_length:,
        ]

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
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        prompt, used_context, budget = (
            self.prompt_builder.fit_rag_prompt(
                question=question,
                contexts=contexts,
                analysis=analysis,
                history=history,
            )
        )

        raw = self.generate_raw(
            prompt,
            max_new_tokens=self.max_new_tokens,
        )

        parsed = (
            self.prompt_builder
            .parse_rag_output(raw)
        )

        validated = (
            self.guards
            .validate_parsed_answer(
                parsed,
                used_context,
            )
        )

        critique = (
            self.guards
            .critique_answer(
                question,
                validated["answer"],
                analysis,
            )
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
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        prompt, used_context, budget = (
            self.prompt_builder
            .fit_rewrite_prompt(
                question=question,
                contexts=contexts,
                draft=draft,
                issues=reasons,
                analysis=analysis,
                history=history,
            )
        )

        raw = self.generate_raw(
            prompt,
            max_new_tokens=self.max_new_tokens,
        )

        parsed = (
            self.prompt_builder
            .parse_rag_output(raw)
        )

        validated = (
            self.guards
            .validate_parsed_answer(
                parsed,
                used_context,
            )
        )

        critique = (
            self.guards
            .critique_answer(
                question,
                validated["answer"],
                analysis,
            )
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
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        self._ensure_ready()

        started = time.perf_counter()

        question = clean_text(question)

        normalized_history = (
            self.normalize_history(
                history,
                current_question=question,
            )
        )

        analysis = (
            retrieval.get("analysis")
            or self.retriever.analyze_question(
                question
            )
        )

        # ----------------------------------------------------
        # Off-topic guard
        # ----------------------------------------------------

        if retrieval.get("is_ood"):
            return {
                "question": question,
                "answer": (
                    self.guards.safe_ood_answer
                ),
                "status": "blocked_off_topic",
                "source_ids": [],
                "source_chunks": [],
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
                "history_message_count": len(
                    normalized_history
                ),
                "tool_trace": (
                    retrieval.get(
                        "tool_trace",
                        [],
                    )
                    + ["final_ood_guard"]
                ),
                "latency_sec": (
                    time.perf_counter()
                    - started
                ),
            }

        # ----------------------------------------------------
        # No-context guard
        # ----------------------------------------------------

        contexts = retrieval.get(
            "final_context",
            [],
        )

        if not contexts:
            return {
                "question": question,
                "answer": (
                    self.guards
                    .safe_insufficient_answer
                ),
                "status": "blocked_no_context",
                "source_ids": [],
                "source_chunks": [],
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
                "history_message_count": len(
                    normalized_history
                ),
                "tool_trace": (
                    retrieval.get(
                        "tool_trace",
                        [],
                    )
                    + ["no_context_guard"]
                ),
                "latency_sec": (
                    time.perf_counter()
                    - started
                ),
            }

        # ----------------------------------------------------
        # Initial Qwen generation
        # ----------------------------------------------------

        first = self.run_generation_pass(
            question=question,
            contexts=contexts,
            analysis=analysis,
            history=normalized_history,
        )

        first_valid = first["validated"]
        first_critique = first["critique"]

        reasons = list(
            dict.fromkeys(
                first_valid["guard_issues"]
                + first_critique.get(
                    "issues",
                    [],
                )
            )
        )

        chosen = first
        rewrite_used = False
        repair_attempted = False

        # ----------------------------------------------------
        # Evidence-only repair
        # ----------------------------------------------------

        should_repair = (
            self.guards
            .enable_completeness_rewrite
            and bool(reasons)
            and (
                self.guards
                .max_rewrite_attempts
                > 0
            )
            and not self.guards.is_refusal(
                first_valid["answer"]
            )
        )

        if should_repair:
            repair_attempted = True

            repaired = self.run_repair_pass(
                question=question,
                contexts=first[
                    "used_context"
                ],
                draft=first_valid["answer"],
                reasons=reasons,
                analysis=analysis,
                history=normalized_history,
            )

            repaired_valid = (
                repaired["validated"]
            )

            repaired_critique = (
                repaired["critique"]
            )

            structurally_better = not (
                repaired_valid[
                    "guard_issues"
                ]
            )

            quality_better = (
                len(
                    repaired_critique.get(
                        "issues",
                        [],
                    )
                )
                < len(
                    first_critique.get(
                        "issues",
                        [],
                    )
                )
            )

            if structurally_better and (
                quality_better
                or bool(
                    first_valid[
                        "guard_issues"
                    ]
                )
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
        source_chunks = validated.get(
            "evidence_chunks",
            [],
        )

        status = "ok"

        if validated["guard_issues"]:
            if (
                "invalid_source_id"
                in validated["guard_issues"]
            ):
                status = (
                    "blocked_invalid_source_id"
                )
            elif (
                "missing_source"
                in validated["guard_issues"]
            ):
                status = (
                    "blocked_missing_source"
                )
            elif (
                "unsupported_year"
                in validated["guard_issues"]
            ):
                status = (
                    "blocked_unsupported_year"
                )
            else:
                status = "blocked_guard"

            answer = (
                self.guards
                .safe_insufficient_answer
            )

            source_ids = []
            source_chunks = []

        elif critique.get("issues"):
            status = (
                "ok_with_quality_warning"
            )

        # ----------------------------------------------------
        # Grounding support score
        # ----------------------------------------------------

        support_score = (
            self.guards.answer_support_score(
                answer,
                source_chunks,
            )
            if source_chunks
            else None
        )

        # ----------------------------------------------------
        # Tool trace
        # ----------------------------------------------------

        tool_trace = (
            retrieval.get(
                "tool_trace",
                [],
            )
            + [
                "qwen_generation",
                "style_polisher",
                "source_year_guard",
                "answer_quality_critic",
            ]
        )

        if normalized_history:
            tool_trace.append(
                "conversation_memory"
            )

        if repair_attempted:
            tool_trace.append(
                "evidence_only_repair"
            )

        if rewrite_used:
            tool_trace.append(
                "repair_accepted"
            )
        elif repair_attempted:
            tool_trace.append(
                "repair_rejected_keep_first"
            )

        return {
            "question": question,
            "answer": answer,
            "status": status,
            "source_ids": source_ids,
            "source_chunks": source_chunks,
            "model_source_ids": (
                chosen["parsed"].get(
                    "source_ids",
                    [],
                )
            ),
            "invalid_source_ids": (
                validated["invalid_ids"]
            ),
            "unsupported_years": (
                validated[
                    "unsupported_years"
                ]
            ),
            "format_ok": (
                chosen["parsed"].get(
                    "format_ok",
                    False,
                )
            ),
            "raw_output": (
                chosen["parsed"].get(
                    "raw_output",
                    "",
                )
            ),
            "retrieval": retrieval,
            "analysis": analysis,
            "prompt_budget": chosen[
                "budget"
            ],
            "support_score": support_score,
            "quality_warnings": (
                critique.get(
                    "issues",
                    [],
                )
            ),
            "rewrite_used": rewrite_used,
            "repair_attempted": (
                repair_attempted
            ),
            "initial_quality_issues": (
                first_critique.get(
                    "issues",
                    [],
                )
            ),
            "history_message_count": len(
                normalized_history
            ),
            "tool_trace": tool_trace,
            "latency_sec": (
                time.perf_counter()
                - started
            ),
        }

    # ========================================================
    # Complete chat pipeline
    # ========================================================

    def chat(
        self,
        question: str,
        final_k: int | None = None,
        history: list[dict[str, str]] | None = None,
        owner_id: str | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_ready()

        question = clean_text(question)

        if not question:
            raise ValueError(
                "Question must not be empty."
            )

        normalized_history = (
            self.normalize_history(
                history,
                current_question=question,
            )
        )

        selected_final_k = (
            int(final_k)
            if final_k is not None
            else self.retriever.final_context_k
        )

        selected_final_k = max(
            1,
            selected_final_k,
        )

        total_started = (
            time.perf_counter()
        )

        retrieval_started = (
            time.perf_counter()
        )

        (
            retrieval_question,
            history_used_for_retrieval,
        ) = self.build_retrieval_question(
            question,
            normalized_history,
        )

        # ----------------------------------------------------
        # Global Vietnamese-history corpus
        # ----------------------------------------------------

        retrieval = self.retriever.retrieve(
            retrieval_question,
            final_k=selected_final_k,
        )

        global_contexts = list(
            retrieval.get(
                "final_context",
                [],
            )
        )

        # Dynamic rules and quality checks must analyze the
        # current question, not the expanded retrieval query.
        retrieval["analysis"] = (
            self.retriever
            .analyze_question(question)
        )

        retrieval["question"] = question
        retrieval["retrieval_question"] = (
            retrieval_question
        )

        retrieval[
            "history_used_for_retrieval"
        ] = history_used_for_retrieval

        retrieval[
            "global_final_context"
        ] = global_contexts

        # ----------------------------------------------------
        # Temporary corpus from uploaded PDF/images
        # ----------------------------------------------------

        temporary_candidates: list[
            dict[str, Any]
        ] = []

        temporary_contexts: list[
            dict[str, Any]
        ] = []

        if (
            self.temporary_retriever
            is not None
            and owner_id
            and conversation_id
        ):
            temporary_candidates = (
                self.temporary_retriever
                .retrieve(
                    owner_id=owner_id,
                    conversation_id=(
                        conversation_id
                    ),
                    question=(
                        retrieval_question
                    ),
                    top_k=max(
                        self.temporary_fetch_k,
                        selected_final_k * 2,
                    ),
                )
            )

            if self.temporary_context_is_relevant(
                question,
                temporary_candidates,
            ):
                temporary_contexts = (
                    temporary_candidates
                )

        retrieval[
            "temporary_candidates"
        ] = temporary_candidates

        retrieval[
            "temporary_context_relevant"
        ] = bool(temporary_contexts)

        # A confident temporary-corpus match can answer a
        # document question even if the global history corpus
        # marks that question as out of domain.
        if (
            retrieval.get("is_ood")
            and temporary_contexts
        ):
            retrieval[
                "global_ood_reason"
            ] = retrieval.get(
                "ood_reason",
                "",
            )

            retrieval["is_ood"] = False
            retrieval["ood_reason"] = (
                "grounded_in_temporary_corpus"
            )

            retrieval.setdefault(
                "tool_trace",
                [],
            ).append(
                "temporary_corpus_ood_override"
            )

        # ----------------------------------------------------
        # Merge and rerank both corpora
        # ----------------------------------------------------

        if not retrieval.get("is_ood"):
            merged_contexts = (
                merge_global_and_temporary_contexts(
                    question=question,
                    global_contexts=global_contexts,
                    temporary_contexts=(
                        temporary_contexts
                    ),
                    rag_service=self.service,
                    final_k=selected_final_k,
                )
            )
        else:
            merged_contexts = []

        retrieval["final_context"] = (
            merged_contexts
        )

        retrieval[
            "temporary_context_count"
        ] = sum(
            1
            for context in merged_contexts
            if context.get("source_kind")
            == "attachment"
        )

        retrieval[
            "global_context_count"
        ] = sum(
            1
            for context in merged_contexts
            if context.get("source_kind")
            != "attachment"
        )

        retrieval[
            "context_title_diversity"
        ] = (
            self.retriever
            .context_title_diversity(
                merged_contexts
            )
        )

        retrieval.setdefault(
            "tool_trace",
            [],
        )

        if temporary_candidates:
            retrieval["tool_trace"].append(
                "temporary_corpus_retrieval:"
                f"{len(temporary_candidates)}"
            )

            if not temporary_contexts:
                retrieval["tool_trace"].append(
                    "temporary_corpus_filtered"
                )

        if temporary_contexts:
            retrieval["tool_trace"].append(
                "global_temporary_context_merge"
            )

        retrieval_latency = (
            time.perf_counter()
            - retrieval_started
        )

        # ----------------------------------------------------
        # Generation, guard and optional repair
        # ----------------------------------------------------

        result = self.answer_from_retrieval(
            question=question,
            retrieval=retrieval,
            history=normalized_history,
        )

        result["retrieval_latency_sec"] = (
            retrieval_latency
        )

        result["total_latency_sec"] = (
            time.perf_counter()
            - total_started
        )

        return result