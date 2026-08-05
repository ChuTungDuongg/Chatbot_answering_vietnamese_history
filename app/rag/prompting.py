import re
from typing import Any

from app.rag.guards import ISSUE_INSTRUCTIONS
from app.rag.retrieval import clean_text
from app.services.rag_service import RAGService


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


SOURCE_BLOCK_RE = re.compile(
    r"Nguồn được dùng\s*:\s*(.*?)(?:\n\s*\n|\n\s*Trả lời\s*:|$)",
    re.I | re.S,
)

ANSWER_SPLIT_RE = re.compile(r"Trả lời\s*:", re.I)


class PromptBuilder:
    def __init__(self, service: RAGService):
        self.service = service

    # ========================================================
    # Config
    # ========================================================

    @property
    def prompt_config(self) -> dict[str, Any]:
        if not self.service.config:
            return {}
        return self.service.config.get("prompt", {}) or {}

    @property
    def max_input_tokens(self) -> int:
        return int(self.prompt_config.get("max_input_tokens", 3600))

    @property
    def max_new_tokens(self) -> int:
        return int(self.prompt_config.get("max_new_tokens", 300))

    @property
    def max_chars_per_chunk(self) -> int:
        return int(self.prompt_config.get("max_chars_per_chunk", 1800))

    @property
    def min_chars_per_chunk(self) -> int:
        return int(self.prompt_config.get("min_chars_per_chunk", 550))

    @property
    def default_system(self) -> str:
        return self.prompt_config.get(
            "default_system",
            "Bạn là trợ lý AI chuyên về lịch sử Việt Nam. "
            "Trả lời trực tiếp đúng trọng tâm, rõ ràng và chính xác. "
            'Không mở đầu bằng các câu rập khuôn như "Theo tài liệu". '
            "Nếu không đủ cơ sở để khẳng định, nói rõ mức độ không chắc chắn.",
        )

    # ========================================================
    # Text helpers
    # ========================================================

    @staticmethod
    def short_text(text: str, max_chars: int) -> str:
        text = clean_text(text)

        if max_chars <= 0:
            return ""

        if len(text) <= max_chars:
            return text

        cut = text[:max_chars]

        last = max(
            cut.rfind(". "),
            cut.rfind("; "),
            cut.rfind("\n"),
            cut.rfind(" "),
        )

        if last > max_chars * 0.65:
            cut = cut[:last]

        return cut.strip() + " ..."

    # ========================================================
    # Dynamic answer rules
    # ========================================================

    @staticmethod
    def build_dynamic_answer_rules(
        question: str,
        analysis: dict[str, Any],
    ) -> list[str]:
        facets = analysis.get("facets", ["general"])

        rules = [
            "Trả lời trực tiếp trọng tâm ngay từ câu đầu; không chỉ tóm tắt tài liệu.",
            "Chỉ dùng thông tin được các đoạn tài liệu bên dưới hỗ trợ; "
            "không tự bổ sung kiến thức ngoài evidence.",
            'Không mở đầu bằng "Theo tài liệu", "Dựa trên tài liệu" hoặc cách nói tương tự.',
            'Không dùng từ "chunk" trong câu trả lời.',
            "Không lặp lại nguyên câu hỏi.",
            "Không suy luận quan hệ nhân vật, triều đại, phe phái hoặc niên đại "
            "nếu evidence không nêu đủ rõ.",
            "Nếu evidence chưa đủ để kết luận chắc chắn, nói rõ phần nào chưa đủ thay vì đoán.",
            "Nguồn được dùng chỉ được chứa chunk_id thực sự có trong Tài liệu tham khảo.",
        ]

        if analysis.get("is_multi_part"):
            rules.append(
                "Câu hỏi có nhiều ý: phải trả lời đủ từng ý; có thể dùng nhãn ngắn "
                "Bối cảnh/Kết quả/Ý nghĩa nếu giúp rõ hơn."
            )

        if "winner" in facets:
            rules.extend(
                [
                    "Câu hỏi hỏi bên thắng: phải trả lời trực tiếp "
                    "thắng/thua/không thể kết luận ở câu đầu.",
                    'Không được suy luận "bên phát động = bên chiến thắng".',
                    "Nếu evidence cho thấy kết quả khác nhau theo quân sự, chiến thuật, "
                    "chiến lược hoặc chính trị, phải tách các khía cạnh đó.",
                ]
            )

        if "compare" in facets:
            rules.extend(
                [
                    "Câu hỏi so sánh: nêu riêng vai trò/đặc điểm của từng bên, "
                    "sau đó mới chỉ ra điểm giống/khác hoặc kết luận.",
                    "Không gán nhân vật, sự kiện hay đặc điểm của đối tượng thứ nhất "
                    "sang đối tượng thứ hai.",
                ]
            )

        if "context" in facets:
            rules.append(
                "Phải nêu hoàn cảnh/bối cảnh dẫn tới sự kiện, không chỉ mô tả sự kiện."
            )

        if "outcome" in facets:
            rules.append(
                "Phải nêu kết quả/kết cục hoặc hệ quả trực tiếp nếu câu hỏi yêu cầu."
            )

        if "significance" in facets:
            rules.append(
                "Phải giải thích ý nghĩa/tác động, không chỉ kể diễn biến."
            )

        if "process" in facets:
            rules.append(
                "Nếu hỏi diễn biến, trình bày các mốc/chặng chính theo trật tự thời gian "
                "khi evidence hỗ trợ."
            )

        if "content" in facets:
            rules.append(
                "Nếu hỏi nội dung văn kiện/hiệp định, ưu tiên các điểm chính "
                "và tách chúng khỏi phần hệ quả."
            )

        return rules

    # ========================================================
    # Context formatting
    # ========================================================

    def build_context_text(
        self,
        contexts: list[dict[str, Any]],
        chars_per_chunk: int,
    ) -> str:
        blocks = []

        for chunk in contexts:
            chunk_id = str(chunk["chunk_id"])
            title = clean_text(chunk.get("title", ""))
            text = self.short_text(chunk.get("text", ""), chars_per_chunk)

            blocks.append(f"[{chunk_id}] {title}\n{text}")

        return "\n\n".join(blocks)

    # ========================================================
    # Initial RAG prompt
    # ========================================================

    def build_rag_user_text(
        self,
        question: str,
        contexts: list[dict[str, Any]],
        chars_per_chunk: int,
        analysis: dict[str, Any],
    ) -> str:
        rules = self.build_dynamic_answer_rules(question, analysis)
        rule_text = "\n".join(
            f"{index}. {rule}"
            for index, rule in enumerate(rules, 1)
        )

        context_text = self.build_context_text(contexts, chars_per_chunk)

        return (
            f"Câu hỏi:\n{clean_text(question)}\n\n"
            f"Yêu cầu bắt buộc:\n{rule_text}\n\n"
            "Định dạng đầu ra bắt buộc:\n"
            "Nguồn được dùng: [chunk_id_1, chunk_id_2]\n"
            "Trả lời: <câu trả lời trực tiếp>\n\n"
            f"Tài liệu tham khảo:\n{context_text}"
        ).strip()

    @staticmethod
    def build_rag_prompt(user_text: str) -> str:
        return f"{IM_START}user\n{user_text}{IM_END}\n{IM_START}assistant\n"

    def fit_rag_prompt(
        self,
        question: str,
        contexts: list[dict[str, Any]],
        analysis: dict[str, Any],
    ) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
        tokenizer = self.service.tokenizer

        if tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded.")

        used_context = list(contexts)
        chars_per_chunk = self.max_chars_per_chunk

        while used_context:
            user_text = self.build_rag_user_text(
                question,
                used_context,
                chars_per_chunk,
                analysis,
            )

            prompt = self.build_rag_prompt(user_text)

            input_tokens = len(
                tokenizer(prompt, add_special_tokens=False)["input_ids"]
            )

            if input_tokens <= self.max_input_tokens:
                return prompt, used_context, {
                    "input_tokens": input_tokens,
                    "chars_per_chunk": chars_per_chunk,
                    "n_context": len(used_context),
                }

            if chars_per_chunk > self.min_chars_per_chunk:
                chars_per_chunk = max(
                    self.min_chars_per_chunk,
                    int(chars_per_chunk * 0.75),
                )
            else:
                used_context = used_context[:-1]

        user_text = self.build_rag_user_text(
            question,
            [],
            0,
            analysis,
        )

        prompt = self.build_rag_prompt(user_text)

        input_tokens = len(
            tokenizer(prompt, add_special_tokens=False)["input_ids"]
        )

        return prompt, [], {
            "input_tokens": input_tokens,
            "chars_per_chunk": 0,
            "n_context": 0,
        }

    # ========================================================
    # Generated output cleanup
    # ========================================================

    def clean_generated(self, text: str) -> str:
        tokenizer = self.service.tokenizer

        if tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded.")

        text = clean_text(text)

        markers = [
            IM_END,
            f"{IM_START}user",
            f"{IM_START}system",
            f"{IM_START}assistant",
        ]

        for marker in markers:
            if marker and marker in text:
                text = text.split(marker, 1)[0]

        special_tokens = [
            IM_START,
            IM_END,
            tokenizer.eos_token or "",
            tokenizer.pad_token or "",
        ]

        for special in special_tokens:
            if special:
                text = text.replace(special, "")

        return clean_text(text)

    # ========================================================
    # Style polishing
    # ========================================================

    @staticmethod
    def polish_answer_style(answer: str) -> str:
        answer = clean_text(answer)

        answer = re.sub(
            r'^(?:(?:theo|dựa trên|căn cứ(?: vào)?)\s+(?:các\s+)?'
            r'(?:tài liệu|đoạn tư liệu)(?:\s+được\s+(?:cung cấp|truy xuất))?'
            r'\s*[:,]?\s*)+',
            "",
            answer,
            flags=re.I,
        ).strip()

        answer = re.sub(
            r"\bChunk\s+(?:cũng\s+)?nêu(?:\s+rằng)?\s*[:,]?\s*",
            "Ngoài ra, ",
            answer,
            flags=re.I,
        )

        answer = re.sub(
            r"\bChunk\s+này\s+(?:cho biết|mô tả|nêu)\s*[:,]?\s*",
            "Cụ thể, ",
            answer,
            flags=re.I,
        )

        answer = re.sub(
            r"\bcác?\s+chunk\b",
            "các đoạn tư liệu",
            answer,
            flags=re.I,
        )

        answer = re.sub(
            r"\bchunk\b",
            "đoạn tư liệu",
            answer,
            flags=re.I,
        )

        answer = clean_text(answer)

        if answer:
            answer = answer[0].upper() + answer[1:]

        return answer

    # ========================================================
    # Parse model output
    # ========================================================

    def parse_rag_output(self, raw: str) -> dict[str, Any]:
        cleaned = self.clean_generated(raw)
        source_ids: list[str] = []

        match = SOURCE_BLOCK_RE.search(cleaned)

        if match:
            source_block = match.group(1)
            groups = re.findall(r"\[([^\[\]]*)\]", source_block)

            if not groups and source_block.strip():
                groups = [source_block]

            for group in groups:
                for source_id in re.split(r"[,;\n]+", group):
                    source_id = source_id.strip().strip("[]\"' ")

                    if source_id:
                        source_ids.append(source_id)

        source_ids = list(dict.fromkeys(source_ids))

        answer_parts = ANSWER_SPLIT_RE.split(cleaned, maxsplit=1)
        answer = clean_text(
            answer_parts[1]
            if len(answer_parts) > 1
            else cleaned
        )

        answer = self.polish_answer_style(answer)

        format_ok = bool(
            re.search(r"Nguồn được dùng\s*:", cleaned, re.I)
            and re.search(r"Trả lời\s*:", cleaned, re.I)
        )

        return {
            "raw_output": cleaned,
            "source_ids": source_ids,
            "answer": answer,
            "format_ok": format_ok,
        }

    # ========================================================
    # Evidence-only repair prompt
    # ========================================================

    def build_rewrite_user_text(
        self,
        question: str,
        contexts: list[dict[str, Any]],
        chars_per_chunk: int,
        draft: str,
        issues: list[str],
        analysis: dict[str, Any],
    ) -> str:
        issue_text = "\n".join(
            "- " + ISSUE_INSTRUCTIONS.get(issue, issue)
            for issue in issues
        )

        rules = self.build_dynamic_answer_rules(question, analysis)

        rule_text = "\n".join(
            f"{index}. {rule}"
            for index, rule in enumerate(rules, 1)
        )

        context_text = self.build_context_text(
            contexts,
            chars_per_chunk,
        )

        evidence_only_rule_number = len(rules) + 1

        return (
            f"Câu hỏi:\n{clean_text(question)}\n\n"
            f"Câu trả lời nháp cần sửa:\n{clean_text(draft)}\n\n"
            f"Các lỗi cần sửa:\n{issue_text}\n\n"
            f"Yêu cầu bắt buộc:\n{rule_text}\n"
            f"{evidence_only_rule_number}. Chỉ sửa bằng evidence bên dưới; "
            "không giữ claim trong bản nháp nếu evidence không hỗ trợ.\n\n"
            "Định dạng đầu ra bắt buộc:\n"
            "Nguồn được dùng: [chunk_id_1, chunk_id_2]\n"
            "Trả lời: <bản trả lời đã sửa>\n\n"
            f"Tài liệu tham khảo:\n{context_text}"
        ).strip()

    def fit_rewrite_prompt(
        self,
        question: str,
        contexts: list[dict[str, Any]],
        draft: str,
        issues: list[str],
        analysis: dict[str, Any],
    ) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
        tokenizer = self.service.tokenizer

        if tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded.")

        used_context = list(contexts)
        chars_per_chunk = min(self.max_chars_per_chunk, 1500)

        while used_context:
            user_text = self.build_rewrite_user_text(
                question,
                used_context,
                chars_per_chunk,
                draft,
                issues,
                analysis,
            )

            prompt = self.build_rag_prompt(user_text)

            input_tokens = len(
                tokenizer(prompt, add_special_tokens=False)["input_ids"]
            )

            if input_tokens <= self.max_input_tokens:
                return prompt, used_context, {
                    "input_tokens": input_tokens,
                    "chars_per_chunk": chars_per_chunk,
                    "n_context": len(used_context),
                }

            if chars_per_chunk > self.min_chars_per_chunk:
                chars_per_chunk = max(
                    self.min_chars_per_chunk,
                    int(chars_per_chunk * 0.75),
                )
            else:
                used_context = used_context[:-1]

        user_text = self.build_rewrite_user_text(
            question,
            [],
            0,
            draft,
            issues,
            analysis,
        )

        prompt = self.build_rag_prompt(user_text)

        input_tokens = len(
            tokenizer(prompt, add_special_tokens=False)["input_ids"]
        )

        return prompt, [], {
            "input_tokens": input_tokens,
            "chars_per_chunk": 0,
            "n_context": 0,
        }