import re
import time
import unicodedata
from collections import Counter, defaultdict
from typing import Any

import bm25s
import numpy as np

from app.services.rag_service import RAGService
from app.telemetry import current_request_telemetry, log_event


# ============================================================
# Text utilities
# ============================================================

def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", clean_text(text).lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d")


def match_norm(text: str) -> str:
    text = strip_accents(text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def query_for_embedding(question: str) -> str:
    return "query: " + clean_text(question)


# ============================================================
# Phase 9 constants
# ============================================================

HISTORY_ANCHORS = [
    "lịch sử Việt Nam các triều đại và nhà nước",
    "khởi nghĩa kháng chiến chiến tranh trong lịch sử Việt Nam",
    "nhân vật lịch sử Việt Nam vua tướng lãnh tụ",
    "hiệp định ngoại giao cách mạng Việt Nam",
    "nhà Lý nhà Trần nhà Lê nhà Nguyễn",
    "Cách mạng tháng Tám chiến tranh Việt Nam",
]

OOD_ANCHORS = [
    "lập trình Python JavaScript sửa lỗi phần mềm",
    "thời tiết hôm nay nhiệt độ dự báo mưa",
    "bóng đá cầu thủ câu lạc bộ tỉ số trận đấu",
    "nấu ăn công thức món ăn nguyên liệu",
    "toán học phương trình tính toán",
    "triệu chứng bệnh thuốc điều trị y tế",
    "giá cổ phiếu tiền điện tử tài chính hôm nay",
    "tình yêu quan hệ cá nhân hẹn hò",
]

META_PATTERNS = [
    r"^(xin chao|chao|hello|hi)\b",
    r"ban co the lam gi|ban lam duoc gi|tro ly.*lam gi",
    r"ban dung nguon nao|nguon nao|lay nguon tu dau",
    r"hybrid rag.*agentic rag|agentic rag.*hybrid rag|hybrid khac agentic",
]

HISTORY_DOMAIN_PATTERNS = [
    r"lich su|su viet|viet nam|dai viet|van lang|au lac",
    r"cach mang|khoi nghia|khang chien|chien dich|chien thang|tran ",
    r"bach dang|dien bien phu|thang tam|geneve|doc lap|bac thuoc|phap thuoc",
    r"nha ly|nha tran|nha le|nha nguyen|tay son|chua nguyen|trinh nguyen",
    r"thoi ly|thoi tran|thoi le|thoi nguyen|phong kien|trieu dai",
    r"vua |tuong |lanh tu|anh hung dan toc|nhan vat lich su",
]

EXPLICIT_OOD_PATTERNS = [
    r"\bpython\b|\bjavascript\b|\bjava\b|lap trinh|viet code|doan code|hello world|debug|\bapi\b",
    r"thoi tiet|nhiet do hom nay|du bao mua|do am hom nay",
    r"\bmessi\b|\bronaldo\b|premier league|champions league|ti so bong da|world cup 20\d\d",
    r"cong thuc nau|nau mon|nau pho|pho bo|chien bao lau|luoc bao lau",
    r"giai phuong trinh|phuong trinh|dao ham|tich phan|tinh \d+\s*[+*/-]|x\^?2",
    r"trieu chung|lieu thuoc|uong thuoc|dau bung|dau hong|viem hong|sot bao nhieu|bi benh|y hoc hien dai",
    r"gia bitcoin|gia co phieu|ty gia hom nay|mua coin",
    r"iphone|samsung|dien thoai nao",
    r"dich cau|dich sang tieng viet|translate",
]

FACET_PATTERNS = [
    (
        "winner",
        r"phe nao.*thang|ben nao.*thang|ai thang|chien thang cua phe|"
        r"thang hay thua|thang loi cua ai",
    ),
    ("compare", r"\bso sanh\b|khac nhau|giong nhau|doi chieu"),
    ("context", r"boi canh|hoan canh"),
    ("cause", r"nguyen nhan|vi sao|tai sao|do dau"),
    ("outcome", r"ket qua|hau qua|ket thuc|ra sao"),
    ("significance", r"y nghia|vai tro|tac dong"),
    ("process", r"dien ra nhu the nao|dien bien|qua trinh"),
    ("content", r"noi dung|dieu khoan|quy dinh"),
    ("features", r"dac diem|noi bat"),
    ("time", r"khi nao|thoi gian nao|vao nam nao|nam nao"),
]

FACET_QUERY_SUFFIX = {
    "winner": "kết quả quân sự chiến thuật chiến lược chính trị bên thắng bên thua tổn thất mục tiêu",
    "compare": "so sánh vai trò điểm giống khác đóng góp xây dựng bảo vệ",
    "context": "bối cảnh hoàn cảnh trước khi nguyên nhân dẫn tới",
    "cause": "nguyên nhân lý do điều kiện dẫn đến",
    "outcome": "kết quả kết thúc thắng lợi thất bại hậu quả hệ quả",
    "significance": "ý nghĩa tác động vai trò đánh dấu mở ra góp phần",
    "process": "diễn biến quá trình các giai đoạn mốc chính",
    "content": "nội dung điều khoản quy định chính",
    "features": "đặc điểm nổi bật tính chất lực lượng hình thức",
    "time": "thời gian niên đại mốc năm",
}

FACET_COVERAGE_TERMS = {
    "winner": [
        "chien thang",
        "thang loi",
        "that bai",
        "uu the",
        "quan su",
        "chien luoc",
        "chinh tri",
        "ton that",
    ],
    "context": [
        "boi canh",
        "hoan canh",
        "truoc khi",
        "sau khi",
        "xam luoc",
        "do ",
        "vi ",
        "khi ",
    ],
    "cause": [
        "nguyen nhan",
        "do ",
        "vi ",
        "dan den",
        "nguyen do",
    ],
    "outcome": [
        "ket qua",
        "ket thuc",
        "thang loi",
        "that bai",
        "cham dut",
        "gianh",
        "buoc",
        "thanh lap",
        "thoai vi",
        "tao co so",
    ],
    "significance": [
        "y nghia",
        "danh dau",
        "khang dinh",
        "mo ra",
        "gop phan",
        "tac dong",
        "vai tro",
        "tao tien de",
    ],
    "process": [
        "dien bien",
        "qua trinh",
        "tien cong",
        "khoi nghia",
        "chien dich",
        "giai doan",
    ],
    "content": [
        "noi dung",
        "dieu khoan",
        "quy dinh",
        "ngung ban",
        "hiep dinh",
        "cam ket",
    ],
    "features": [
        "dac diem",
        "noi bat",
        "tinh chat",
        "luc luong",
    ],
}

FACET_TO_METADATA = {
    "context": {"boi canh"},
    "cause": {"nguyen nhan"},
    "outcome": {"ket qua"},
    "winner": {"ket qua"},
    "significance": {"y nghia"},
    "process": {"dien bien"},
    "content": {"noi dung"},
    "features": {"dac diem"},
}


def extract_comparison_targets(question: str) -> list[str]:
    original = clean_text(question)
    normalized = match_norm(original)
    if not re.search(r"\bso sanh\b|khac nhau|giong nhau|doi chieu", normalized):
        return []

    body = re.sub(
        r"^\s*(hãy\s+|hay\s+)?(so\s+sánh|so sanh|đối\s+chiếu|doi chieu)\s+",
        "",
        original,
        flags=re.I,
    )
    body = re.sub(r"\s+(khác nhau|giong nhau|giống nhau)\s+(như thế nào|ra sao)?\??\s*$", "", body, flags=re.I)
    body = re.sub(r"\s+(như thế nào|ra sao)\??\s*$", "", body, flags=re.I)
    body = body.strip(" .?!:;")
    if not body:
        return []

    parts = re.split(r"\s+(?:và|va|với|voi)\s+|[,;/]+", body, maxsplit=2, flags=re.I)
    targets = [part.strip(" .?!:;") for part in parts if part.strip(" .?!:;")]
    return targets[:2] if len(targets) >= 2 else []


def text_matches_target(text: str, target: str) -> bool:
    target_norm = match_norm(target)
    text_norm = match_norm(text)
    if not target_norm or not text_norm:
        return False
    target_terms = [term for term in target_norm.split() if len(term) > 2]
    if not target_terms:
        return target_norm in text_norm
    hits = sum(1 for term in target_terms if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text_norm))
    return target_norm in text_norm or hits >= max(1, min(len(target_terms), 2))


COMPARISON_TARGET_QUERY_SUFFIX = (
    "bối cảnh mục tiêu tính chất lực lượng hình thức đấu tranh kết quả hệ quả ý nghĩa lịch sử"
)


def build_comparison_target_queries(
    question: str,
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build deterministic compare queries without a generative query planner."""
    targets = extract_comparison_targets(question)
    if len(targets) < 2:
        return {}
    analysis = analysis or {}
    requested_facets = [
        facet
        for facet in analysis.get("facets", [])
        if facet not in {"general", "compare", "time"} and facet in FACET_QUERY_SUFFIX
    ]
    facet_suffix = " ".join(FACET_QUERY_SUFFIX[facet] for facet in requested_facets[:2]).strip()
    intent_suffix = facet_suffix or COMPARISON_TARGET_QUERY_SUFFIX
    return {
        "targets": {"target_a": targets[0], "target_b": targets[1]},
        "target_a_query": clean_text(f"{targets[0]} {intent_suffix}"),
        "target_b_query": clean_text(f"{targets[1]} {intent_suffix}"),
        "global_query": clean_text(question),
        "strategy": "target_entity+requested_historical_facet+comparison_dimensions",
    }


NON_EVENT_SUBJECT_TITLE_CUES = {
    "tuong dai",
    "bao tang",
    "duong",
    "pho",
    "thanh pho",
    "san bay",
    "truong",
    "ky niem",
    "le hoi",
    "dia danh",
    "di tich",
}
EVENT_TARGET_PREFIXES = {
    "chien thang",
    "chien dich",
    "tran",
    "cach mang",
    "cuoc cach mang",
    "tong khoi nghia",
    "cuoc khoi nghia",
    "khoi nghia",
    "phong trao",
    "su kien",
}


def _event_target_non_subject_penalty(target: str, title: str) -> float:
    target_norm = match_norm(target)
    title_norm = match_norm(title)
    if not target_norm or not title_norm:
        return 0.0
    is_event = any(target_norm == prefix or target_norm.startswith(prefix + " ") for prefix in EVENT_TARGET_PREFIXES)
    if not is_event:
        return 0.0
    return 0.28 if any(cue in title_norm for cue in NON_EVENT_SUBJECT_TITLE_CUES) else 0.0


def _target_direct_relevance(target: str, chunk: dict[str, Any]) -> dict[str, Any]:
    target_norm = match_norm(target)
    title_norm = match_norm(chunk.get("title", ""))
    text_norm = match_norm(chunk.get("text", ""))
    metadata_norm = match_norm(str(chunk.get("metadata") or ""))
    target_terms = [term for term in target_norm.split() if len(term) > 2]
    title_hits = sum(1 for term in target_terms if re.search(rf"\b{re.escape(term)}\b", title_norm))
    text_hits = sum(1 for term in target_terms if re.search(rf"\b{re.escape(term)}\b", text_norm))
    title_ratio = title_hits / max(len(target_terms), 1)
    text_ratio = text_hits / max(len(target_terms), 1)
    reasons: list[str] = []
    score = 0.0
    if target_norm and target_norm in title_norm:
        score = 1.0
        reasons.append("title_phrase")
    elif title_ratio >= 0.67:
        score = 0.82
        reasons.append("title_terms")
    elif target_norm and target_norm in metadata_norm:
        score = 0.62
        reasons.append("metadata_phrase")
    elif target_norm and target_norm in text_norm:
        score = 0.42
        reasons.append("text_phrase")
    elif text_ratio >= 0.67:
        score = 0.32
        reasons.append("text_terms")
    non_subject_penalty = _event_target_non_subject_penalty(target, title_norm)
    if non_subject_penalty:
        score -= non_subject_penalty
        reasons.append("non_event_subject_title_penalty")
    incidental_penalty = 0.18 if score and not any(reason.startswith("title_") for reason in reasons) else 0.0
    direct_subject_score = max(0.0, score - incidental_penalty)
    return {
        "score": direct_subject_score,
        "incidental_penalty": incidental_penalty,
        "direct": any(reason.startswith("title_") for reason in reasons) and not non_subject_penalty,
        "direct_subject_score": direct_subject_score,
        "reasons": reasons,
    }


def balance_comparison_candidates(
    question: str,
    candidates: list[dict[str, Any]],
    final_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reserve useful model-visible capacity for both compare targets."""
    targets = extract_comparison_targets(question)
    if len(targets) < 2 or not candidates:
        return candidates[:final_k], {}

    annotated: list[dict[str, Any]] = []
    for candidate in candidates:
        item = candidate
        relevance_a = _target_direct_relevance(targets[0], item)
        relevance_b = _target_direct_relevance(targets[1], item)
        item["comparison_target_relevance"] = {
            "target_a": relevance_a,
            "target_b": relevance_b,
        }
        best_label, best_relevance = max(
            (("target_a", relevance_a), ("target_b", relevance_b)),
            key=lambda pair: pair[1]["score"],
        )
        if relevance_a["score"] >= 0.45 and relevance_b["score"] >= 0.45 and abs(relevance_a["score"] - relevance_b["score"]) <= 0.12:
            label = "shared"
        elif best_relevance["score"] > 0:
            label = best_label
        else:
            label = "unknown"
        role_labels = set(item.get("retrieval_query_roles") or [])
        if "target_b" in role_labels and "target_a" not in role_labels:
            label = "target_b"
        elif "target_a" in role_labels and "target_b" not in role_labels:
            label = "target_a"
        item["comparison_target"] = label
        item["incidental_target_penalty"] = float(best_relevance["incidental_penalty"])
        item["comparison_adjusted_score"] = float(
            item.get("final_retrieval_score", 0.0)
            + 0.12 * best_relevance["score"]
            - best_relevance["incidental_penalty"]
        )
        annotated.append(item)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    reserved_ids: dict[str, list[str]] = {"target_a": [], "target_b": []}
    reserve_per_target = min(2, max(1, final_k // 3))
    for label, target in (("target_a", targets[0]), ("target_b", targets[1])):
        ranked = sorted(
            annotated,
            key=lambda item: (
                item["comparison_target_relevance"][label]["direct"],
                item["comparison_target_relevance"][label]["score"],
                item.get("final_retrieval_score", 0.0),
            ),
            reverse=True,
        )
        for item in ranked:
            chunk_id = str(item.get("chunk_id") or "")
            directness = item["comparison_target_relevance"][label]
            if not chunk_id or chunk_id in selected_ids or directness["score"] < 0.24:
                continue
            if len(reserved_ids[label]) >= reserve_per_target:
                break
            selected.append(item)
            selected_ids.add(chunk_id)
            reserved_ids[label].append(chunk_id)

    for item in sorted(annotated, key=lambda value: value["comparison_adjusted_score"], reverse=True):
        if len(selected) >= final_k:
            break
        chunk_id = str(item.get("chunk_id") or "")
        if chunk_id and chunk_id not in selected_ids:
            selected.append(item)
            selected_ids.add(chunk_id)

    report = {
        "targets": {"target_a": targets[0], "target_b": targets[1]},
        "reserve_per_target": reserve_per_target,
        "reserved_ids": reserved_ids,
        "target_a_candidate_count": sum(
            item["comparison_target_relevance"]["target_a"]["score"] >= 0.24 for item in annotated
        ),
        "target_b_candidate_count": sum(
            item["comparison_target_relevance"]["target_b"]["score"] >= 0.24 for item in annotated
        ),
        "incidental_penalty_ids": [
            str(item.get("chunk_id"))
            for item in annotated
            if item.get("incidental_target_penalty", 0.0) > 0
        ],
        "final_ids": [str(item.get("chunk_id")) for item in selected[:final_k]],
    }
    return selected[:final_k], report


# ============================================================
# Hybrid Retriever
# ============================================================

class HybridRetriever:
    def __init__(self, service: RAGService):
        self.service = service
        self.history_anchor_embs: np.ndarray | None = None
        self.ood_anchor_embs: np.ndarray | None = None
        self._anchors_ready = False

    # ========================================================
    # Configuration
    # ========================================================

    @property
    def retrieval_config(self) -> dict[str, Any]:
        if not self.service.config:
            return {}
        return self.service.config.get("retrieval", {}) or {}

    def _cfg(self, name: str, default: Any) -> Any:
        return self.retrieval_config.get(name, default)

    @property
    def dense_fetch_k(self) -> int:
        return int(self._cfg("dense_fetch_k", 80))

    @property
    def bm25_fetch_k(self) -> int:
        return int(self._cfg("bm25_fetch_k", 80))

    @property
    def rrf_k(self) -> int:
        return int(self._cfg("rrf_k", 60))

    @property
    def rrf_top_k(self) -> int:
        return int(self._cfg("rrf_top_k", 20))

    @property
    def final_context_k(self) -> int:
        return int(self._cfg("final_context_k", 6))

    @property
    def rerank_batch_size(self) -> int:
        return int(self._cfg("rerank_batch_size", 32))

    @property
    def max_query_variants(self) -> int:
        return int(self._cfg("max_query_variants", 3))

    @property
    def query_expansion_weight(self) -> float:
        return float(self._cfg("query_expansion_weight", 0.82))

    @property
    def max_chunks_per_title(self) -> int:
        return int(self._cfg("max_chunks_per_title", 2))

    @property
    def enable_context_diversity(self) -> bool:
        return bool(self._cfg("enable_context_diversity", True))

    @property
    def metadata_max_bonus(self) -> float:
        return float(self._cfg("metadata_max_bonus", 0.18))

    @property
    def intent_facet_bonus(self) -> float:
        return float(self._cfg("intent_facet_bonus", 0.025))

    def _guard_cfg(self, name: str, default: float) -> float:
        config = self.service.config or {}

        for section_name in ("guards", "guard", "ood", "retrieval"):
            section = config.get(section_name, {})
            if isinstance(section, dict) and name in section:
                return float(section[name])

        return float(default)

    @property
    def ood_anchor_margin(self) -> float:
        return self._guard_cfg("ood_anchor_margin", 0.02)

    @property
    def secondary_ood_margin(self) -> float:
        return self._guard_cfg("secondary_ood_margin", -0.06)

    @property
    def secondary_min_dense(self) -> float:
        return self._guard_cfg("secondary_min_dense", 0.28)

    # ========================================================
    # Runtime validation
    # ========================================================

    def _ensure_ready(self) -> None:
        if not self.service.loaded:
            raise RuntimeError("RAGService has not been loaded.")

        if self.service.embedder is None:
            raise RuntimeError(
                "Retrieval runtime is unavailable. "
                "Use APP_MODE=retrieval-only or APP_MODE=full."
            )

        if self.service.reranker is None:
            raise RuntimeError("Reranker is not loaded.")

        if self.service.faiss_index is None:
            raise RuntimeError("FAISS index is not loaded.")

        if self.service.bm25 is None:
            raise RuntimeError("BM25 index is not loaded.")

        if not self.service.chunks:
            raise RuntimeError("Corpus is not loaded.")

    # ========================================================
    # Anchor embeddings / OOD
    # ========================================================

    def _ensure_anchor_embeddings(self) -> None:
        if self._anchors_ready:
            return

        anchor_texts = [
            "query: " + text
            for text in HISTORY_ANCHORS + OOD_ANCHORS
        ]

        anchor_embeddings = self.service.embedder.encode(
            anchor_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        history_count = len(HISTORY_ANCHORS)

        self.history_anchor_embs = anchor_embeddings[:history_count]
        self.ood_anchor_embs = anchor_embeddings[history_count:]
        self._anchors_ready = True

    def intent_scores(self, question: str) -> dict[str, Any]:
        self._ensure_anchor_embeddings()

        q_emb = self.service.embedder.encode(
            [query_for_embedding(question)],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")[0]

        history_score = float(np.max(self.history_anchor_embs @ q_emb))
        ood_score = float(np.max(self.ood_anchor_embs @ q_emb))

        normalized = match_norm(question)

        explicit_ood = any(
            re.search(pattern, normalized, flags=re.I)
            for pattern in EXPLICIT_OOD_PATTERNS
        )

        return {
            "history_anchor": history_score,
            "ood_anchor": ood_score,
            "margin": history_score - ood_score,
            "explicit_ood": bool(explicit_ood),
            "query_embedding": q_emb,
        }

    # ========================================================
    # Question analysis
    # ========================================================

    def classify_question(self, question: str) -> dict[str, Any]:
        """Run the lightweight anchor intent guard without corpus retrieval."""
        question = clean_text(question)
        normalized = match_norm(question)
        intent = self.intent_scores(question)
        public_intent = {key: value for key, value in intent.items() if key != "query_embedding"}
        explicit_meta = any(re.search(pattern, normalized, flags=re.I) for pattern in META_PATTERNS)
        explicit_history = any(re.search(pattern, normalized, flags=re.I) for pattern in HISTORY_DOMAIN_PATTERNS)
        explicit_year = bool(re.search(r"(?<!\d)(9[0-9]{2}|1[0-9]{3}|20[0-9]{2})(?!\d)", normalized))
        comparison_targets = extract_comparison_targets(question)

        if explicit_meta:
            result = "meta"
            reason = "meta_request"
        elif explicit_history or comparison_targets or (explicit_year and intent["history_anchor"] >= intent["ood_anchor"]):
            result = "in_domain"
            reason = "history_signal"
        elif intent["explicit_ood"]:
            result = "out_of_domain"
            reason = "explicit_ood"
        elif intent["margin"] < self.secondary_ood_margin:
            result = "out_of_domain"
            reason = "anchor_guard"
        elif abs(float(intent["margin"])) < self.ood_anchor_margin:
            result = "ambiguous"
            reason = "low_domain_margin"
        else:
            result = "in_domain"
            reason = "history_anchor"
        telemetry = current_request_telemetry()
        if telemetry is not None:
            telemetry.domain_gate_result = result
            telemetry.domain_gate_reason = reason
            telemetry.history_anchor = float(intent["history_anchor"])
            telemetry.ood_anchor = float(intent["ood_anchor"])
            telemetry.domain_margin = float(intent["margin"])
        return {
            "is_ood": result == "out_of_domain",
            "ood_reason": reason if result == "out_of_domain" else "",
            "domain_gate_result": result,
            "domain_gate_reason": reason,
            "history_anchor": float(intent["history_anchor"]),
            "ood_anchor": float(intent["ood_anchor"]),
            "domain_margin": float(intent["margin"]),
            "comparison_targets": comparison_targets,
            "intent": public_intent,
        }

    def analyze_question(self, question: str) -> dict[str, Any]:
        question = clean_text(question)
        normalized = match_norm(question)

        facets = [
            name
            for name, pattern in FACET_PATTERNS
            if re.search(pattern, normalized, flags=re.I)
        ]

        if "winner" in facets and "outcome" not in facets:
            facets.append("outcome")

        if not facets:
            facets = ["general"]

        years = sorted(
            {
                int(value)
                for value in re.findall(r"(?<!\d)(\d{3,4})(?!\d)", question)
            }
        )

        meaningful_facets = [
            facet
            for facet in facets
            if facet != "time"
        ]

        comparison_targets = extract_comparison_targets(question) if "compare" in facets else []

        return {
            "question": question,
            "facet": facets[0],
            "facets": facets,
            "years": years,
            "comparison_targets": comparison_targets,
            "is_multi_part": len(meaningful_facets) >= 2,
        }

    # ========================================================
    # Query planning
    # ========================================================

    def plan_query_variants(self, question: str) -> list[str]:
        analysis = self.analyze_question(question)
        base = clean_text(question)

        variants = [base]

        for facet in analysis["facets"]:
            suffix = FACET_QUERY_SUFFIX.get(facet)

            if suffix:
                variants.append(clean_text(f"{base} {suffix}"))

            if len(variants) >= self.max_query_variants:
                break

        return list(dict.fromkeys(variants))[:self.max_query_variants]

    # ========================================================
    # Facet coverage
    # ========================================================

    def context_covers_facet(
        self,
        chunk: dict[str, Any],
        facet: str,
    ) -> bool:
        if facet in {"general", "compare", "time"}:
            return False

        metadata = chunk.get("metadata") or {}

        metadata_facets = {
            match_norm(value)
            for value in (metadata.get("content_facets") or [])
            if clean_text(value)
        }

        if FACET_TO_METADATA.get(facet, set()) & metadata_facets:
            return True

        body = match_norm(
            f"{chunk.get('title', '')} {chunk.get('text', '')}"
        )

        return any(
            term in body
            for term in FACET_COVERAGE_TERMS.get(facet, [])
        )

    # ========================================================
    # Metadata soft boost
    # ========================================================

    def metadata_bonus(
        self,
        question: str,
        chunk: dict[str, Any],
        analysis: dict[str, Any] | None = None,
    ) -> tuple[float, list[str]]:
        analysis = analysis or self.analyze_question(question)
        metadata = chunk.get("metadata") or {}

        normalized_question = match_norm(question)

        question_years = {
            int(value)
            for value in re.findall(r"(?<!\d)(\d{3,4})(?!\d)", question)
        }

        bonus = 0.0
        hits: list[str] = []

        metadata_years = {
            int(year)
            for year in metadata.get("years", [])
            if str(year).isdigit()
        }

        year_hits = sorted(question_years & metadata_years)

        if year_hits:
            bonus += min(0.07, 0.035 * len(year_hits))
            hits.append("years=" + ",".join(map(str, year_hits)))

        field_weights = {
            "people": 0.055,
            "events": 0.045,
            "documents": 0.055,
            "dynasties": 0.055,
            "locations": 0.030,
            "periods": 0.030,
            "topics": 0.012,
        }

        for field, weight in field_weights.items():
            matched = []

            for item in metadata.get(field, []) or []:
                normalized_item = match_norm(item)

                if len(normalized_item) < 4:
                    continue

                pattern = (
                    rf"(?<![a-z0-9])"
                    rf"{re.escape(normalized_item)}"
                    rf"(?![a-z0-9])"
                )

                if re.search(pattern, normalized_question):
                    matched.append(str(item))

            if matched:
                bonus += weight * min(2, len(matched))
                hits.append(f"{field}=" + "|".join(matched[:2]))

        metadata_content = {
            match_norm(value)
            for value in (metadata.get("content_facets") or [])
            if clean_text(value)
        }

        for facet in analysis.get("facets", []):
            if FACET_TO_METADATA.get(facet, set()) & metadata_content:
                bonus += self.intent_facet_bonus
                hits.append(f"intent_facet={facet}")

        return min(self.metadata_max_bonus, bonus), hits

    # ========================================================
    # Dense search
    # ========================================================

    def dense_search(
        self,
        question: str,
        k: int | None = None,
    ) -> list[tuple[int, float]]:
        k = k or self.dense_fetch_k

        embedding = self.service.embedder.encode(
            [query_for_embedding(question)],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indexes = self.service.faiss_index.search(
            embedding,
            min(k, self.service.faiss_index.ntotal),
        )

        return [
            (int(index), float(score))
            for index, score in zip(indexes[0], scores[0])
            if int(index) >= 0
        ]

    # ========================================================
    # BM25 search
    # ========================================================

    def bm25_search(
        self,
        question: str,
        k: int | None = None,
    ) -> list[tuple[int, float]]:
        k = k or self.bm25_fetch_k

        query_tokens = bm25s.tokenize(
            [match_norm(question)],
            stopwords=None,
            stemmer=None,
        )

        indexes, scores = self.service.bm25.retrieve(
            query_tokens,
            k=min(k, len(self.service.chunks)),
        )

        index_values = np.asarray(indexes[0]).tolist()
        score_values = np.asarray(scores[0]).tolist()

        return [
            (int(index), float(score))
            for index, score in zip(index_values, score_values)
        ]

    # ========================================================
    # Weighted Reciprocal Rank Fusion
    # ========================================================

    def multi_query_rrf(
        self,
        runs: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        top_k = top_k or self.rrf_top_k

        fused = defaultdict(float)

        info = defaultdict(
            lambda: {
                "retrieval_hits": [],
                "retrieval_query_roles": [],
                "best_dense_score": None,
                "best_bm25_score": None,
            }
        )

        for query_index, run in enumerate(runs):
            weight = float(run.get("weight", 1.0 if query_index == 0 else self.query_expansion_weight))
            query_role = str(run.get("role") or f"query_{query_index}")

            for rank, (corpus_index, score) in enumerate(run["dense"], 1):
                fused[corpus_index] += weight / (self.rrf_k + rank)

                info[corpus_index]["retrieval_hits"].append(
                    f"dense:q{query_index}@{rank}"
                )
                info[corpus_index]["retrieval_query_roles"].append(query_role)

                current = info[corpus_index]["best_dense_score"]

                if current is None or score > current:
                    info[corpus_index]["best_dense_score"] = float(score)

            for rank, (corpus_index, score) in enumerate(run["bm25"], 1):
                fused[corpus_index] += weight / (self.rrf_k + rank)

                info[corpus_index]["retrieval_hits"].append(
                    f"bm25:q{query_index}@{rank}"
                )
                info[corpus_index]["retrieval_query_roles"].append(query_role)

                current = info[corpus_index]["best_bm25_score"]

                if current is None or score > current:
                    info[corpus_index]["best_bm25_score"] = float(score)

        ordered = sorted(
            fused,
            key=fused.get,
            reverse=True,
        )[:top_k]

        output = []

        for corpus_index in ordered:
            chunk = dict(self.service.chunks[corpus_index])

            chunk["_corpus_idx"] = corpus_index
            chunk["rrf_score"] = float(fused[corpus_index])
            chunk.update(info[corpus_index])
            chunk["retrieval_query_roles"] = list(dict.fromkeys(chunk["retrieval_query_roles"]))

            output.append(chunk)

        return output

    # ========================================================
    # Score normalization
    # ========================================================

    @staticmethod
    def minmax(values: list[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)

        if len(array) == 0:
            return array

        low = float(array.min())
        high = float(array.max())

        if high - low < 1e-9:
            return np.ones_like(array) * 0.5

        return (array - low) / (high - low)

    # ========================================================
    # Context diversity
    # ========================================================

    def select_diverse_contexts(
        self,
        candidates: list[dict[str, Any]],
        analysis: dict[str, Any],
        final_k: int,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        if not self.enable_context_diversity:
            return candidates[:final_k]

        selected = []
        selected_ids = set()
        title_counts = Counter()

        def title_key(chunk: dict[str, Any]) -> str:
            return (
                match_norm(chunk.get("title", ""))
                or str(chunk.get("chunk_id", ""))
            )

        def can_add(
            chunk: dict[str, Any],
            cap: int | None = None,
        ) -> bool:
            title_cap = cap if cap is not None else self.max_chunks_per_title
            chunk_id = str(chunk["chunk_id"])

            return (
                chunk_id not in selected_ids
                and title_counts[title_key(chunk)] < title_cap
            )

        def add(chunk: dict[str, Any]) -> None:
            selected.append(chunk)
            selected_ids.add(str(chunk["chunk_id"]))
            title_counts[title_key(chunk)] += 1

        # First pass: cover requested facets.
        for facet in analysis.get("facets", []):
            if facet in {"general", "compare", "time"}:
                continue

            for chunk in candidates:
                if can_add(chunk) and self.context_covers_facet(chunk, facet):
                    add(chunk)
                    break

            if len(selected) >= final_k:
                break

        # Second pass: rank order while respecting title cap.
        for chunk in candidates:
            if len(selected) >= final_k:
                break

            if can_add(chunk):
                add(chunk)

        # Third pass: remove title cap if contexts are still insufficient.
        if len(selected) < final_k:
            for chunk in candidates:
                if len(selected) >= final_k:
                    break

                if str(chunk["chunk_id"]) not in selected_ids:
                    add(chunk)

        selected.sort(
            key=lambda item: item.get("final_retrieval_score", 0.0),
            reverse=True,
        )

        return selected[:final_k]

    @staticmethod
    def context_title_diversity(
        contexts: list[dict[str, Any]],
    ) -> float:
        if not contexts:
            return 0.0

        unique_titles = {
            match_norm(chunk.get("title", ""))
            for chunk in contexts
        }

        return len(unique_titles) / len(contexts)

    # ========================================================
    # Full Phase 9 retrieval pipeline
    # ========================================================

    def retrieve(
        self,
        question: str,
        final_k: int | None = None,
    ) -> dict[str, Any]:
        retrieve_started = time.perf_counter()
        telemetry = current_request_telemetry()
        request_id = telemetry.request_id if telemetry is not None else None
        embedding_ms = 0.0
        faiss_ms = 0.0
        bm25_ms = 0.0
        fusion_ms = 0.0
        reranker_ms = 0.0
        self._ensure_ready()

        question = clean_text(question)

        if not question:
            raise ValueError("Question must not be empty.")

        final_k = final_k or self.final_context_k

        analysis = self.analyze_question(question)
        comparison_query_plan = build_comparison_target_queries(question, analysis)
        if comparison_query_plan:
            query_specs = [
                {"query": comparison_query_plan["global_query"], "role": "global", "weight": 0.7},
                {"query": comparison_query_plan["target_a_query"], "role": "target_a", "weight": 1.0},
                {"query": comparison_query_plan["target_b_query"], "role": "target_b", "weight": 1.0},
            ]
            query_variants = [spec["query"] for spec in query_specs]
        else:
            query_variants = self.plan_query_variants(question)
            query_specs = [
                {
                    "query": query,
                    "role": "global" if index == 0 else "global_expansion",
                    "weight": 1.0 if index == 0 else self.query_expansion_weight,
                }
                for index, query in enumerate(query_variants)
            ]
        classification = self.classify_question(question)
        public_intent = classification.get("intent", {})

        # ----------------------------------------------------
        # Shared domain gate: no corpus retrieval for scoped exits.
        # ----------------------------------------------------

        if classification.get("domain_gate_result") in {"out_of_domain", "meta", "ambiguous"}:
            gate_result = str(classification.get("domain_gate_result"))
            telemetry = current_request_telemetry()
            if telemetry is not None:
                telemetry.retrieval_skipped_due_to_ood = gate_result == "out_of_domain"
                telemetry.llm_calls_skipped_due_to_ood = gate_result in {"out_of_domain", "meta", "ambiguous"}
            result = {
                "question": question,
                "is_ood": gate_result == "out_of_domain",
                "ood_reason": str(classification.get("ood_reason") or ""),
                "domain_gate_result": gate_result,
                "domain_gate_reason": str(classification.get("domain_gate_reason") or ""),
                "intent": public_intent,
                "analysis": analysis,
                "query_variants": query_variants,
                "target_specific_queries": comparison_query_plan,
                "target_retrieval_results": {},
                "comparison_balance": {},
                "candidates20": [],
                "final_context": [],
                "max_dense": None,
                "context_title_diversity": 0.0,
                "tool_trace": [
                    "question_analyzer",
                    "query_planner",
                    f"domain_gate:{gate_result}",
                ],
            }
            log_event(
                "RETRIEVAL_COMPLETE",
                request_id=request_id,
                embedding_ms=embedding_ms,
                faiss_ms=0.0,
                bm25_ms=0.0,
                fusion_ms=0.0,
                reranker_ms=0.0,
                total_ms=(time.perf_counter() - retrieve_started) * 1000,
                candidate_count=0,
                reranker_pair_count=0,
                final_count=0,
            )
            return result

        # ----------------------------------------------------
        # Dense + BM25 for all query variants
        # ----------------------------------------------------

        runs = []

        for query_spec in query_specs:
            query = str(query_spec["query"])
            embedding_started = time.perf_counter()
            embedding = self.service.embedder.encode(
                [query_for_embedding(query)],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")
            embedding_ms += (time.perf_counter() - embedding_started) * 1000
            faiss_started = time.perf_counter()
            scores, indexes = self.service.faiss_index.search(
                embedding,
                min(self.dense_fetch_k, self.service.faiss_index.ntotal),
            )
            dense = [
                (int(index), float(score))
                for index, score in zip(indexes[0], scores[0])
                if int(index) >= 0
            ]
            faiss_ms += (time.perf_counter() - faiss_started) * 1000
            bm25_started = time.perf_counter()
            bm25 = self.bm25_search(query)
            bm25_ms += (time.perf_counter() - bm25_started) * 1000
            runs.append(
                {
                    "query": query,
                    "role": query_spec["role"],
                    "weight": query_spec["weight"],
                    "dense": dense,
                    "bm25": bm25,
                }
            )

        global_run = next((run for run in runs if run.get("role") == "global"), runs[0] if runs else None)
        original_dense = global_run["dense"] if global_run else []

        max_dense = (
            original_dense[0][1]
            if original_dense
            else -1.0
        )

        # ----------------------------------------------------
        # OOD guard 2
        # ----------------------------------------------------

        if (
            float(public_intent.get("margin", 0.0)) < self.secondary_ood_margin
            and max_dense < self.secondary_min_dense
        ):
            result = {
                "question": question,
                "is_ood": True,
                "ood_reason": "weak_history_retrieval+anchor_guard",
                "domain_gate_result": "out_of_domain",
                "domain_gate_reason": "weak_history_retrieval+anchor_guard",
                "intent": public_intent,
                "analysis": analysis,
                "query_variants": query_variants,
                "target_specific_queries": comparison_query_plan,
                "target_retrieval_results": {},
                "comparison_balance": {},
                "candidates20": [],
                "final_context": [],
                "max_dense": float(max_dense),
                "context_title_diversity": 0.0,
                "tool_trace": [
                    "question_analyzer",
                    "query_planner",
                    "multi_query_retrieval",
                    "ood_guard:block",
                ],
            }
            log_event(
                "RETRIEVAL_COMPLETE",
                request_id=request_id,
                embedding_ms=embedding_ms,
                faiss_ms=faiss_ms,
                bm25_ms=bm25_ms,
                fusion_ms=fusion_ms,
                reranker_ms=reranker_ms,
                total_ms=(time.perf_counter() - retrieve_started) * 1000,
                candidate_count=0,
                reranker_pair_count=0,
                final_count=0,
            )
            return result

        # ----------------------------------------------------
        # RRF
        # ----------------------------------------------------

        fusion_started = time.perf_counter()
        candidates = self.multi_query_rrf(
            runs,
            top_k=self.rrf_top_k,
        )
        if comparison_query_plan:
            by_corpus_index = {int(item["_corpus_idx"]): item for item in candidates}
            per_target_k = max(6, final_k * 3)
            for role in ("target_a", "target_b"):
                role_runs = [run for run in runs if run.get("role") == role]
                for item in self.multi_query_rrf(role_runs, top_k=per_target_k):
                    corpus_index = int(item["_corpus_idx"])
                    existing = by_corpus_index.get(corpus_index)
                    if existing is None:
                        candidates.append(item)
                        by_corpus_index[corpus_index] = item
                        continue
                    existing["retrieval_hits"] = list(dict.fromkeys([
                        *existing.get("retrieval_hits", []),
                        *item.get("retrieval_hits", []),
                    ]))
                    existing["retrieval_query_roles"] = list(dict.fromkeys([
                        *existing.get("retrieval_query_roles", []),
                        *item.get("retrieval_query_roles", []),
                    ]))
                    for score_key in ("best_dense_score", "best_bm25_score"):
                        values = [value for value in (existing.get(score_key), item.get(score_key)) if value is not None]
                        existing[score_key] = max(values) if values else None
        fusion_ms = (time.perf_counter() - fusion_started) * 1000

        if not candidates:
            result = {
                "question": question,
                "is_ood": False,
                "ood_reason": "",
                "domain_gate_result": "in_domain",
                "domain_gate_reason": str(classification.get("domain_gate_reason") or ""),
                "intent": public_intent,
                "analysis": analysis,
                "query_variants": query_variants,
                "target_specific_queries": comparison_query_plan,
                "target_retrieval_results": {},
                "comparison_balance": {},
                "candidates20": [],
                "final_context": [],
                "max_dense": float(max_dense),
                "context_title_diversity": 0.0,
                "tool_trace": [
                    "question_analyzer",
                    "query_planner",
                    "multi_query_retrieval:no_candidates",
                ],
            }
            log_event(
                "RETRIEVAL_COMPLETE",
                request_id=request_id,
                embedding_ms=embedding_ms,
                faiss_ms=faiss_ms,
                bm25_ms=bm25_ms,
                fusion_ms=fusion_ms,
                reranker_ms=reranker_ms,
                total_ms=(time.perf_counter() - retrieve_started) * 1000,
                candidate_count=0,
                reranker_pair_count=0,
                final_count=0,
            )
            return result

        # ----------------------------------------------------
        # Cross-encoder reranking
        # ----------------------------------------------------

        pairs = [
            [
                question,
                clean_text(
                    f"{chunk.get('title', '')}\n"
                    f"{chunk.get('text', '')}"
                ),
            ]
            for chunk in candidates
        ]

        reranker_started = time.perf_counter()
        reranker_scores = np.asarray(
            self.service.reranker.predict(
                pairs,
                batch_size=self.rerank_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        ).reshape(-1).astype(float)
        reranker_ms = (time.perf_counter() - reranker_started) * 1000

        reranker_norm = self.minmax(reranker_scores.tolist())

        rrf_norm = self.minmax(
            [chunk["rrf_score"] for chunk in candidates]
        )

        # ----------------------------------------------------
        # Final Phase 9 score
        #
        # 0.72 * reranker
        # + 0.28 * RRF
        # + metadata soft boost
        # ----------------------------------------------------

        for index, chunk in enumerate(candidates):
            bonus, hits = self.metadata_bonus(
                question,
                chunk,
                analysis=analysis,
            )

            chunk["reranker_score"] = float(reranker_scores[index])
            chunk["reranker_norm"] = float(reranker_norm[index])
            chunk["rrf_norm"] = float(rrf_norm[index])
            chunk["metadata_bonus"] = float(bonus)
            chunk["metadata_hits"] = hits

            chunk["final_retrieval_score"] = float(
                0.72 * reranker_norm[index]
                + 0.28 * rrf_norm[index]
                + bonus
            )

        candidates.sort(
            key=lambda item: item["final_retrieval_score"],
            reverse=True,
        )

        # ----------------------------------------------------
        # Final context diversity
        # ----------------------------------------------------
        if comparison_query_plan:
            final_context, comparison_balance = balance_comparison_candidates(
                question,
                candidates,
                final_k,
            )
        else:
            final_context = self.select_diverse_contexts(
                candidates,
                analysis,
                final_k,
            )
            comparison_balance = {}

        target_retrieval_results: dict[str, list[dict[str, Any]]] = {}
        if comparison_query_plan:
            for role in ("target_a", "target_b", "global"):
                role_items = [
                    item
                    for item in candidates
                    if role in item.get("retrieval_query_roles", [])
                ]
                role_items.sort(
                    key=lambda item: (
                        item.get("comparison_target_relevance", {}).get(role, {}).get("direct", False),
                        item.get("comparison_target_relevance", {}).get(role, {}).get("score", 0.0),
                        item.get("final_retrieval_score", 0.0),
                    ),
                    reverse=True,
                )
                target_retrieval_results[role] = role_items[:10]

        trace_candidates = candidates[:self.rrf_top_k]
        trace_ids = {str(item.get("chunk_id")) for item in trace_candidates}
        trace_candidates.extend(
            item for item in final_context if str(item.get("chunk_id")) not in trace_ids
        )

        result = {
            "question": question,
            "is_ood": False,
            "ood_reason": "",
            "domain_gate_result": "in_domain",
            "domain_gate_reason": str(classification.get("domain_gate_reason") or ""),
            "intent": public_intent,
            "analysis": analysis,
            "query_variants": query_variants,
            "candidates20": trace_candidates,
            "final_context": final_context,
            "target_specific_queries": comparison_query_plan,
            "target_retrieval_results": target_retrieval_results,
            "comparison_balance": comparison_balance,
            "max_dense": float(max_dense),
            "context_title_diversity": self.context_title_diversity(
                final_context
            ),
            "tool_trace": [
                "question_analyzer",
                "query_planner",
                f"multi_query_faiss_bm25:{len(query_variants)}q",
                *( ["compare:target_specific_retrieval", "compare:target_balanced_pool"] if comparison_query_plan else [] ),
                "weighted_rrf:top20",
                "cross_encoder_reranker",
                "metadata_intent_boost",
                "evidence_diversity",
            ],
        }
        log_event(
            "RETRIEVAL_COMPLETE",
            request_id=request_id,
            embedding_ms=embedding_ms,
            faiss_ms=faiss_ms,
            bm25_ms=bm25_ms,
            fusion_ms=fusion_ms,
            reranker_ms=reranker_ms,
            total_ms=(time.perf_counter() - retrieve_started) * 1000,
            candidate_count=len(candidates),
            reranker_pair_count=len(pairs),
            final_count=len(final_context),
        )
        return result
