from __future__ import annotations

import hashlib
import heapq
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..citations import extract_evidence_citations, format_evidence_citation
from ..dedup import normalized_question
from ..io_utils import iter_jsonl
from ..schema import (
    SEARCH_HISTORY_TOOL,
    SEARCH_WEB_TOOL,
    SEARCH_WIKIPEDIA_TOOL,
    canonical_id,
    make_trajectory,
    tool_call,
)
from ..teacher.base import Teacher, TeacherRequest


TASK_TYPES = (
    "factual", "cause", "significance", "compare", "summary", "multihop",
    "verification", "hard_negative", "insufficient_evidence",
)
SUBJECT_TYPES = (
    "person", "event", "organization", "state", "dynasty", "document",
    "location", "date", "topic",
)
ANALYTICAL_SUBJECTS = {"event", "organization", "state", "dynasty", "document"}
MEANINGFUL_SUBJECTS = {"person", *ANALYTICAL_SUBJECTS}
FACTUAL_SUBJECTS = set(MEANINGFUL_SUBJECTS)
COMPARE_SUBJECTS = set(MEANINGFUL_SUBJECTS)
SUMMARY_SUBJECTS = set(MEANINGFUL_SUBJECTS)
SYNTHETIC_MARKER = "Z-1901"
CUSTOM_HISTORY_SYSTEM_PROMPT = (
    "Bạn là trợ lý trung tâm về lịch sử. Hãy trả lời bằng tiếng Việt, dùng công cụ khi cần, "
    "chỉ kết luận từ bằng chứng quan sát được và trích dẫn ID nguồn."
)

TASK_TERMS: dict[str, tuple[str, ...]] = {
    "factual": ("lịch sử", "dấu mốc", "hoạt động", "đặc điểm", "năm", "thành lập"),
    "cause": ("nguyên nhân", "do", "vì", "dẫn đến", "bối cảnh", "điều kiện", "hình thành", "thành lập"),
    "significance": ("ý nghĩa", "tác động", "vai trò", "đánh dấu", "mở ra", "góp phần", "kết quả"),
    "summary": ("bối cảnh", "nguyên nhân", "diễn biến", "mốc", "kết quả", "ý nghĩa", "tác động"),
    "multihop": ("bối cảnh", "nguyên nhân", "điều kiện", "kết quả", "hệ quả", "ý nghĩa", "tác động"),
    "compare": ("vai trò", "hoạt động", "đóng góp", "bối cảnh", "diễn biến", "kết quả", "ý nghĩa"),
    "verification": ("năm", "diễn ra", "thành lập", "ký", "kết quả", "đánh dấu"),
    "hard_negative": ("khó khăn", "hạn chế", "thất bại", "suy yếu", "bất lợi", "khủng hoảng", "sa lầy"),
    "insufficient_evidence": (SYNTHETIC_MARKER.casefold(),),
}
METADATA_FIELDS = {
    "people": "person", "events": "event", "documents": "document",
    "organizations": "organization", "states": "state", "dynasties": "dynasty",
    "locations": "location", "dates": "date", "topics": "topic",
}
TITLE_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("document", ("hiệp định", "hiệp ước", "hòa ước", "tuyên ngôn", "chiếu ", "sắc lệnh", "văn kiện")),
    ("event", (
        "hội nghị", "đảo chính", "trận ", "chiến thắng", "chiến dịch", "khởi nghĩa",
        "cách mạng", "phong trào", "chiến tranh", "cuộc chiến", "tổng tiến công",
        "biến cố", "giải bóng đá", "đại hội thể thao",
    )),
    ("organization", (
        "đảng ", "hội ", "mặt trận", "việt minh", "tổ chức", "quân đội",
        "không quân", "hải quân", "lục quân", "quân chủng", "binh chủng",
        "lực lượng", "chính phủ", "ủy ban", "liên minh", "trường quân",
    )),
    ("state", ("việt nam dân chủ cộng hòa", "việt nam cộng hòa", "đại việt", "đại nam", "đại cồ việt", "vương quốc", "quốc gia", "chính quyền")),
    ("dynasty", ("triều đại", "vương triều", "triều ", "chúa nguyễn", "chúa trịnh")),
    ("location", (
        "thành phố", "tỉnh ", "huyện ", "quận ", "xã ", "phường ",
        "thị trấn", "thị xã", "làng ", "thôn ", "ấp ", "bản ", "buôn ",
        "địa danh", "trụ sở", "sông ", "núi ", "đảo ", "vịnh ",
        "cần thơ", "hà nội", "sài gòn",
    )),
)
STRONG_LOCATION_LABELS = {
    "thanh pho", "tinh", "huyen", "xa", "phuong", "thi tran",
    "thi xa", "lang", "thon", "ap", "ban", "buon", "dia danh", "tru so",
}
PERSON_TEXT_CUES = (
    "sinh nam", "sinh ngay", "qua doi", "mat nam", "ten that", "ong la",
    "ba la", "vi vua", "danh tuong", "nha cach mang", "chinh tri gia",
    "nhan vat lich su", "anh hung dan toc", "cuoc doi", "tieu su",
    "giao si", "linh muc", "giam muc", "tong giam muc", "tuong linh",
)
PERSON_ACTIVITY_CUES = (
    "hoat dong", "lanh dao", "chi huy", "tham gia", "giu chuc", "dam nhiem",
    "duoc bo nhiem", "duoc phong", "sang lap", "dung dau", "phuc vu",
)
ROLE_FACET_TERMS: dict[str, tuple[str, ...]] = {
    "biography_timeline": (*PERSON_TEXT_CUES, *PERSON_ACTIVITY_CUES, "nam", "tu nam", "den nam"),
    "role_contribution": (
        "vai tro", "dong gop", "gop phan", "lanh dao", "chi huy", "sang lap",
        "anh huong", "hoat dong", "thuc day", "bao ve", "xay dung",
    ),
    "context_cause": (
        "nguyen nhan", "boi canh", "dieu kien", "do", "vi", "bat nguon",
        "xuat phat", "hinh thanh", "thanh lap", "ra doi", "truoc do", "tien de",
    ),
    "context_timeline": (*TASK_TERMS["cause"], "dien bien", "moc", "nam", "thoi ky", "giai doan"),
    "result_significance": (
        "y nghia", "tac dong", "he qua", "hau qua", "ket qua", "vai tro",
        "danh dau", "gop phan", "dan toi", "lam thay doi", "cham dut",
        "ket thuc", "thuc day", "tao dieu kien", "khang dinh",
        "anh huong", "de lai", "suy yeu", "phat trien", "duoc xem la",
    ),
    "corrective_facet": (
        *TASK_TERMS["hard_negative"], "thieu", "khong du", "thua", "ton that",
        "mat", "bi danh bai",
    ),
    "wrong_facet": ("thanh cong", "thang loi", "uu the", "thuan loi"),
}
NON_PERSON_TOPIC_PREFIXES = (
    "chu", "tieng", "van hoa", "ngon ngu", "chu viet", "van hoc",
    "ly hoc", "nho giao", "phat giao", "chu nghia", "quan ham",
)
CULTURAL_TOPIC_CUES = (
    "xẩm", "ca trù", "chèo", "tuồng", "quan họ", "dân ca", "nghệ thuật",
    "âm nhạc", "nhã nhạc", "tín ngưỡng", "tôn giáo", "triết học", "học thuyết",
)
DYNASTY_NON_CUES = {
    "hat", "nuoc", "van", "tho", "may", "hang", "bao", "xuat ban",
    "nghien cuu", "truong", "ga", "khach", "rong",
}
VIETNAMESE_LANGUAGE_WORDS = (
    "la", "cua", "va", "duoc", "trong", "nhung", "mot", "nam", "tai",
    "voi", "da", "co", "tu", "den", "sau", "truoc",
)
HISTORICAL_SCOPE_CUES = (
    "lich su", "chien tranh", "chien dich", "khoi nghia", "cach mang",
    "phong trao", "thanh lap", "hinh thanh", "ra doi", "trieu dai", "vuong trieu",
    "thuoc dia", "khang chien", "xung dot", "sinh nam", "qua doi", "tri vi",
    "lanh dao", "chi huy", "tuong linh", "nhan vat", "dong gop", "hoat dong",
    "chinh tri", "quan su",
)
HISTORICAL_EVENT_TITLE_CUES = (
    "hoi nghi", "dao chinh", "chien tranh", "cuoc chien", "tran", "chien dich",
    "chien thang", "khoi nghia", "cach mang", "phong trao", "tong tien cong",
    "bien co",
)
HISTORICAL_ORGANIZATION_CUES = (
    "dang", "mat tran", "quan doi", "khong quan", "hai quan", "luc quan",
    "quan chung", "chinh phu", "chinh quyen", "giao hoi", "ton giao",
)
HISTORICAL_PERSON_ROLE_CUES = (
    "vi vua", "hoang de", "danh tuong", "tuong linh", "nha cach mang",
    "chinh tri gia", "anh hung dan toc", "lanh dao", "chi huy", "si quan",
    "tinh bao", "giao si", "linh muc", "giam muc", "tong giam muc", "tri vi",
)
HISTORICAL_NARRATIVE_CUES = (
    "boi canh", "nguyen nhan", "dien bien", "he qua", "hau qua", "thuoc dia",
    "khang chien", "xung dot", "thanh lap", "hinh thanh", "ra doi", "suy vong",
)
NON_HISTORY_SUBJECT_CUES = (
    "cuoc thi ca hat", "chuong trinh truyen hinh", "than tuong am nhac",
    "giai bong da", "vo dich u 21", "dai hoi the thao", "sea games",
    "bang tong sap huy chuong", "bang xep hang", "vong bang", "tran thang 3 diem",
)
ALIAS_METADATA_FIELDS = (
    "aliases", "alias", "alternative_names", "alternate_names", "other_names",
    "also_known_as", "names",
)
REFERENCE_NOISE_CUES = (
    "xem them", "tham khao", "chu thich", "lien ket ngoai", "muc luc",
    "the loai", "danh sach bai", "trang dinh huong", "nguon tham khao",
)


@dataclass(frozen=True)
class CustomBuildConfig:
    task_counts: dict[str, int] = field(default_factory=lambda: {
        "factual": 20, "cause": 10, "significance": 10, "compare": 8,
        "summary": 10, "multihop": 10, "verification": 6,
        "hard_negative": 6, "insufficient_evidence": 6,
    })
    top_k: int = 6
    seed: int = 42
    max_corpus_records: int = 10_000
    observation_char_budget: int = 12_000
    trajectory_observation_char_budget: int = 6_000
    max_result_text_chars: int = 1_600
    max_candidate_attempts_per_task: int = 10_000

    def __post_init__(self) -> None:
        unknown = set(self.task_counts) - set(TASK_TYPES)
        if unknown:
            raise ValueError(f"unknown custom task types: {sorted(unknown)}")
        if any(value < 0 for value in self.task_counts.values()):
            raise ValueError("custom task counts must be non-negative")
        if self.top_k < 1 or self.max_corpus_records < 1:
            raise ValueError("top_k and max_corpus_records must be positive")
        if self.observation_char_budget < 256:
            raise ValueError("observation_char_budget must be at least 256")
        if self.trajectory_observation_char_budget < 768:
            raise ValueError("trajectory_observation_char_budget must be at least 768")
        if self.max_result_text_chars < 64:
            raise ValueError("max_result_text_chars must be at least 64")
        if self.max_candidate_attempts_per_task < 1:
            raise ValueError("max_candidate_attempts_per_task must be positive")


@dataclass(frozen=True)
class QueryPlan:
    tool_name: str
    query: str
    top_k: int
    role: str
    required: bool = True
    expected_empty: bool = False
    is_fallback: bool = False

    def arguments(self) -> dict[str, Any]:
        arguments: dict[str, Any] = {"query": self.query, "top_k": self.top_k}
        if self.tool_name == "search_wikipedia":
            arguments["language"] = "vi"
        return arguments


@dataclass(frozen=True)
class Observation:
    plan: QueryPlan
    results: list[dict[str, Any]]


def _plain(text: Any) -> str:
    return " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())


def _match_norm(text: Any) -> str:
    value = unicodedata.normalize("NFKD", _plain(text).casefold())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.replace("đ", "d")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


def _entity_norm(text: Any) -> str:
    """Normalize spacing/punctuation while preserving Vietnamese distinctions."""
    value = unicodedata.normalize("NFKC", _plain(text).casefold())
    return " ".join(re.sub(r"[^\w]+", " ", value, flags=re.UNICODE).replace("_", " ").split())


def _title(row: dict[str, Any]) -> str:
    return _plain(row.get("title"))


def _entity_matches_title(title: str, values: Any) -> bool:
    title_norm = _entity_norm(title)
    if not title_norm or not isinstance(values, list):
        return False
    for value in values:
        entity = _entity_norm(value)
        if entity and (entity == title_norm or (len(entity) >= 6 and entity in title_norm)):
            return True
    return False


def _title_without_parenthetical(title: str) -> str:
    return _plain(re.sub(r"\s*\([^()]*\)\s*$", "", title)).strip()


def title_implied_subject_type(title: str) -> str | None:
    """Return only strong non-person title evidence; otherwise remain uncertain."""
    title = _plain(title)
    title_norm = _match_norm(title)
    if not title_norm:
        return None
    if re.fullmatch(r"(?:ngay\s+)?\d{1,2}(?:\s+thang\s+\d{1,2})?(?:\s+nam\s+\d{3,4})?|\d{3,4}", title_norm):
        return "date"
    parenthetical_labels = {
        _match_norm(value)
        for value in re.findall(r"\(([^()]*)\)", title, flags=re.UNICODE)
    }
    if any(
        value == label or value.startswith(f"{label} ")
        for value in parenthetical_labels
        for label in STRONG_LOCATION_LABELS
    ) or any(
        title_norm == label or title_norm.startswith(f"{label} ")
        for label in STRONG_LOCATION_LABELS
    ):
        return "location"
    if any(
        title_norm == prefix or title_norm.startswith(f"{prefix} ")
        for prefix in NON_PERSON_TOPIC_PREFIXES
    ):
        return "topic"
    cultural_title = _entity_norm(_title_without_parenthetical(title))
    if any(
        cultural_title == _entity_norm(cue)
        or cultural_title.startswith(f"{_entity_norm(cue)} ")
        for cue in CULTURAL_TOPIC_CUES
    ):
        return "topic"
    title_tokens = title_norm.split()
    if len(title_tokens) >= 2 and title_tokens[0] == "nha" and title_tokens[1] not in DYNASTY_NON_CUES:
        return "dynasty"
    title_words = " " + " ".join(
        re.findall(r"\w+", title.casefold(), flags=re.UNICODE)
    ) + " "
    for subject_type, cues in TITLE_CUES:
        for cue in cues:
            cue_words = " ".join(re.findall(r"\w+", cue.casefold(), flags=re.UNICODE))
            if cue_words and f" {cue_words} " in title_words:
                return subject_type
    if "quan su" in parenthetical_labels:
        return "topic"
    return None


def _contains_norm_phrase(text: Any, phrase: Any) -> bool:
    text_norm = _entity_norm(text)
    phrase_norm = _entity_norm(phrase)
    return bool(text_norm and phrase_norm and f" {phrase_norm} " in f" {text_norm} ")


def _biographical_person_evidence(title: str, text: Any) -> bool:
    sentences = _split_sentences(text)
    if not sentences:
        sentences = [_plain(text)] if _plain(text) else []
    for sentence in sentences:
        sentence_norm = _match_norm(sentence)
        mentions_title = _contains_norm_phrase(sentence, title)
        has_biography_cue = any(
            f" {cue} " in f" {sentence_norm} " for cue in PERSON_TEXT_CUES
        )
        has_activity_cue = any(
            f" {cue} " in f" {sentence_norm} " for cue in PERSON_ACTIVITY_CUES
        )
        has_lifespan = bool(re.search(r"\b\d{3,4}\s*[-–—]\s*\d{3,4}\b", sentence))
        if mentions_title and (has_biography_cue or has_activity_cue or has_lifespan):
            return True
    if sentences:
        first_norm = _match_norm(sentences[0])
        return any(first_norm.startswith(cue) for cue in PERSON_TEXT_CUES)
    return False


def _has_dynasty_evidence(row: dict[str, Any], implied_type: str | None) -> bool:
    if implied_type == "dynasty":
        return True
    title = _title(row)
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if _entity_matches_title(title, metadata.get("dynasties")):
        return True
    for sentence in _split_sentences(row.get("text")):
        if not _contains_norm_phrase(sentence, title):
            continue
        sentence_norm = _match_norm(sentence)
        if any(
            f" {cue} " in f" {sentence_norm} "
            for cue in ("trieu dai", "vuong trieu", "trieu", "hoang trieu")
        ):
            return True
    return False


def _text_implied_subject_type(row: dict[str, Any]) -> str | None:
    """Use only target-linked definitional language to correct noisy extracted labels."""
    title = _title(row)
    for sentence in _split_sentences(row.get("text")):
        if not _contains_norm_phrase(sentence, title):
            continue
        sentence_norm = _match_norm(sentence)
        if re.search(r"\bla (?:mot )?(?:vuong quoc|de quoc|quoc gia|nha nuoc|chinh the)\b", sentence_norm):
            return "state"
        if re.search(
            r"\bla (?:mot )?(?:cuoc thi|chuong trinh truyen hinh|chuong trinh thuc te)\b",
            sentence_norm,
        ):
            return "topic"
        if re.search(r"\bla (?:mot )?(?:giai dau|su kien|cuoc hop|hoi nghi)\b", sentence_norm):
            return "event"
        if re.search(r"\bla (?:mot )?(?:loai hinh|the loai|khai niem|hoc thuyet|cap bac)\b", sentence_norm):
            return "topic"
    return None


def _semantic_title_overrides(declared_type: str, implied_type: str | None) -> bool:
    if implied_type is None or implied_type == declared_type:
        return False
    if declared_type in {"person", "dynasty"}:
        return True
    if implied_type in {"topic", "location"}:
        return True
    if implied_type in {"event", "state"} and declared_type in {"organization", "state", "event"}:
        return True
    if implied_type == "organization" and declared_type in {"state", "event"}:
        return True
    if declared_type == "topic" and implied_type in {"organization", "dynasty", "event", "state"}:
        return True
    return {declared_type, implied_type} == {"organization", "dynasty"}


def classify_subject(row: dict[str, Any]) -> str:
    """Classify conservatively: strong evidence is required to emit ``person``."""
    title = _title(row)
    implied_type = title_implied_subject_type(title) or _text_implied_subject_type(row)
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    explicit = _match_norm(
        metadata.get("subject_type") or metadata.get("entity_type") or row.get("subject_type")
    )
    if explicit in SUBJECT_TYPES:
        if _semantic_title_overrides(explicit, implied_type):
            return str(implied_type)
        if explicit == "dynasty" and not _has_dynasty_evidence(row, implied_type):
            return "topic"
        return explicit
    for field, subject_type in METADATA_FIELDS.items():
        if _entity_matches_title(title, metadata.get(field)):
            if _semantic_title_overrides(subject_type, implied_type):
                return str(implied_type)
            if subject_type == "dynasty" and not _has_dynasty_evidence(row, implied_type):
                return "topic"
            return subject_type
    if implied_type is not None:
        return implied_type
    if _biographical_person_evidence(title, row.get("text")):
        return "person"
    return "topic"


def custom_history_eligibility_signals(row: dict[str, Any]) -> tuple[str, ...]:
    """Return language/history evidence for the country-neutral custom dataset scope."""
    title = _title(row)
    combined = _plain(f"{title}. {_plain(row.get('text'))}")
    combined_norm = _match_norm(combined)
    combined_padded = f" {combined_norm} "
    signals: list[str] = []
    subject_type = classify_subject(row)
    original_norm = _entity_norm(combined)
    if re.search(r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", original_norm):
        signals.append("language:vietnamese_diacritics")
    language_word_count = sum(
        f" {word} " in combined_padded for word in VIETNAMESE_LANGUAGE_WORDS
    )
    if language_word_count >= 2:
        signals.append("language:vietnamese_words")
    if isinstance(row.get("history_score"), (int, float)) and row["history_score"] > 0:
        signals.append("weak:history_score")
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if any(metadata.get(field) for field in ("events", "dynasties", "years", "periods")):
        signals.append("weak:history_metadata")
    if re.search(r"\b\d{3,4}\b", combined):
        signals.append("weak:year")
    for cue in HISTORICAL_SCOPE_CUES:
        if f" {cue} " in combined_padded:
            signals.append(f"weak:{cue}")
    title_norm = _match_norm(title)
    title_padded = f" {title_norm} "
    excluded_scope = any(f" {cue} " in combined_padded for cue in NON_HISTORY_SUBJECT_CUES)
    if not excluded_scope:
        if subject_type in {"state", "dynasty", "document"}:
            signals.append(f"history:strong_subject:{subject_type}")
        if subject_type == "event" and any(
            f" {cue} " in title_padded for cue in HISTORICAL_EVENT_TITLE_CUES
        ):
            signals.append("history:strong_event")
        if subject_type == "organization" and any(
            f" {cue} " in combined_padded for cue in HISTORICAL_ORGANIZATION_CUES
        ) and (
            re.search(r"\b\d{3,4}\b", combined)
            or any(f" {cue} " in combined_padded for cue in ("thanh lap", "hinh thanh", "ra doi", "lich su"))
        ):
            signals.append("history:strong_organization")
        if subject_type == "person" and any(
            f" {cue} " in combined_padded for cue in HISTORICAL_PERSON_ROLE_CUES
        ):
            signals.append("history:strong_person")
        narrative_count = sum(
            f" {cue} " in combined_padded for cue in HISTORICAL_NARRATIVE_CUES
        )
        if narrative_count >= 2:
            signals.append("history:strong_narrative")
    return tuple(dict.fromkeys(signals))


def is_custom_history_eligible(row: dict[str, Any]) -> bool:
    """Whether a meaningful subject has Vietnamese-language historical evidence."""
    signals = custom_history_eligibility_signals(row)
    return (
        classify_subject(row) in MEANINGFUL_SUBJECTS
        and any(signal.startswith("language:") for signal in signals)
        and any(signal.startswith("history:strong") for signal in signals)
    )


def vietnam_history_relevance_signals(row: dict[str, Any]) -> tuple[str, ...]:
    """Compatibility alias for the former, overly narrow domain contract."""
    return custom_history_eligibility_signals(row)


def is_vietnam_history_relevant(row: dict[str, Any]) -> bool:
    """Compatibility alias; foreign Vietnamese-language history is intentionally allowed."""
    return is_custom_history_eligible(row)


def task_eligible(row: dict[str, Any], task_type: str) -> bool:
    subject_type = classify_subject(row)
    if task_type == "factual":
        return subject_type in FACTUAL_SUBJECTS
    if task_type in {"cause", "significance", "multihop"}:
        return subject_type in ANALYTICAL_SUBJECTS
    if task_type == "summary":
        return subject_type in SUMMARY_SUBJECTS
    if task_type == "compare":
        return subject_type in COMPARE_SUBJECTS
    if task_type in {"verification", "insufficient_evidence"}:
        return subject_type in MEANINGFUL_SUBJECTS
    if task_type == "hard_negative":
        return subject_type in {"event", "organization", "state", "dynasty"}
    return False


def compare_subjects_compatible(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Allow same-type pairs plus a narrowly nested state/institution relationship."""
    first_type = classify_subject(first)
    second_type = classify_subject(second)
    if first_type == second_type:
        return first_type in COMPARE_SUBJECTS
    if {first_type, second_type} != {"state", "organization"}:
        return False
    first_title = _entity_norm(_title(first))
    second_title = _entity_norm(_title(second))
    return bool(
        first_title and second_title
        and (
            f" {first_title} " in f" {second_title} "
            or f" {second_title} " in f" {first_title} "
        )
    )


def inspect_corpus(path: str | Path, *, max_records: int = 1000) -> dict[str, Any]:
    sources: Counter[str] = Counter()
    facets: Counter[str] = Counter()
    subjects: Counter[str] = Counter()
    titles: set[str] = set()
    rows = malformed = 0
    for record in iter_jsonl(path):
        rows += 1
        if not record.get("chunk_id") or not record.get("text"):
            malformed += 1
        sources[str(record.get("source_type") or record.get("source") or "unknown")] += 1
        titles.add(_title(record))
        subjects[classify_subject(record)] += 1
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        for facet in metadata.get("content_facets", record.get("content_facets", [])) or []:
            facets[str(facet)] += 1
        if rows >= max_records:
            break
    return {
        "path": str(Path(path).resolve()), "sampled_records": rows,
        "malformed_records": malformed, "unique_titles": len(titles),
        "sources": dict(sources), "subject_types": dict(subjects),
        "content_facets": dict(facets), "read_only": True,
    }


def load_seed_records(path: str | Path, *, limit: int, seed: int) -> list[dict[str, Any]]:
    """Keep the best deterministic hash sample while streaming the full corpus."""
    if limit < 1:
        raise ValueError("seed record limit must be positive")
    heap: list[tuple[int, str, dict[str, Any]]] = []
    for row in iter_jsonl(path):
        chunk_id = str(row.get("chunk_id") or "").strip()
        if not chunk_id or not _plain(row.get("text")):
            continue
        score = int(hashlib.sha256(f"{seed}:{chunk_id}".encode()).hexdigest(), 16)
        entry = (-score, chunk_id, row)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    return [row for _, _, row in sorted(heap, key=lambda item: (-item[0], item[1]))]


def _split_sentences(text: Any) -> list[str]:
    clean = _plain(text)
    if not clean:
        return []
    sentences = re.split(r"(?<=[.!?])\s+|\s*[;•]\s*", clean)
    return [sentence.strip(" -") for sentence in sentences if len(sentence.strip(" -")) >= 20]


def _text_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [_plain(value)] if _plain(value) else []
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value if _plain(item)]
    return []


def _target_aliases(row: dict[str, Any]) -> tuple[str, ...]:
    title_norm = _entity_norm(_title(row))
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    aliases: list[str] = []
    for field in ALIAS_METADATA_FIELDS:
        aliases.extend(_text_values(metadata.get(field)))
        aliases.extend(_text_values(row.get(field)))
    for field in METADATA_FIELDS:
        for value in _text_values(metadata.get(field)):
            value_norm = _entity_norm(value)
            if value_norm == title_norm:
                aliases.append(value)
    return tuple(dict.fromkeys(alias for alias in aliases if _match_norm(alias)))


def _entity_variants(target_title: str, target_aliases: Iterable[str] = ()) -> tuple[str, ...]:
    values = [target_title, _title_without_parenthetical(target_title), *target_aliases]
    return tuple(dict.fromkeys(value for value in (_entity_norm(item) for item in values) if value))


def _subject_entity_variants(
    target_title: str, target_subject_type: str, target_aliases: Iterable[str] = (),
) -> tuple[str, ...]:
    variants = _entity_variants(target_title, target_aliases)
    if target_subject_type != "person" or len(_entity_norm(target_title).split()) < 2:
        return variants
    # A surname or other one-token alias cannot identify a multi-token person.
    return tuple(variant for variant in variants if len(variant.split()) >= 2)


def _contains_entity_variant(text: Any, variants: Iterable[str]) -> bool:
    text_norm = _entity_norm(text)
    padded = f" {text_norm} "
    return any(f" {variant} " in padded for variant in variants if variant)


def result_text_mentions_target(
    result: dict[str, Any], *, target_title: str, target_aliases: Iterable[str] = (),
) -> bool:
    """Whether the actual evidence text explicitly names the target or a strong alias."""
    variants = _entity_variants(target_title, target_aliases)
    if len(_entity_norm(target_title).split()) >= 2:
        variants = tuple(variant for variant in variants if len(variant.split()) >= 2)
    return _contains_entity_variant(result.get("text"), variants)


def _metadata_mentions_target(metadata: Any, variants: tuple[str, ...]) -> bool:
    if not isinstance(metadata, dict):
        return False
    fields = (*METADATA_FIELDS, *ALIAS_METADATA_FIELDS, "entities", "entity", "subjects")
    for field in fields:
        for value in _text_values(metadata.get(field)):
            if _entity_norm(value) in variants:
                return True
    return False


def is_result_relevant_to_target(
    result: dict[str, Any], *, target_title: str, target_subject_type: str,
    retrieval_role: str, target_aliases: Iterable[str] = (),
) -> bool:
    """Deterministically reject lexical hits that are not anchored to the target."""
    del retrieval_role
    variants = _subject_entity_variants(target_title, target_subject_type, target_aliases)
    if not variants:
        return False
    result_title_variants = _entity_variants(_title(result))
    title_match = bool(set(variants) & set(result_title_variants))
    return (
        title_match
        or _contains_entity_variant(result.get("text"), variants)
        or _metadata_mentions_target(result.get("metadata"), variants)
    )


def _sentence_score(sentence: str, query: str, task_type: str) -> tuple[int, int, int]:
    sentence_norm = _match_norm(sentence)
    query_terms = {term for term in _match_norm(query).split() if len(term) > 2}
    task_terms = {_match_norm(term) for term in TASK_TERMS.get(task_type, ())}
    return (
        sum(term and term in sentence_norm for term in task_terms),
        sum(term in sentence_norm for term in query_terms),
        int(bool(re.search(r"\b\d{3,4}\b", sentence))),
    )


def _target_formation_score(
    sentence_norm: str, target_title: str, target_aliases: Iterable[str],
) -> int:
    formation = r"(?:hinh thanh|thanh lap|ra doi)"
    for value in (target_title, *target_aliases):
        target_norm = _match_norm(value)
        if not target_norm:
            continue
        target_pattern = re.escape(target_norm)
        if re.search(
            rf"(?:{target_pattern}(?: [a-z0-9]+){{0,5}} (?:duoc )?{formation}|"
            rf"{formation}(?: cua| nen| thanh)? {target_pattern})",
            sentence_norm,
        ):
            return 1
    return 0


def _strong_formation_score(sentence_norm: str) -> int:
    """Recognize formation statements even when the exact-title article uses an implicit subject."""
    return int(bool(re.search(
        r"^(?:da |chinh thuc )?(?:duoc )?(?:hinh thanh|thanh lap|ra doi)(?: |$)",
        sentence_norm,
    )))


def _target_leading_cause_score(
    sentence_norm: str, target_title: str, target_aliases: Iterable[str],
) -> int:
    """Count ``dẫn đến`` only when the stated circumstance leads to the target."""
    for value in (target_title, *target_aliases):
        target_norm = _match_norm(value)
        if target_norm and re.search(
            rf"(?:dan den|dan toi)(?: su| viec| qua trinh)?(?: [a-z0-9]+){{0,5}} {re.escape(target_norm)}(?: |$)",
            sentence_norm,
        ):
            return 1
    return 0


def _causal_relation_score(sentence_norm: str) -> int:
    """Recognize a cause attached to an event/origin predicate, not every lexical do/vì."""
    event_predicate = (
        r"(?:dien ra|xay ra|no ra|bung no|phat sinh|duoc ban hanh|duoc ky|duoc cong bo|"
        r"duoc thanh lap|hinh thanh|ra doi|gianh thang loi|that bai)"
    )
    predicate_then_cause = re.search(
        rf"\b{event_predicate}(?: [a-z0-9]+){{0,8}} (?:do|vi)\b",
        sentence_norm,
    )
    cause_then_predicate = re.search(
        rf"^(?:do|vi)(?: [a-z0-9]+){{2,18}} {event_predicate}\b",
        sentence_norm,
    )
    return int(bool(predicate_then_cause or cause_then_predicate))


def _facet_score(
    sentence: str, *, query: str, task_type: str, retrieval_role: str,
    target_subject_type: str, target_title: str, target_aliases: Iterable[str],
) -> int:
    sentence_norm = _match_norm(sentence)
    padded = f" {sentence_norm} "
    terms = ROLE_FACET_TERMS.get(retrieval_role, TASK_TERMS.get(task_type, ()))
    if retrieval_role == "context_cause":
        formation_terms = {"do", "vi", "hinh thanh", "thanh lap", "ra doi"}
        score = sum(
            f" {_match_norm(term)} " in padded
            for term in terms
            if _match_norm(term) and _match_norm(term) not in formation_terms
        )
        score += _causal_relation_score(sentence_norm)
        score += _target_formation_score(sentence_norm, target_title, target_aliases)
        score += _strong_formation_score(sentence_norm)
        score += _target_leading_cause_score(sentence_norm, target_title, target_aliases)
    else:
        score = sum(f" {_match_norm(term)} " in padded for term in terms if _match_norm(term))
        if retrieval_role == "result_significance":
            # Preserve the dấu distinction: ``củng cố`` is impact evidence, ``cũng có`` is not.
            score += int(" củng cố " in f" {_entity_norm(sentence)} ")
    if retrieval_role in {"biography_timeline", "context_timeline", "target_a", "target_b"} and re.search(
        r"\b\d{3,4}\b", sentence,
    ):
        score += 1
    if retrieval_role in {"claim_support", "corroboration", "external_corroboration"}:
        target_terms = {
            term
            for value in (target_title, *target_aliases)
            for term in _match_norm(value).split()
        }
        query_terms = {
            term for term in _match_norm(query).split()
            if len(term) > 2
            and term not in target_terms
            and term not in {"lich", "nguon", "kiem", "chung", "doi", "chieu", "thoi", "gian"}
        }
        score += min(4, sum(f" {term} " in padded for term in query_terms))
    if retrieval_role in {"target_a", "target_b"} and target_subject_type == "person":
        person_terms = (*PERSON_TEXT_CUES, *PERSON_ACTIVITY_CUES, *ROLE_FACET_TERMS["role_contribution"])
        score += sum(f" {_match_norm(term)} " in padded for term in person_terms)
    return score


def _facet_is_required(retrieval_role: str) -> bool:
    return retrieval_role in {
        "biography_timeline", "role_contribution", "context_cause", "context_timeline",
        "result_significance", "corrective_facet", "claim_support",
        "unsupported_claim", "target_a", "target_b",
    }


def _minimum_facet_score(retrieval_role: str) -> int:
    return 2 if retrieval_role == "claim_support" else 1


def is_result_facet_relevant(
    result: dict[str, Any], *, query: str, task_type: str, retrieval_role: str,
    target_subject_type: str, target_title: str = "",
    target_aliases: Iterable[str] = (),
) -> bool:
    """Require explicit role-specific facet evidence for every hard-gated role."""
    if not _facet_is_required(retrieval_role):
        return True
    if retrieval_role == "unsupported_claim":
        return SYNTHETIC_MARKER.casefold() in _plain(result.get("text")).casefold()
    for sentence in _split_sentences(result.get("text")):
        sentence_norm = _match_norm(sentence)
        if any(f" {cue} " in f" {sentence_norm} " for cue in REFERENCE_NOISE_CUES):
            continue
        score = _facet_score(
            sentence,
            query=query,
            task_type=task_type,
            retrieval_role=retrieval_role,
            target_subject_type=target_subject_type,
            target_title=target_title,
            target_aliases=target_aliases,
        )
        if score >= _minimum_facet_score(retrieval_role):
            return True
    return False


def _sentence_rank(
    sentence: str, *, result_title: str, query: str, task_type: str,
    target_title: str, target_subject_type: str, retrieval_role: str,
    target_aliases: Iterable[str], excluded_titles: Iterable[str],
) -> tuple[int, int, tuple[int, int, int]] | None:
    sentence_norm = _match_norm(sentence)
    if any(f" {cue} " in f" {sentence_norm} " for cue in REFERENCE_NOISE_CUES):
        return None
    target_variants = _subject_entity_variants(
        target_title, target_subject_type, target_aliases,
    )
    excluded_variants = tuple(
        variant
        for excluded in excluded_titles
        for variant in _entity_variants(excluded)
        if variant not in target_variants
        and not any(f" {variant} " in f" {target_variant} " for target_variant in target_variants)
    )
    if excluded_variants and _contains_entity_variant(sentence, excluded_variants):
        return None
    explicit_target = _contains_entity_variant(sentence, target_variants)
    title_target = bool(set(_entity_variants(result_title)) & set(target_variants))
    entity_strength = 2 if explicit_target else 1 if title_target else 0
    if entity_strength == 0:
        return None
    person_specific_roles = {
        "factual", "biography_timeline", "role_contribution", "target_a", "target_b",
        "claim_support",
    }
    if target_subject_type == "person" and retrieval_role in person_specific_roles and not explicit_target:
        return None
    if retrieval_role == "unsupported_claim" and SYNTHETIC_MARKER.casefold() not in sentence.casefold():
        return None
    facet_score = _facet_score(
        sentence, query=query, task_type=task_type, retrieval_role=retrieval_role,
        target_subject_type=target_subject_type, target_title=target_title,
        target_aliases=target_aliases,
    )
    minimum_facet_score = _minimum_facet_score(retrieval_role)
    if _facet_is_required(retrieval_role) and facet_score < minimum_facet_score:
        return None
    return entity_strength, facet_score, _sentence_score(sentence, query, task_type)


def _compact_text(
    text: Any, *, query: str, task_type: str, max_chars: int,
    result_title: str = "", target_title: str = "", target_subject_type: str = "topic",
    retrieval_role: str = "", target_aliases: Iterable[str] = (),
    excluded_titles: Iterable[str] = (),
) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return _plain(text)[:max_chars] if not target_title else ""
    if target_title:
        candidates = [
            (
                index,
                sentence,
                _sentence_rank(
                    sentence, result_title=result_title, query=query, task_type=task_type,
                    target_title=target_title, target_subject_type=target_subject_type,
                    retrieval_role=retrieval_role, target_aliases=target_aliases,
                    excluded_titles=excluded_titles,
                ),
            )
            for index, sentence in enumerate(sentences)
        ]
        ranked = sorted(
            (item for item in candidates if item[2] is not None),
            key=lambda item: (item[2], -item[0]),
            reverse=True,
        )
        if any(item[2][0] == 2 for item in ranked):
            ranked = [item for item in ranked if item[2][0] == 2]
    else:
        ranked = [
            (index, sentence, _sentence_score(sentence, query, task_type))
            for index, sentence in sorted(
                enumerate(sentences),
                key=lambda item: (_sentence_score(item[1], query, task_type), -item[0]),
                reverse=True,
            )
        ]
    selected: list[tuple[int, str]] = []
    chars = 0
    for index, sentence, _ in ranked:
        addition = len(sentence) + int(bool(selected))
        if chars + addition > max_chars and selected:
            continue
        if not selected and addition > max_chars:
            sentence = sentence[:max_chars].rsplit(" ", 1)[0] or sentence[:max_chars]
            addition = len(sentence)
        selected.append((index, sentence))
        chars += addition
        if chars >= max_chars or len(selected) >= 4:
            break
    return " ".join(sentence for _, sentence in sorted(selected))[:max_chars].strip()


def _compact_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    allowed = (
        "content_facets", "events", "people", "documents", "organizations", "states",
        "dynasties", "locations", "periods", "years", *ALIAS_METADATA_FIELDS,
    )
    return {key: metadata[key] for key in allowed if metadata.get(key) not in (None, [], "")}


def _compact_result(
    result: dict[str, Any], plan: QueryPlan, task_type: str, max_text_chars: int, *,
    target_title: str = "", target_subject_type: str = "topic",
    target_aliases: Iterable[str] = (), excluded_titles: Iterable[str] = (),
) -> dict[str, Any] | None:
    chunk_id = str(result.get("chunk_id") or result.get("evidence_id") or "").strip()
    if target_title and not is_result_relevant_to_target(
        result, target_title=target_title, target_subject_type=target_subject_type,
        retrieval_role=plan.role, target_aliases=target_aliases,
    ):
        return None
    text = _compact_text(
        result.get("text"), query=plan.query, task_type=task_type, max_chars=max_text_chars,
        result_title=_title(result), target_title=target_title,
        target_subject_type=target_subject_type, retrieval_role=plan.role,
        target_aliases=target_aliases, excluded_titles=excluded_titles,
    )
    if not chunk_id or not text:
        return None
    compact: dict[str, Any] = {
        "chunk_id": chunk_id, "title": _plain(result.get("title")), "text": text,
        "source_kind": str(result.get("source_kind") or "history"),
        "retrieval_role": plan.role,
    }
    for key in ("url", "source", "source_type"):
        if result.get(key):
            compact[key] = result[key]
    for key in ("final_retrieval_score", "rerank_score", "score"):
        if isinstance(result.get(key), (int, float)):
            compact[key] = result[key]
            break
    if isinstance(result.get("history_score"), (int, float)):
        compact["history_score"] = result["history_score"]
    metadata = _compact_metadata(result.get("metadata"))
    if metadata:
        compact["metadata"] = metadata
    return compact


def compact_observation(
    results: list[dict[str, Any]], plan: QueryPlan, *, task_type: str,
    observation_char_budget: int, max_result_text_chars: int,
    target_title: str = "", target_subject_type: str = "topic",
    target_aliases: Iterable[str] = (), excluded_titles: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Return a compact evidence list whose serialized form respects the budget."""
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in results:
        compact = _compact_result(
            raw, plan, task_type, max_result_text_chars,
            target_title=target_title, target_subject_type=target_subject_type,
            target_aliases=target_aliases, excluded_titles=excluded_titles,
        )
        if compact is None or compact["chunk_id"] in seen_ids:
            continue
        if len(json.dumps([*selected, compact], ensure_ascii=False, sort_keys=True)) <= observation_char_budget:
            selected.append(compact)
            seen_ids.add(compact["chunk_id"])
            continue
        if selected:
            break
        while len(compact["text"]) > 32:
            overflow = len(json.dumps([compact], ensure_ascii=False, sort_keys=True)) - observation_char_budget
            if overflow <= 0:
                selected.append(compact)
                break
            compact["text"] = compact["text"][:max(32, len(compact["text"]) - overflow - 8)].rstrip()
        break
    return selected


def _concrete_claim(row: dict[str, Any]) -> str:
    title_norm = _match_norm(_title(row))
    ranked = sorted(
        enumerate(_split_sentences(row.get("text"))),
        key=lambda item: (
            int(bool(title_norm and title_norm in _match_norm(item[1]))),
            int(bool(re.search(r"\b\d{3,4}\b", item[1]))), -item[0],
        ),
        reverse=True,
    )
    if not ranked:
        raise ValueError(f"cannot derive a concrete claim from chunk {row.get('chunk_id')}")
    return ranked[0][1][:500].rstrip()


def _unsupported_claim(title: str) -> str:
    return f"{SYNTHETIC_MARKER} xác nhận rằng {title} diễn ra trên Sao Hỏa và kết thúc vào năm 1901."


def _question(task_type: str, primary: dict[str, Any], secondary: dict[str, Any] | None = None) -> tuple[str, str | None]:
    title = _title(primary)
    subject_type = classify_subject(primary)
    other = _title(secondary or {})
    claim: str | None = None
    if task_type == "factual":
        question = (
            f"{title} là ai và những dấu mốc lịch sử chính trong cuộc đời, hoạt động của nhân vật này là gì?"
            if subject_type == "person" else
            f"Những dấu mốc, đặc điểm và vai trò lịch sử chính của {title} là gì?"
        )
    elif task_type == "cause":
        question = (
            f"Những bối cảnh và điều kiện nào dẫn đến việc ban hành hoặc ký kết {title}?"
            if subject_type == "document" else
            f"Những bối cảnh và điều kiện nào dẫn đến sự hình thành của {title}?"
            if subject_type in {"organization", "state", "dynasty"} else
            f"Những nguyên nhân và điều kiện lịch sử nào dẫn đến {title}?"
        )
    elif task_type == "significance":
        question = f"{title} có ý nghĩa, vai trò và tác động lịch sử như thế nào?"
    elif task_type == "compare":
        question = (
            f"So sánh vai trò, hoạt động và đóng góp lịch sử của {title} và {other}."
            if subject_type == "person" else
            f"So sánh {title} và {other} theo bối cảnh, diễn biến, kết quả và ý nghĩa."
        )
    elif task_type == "summary":
        if subject_type == "person":
            question = (
                f"Hãy tóm tắt có cấu trúc về {title}, gồm bối cảnh hoặc tiểu sử, "
                "các dấu mốc hoạt động, vai trò và đóng góp lịch sử."
            )
        elif subject_type == "event":
            question = (
                f"Hãy tóm tắt có cấu trúc về {title}, gồm bối cảnh, diễn biến chính, "
                "kết quả và ý nghĩa lịch sử."
            )
        else:
            question = (
                f"Hãy tóm tắt có cấu trúc về {title}, gồm bối cảnh hoặc sự hình thành, "
                "những phát triển chính, vai trò, kết quả và ý nghĩa lịch sử."
            )
    elif task_type == "multihop":
        question = f"Bối cảnh và nguyên nhân của {title} liên hệ như thế nào với kết quả, hệ quả và ý nghĩa của đối tượng này?"
    elif task_type == "verification":
        claim = _concrete_claim(primary)
        question = f'Hãy kiểm chứng nhận định sau bằng các nguồn truy xuất: "{claim}"'
    elif task_type == "hard_negative":
        question = f"Những khó khăn, hạn chế hoặc nguyên nhân suy yếu, thất bại liên quan đến {title} là gì?"
    elif task_type == "insufficient_evidence":
        claim = _unsupported_claim(title)
        question = f'Hãy kiểm chứng nhận định sau bằng bằng chứng truy xuất: "{claim}"'
    else:
        raise ValueError(f"unsupported task type: {task_type}")
    return question, claim


def build_query_plans(
    task_type: str, primary: dict[str, Any], *, secondary: dict[str, Any] | None,
    question: str, claim: str | None, top_k: int,
) -> list[QueryPlan]:
    del question
    title = _title(primary)
    if task_type == "factual":
        return [QueryPlan("search_history", f"{title} lịch sử dấu mốc hoạt động đặc điểm", top_k, "factual")]
    if task_type == "cause":
        return [QueryPlan("search_history", f"{title} bối cảnh nguyên nhân điều kiện hình thành", top_k, "context_cause")]
    if task_type == "significance":
        return [QueryPlan("search_history", f"{title} ý nghĩa tác động vai trò kết quả", top_k, "result_significance")]
    if task_type == "summary":
        if classify_subject(primary) == "person":
            return [
                QueryPlan("search_history", f"{title} tiểu sử cuộc đời dấu mốc hoạt động", top_k, "biography_timeline"),
                QueryPlan("search_history", f"{title} vai trò đóng góp lịch sử", top_k, "role_contribution"),
            ]
        return [
            QueryPlan("search_history", f"{title} bối cảnh nguyên nhân hình thành diễn biến mốc thời gian", top_k, "context_timeline"),
            QueryPlan("search_history", f"{title} kết quả ý nghĩa tác động vai trò", top_k, "result_significance"),
        ]
    if task_type == "multihop":
        return [
            QueryPlan("search_history", f"{title} bối cảnh nguyên nhân điều kiện hình thành", top_k, "context_cause"),
            QueryPlan("search_history", f"{title} kết quả hệ quả ý nghĩa tác động", top_k, "result_significance"),
        ]
    if task_type == "compare":
        if secondary is None:
            raise ValueError("compare query planning requires a secondary subject")
        facets = "vai trò hoạt động đóng góp lịch sử" if classify_subject(primary) == "person" else "bối cảnh diễn biến kết quả ý nghĩa"
        return [
            QueryPlan("search_history", f"{title} {facets}", top_k, "target_a"),
            QueryPlan("search_history", f"{_title(secondary)} {facets}", top_k, "target_b"),
        ]
    if task_type == "verification":
        return [
            QueryPlan("search_history", f"{title} {claim}", top_k, "claim_support"),
            QueryPlan(
                "search_history", f"{title} mốc thời gian nguồn đối chiếu kết quả",
                top_k, "corroboration", required=False, expected_empty=True,
            ),
        ]
    if task_type == "hard_negative":
        return [
            QueryPlan(
                "search_history", f"{title} thành công thắng lợi ưu thế", top_k,
                "wrong_facet", required=False, expected_empty=True,
            ),
            QueryPlan("search_history", f"{title} khó khăn hạn chế thất bại suy yếu bất lợi", top_k, "corrective_facet"),
        ]
    if task_type == "insufficient_evidence":
        return [QueryPlan(
            "search_history", f"{title} {claim}", top_k, "unsupported_claim",
            required=False, expected_empty=True,
        )]
    raise ValueError(f"unsupported task type: {task_type}")


def _search_observations(
    plans: list[QueryPlan], retriever: Any, *, task_type: str,
    primary: dict[str, Any], secondary: dict[str, Any] | None,
    config: CustomBuildConfig, observation_slots: int | None = None,
) -> list[Observation]:
    slots = observation_slots or len(plans)
    if slots < 1:
        return []
    slot_budget = min(
        config.observation_char_budget,
        config.trajectory_observation_char_budget // slots,
    )
    observations = []
    for plan in plans:
        target = secondary if plan.role == "target_b" and secondary is not None else primary
        target_title = _title(target)
        target_subject_type = classify_subject(target)
        aliases = _target_aliases(target)
        excluded_titles = (
            (_title(primary),) if plan.role == "target_b" else
            (_title(secondary),) if plan.role == "target_a" and secondary is not None else
            ()
        )
        compact = compact_observation(
            retriever.search(plan.query, top_k=plan.top_k), plan, task_type=task_type,
            observation_char_budget=slot_budget,
            max_result_text_chars=config.max_result_text_chars,
            target_title=target_title, target_subject_type=target_subject_type,
            target_aliases=aliases, excluded_titles=excluded_titles,
        )
        selected_plan = plan
        fallback_roles = {
            "cause": {"context_cause"},
            "significance": {"result_significance"},
            "summary": {"context_timeline", "result_significance", "biography_timeline", "role_contribution"},
            "multihop": {"context_cause", "result_significance"},
            "verification": {"corroboration"},
            "compare": {"target_a", "target_b"},
            "hard_negative": {"corrective_facet"},
        }
        if not compact and plan.role in fallback_roles.get(task_type, set()):
            fallback_plan = QueryPlan(
                plan.tool_name,
                f"{target_title} lịch sử",
                plan.top_k,
                plan.role,
                required=plan.required,
                expected_empty=plan.expected_empty,
                is_fallback=True,
            )
            compact = compact_observation(
                retriever.search(fallback_plan.query, top_k=fallback_plan.top_k),
                fallback_plan,
                task_type=task_type,
                observation_char_budget=slot_budget,
                max_result_text_chars=config.max_result_text_chars,
                target_title=target_title, target_subject_type=target_subject_type,
                target_aliases=aliases, excluded_titles=excluded_titles,
            )
            selected_plan = fallback_plan
        observations.append(Observation(selected_plan, compact))
    return observations


def _required_observations_present(observations: list[Observation]) -> bool:
    return all(observation.results for observation in observations if observation.plan.required)


def _result_sentences(results: list[dict[str, Any]], task_type: str, *, limit: int = 3) -> list[tuple[str, str]]:
    candidates: list[tuple[tuple[int, int, int], str, str, int]] = []
    best_by_chunk: dict[str, tuple[tuple[int, int, int], str, str, int]] = {}
    for result_index, result in enumerate(results):
        chunk_id = str(result.get("chunk_id") or "")
        for sentence_index, sentence in enumerate(_split_sentences(result.get("text"))):
            if not chunk_id:
                continue
            candidate = (_sentence_score(sentence, "", task_type), chunk_id, sentence, -(result_index * 100 + sentence_index))
            if chunk_id not in best_by_chunk or (candidate[0], candidate[3]) > (best_by_chunk[chunk_id][0], best_by_chunk[chunk_id][3]):
                best_by_chunk[chunk_id] = candidate
    candidates.extend(best_by_chunk.values())
    candidates.sort(key=lambda item: (item[0], item[3], item[1]), reverse=True)
    return [(chunk_id, sentence) for _, chunk_id, sentence, _ in candidates[:limit]]


def _format_evidence(items: list[tuple[str, str]]) -> list[str]:
    return [f"- {sentence} {format_evidence_citation(chunk_id)}" for chunk_id, sentence in items]


def _observed_ids(observations: list[Observation]) -> set[str]:
    return {str(result["chunk_id"]) for obs in observations for result in obs.results if result.get("chunk_id")}


def _answer(task_type: str, observations: list[Observation], *, claim: str | None) -> str:
    by_role = {obs.plan.role: obs.results for obs in observations}
    if task_type == "insufficient_evidence":
        return (
            f"chưa đủ bằng chứng: các kết quả quan sát không trực tiếp chứng minh nhận định mang dấu {SYNTHETIC_MARKER}. "
            "Không thể dùng các đoạn liên quan nhưng không khớp làm bằng chứng cho nhận định này."
        )
    if task_type == "verification":
        claim_key = normalized_question(claim or "")
        supported = [
            (str(result["chunk_id"]), claim or "")
            for obs in observations for result in obs.results
            if claim_key and claim_key in normalized_question(str(result.get("text") or ""))
        ]
        if not supported:
            return "Chưa đủ bằng chứng: kết quả truy xuất chỉ liên quan đến chủ đề nhưng không trực tiếp chứng minh nhận định đã nêu."
        return "Nhận định được bằng chứng truy xuất hỗ trợ trực tiếp:\n" + "\n".join(_format_evidence(supported[:2]))
    if task_type == "compare":
        left = _result_sentences(by_role.get("target_a", []), task_type, limit=2)
        right = _result_sentences(by_role.get("target_b", []), task_type, limit=2)
        if not left or not right:
            evidence = _format_evidence(left + right)
            return "Chưa đủ bằng chứng cân bằng cho cả hai đối tượng để so sánh đáng tin cậy." + (("\n" + "\n".join(evidence)) if evidence else "")
        return "Bằng chứng cho đối tượng thứ nhất:\n" + "\n".join(_format_evidence(left)) + "\nBằng chứng cho đối tượng thứ hai:\n" + "\n".join(_format_evidence(right))
    if task_type in {"summary", "multihop"}:
        biography = _result_sentences(by_role.get("biography_timeline", []), "factual", limit=2)
        contribution = _result_sentences(by_role.get("role_contribution", []), "significance", limit=2)
        context = _result_sentences(
            by_role.get("context_timeline", []) + by_role.get("context_cause", []), "cause", limit=2,
        )
        outcomes = _result_sentences(by_role.get("result_significance", []), "significance", limit=2)
        if not biography and not contribution and not context and not outcomes:
            return "Chưa đủ bằng chứng quan sát để xây dựng câu trả lời có cấu trúc."
        parts = []
        if biography:
            parts.append("Tiểu sử và các dấu mốc hoạt động:\n" + "\n".join(_format_evidence(biography)))
        if contribution:
            parts.append("Vai trò và đóng góp lịch sử:\n" + "\n".join(_format_evidence(contribution)))
        if context:
            parts.append("Bối cảnh/nguyên nhân:\n" + "\n".join(_format_evidence(context)))
        if outcomes:
            parts.append("Kết quả/ý nghĩa:\n" + "\n".join(_format_evidence(outcomes)))
        return "\n".join(parts)
    relevant = [
        result for obs in observations
        if task_type != "hard_negative" or obs.plan.role == "corrective_facet"
        for result in obs.results
    ]
    selected = _result_sentences(relevant, task_type, limit=3)
    if not selected:
        return "Chưa đủ bằng chứng quan sát phù hợp với đúng khía cạnh được hỏi."
    prefix = {
        "cause": "Bằng chứng về bối cảnh và nguyên nhân:",
        "significance": "Bằng chứng về ý nghĩa và tác động:",
        "hard_negative": "Kết quả sai khía cạnh không được dùng; bằng chứng phù hợp về khó khăn hoặc hạn chế:",
        "factual": "Các dữ kiện lịch sử chính từ bằng chứng quan sát:",
    }.get(task_type, "Từ bằng chứng quan sát:")
    return prefix + "\n" + "\n".join(_format_evidence(selected))


def _stable_source_group(row: dict[str, Any]) -> str:
    return str(row.get("url") or row.get("source_sha1") or row.get("title") or row.get("chunk_id"))


def _trajectory(
    *, task_type: str, question: str, claim: str | None,
    observations: list[Observation], primary: dict[str, Any], secondary: dict[str, Any] | None,
    trajectory_observation_char_budget: int,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": CUSTOM_HISTORY_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    for index, obs in enumerate(observations, 1):
        call_id = f"call_{obs.plan.tool_name}_{index:04d}"
        messages.extend([
            {"role": "assistant", "content": None, "tool_calls": [tool_call(call_id, obs.plan.tool_name, obs.plan.arguments())]},
            {"role": "tool", "name": obs.plan.tool_name, "tool_call_id": call_id, "content": json.dumps(obs.results, ensure_ascii=False, sort_keys=True)},
        ])
    answer = _answer(task_type, observations, claim=claim)
    messages.append({"role": "assistant", "content": answer})
    observed_ids = _observed_ids(observations)
    parsed_citations = extract_evidence_citations(answer, observed_ids)
    cited_ids = list(parsed_citations.citations)
    if parsed_citations.unknown_ids or not set(cited_ids).issubset(observed_ids):
        raise ValueError("deterministic answer cited evidence absent from compact observations")

    primary_group = _stable_source_group(primary)
    source_groups = [primary_group]
    if secondary is not None:
        source_groups.append(_stable_source_group(secondary))
    for obs in observations:
        if obs.plan.tool_name != "search_history":
            source_groups.extend(str(result.get("url") or result.get("chunk_id")) for result in obs.results if result.get("url") or result.get("chunk_id"))
    source_groups = list(dict.fromkeys(source_groups))
    pair_key = None
    if secondary is not None:
        pair_key = "||".join(sorted((_match_norm(_title(primary)), _match_norm(_title(secondary)))))
    observation_chars = [len(json.dumps(obs.results, ensure_ascii=False, sort_keys=True)) for obs in observations]
    tool_map = {"search_history": SEARCH_HISTORY_TOOL, "search_wikipedia": SEARCH_WIKIPEDIA_TOOL, "search_web": SEARCH_WEB_TOOL}
    used_tools = list(dict.fromkeys(obs.plan.tool_name for obs in observations))
    return make_trajectory(
        trajectory_id=canonical_id("custom_history_v4", {
            "task": task_type, "primary": primary.get("chunk_id"),
            "secondary": (secondary or {}).get("chunk_id"), "question": question,
        }),
        source_dataset="custom_history", task_type=task_type,
        difficulty="hard" if task_type in {"compare", "multihop", "verification", "hard_negative", "insufficient_evidence"} else "medium",
        tools=[tool_map[name] for name in used_tools], messages=messages,
        provenance={
            "builder_version": "v4", "dataset_id": "local_enriched_vietnamese_history_corpus",
            "corpus_chunk_id": primary["chunk_id"],
            "secondary_corpus_chunk_id": (secondary or {}).get("chunk_id"),
            "primary_title": _title(primary), "secondary_title": _title(secondary or {}) or None,
            "primary_aliases": list(_target_aliases(primary)),
            "secondary_aliases": list(_target_aliases(secondary)) if secondary is not None else [],
            "subject_type": classify_subject(primary),
            "custom_history_eligible": is_custom_history_eligible(primary),
            "custom_history_eligibility_signals": list(custom_history_eligibility_signals(primary)),
            "vietnam_history_relevant": is_custom_history_eligible(primary),
            "vietnam_history_relevance_signals": list(custom_history_eligibility_signals(primary)),
            "seed_history_score": primary.get("history_score"),
            "seed_content_facets": (
                primary.get("metadata", {}).get("content_facets", [])
                if isinstance(primary.get("metadata"), dict) else []
            ),
            "secondary_subject_type": classify_subject(secondary) if secondary is not None else None,
            "secondary_custom_history_eligible": (
                is_custom_history_eligible(secondary) if secondary is not None else None
            ),
            "secondary_custom_history_eligibility_signals": (
                list(custom_history_eligibility_signals(secondary)) if secondary is not None else []
            ),
            "secondary_vietnam_history_relevant": (
                is_custom_history_eligible(secondary) if secondary is not None else None
            ),
            "secondary_vietnam_history_relevance_signals": (
                list(custom_history_eligibility_signals(secondary)) if secondary is not None else []
            ),
            "compare_pair_key": pair_key,
            "concrete_claim": claim if task_type == "verification" else None,
            "synthetic_claim": claim if task_type == "insufficient_evidence" else None,
            "source_document_id": primary_group, "source_group": primary_group,
            "source_groups": source_groups, "grounded": True,
            "evidence_ids": cited_ids, "observed_evidence_ids": sorted(observed_ids),
            "retrieval_queries": [
                {
                    "tool": obs.plan.tool_name, "query": obs.plan.query,
                    "top_k": obs.plan.top_k, "role": obs.plan.role,
                    "required": obs.plan.required,
                    "expected_empty": obs.plan.expected_empty,
                    "is_fallback": obs.plan.is_fallback,
                }
                for obs in observations
            ],
            "observation_chars": observation_chars,
            "trajectory_observation_chars": sum(observation_chars),
            "trajectory_observation_char_budget": trajectory_observation_char_budget,
            "observation_result_counts": [len(obs.results) for obs in observations],
            "external_verification": any(name != "search_history" for name in used_tools),
            "requires_final_answer": True, "corpus_read_only": True,
        },
    )


def _find_compare_secondary(
    records: list[dict[str, Any]], primary_index: int, primary: dict[str, Any], used_pairs: set[str],
) -> dict[str, Any] | None:
    primary_title = _match_norm(_title(primary))
    for offset in range(1, len(records)):
        candidate = records[(primary_index + offset) % len(records)]
        candidate_title = _match_norm(_title(candidate))
        pair_key = "||".join(sorted((primary_title, candidate_title)))
        if (
            candidate_title and candidate_title != primary_title
            and compare_subjects_compatible(primary, candidate)
            and is_custom_history_eligible(candidate)
            and task_eligible(candidate, "compare") and pair_key not in used_pairs
        ):
            return candidate
    return None


def _apply_teacher_to_rows(rows: list[dict[str, Any]], teacher: Teacher, *, seed: int) -> list[dict[str, Any]]:
    if not rows:
        return rows
    requests = []
    for index, row in enumerate(rows):
        evidence = [json.loads(message["content"]) for message in row["messages"] if message.get("role") == "tool"]
        requests.append(TeacherRequest(
            task_type=str(row["task_type"]),
            question=str(next(message["content"] for message in row["messages"] if message["role"] == "user")),
            evidence=json.dumps(evidence, ensure_ascii=False),
            allowed_evidence_ids=tuple((row.get("provenance") or {}).get("observed_evidence_ids", [])),
            seed=seed + index,
        ))
    responses = teacher.generate(requests)
    if len(responses) != len(rows):
        raise ValueError("teacher must return exactly one answer per request")
    for row, response, request in zip(rows, responses, requests):
        answer = _plain(response.answer)
        parsed_citations = extract_evidence_citations(answer, request.allowed_evidence_ids)
        citations = set(parsed_citations.citations)
        explicitly_insufficient = "chưa đủ bằng chứng" in answer.casefold()
        if (
            answer and not parsed_citations.unknown_ids
            and citations.issubset(set(request.allowed_evidence_ids))
            and (citations or not request.allowed_evidence_ids or explicitly_insufficient)
        ):
            row["messages"][-1]["content"] = answer
            row["provenance"]["evidence_ids"] = sorted(citations)
            row["provenance"]["teacher_enhanced"] = True
        else:
            row["provenance"]["teacher_enhanced"] = False
            row["provenance"]["teacher_fallback_reason"] = "empty answer or citation outside allowed evidence IDs"
    return rows


def build_custom_trajectories(
    corpus_path: str | Path, retriever: Any, *, config: CustomBuildConfig,
    completed_ids: set[str] | None = None, teacher: Teacher | None = None,
    external_retriever: Any | None = None,
) -> Iterable[dict[str, Any]]:
    """Build deterministic V4 rows; CLI teacher enhancement is a later stage."""
    completed = completed_ids if completed_ids is not None else set()
    records = load_seed_records(corpus_path, limit=config.max_corpus_records, seed=config.seed)
    if not records:
        raise ValueError("corpus contains no usable records")
    seen_questions: set[str] = set()
    used_titles = {task: set() for task in TASK_TYPES}
    used_compare_pairs: set[str] = set()
    for task_type in TASK_TYPES:
        wanted = config.task_counts.get(task_type, 0)
        selected = 0
        candidate_attempts = 0
        task_rows: list[dict[str, Any]] = []
        for primary_index, primary in enumerate(records):
            if selected >= wanted or candidate_attempts >= config.max_candidate_attempts_per_task:
                break
            if not is_custom_history_eligible(primary) or not task_eligible(primary, task_type):
                continue
            title_key = _match_norm(_title(primary))
            if not title_key:
                continue
            secondary = None
            if task_type == "compare":
                secondary = _find_compare_secondary(records, primary_index, primary, used_compare_pairs)
                if secondary is None:
                    continue
            elif title_key in used_titles[task_type]:
                continue
            question, claim = _question(task_type, primary, secondary)
            question_key = normalized_question(question)
            if not question_key or question_key in seen_questions:
                continue
            pair_key = None
            if secondary is not None:
                pair_key = "||".join(sorted((title_key, _match_norm(_title(secondary)))))
                if pair_key in used_compare_pairs:
                    continue
            candidate_attempts += 1
            plans = build_query_plans(
                task_type, primary, secondary=secondary, question=question,
                claim=claim, top_k=config.top_k,
            )
            has_external_slot = task_type == "verification" and external_retriever is not None
            observation_slots = len(plans) + int(has_external_slot)
            observations = _search_observations(
                plans,
                retriever,
                task_type=task_type,
                primary=primary,
                secondary=secondary,
                config=config,
                observation_slots=observation_slots,
            )
            if not _required_observations_present(observations):
                continue
            if task_type == "verification" and external_retriever is not None:
                plan = QueryPlan(
                    "search_wikipedia",
                    f"{_title(primary)} kiểm chứng mốc thời gian lịch sử",
                    config.top_k,
                    "external_corroboration",
                    required=False,
                    expected_empty=True,
                )
                slot_budget = min(
                    config.observation_char_budget,
                    config.trajectory_observation_char_budget // observation_slots,
                )
                observations.append(Observation(plan, compact_observation(
                    external_retriever.search(plan.tool_name, plan.query, top_k=plan.top_k),
                    plan, task_type=task_type,
                    observation_char_budget=slot_budget,
                    max_result_text_chars=config.max_result_text_chars,
                    target_title=_title(primary),
                    target_subject_type=classify_subject(primary),
                    target_aliases=_target_aliases(primary),
                )))
            row = _trajectory(
                task_type=task_type, question=question, claim=claim,
                observations=observations, primary=primary, secondary=secondary,
                trajectory_observation_char_budget=config.trajectory_observation_char_budget,
            )
            seen_questions.add(question_key)
            used_titles[task_type].add(title_key)
            if pair_key:
                used_compare_pairs.add(pair_key)
            selected += 1
            if row["id"] not in completed:
                task_rows.append(row)
        if selected < wanted:
            raise ValueError(
                "eligible, retrievable unique corpus subjects could produce only "
                f"{selected}/{wanted} {task_type} trajectories after {candidate_attempts} candidate attempts"
            )
        if teacher is not None:
            task_rows = _apply_teacher_to_rows(task_rows, teacher, seed=config.seed + selected)
        yield from task_rows


def build_no_tool_trajectories() -> list[dict[str, Any]]:
    pairs = [
        ("Xin chào!", "Xin chào! Tôi có thể hỗ trợ bạn tìm hiểu lịch sử Việt Nam."),
        ("Bạn có thể làm gì?", "Tôi có thể hỗ trợ tra cứu, đối chiếu và giải thích các vấn đề lịch sử Việt Nam."),
    ]
    return [
        make_trajectory(
            trajectory_id=canonical_id("custom_history_no_tool", {"index": index, "question": question}),
            source_dataset="custom_history", task_type="no_tool", difficulty="easy", tools=[],
            messages=[{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
            provenance={
                "builder_version": "v4", "synthetic": True, "subject_type": "topic",
                "source_group": "no-tool", "source_groups": ["no-tool"],
                "requires_final_answer": True,
            },
        )
        for index, (question, answer) in enumerate(pairs)
    ]
