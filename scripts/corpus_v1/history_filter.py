"""Deterministic high-recall relevance heuristic, not a calibrated classifier."""

import re
import unicodedata


YEAR = re.compile(r"(?<!\d)(?:[1-9]\d{2,3})(?!\d)|\b(?:thế kỷ|thế kỉ|TCN|SCN)\b", re.I)
STRONG = ("lịch sử", "triều đại", "vương quốc", "đế quốc", "chiến tranh", "trận đánh",
          "cách mạng", "khởi nghĩa", "hiệp ước", "thuộc địa", "khảo cổ", "di sản",
          "di tích", "văn minh", "cổ đại", "trung đại", "nhà nước", "hoàng đế",
          "vua ", "hoàng hậu", "tướng quân", "chiến dịch", "thời kỳ", "thời đại",
          "độc lập", "phong kiến", "thành cổ", "đình", "chùa", "lăng", "cung điện")
CONTEXT = ("sinh", "mất", "qua đời", "thành lập", "được xây dựng", "thủ đô",
           "tỉnh", "thành phố", "quốc gia", "vùng", "địa danh", "chính trị",
           "quân sự", "ngoại giao", "kinh tế", "văn hóa", "tôn giáo", "học giả",
           "nhà văn", "nghệ sĩ", "nhà thám hiểm", "nhà lãnh đạo", "tổng thống",
           "thủ tướng", "vương triều", "dân tộc", "đền", "đô thị")
CLEARLY_UNRELATED = ("công thức nấu ăn", "hướng dẫn cài đặt", "mã nguồn python",
                     "phiên bản phần mềm", "thông số điện thoại", "lịch thi đấu hôm nay")


def _fold(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def classify(title: str, text: str) -> tuple[int, str, list[str]]:
    title = _fold(title)
    body = _fold(text)
    sample = body[:20000]
    strong_title = [term for term in STRONG if term in title]
    strong_body = [term for term in STRONG if term in sample]
    context_title = [term for term in CONTEXT if term in title]
    context_body = [term for term in CONTEXT if term in sample]
    dated = bool(YEAR.search(sample))
    reasons = (["historical_title"] if strong_title else []) + (["historical_text"] if strong_body else [])
    reasons += (["person_place_context"] if context_title or context_body else [])
    reasons += (["date_or_period"] if dated else [])
    score = min(10, 4 * bool(strong_title) + 2 * bool(strong_body) +
                2 * bool(context_title) + bool(context_body) + int(dated))
    if strong_title or (strong_body and (context_body or dated)) or (context_title and dated):
        return score, "KEEP", reasons
    if any(term in title for term in CLEARLY_UNRELATED) and not strong_title and not strong_body:
        return score, "DROP", ["clearly_unrelated_title"]
    if not title.strip() or len(body.strip()) < 40:
        return score, "DROP", ["missing_title_or_near_empty"]
    # Ambiguity is retained. This intentionally admits some general geography and biography.
    return score, "REVIEW", reasons or ["ambiguous_broad_recall"]
