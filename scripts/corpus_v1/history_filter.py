"""Broad history relevance heuristic using UVW's local Wikidata category cue."""

import re
import unicodedata


FILTER_VERSION = "broad_history_category_v2"
YEAR = re.compile(r"(?<!\d)[1-9]\d{2,3}(?!\d)|\b(?:thế kỷ|thế kỉ|TCN|SCN)\b", re.I)
HISTORICAL = ("lịch sử", "triều đại", "vương triều", "vương quốc", "đế quốc", "chiến tranh",
              "trận đánh", "cách mạng", "khởi nghĩa", "hiệp ước", "hiệp định", "thuộc địa",
              "khảo cổ", "di sản", "di tích", "văn minh", "cổ đại", "trung đại", "nhà nước",
              "hoàng đế", "vua", "hoàng hậu", "tướng quân", "chiến dịch", "thời kỳ",
              "thời đại", "phong kiến", "thành cổ", "cung điện", "độc lập", "đế chế",
              "lãnh thổ", "thực dân", "phong trào", "sắc lệnh", "cựu quốc gia")
PERSON_PLACE = ("người", "nhân vật", "chính trị gia", "nhà văn", "nhà thơ", "học giả",
                "nhà khoa học", "nhà sử học", "nghệ sĩ", "họa sĩ", "nhà thám hiểm",
                "tổng thống", "thủ tướng", "tướng", "quân nhân", "giáo sĩ", "tu sĩ",
                "thành phố", "đô thị", "khu định cư", "xã của", "xã tại", "tỉnh",
                "quốc gia", "đất nước", "vùng", "khu vực", "địa danh", "đảo", "sông",
                "núi", "công trình", "tòa nhà", "lâu đài", "đền", "chùa", "nhà thờ",
                "di tích", "di sản", "tôn giáo", "văn hóa", "dân tộc", "tổ chức chính trị",
                "tổ chức quân sự", "quân đội", "triều đình", "văn kiện")
NEGATIVE_CATEGORY = ("đơn vị phân loại", "loài thực vật", "loài động vật", "chi thực vật",
                     "tiểu hành tinh", "hợp chất hóa học", "phân tử", "protein", "gene",
                     "phần mềm", "gói phần mềm", "thiết bị điện tử", "điện thoại thông minh",
                     "trò chơi điện tử", "phiên bản trò chơi", "tập phim", "tập truyền hình",
                     "trận đấu bóng đá", "trận bóng đá", "mục nội bộ Wikimedia",
                     "trang định hướng Wikimedia", "taxon", "chemical compound",
                     "asteroid", "video game", "television episode", "software release")
NEGATIVE_TITLE = ("phiên bản phần mềm", "thông số kỹ thuật", "lịch thi đấu", "công thức nấu ăn",
                  "mã nguồn", "hướng dẫn cài đặt", "package documentation", "programming package")


def _fold(value: str | None) -> str:
    return unicodedata.normalize("NFC", value or "").casefold()


def _hits(value: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", value)]


def classify(title: str, text: str, main_category: str | None = None) -> tuple[int, str, list[str]]:
    title, body, category = _fold(title), _fold(text)[:20000], _fold(main_category)
    title_history = _hits(title, HISTORICAL)
    body_history = _hits(body, HISTORICAL)
    category_history = _hits(category, HISTORICAL)
    category_broad = _hits(category, PERSON_PLACE)
    title_broad = _hits(title, PERSON_PLACE)
    negative_category = _hits(category, NEGATIVE_CATEGORY)
    negative_title = _hits(title, NEGATIVE_TITLE)
    dated = bool(YEAR.search(body))
    reasons = []
    if title_history:
        reasons.append("historical_title")
    if body_history:
        reasons.append("historical_text")
    if category_history:
        reasons.append("historical_category")
    if category_broad:
        reasons.append("person_place_heritage_category")
    if title_broad:
        reasons.append("person_place_heritage_title")
    if dated:
        reasons.append("date_or_period")
    score = min(10, 4 * bool(title_history) + 3 * bool(category_history) +
                2 * bool(body_history) + 2 * bool(category_broad) +
                bool(title_broad) + int(dated))
    # Negative category must win over incidental years or infobox place names.
    # Strong, article-specific history cues can still rescue a category mismatch.
    if negative_category or negative_title:
        if title_history or category_history or len(body_history) >= 2:
            return score, "REVIEW", reasons + ["negative_signal_with_history_exception"]
        return score, "DROP", reasons + ["unrelated_category" if negative_category else "unrelated_title"]
    if not title.strip() or len(body.strip()) < 40:
        return score, "DROP", reasons + ["missing_title_or_near_empty"]
    if title_history or category_history or (body_history and (category_broad or title_broad or dated)):
        return score, "KEEP", reasons
    # People, places, heritage, religions and organizations stay for human review.
    if category_broad or title_broad or body_history:
        return score, "REVIEW", reasons
    # Missing or unknown categories are not alone a reason for exclusion.
    return score, "REVIEW", reasons + ["unknown_or_ambiguous_category"]
