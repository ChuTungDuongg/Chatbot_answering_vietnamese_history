"""Deterministic broad-history filter informed by UVW's local category metadata."""

from __future__ import annotations

import re
import unicodedata


FILTER_VERSION = "broad_history_category_v3"
YEAR = re.compile(r"(?<!\d)[1-9]\d{2,3}(?!\d)|\b(?:thế kỷ|thế kỉ|TCN|SCN)\b", re.I)

# A title/category match is stronger than the same word buried in article text.
# Body evidence requires distinct cues; generic dates and periods never promote alone.
HISTORICAL_TERMS = (
    "lịch sử", "triều đại", "vương triều", "vương quốc", "đế quốc", "đế chế",
    "chiến tranh", "trận đánh", "chiến dịch", "cách mạng", "khởi nghĩa", "hiệp ước", "hiệp định",
    "thuộc địa", "thực dân", "khảo cổ", "di sản", "di tích", "văn minh",
    "cổ đại", "trung đại", "hoàng đế", "quốc vương", "vua", "hoàng hậu",
    "tướng quân", "võ tướng", "phong kiến", "thành cổ", "cung điện",
    "cựu quốc gia", "thủ đô cũ", "lăng mộ", "hầm mộ", "historical",
    "history", "dynasty", "kingdom", "empire", "war", "battle", "campaign", "revolution",
    "treaty", "colonial", "archaeology", "archaeological", "heritage",
    "monument", "ancient", "medieval", "former state", "emperor", "catacomb",
)
HISTORICAL = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(term) for term in sorted(HISTORICAL_TERMS, key=len, reverse=True))
    + r")(?!\w)"
)

# Category keyword families, rather than exact category names, cover Vietnamese
# and English labels. The patterns intentionally distinguish major places from
# generic municipality/commune categories and biographies from sports/fiction.
CATEGORY_FAMILIES = {
    "wikimedia_maintenance": re.compile(
        r"trang định hướng|trang bảo trì|trang điều hướng|mục nội bộ wikimedia"
        r"|thể loại wikimedia"
        r"|wikimedia.*(?:disambiguation|category|maintenance|navigation)"
        r"|(?:disambiguation page|wikipedia category)"
    ),
    "wikimedia_list": re.compile(r"bài viết danh sách wikimedia|wikimedia list|list article"),
    "taxonomy": re.compile(
        r"đơn vị phân loại|loài (?:thực vật|động vật)|chi (?:thực vật|động vật)"
        r"|\btaxon(?:omy)?\b|biological species|plant species|animal species|genus of"
    ),
    "technical": re.compile(
        r"cấu trúc dữ liệu|thuật toán|đối tượng toán học|khái niệm toán học|khái niệm vật lý"
        r"|tầng khí quyển|tầng bình lưu|hợp chất hóa học|phân tử|\bprotein\b|\bgene\b"
        r"|data structure|algorithm|mathematical (?:object|concept)|atmospheric layer"
        r"|chemical (?:compound|entity)|\basteroid\b|tiểu hành tinh"
        r"|phương tiện giao thông|rail transport|transportation vehicle"
    ),
    "software_product": re.compile(
        r"phần mềm|gói lập trình|thiết bị điện tử|điện thoại thông minh|mẫu điện thoại"
        r"|trò chơi điện tử|phiên bản trò chơi|software|programming package|smartphone"
        r"|consumer electronic|product model|video game|game version"
    ),
    "sports": re.compile(
        r"cầu thủ|vận động viên|bóng đá|trận đấu|giải thể thao|sports? (?:player|match|fixture)"
        r"|football(?:er| player| match| association)?|athlete|soccer"
    ),
    "entertainment_fiction": re.compile(
        r"nhân vật hư cấu|đồ vật hư cấu|đồ vật phép thuật|truyện giả tưởng|tập phim"
        r"|tập truyền hình|người nổi tiếng|diễn viên truyền hình|fictional|fantasy"
        r"|television episode|celebrity|entertainment personality"
    ),
    "generic_municipality": re.compile(
        r"xã của|xã tại|đô thị của|khu định cư|municipalit(?:y|ies)|commune|settlement"
        r"|village|ortsteil|gemeinde"
    ),
    "major_place": re.compile(
        r"thành phố|thủ đô|quốc gia|quốc đảo|tỉnh|vùng lịch sử|bang của|cựu quốc gia"
        r"|\bcity\b|\bcountry\b|\bcapital\b|\bprovince\b|historical region"
        r"|former state|nation state"
    ),
    "other_geography": re.compile(r"địa danh|đảo|sông|núi|vùng địa lý|\briver\b|\bisland\b|mountain|geographic region"),
    "person": re.compile(r"\bngười\b|nhân vật|biograph|\bperson\b|\bpeople\b"),
    "historical_person_role": re.compile(
        r"chính trị gia|quân nhân|võ tướng|tướng lĩnh|lãnh đạo|nhà cách mạng|học giả"
        r"|nhà văn|nhà thơ|nhà sử học|nhà thám hiểm|họa sĩ|nghệ sĩ|giáo sĩ|tu sĩ"
        r"|samurai|politician|military personnel|general|revolutionary|scholar"
        r"|writer|artist|explorer|religious figure|historian"
    ),
    "heritage": re.compile(
        r"di sản|di tích|khảo cổ|lăng mộ|hầm mộ|đền|chùa|nhà thờ|cung điện|lâu đài"
        r"|heritage|monument|archaeolog|catacomb|temple|palace|castle|historic site"
    ),
    "politics_military": re.compile(
        r"tổ chức chính trị|tổ chức quân sự|quân đội|chính phủ|đảng chính trị"
        r"|political organization|military organization|government|political party"
    ),
    "culture_religion": re.compile(
        r"tôn giáo|văn hóa|dân tộc|religion|culture|ethnic group"
    ),
}
NEGATIVE_TITLE = re.compile(
    r"phiên bản phần mềm|thông số kỹ thuật|lịch thi đấu|công thức nấu ăn|mã nguồn"
    r"|hướng dẫn cài đặt|package documentation|programming package"
    r"|bóng đá|cầu thủ|vận động viên|football|soccer|athlete"
)
FICTION_TITLE = re.compile(r"trong truyện|nhân vật hư cấu|đồ vật phép thuật|fictional|fantasy (?:object|character)")
META_TITLE = re.compile(r"^(?:thể loại|category|wikipedia|bản mẫu|template):")


def _fold(value: str | None, *, compact: bool = False) -> str:
    folded = unicodedata.normalize("NFC", value or "").casefold()
    return " ".join(folded.split()) if compact else folded


def _historical_hits(value: str) -> set[str]:
    return {match.group() for match in HISTORICAL.finditer(value)}


def _families(category: str) -> set[str]:
    return {name for name, pattern in CATEGORY_FAMILIES.items() if pattern.search(category)}


def classify(title: str, text: str, main_category: str | None = None) -> tuple[int, str, list[str]]:
    title = _fold(title, compact=True)
    body = _fold(text)[:20000]
    category = _fold(main_category, compact=True)
    families = _families(category)
    title_history = _historical_hits(title)
    category_history = _historical_hits(category)
    body_history = _historical_hits(body)
    body_rich = len(body_history) >= 2
    dated = bool(YEAR.search(body))
    strong = bool(title_history or category_history or body_rich)
    person = bool(families & {"person", "historical_person_role"})
    major_place = "major_place" in families
    generic_place = "generic_municipality" in families
    contextual = bool(families & {"person", "historical_person_role", "major_place",
                                  "generic_municipality", "other_geography", "heritage",
                                  "politics_military", "culture_religion"})
    title_context = bool(re.search(
        r"(?<!\w)(?:thành phố|quốc gia|thủ đô|tỉnh|vùng|đền|chùa|nhà thờ|sông|đảo|"
        r"chính trị gia|nhà văn|học giả|nhà thám hiểm|city|country|capital|temple|scholar)(?!\w)", title))
    reasons: list[str] = []
    if title_history:
        reasons.append("historical_title")
    if category_history:
        reasons.append("historical_category")
    if body_rich:
        reasons.append("historical_text")
    elif body_history:
        reasons.append("isolated_historical_text")
    if dated:
        reasons.append("date_or_period")
    if dated and not (strong or contextual or title_context):
        reasons.append("date_or_period_only")
    if body_rich and not (title_history or category_history or contextual or title_context):
        reasons.append("historical_text_only")
    for family in ("historical_person_role", "person", "major_place", "generic_municipality",
                   "other_geography", "heritage", "politics_military", "culture_religion"):
        if family in families:
            reasons.append(f"{family}_category")

    # Dates are diagnostic. They add only a small supporting point when an
    # independent semantic signal exists and never change a decision alone.
    score = min(10, 4 * bool(title_history) + 4 * bool(category_history) +
                3 * body_rich + int(bool(body_history) and not body_rich) +
                2 * bool(families & {"historical_person_role", "heritage", "politics_military"}) +
                int(person or major_place or title_context) + int(dated and (strong or contextual or title_context)))

    if not title or len(body.strip()) < 40:
        return score, "DROP", reasons + ["missing_title_or_near_empty"]
    if META_TITLE.search(title) or "wikimedia_maintenance" in families:
        return score, "DROP", reasons + ["wikimedia_maintenance"]
    if "wikimedia_list" in families:
        if title_history or category_history or body_rich:
            return score, "REVIEW", reasons + ["historical_wikimedia_list"]
        return score, "DROP", reasons + ["generic_wikimedia_list"]
    if FICTION_TITLE.search(title) or "entertainment_fiction" in families:
        if title_history or category_history:
            return score, "REVIEW", reasons + ["fictional_topic_with_history_context"]
        return score, "DROP", reasons + ["fictional_or_entertainment_topic"]
    negative_family = families & {"taxonomy", "technical", "software_product", "sports"}
    if negative_family or NEGATIVE_TITLE.search(title):
        if title_history or category_history or (body_rich and len(body_history) >= 3):
            return score, "REVIEW", reasons + ["unrelated_category_with_history_exception"]
        return score, "DROP", reasons + ["unrelated_category" if negative_family else "unrelated_title"]

    if generic_place and not (title_history or category_history or body_rich):
        if len(body.strip()) < 900 or re.search(r"dân số|population|độ cao|elevation", body):
            return score, "DROP", reasons + ["generic_municipality_stub"]
        return score, "REVIEW", reasons + ["substantive_generic_place", "weak_signal_review"]

    if generic_place and body_rich and len(body_history) == 2 and not (title_history or category_history):
        return score, "REVIEW", reasons + ["historical_generic_place"]

    if title_history or category_history:
        return score, "KEEP", reasons
    if body_rich and (contextual or title_context or len(body_history) >= 3):
        return score, "KEEP", reasons
    if body_rich:
        return score, "REVIEW", reasons
    if body_history and (contextual or title_context):
        return score, "REVIEW", reasons + ["weak_signal_review"]
    if person or major_place or "heritage" in families or "politics_military" in families:
        return score, "REVIEW", reasons + ["weak_signal_review"]
    if title_context or families & {"other_geography", "culture_religion"}:
        if len(body.strip()) >= 200:
            return score, "REVIEW", reasons + ["weak_signal_review"]
    return score, "DROP", reasons + ["insufficient_historical_evidence"]
