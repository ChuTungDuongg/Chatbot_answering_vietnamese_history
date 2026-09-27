"""Semantic regressions for the deterministic UVW history filter."""

import pytest

from scripts.corpus_v1.history_filter import FILTER_VERSION, classify


def article(title, category=None, text=None):
    return classify(title, text or ("Bài viết cung cấp thông tin về chủ đề và các nguồn tham khảo. " * 3), category)


@pytest.mark.parametrize("title,category,text", [
    ("Đế quốc Đông La Mã", "cựu quốc gia", None),
    ("Napoleon Bonaparte", "chính trị gia", None),
    ("Shibata Katsuie", "samurai", None),
    ("Hầm mộ Paris", "di sản", None),
    ("Cabo Verde", "quốc gia", None),
    ("Hà Nội", "thành phố", "Hà Nội có di tích lịch sử và từng là kinh đô. " * 3),
    ("Angkor Wat", "archaeological site", None),
    ("Roman Empire", "former state", None),
    ("French Revolution", "political history", None),
    ("World War II", "war", None),
    ("Cung điện cổ", "cung điện", None),
    ("Di chỉ khảo cổ", "archaeological site", None),
    ("Cộng hòa cũ", "former state", None),
    ("Hiệp ước lịch sử", "treaty", None),
    ("Đền cổ", "historical religious site", None),
    ("Ancient civilization", "culture", None),
    ("Học giả Đại Việt", "scholar", None),
    ("Nhà văn", "writer", "Tác phẩm của ông phản ánh đời sống xã hội. " * 3),
    ("Một cuộc nổi dậy", None, "Cuộc cách mạng dẫn đến chiến tranh và thành lập vương quốc. " * 3),
])
def test_historically_useful_topics_not_dropped(title, category, text):
    assert article(title, category, text)[1] in {"KEEP", "REVIEW"}


@pytest.mark.parametrize("title,category,text", [
    ("Cây đỏ đen", "cấu trúc dữ liệu", "Cấu trúc dữ liệu này được mô tả năm 1972 và sửa đổi năm 1978. " * 3),
    ("Tầng bình lưu", "atmospheric layer", "Đây là một tầng khí quyển được nghiên cứu vào năm 1972. " * 3),
    ("Gói phần mềm", "software package", "Phiên bản mới được phát hành năm 2024. " * 4),
    ("Điện thoại mẫu", "smartphone model", "Thiết bị được ra mắt năm 2025 và có màn hình mới. " * 3),
    ("Trận bóng đá", "sports match", "Trận đấu diễn ra năm 2024 với tỷ số hai một. " * 3),
    ("Hiệp hội bóng đá Anh", "tổ chức thể thao", "Tổ chức này được thành lập năm 1863. " * 3),
    ("Tàu hỏa", "phương tiện giao thông", "Phương tiện này được sử dụng từ năm 1825. " * 3),
    ("Loài cây", "đơn vị phân loại", "Loài được mô tả khoa học năm 1972. " * 3),
    ("Hợp chất", "chemical entity", "Hợp chất được tổng hợp năm 1998. " * 3),
    ("X", "trang định hướng Wikimedia", "Trang này liệt kê nhiều mục có cùng tên. " * 3),
    ("Thể loại:X", "thể loại Wikimedia", "Trang này tập hợp các bài cùng chủ đề. " * 3),
    ("X", "xã của Pháp", "X là một xã thuộc tỉnh Y. Dân số là 100 và độ cao 50 m. " * 3),
    ("Đồ vật phép thuật trong truyện Harry Potter", "đồ vật phép thuật", "Vật này xuất hiện trong lịch sử của truyện. " * 3),
    ("Gói A", None, "Phát hành năm 2024 với nhiều chức năng mới. " * 3),
])
def test_clear_unrelated_or_low_value_topics_drop(title, category, text):
    assert article(title, category, text)[1] == "DROP"


def test_dates_are_diagnostic_and_never_rescue_unrelated_subjects():
    plain = article("Một đối tượng", None, "Đối tượng được ghi nhận năm 1972. " * 4)
    assert plain[1] == "DROP"
    assert "date_or_period_only" in plain[2]
    for category in ("cấu trúc dữ liệu", "đơn vị phân loại", "atmospheric layer"):
        assert article("Một đối tượng", category, "Đối tượng được ghi nhận năm 1972. " * 4)[1] == "DROP"


def test_body_history_needs_multiple_distinct_cues():
    one = article("Một chủ đề", None, "Lịch sử được đề cập một lần, sau đó là dữ liệu thường. " * 4)
    rich = article("Một chủ đề", None, "Chiến tranh dẫn đến cách mạng và hiệp ước giữa các nước. " * 3)
    assert one[1] == "DROP"
    assert "historical_text" not in one[2]
    assert rich[1] == "KEEP"
    assert "historical_text_only" in rich[2]


def test_categories_normalize_unicode_case_and_space():
    assert article("Napoleon Bonaparte", "  CHÍNH   TRỊ GIA  ")[1] == "REVIEW"
    assert article("Cây đỏ đen", "  DATA   STRUCTURE  ")[1] == "DROP"
    assert article("X", " WIKIMEDIA   DISAMBIGUATION   PAGE ")[1] == "DROP"


def test_lists_geography_people_and_weak_signals():
    assert article("Danh sách hoàng đế Việt Nam", "bài viết danh sách Wikimedia")[1] == "REVIEW"
    assert article("Danh sách sản phẩm", "bài viết danh sách Wikimedia")[1] == "DROP"
    assert article("Paris", "city")[1] == "REVIEW"
    assert article("Cầu thủ bóng đá hiện đại", "người")[1] == "DROP"
    assert article("Một nhân vật", "người")[1] == "REVIEW"
    assert article("Một xã", "commune", "X là xã cũ và có di tích khảo cổ từ thời cổ đại. " * 4)[1] in {"KEEP", "REVIEW"}
    assert article("Một cuộc chiến", None, "Cuộc chiến tranh dẫn đến cách mạng. " * 4)[1] in {"KEEP", "REVIEW"}


def test_determinism_and_version():
    assert FILTER_VERSION == "broad_history_category_v3"
    inputs = ("Angkor Wat", "Di tích khảo cổ được xây vào thế kỷ 12. " * 3, "ARCHAEOLOGICAL   SITE")
    assert classify(*inputs) == classify(*inputs)
