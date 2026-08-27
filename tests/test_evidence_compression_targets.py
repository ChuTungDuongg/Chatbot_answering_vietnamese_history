from __future__ import annotations

from training.evidence_agent.prepare_dataset import GENERIC_SUMMARIES, grounded_compression


def test_compression_is_nonempty_shorter_and_extractively_grounded():
    relevant = "Chiến dịch Điện Biên Phủ kết thúc thắng lợi ngày 7/5/1954 và Pháp chịu thất bại quân sự lớn."
    source = (
        "Nhà Trần tồn tại trong một giai đoạn lịch sử khác và đoạn này không trả lời câu hỏi. "
        + relevant
        + " Một chi tiết dài khác nói về địa lý chung và không cần thiết cho câu trả lời đang xét."
    )
    claims, compressed = grounded_compression(
        "Điện Biên Phủ kết thúc khi nào và tác động quân sự ra sao?",
        "Kết thúc ngày 7/5/1954 và gây thất bại quân sự lớn cho Pháp.",
        source,
        max_chars=140,
    )

    assert claims
    assert compressed
    assert len(compressed) < len(source)
    assert all(claim in source for claim in claims)
    assert compressed == " ".join(claims)
    assert compressed not in GENERIC_SUMMARIES

