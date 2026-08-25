from __future__ import annotations

from app.tools.evidence_tools import SessionEvidenceStore


def test_session_evidence_store_dedupes_and_searches():
    store = SessionEvidenceStore(max_items=4)
    store.add_many(
        [
            {"chunk_id": "a", "title": "Bach Dang", "text": "Ngo Quyen chien thang Bach Dang"},
            {"chunk_id": "a", "title": "Bach Dang", "text": "duplicate"},
            {"chunk_id": "b", "title": "Lam Son", "text": "Le Loi khoi nghia Lam Son"},
        ]
    )
    assert len(store.all()) == 2
    assert store.get("a")["text"] == "duplicate"
    assert store.search("Le Loi", top_k=1)[0]["chunk_id"] == "b"


def test_sessions_are_isolated_and_removable():
    store = SessionEvidenceStore()
    store.add_documents("session-a", [{"chunk_id": "a", "text": "Ngo Quyen"}])
    store.add_documents("session-b", [{"chunk_id": "b", "text": "Le Loi"}])
    assert [row["chunk_id"] for row in store.all("session-a")] == ["a"]
    assert [row["chunk_id"] for row in store.all("session-b")] == ["b"]
    store.remove_session("session-a")
    assert store.all("session-a") == []
