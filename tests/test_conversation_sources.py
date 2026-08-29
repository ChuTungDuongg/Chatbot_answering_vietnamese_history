import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI

from app.api.conversations import router as conversations_router
from app.chat.store import ConversationStore
from app.schemas import MessageItem


OWNER_ID = "sources-test-client"


def _api_app(store: ConversationStore) -> FastAPI:
    app = FastAPI()
    app.state.chat_store = store
    app.include_router(conversations_router)
    return app


def _get_conversation(
    store: ConversationStore,
    conversation_id: str,
) -> tuple[int, dict[str, object]]:
    app = _api_app(store)
    sent: list[dict[str, object]] = []
    request_sent = False
    path = f"/api/v1/conversations/{conversation_id}"

    async def receive() -> dict[str, object]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {
                "type": "http.request",
                "body": b"",
                "more_body": False,
            }
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"x-client-id", OWNER_ID.encode("ascii"))],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    asyncio.run(app(scope, receive, send))

    status_code = next(
        int(message["status"])
        for message in sent
        if message["type"] == "http.response.start"
    )
    body = b"".join(
        bytes(message.get("body", b""))
        for message in sent
        if message["type"] == "http.response.body"
    )
    return status_code, json.loads(body)


def _message_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": uuid4(),
        "conversation_id": uuid4(),
        "role": "assistant",
        "content": "Câu trả lời",
        "created_at": datetime.now(timezone.utc),
    }
    payload.update(overrides)
    return payload


def test_message_response_normalizes_null_and_missing_sources() -> None:
    assert MessageItem.model_validate(_message_payload(sources=None)).sources == []
    assert MessageItem.model_validate(_message_payload()).sources == []


def test_new_messages_store_empty_sources_and_detail_returns_200(tmp_path) -> None:
    store = ConversationStore(tmp_path / "chat.db")
    conversation = store.create_conversation(OWNER_ID)

    store.add_message(
        owner_id=OWNER_ID,
        conversation_id=conversation["id"],
        role="user",
        content="Một câu hỏi không có nguồn",
    )
    store.add_message(
        owner_id=OWNER_ID,
        conversation_id=conversation["id"],
        role="assistant",
        content="Một câu trả lời không có nguồn",
        sources=None,
    )

    with store.connection() as connection:
        stored_values = [
            row["sources_json"]
            for row in connection.execute(
                "SELECT sources_json FROM messages ORDER BY created_at ASC"
            ).fetchall()
        ]
    assert stored_values == ["[]", "[]"]

    status_code, body = _get_conversation(store, conversation["id"])

    assert status_code == 200
    assert [message["sources"] for message in body["messages"]] == [[], []]


def test_legacy_null_sources_detail_returns_200(tmp_path) -> None:
    database_path = tmp_path / "legacy.db"
    conversation_id = str(uuid4())
    message_id = str(uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE conversations (
                id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources_json TEXT,
                status TEXT NOT NULL DEFAULT 'done',
                created_at TEXT NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT INTO conversations VALUES (?, ?, ?, ?, ?)",
            (conversation_id, OWNER_ID, "Hội thoại cũ", timestamp, timestamp),
        )
        connection.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                message_id,
                conversation_id,
                "assistant",
                "Tin nhắn cũ",
                None,
                "done",
                timestamp,
            ),
        )

    store = ConversationStore(database_path)
    status_code, body = _get_conversation(store, conversation_id)

    assert status_code == 200
    assert body["messages"][0]["sources"] == []


def test_non_empty_sources_are_preserved_in_detail(tmp_path) -> None:
    store = ConversationStore(tmp_path / "chat.db")
    conversation = store.create_conversation(OWNER_ID)
    source = {
        "chunk_id": "history-42",
        "title": "Đại Việt sử ký toàn thư",
        "source_kind": "history",
        "attachment_id": None,
        "page_number": 42,
    }
    store.add_message(
        owner_id=OWNER_ID,
        conversation_id=conversation["id"],
        role="assistant",
        content="Câu trả lời có dẫn nguồn",
        sources=[source],
    )

    with store.connection() as connection:
        stored = connection.execute(
            "SELECT sources_json FROM messages"
        ).fetchone()["sources_json"]
    assert json.loads(stored) == [source]

    status_code, body = _get_conversation(store, conversation["id"])

    assert status_code == 200
    assert body["messages"][0]["sources"] == [source]
