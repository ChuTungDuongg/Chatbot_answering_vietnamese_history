import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import UUID, uuid4

import numpy as np


DEFAULT_CONVERSATION_TITLE = "Cuộc trò chuyện mới"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_id(value: str | UUID) -> str:
    return str(value)


def serialize_sources(
    sources: list[dict[str, Any]] | None,
) -> str:
    return json.dumps(
        sources or [],
        ensure_ascii=False,
        default=str,
    )


def deserialize_sources(value: str | None) -> list[dict[str, Any]]:
    if not value:
        return []

    try:
        result = json.loads(value)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


def clean_title(value: str | None) -> str:
    title = " ".join((value or "").split()).strip()

    if not title:
        return DEFAULT_CONVERSATION_TITLE

    return title[:120]


def title_from_question(question: str) -> str:
    title = " ".join(question.split()).strip()

    if len(title) <= 60:
        return title

    return title[:57].rstrip() + "..."


class ConversationStore:
    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._write_lock = threading.RLock()
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=30,
            check_same_thread=False,
        )

        connection.row_factory = sqlite3.Row

        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA busy_timeout = 30000")

        return connection

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()

        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._write_lock:
            with self.connection() as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS conversations (
                        id TEXT PRIMARY KEY,
                        owner_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS
                    idx_conversations_owner_updated
                    ON conversations(owner_id, updated_at DESC);

                    CREATE TABLE IF NOT EXISTS messages (
                        id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL,
                        role TEXT NOT NULL
                            CHECK(role IN ('user', 'assistant')),
                        content TEXT NOT NULL,
                        sources_json TEXT NOT NULL DEFAULT '[]',
                        status TEXT NOT NULL DEFAULT 'done',
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(conversation_id)
                            REFERENCES conversations(id)
                            ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS
                    idx_messages_conversation_created
                    ON messages(conversation_id, created_at);

                    CREATE TABLE IF NOT EXISTS attachments (
                        id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        mime_type TEXT NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        status TEXT NOT NULL
                            CHECK(
                                status IN (
                                    'processing',
                                    'ready',
                                    'failed'
                                )
                            ),
                        chunk_count INTEGER NOT NULL DEFAULT 0,
                        error TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(conversation_id)
                            REFERENCES conversations(id)
                            ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS
                    idx_attachments_conversation
                    ON attachments(conversation_id, created_at);

                    CREATE TABLE IF NOT EXISTS temporary_chunks (
                        id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL,
                        attachment_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        text TEXT NOT NULL,
                        page_number INTEGER,
                        embedding BLOB NOT NULL,
                        embedding_dim INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(conversation_id)
                            REFERENCES conversations(id)
                            ON DELETE CASCADE,
                        FOREIGN KEY(attachment_id)
                            REFERENCES attachments(id)
                            ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS
                    idx_temporary_chunks_conversation
                    ON temporary_chunks(conversation_id);

                    CREATE INDEX IF NOT EXISTS
                    idx_temporary_chunks_attachment
                    ON temporary_chunks(attachment_id);
                    """
                )

                connection.commit()

    @staticmethod
    def _row_to_dict(
        row: sqlite3.Row | None,
    ) -> dict[str, Any] | None:
        if row is None:
            return None

        return dict(row)

    def conversation_exists(
        self,
        owner_id: str,
        conversation_id: str | UUID,
    ) -> bool:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM conversations
                WHERE id = ? AND owner_id = ?
                """,
                (
                    normalize_id(conversation_id),
                    owner_id,
                ),
            ).fetchone()

        return row is not None

    def create_conversation(
        self,
        owner_id: str,
        title: str | None = None,
    ) -> dict[str, Any]:
        conversation_id = str(uuid4())
        timestamp = utc_now()

        with self._write_lock:
            with self.connection() as connection:
                connection.execute(
                    """
                    INSERT INTO conversations (
                        id,
                        owner_id,
                        title,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        conversation_id,
                        owner_id,
                        clean_title(title),
                        timestamp,
                        timestamp,
                    ),
                )

                connection.commit()

        result = self.get_conversation(
            owner_id,
            conversation_id,
        )

        if result is None:
            raise RuntimeError(
                "Conversation was created but could not be loaded."
            )

        return result

    def list_conversations(
        self,
        owner_id: str,
    ) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    c.id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    (
                        SELECT COUNT(*)
                        FROM messages m
                        WHERE m.conversation_id = c.id
                    ) AS message_count,
                    (
                        SELECT COUNT(*)
                        FROM attachments a
                        WHERE a.conversation_id = c.id
                    ) AS attachment_count
                FROM conversations c
                WHERE c.owner_id = ?
                ORDER BY c.updated_at DESC
                """,
                (owner_id,),
            ).fetchall()

        return [dict(row) for row in rows]

    def get_conversation(
        self,
        owner_id: str,
        conversation_id: str | UUID,
    ) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT
                    c.id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    (
                        SELECT COUNT(*)
                        FROM messages m
                        WHERE m.conversation_id = c.id
                    ) AS message_count,
                    (
                        SELECT COUNT(*)
                        FROM attachments a
                        WHERE a.conversation_id = c.id
                    ) AS attachment_count
                FROM conversations c
                WHERE c.id = ? AND c.owner_id = ?
                """,
                (
                    normalize_id(conversation_id),
                    owner_id,
                ),
            ).fetchone()

        return self._row_to_dict(row)

    def update_conversation_title(
        self,
        owner_id: str,
        conversation_id: str | UUID,
        title: str,
    ) -> dict[str, Any] | None:
        conversation_id = normalize_id(conversation_id)

        with self._write_lock:
            with self.connection() as connection:
                cursor = connection.execute(
                    """
                    UPDATE conversations
                    SET title = ?, updated_at = ?
                    WHERE id = ? AND owner_id = ?
                    """,
                    (
                        clean_title(title),
                        utc_now(),
                        conversation_id,
                        owner_id,
                    ),
                )

                connection.commit()

        if cursor.rowcount == 0:
            return None

        return self.get_conversation(
            owner_id,
            conversation_id,
        )

    def delete_conversation(
        self,
        owner_id: str,
        conversation_id: str | UUID,
    ) -> bool:
        with self._write_lock:
            with self.connection() as connection:
                cursor = connection.execute(
                    """
                    DELETE FROM conversations
                    WHERE id = ? AND owner_id = ?
                    """,
                    (
                        normalize_id(conversation_id),
                        owner_id,
                    ),
                )

                connection.commit()

        return cursor.rowcount > 0

    def add_message(
        self,
        owner_id: str,
        conversation_id: str | UUID,
        role: str,
        content: str,
        sources: list[dict[str, Any]] | None = None,
        status: str = "done",
    ) -> dict[str, Any]:
        if role not in {"user", "assistant"}:
            raise ValueError(
                "Message role must be 'user' or 'assistant'."
            )

        content = content.strip()

        if not content:
            raise ValueError(
                "Message content must not be empty."
            )

        conversation_id = normalize_id(conversation_id)
        message_id = str(uuid4())
        timestamp = utc_now()

        with self._write_lock:
            with self.connection() as connection:
                conversation = connection.execute(
                    """
                    SELECT title
                    FROM conversations
                    WHERE id = ? AND owner_id = ?
                    """,
                    (
                        conversation_id,
                        owner_id,
                    ),
                ).fetchone()

                if conversation is None:
                    raise LookupError(
                        "Conversation not found."
                    )

                connection.execute(
                    """
                    INSERT INTO messages (
                        id,
                        conversation_id,
                        role,
                        content,
                        sources_json,
                        status,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message_id,
                        conversation_id,
                        role,
                        content,
                        serialize_sources(sources),
                        status,
                        timestamp,
                    ),
                )

                next_title = conversation["title"]

                if (
                    role == "user"
                    and next_title
                    == DEFAULT_CONVERSATION_TITLE
                ):
                    next_title = title_from_question(content)

                connection.execute(
                    """
                    UPDATE conversations
                    SET title = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        next_title,
                        timestamp,
                        conversation_id,
                    ),
                )

                connection.commit()

        message = self.get_message(
            owner_id,
            conversation_id,
            message_id,
        )

        if message is None:
            raise RuntimeError(
                "Message was created but could not be loaded."
            )

        return message

    def get_message(
        self,
        owner_id: str,
        conversation_id: str | UUID,
        message_id: str | UUID,
    ) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT
                    m.id,
                    m.conversation_id,
                    m.role,
                    m.content,
                    m.sources_json,
                    m.status,
                    m.created_at
                FROM messages m
                JOIN conversations c
                    ON c.id = m.conversation_id
                WHERE
                    m.id = ?
                    AND m.conversation_id = ?
                    AND c.owner_id = ?
                """,
                (
                    normalize_id(message_id),
                    normalize_id(conversation_id),
                    owner_id,
                ),
            ).fetchone()

        if row is None:
            return None

        result = dict(row)
        result["sources"] = deserialize_sources(
            result.pop("sources_json", None)
        )

        return result

    def list_messages(
        self,
        owner_id: str,
        conversation_id: str | UUID,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        conversation_id = normalize_id(conversation_id)

        with self.connection() as connection:
            if limit is None:
                rows = connection.execute(
                    """
                    SELECT
                        m.id,
                        m.conversation_id,
                        m.role,
                        m.content,
                        m.sources_json,
                        m.status,
                        m.created_at
                    FROM messages m
                    JOIN conversations c
                        ON c.id = m.conversation_id
                    WHERE
                        m.conversation_id = ?
                        AND c.owner_id = ?
                    ORDER BY m.created_at ASC
                    """,
                    (
                        conversation_id,
                        owner_id,
                    ),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT *
                    FROM (
                        SELECT
                            m.id,
                            m.conversation_id,
                            m.role,
                            m.content,
                            m.sources_json,
                            m.status,
                            m.created_at
                        FROM messages m
                        JOIN conversations c
                            ON c.id = m.conversation_id
                        WHERE
                            m.conversation_id = ?
                            AND c.owner_id = ?
                        ORDER BY m.created_at DESC
                        LIMIT ?
                    )
                    ORDER BY created_at ASC
                    """,
                    (
                        conversation_id,
                        owner_id,
                        max(1, limit),
                    ),
                ).fetchall()

        messages: list[dict[str, Any]] = []

        for row in rows:
            message = dict(row)
            message["sources"] = deserialize_sources(
                message.pop("sources_json", None)
            )
            messages.append(message)

        return messages

    def get_recent_history(
        self,
        owner_id: str,
        conversation_id: str | UUID,
        limit: int = 6,
    ) -> list[dict[str, str]]:
        messages = self.list_messages(
            owner_id,
            conversation_id,
            limit=limit,
        )

        return [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in messages
            if message["status"] == "done"
        ]

    def create_attachment(
        self,
        owner_id: str,
        conversation_id: str | UUID,
        filename: str,
        mime_type: str,
        size_bytes: int,
    ) -> dict[str, Any]:
        conversation_id = normalize_id(conversation_id)
        attachment_id = str(uuid4())
        timestamp = utc_now()

        if not self.conversation_exists(
            owner_id,
            conversation_id,
        ):
            raise LookupError("Conversation not found.")

        with self._write_lock:
            with self.connection() as connection:
                connection.execute(
                    """
                    INSERT INTO attachments (
                        id,
                        conversation_id,
                        filename,
                        mime_type,
                        size_bytes,
                        status,
                        chunk_count,
                        error,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, 'processing', 0, NULL, ?)
                    """,
                    (
                        attachment_id,
                        conversation_id,
                        filename,
                        mime_type,
                        size_bytes,
                        timestamp,
                    ),
                )

                connection.execute(
                    """
                    UPDATE conversations
                    SET updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        timestamp,
                        conversation_id,
                    ),
                )

                connection.commit()

        attachment = self.get_attachment(
            owner_id,
            conversation_id,
            attachment_id,
        )

        if attachment is None:
            raise RuntimeError(
                "Attachment was created but could not be loaded."
            )

        return attachment

    def get_attachment(
        self,
        owner_id: str,
        conversation_id: str | UUID,
        attachment_id: str | UUID,
    ) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT
                    a.id,
                    a.conversation_id,
                    a.filename,
                    a.mime_type,
                    a.size_bytes,
                    a.status,
                    a.chunk_count,
                    a.error,
                    a.created_at
                FROM attachments a
                JOIN conversations c
                    ON c.id = a.conversation_id
                WHERE
                    a.id = ?
                    AND a.conversation_id = ?
                    AND c.owner_id = ?
                """,
                (
                    normalize_id(attachment_id),
                    normalize_id(conversation_id),
                    owner_id,
                ),
            ).fetchone()

        return self._row_to_dict(row)

    def list_attachments(
        self,
        owner_id: str,
        conversation_id: str | UUID,
    ) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    a.id,
                    a.conversation_id,
                    a.filename,
                    a.mime_type,
                    a.size_bytes,
                    a.status,
                    a.chunk_count,
                    a.error,
                    a.created_at
                FROM attachments a
                JOIN conversations c
                    ON c.id = a.conversation_id
                WHERE
                    a.conversation_id = ?
                    AND c.owner_id = ?
                ORDER BY a.created_at ASC
                """,
                (
                    normalize_id(conversation_id),
                    owner_id,
                ),
            ).fetchall()

        return [dict(row) for row in rows]

    def update_attachment_status(
        self,
        owner_id: str,
        conversation_id: str | UUID,
        attachment_id: str | UUID,
        status: str,
        chunk_count: int = 0,
        error: str | None = None,
    ) -> dict[str, Any] | None:
        if status not in {
            "processing",
            "ready",
            "failed",
        }:
            raise ValueError(
                "Invalid attachment status."
            )

        with self._write_lock:
            with self.connection() as connection:
                cursor = connection.execute(
                    """
                    UPDATE attachments
                    SET
                        status = ?,
                        chunk_count = ?,
                        error = ?
                    WHERE
                        id = ?
                        AND conversation_id = ?
                        AND conversation_id IN (
                            SELECT id
                            FROM conversations
                            WHERE owner_id = ?
                        )
                    """,
                    (
                        status,
                        max(0, chunk_count),
                        error,
                        normalize_id(attachment_id),
                        normalize_id(conversation_id),
                        owner_id,
                    ),
                )

                connection.commit()

        if cursor.rowcount == 0:
            return None

        return self.get_attachment(
            owner_id,
            conversation_id,
            attachment_id,
        )

    def delete_attachment(
        self,
        owner_id: str,
        conversation_id: str | UUID,
        attachment_id: str | UUID,
    ) -> bool:
        with self._write_lock:
            with self.connection() as connection:
                cursor = connection.execute(
                    """
                    DELETE FROM attachments
                    WHERE
                        id = ?
                        AND conversation_id = ?
                        AND conversation_id IN (
                            SELECT id
                            FROM conversations
                            WHERE owner_id = ?
                        )
                    """,
                    (
                        normalize_id(attachment_id),
                        normalize_id(conversation_id),
                        owner_id,
                    ),
                )

                connection.commit()

        return cursor.rowcount > 0

    def replace_temporary_chunks(
        self,
        owner_id: str,
        conversation_id: str | UUID,
        attachment_id: str | UUID,
        chunks: list[dict[str, Any]],
    ) -> None:
        conversation_id = normalize_id(conversation_id)
        attachment_id = normalize_id(attachment_id)

        if self.get_attachment(
            owner_id,
            conversation_id,
            attachment_id,
        ) is None:
            raise LookupError("Attachment not found.")

        timestamp = utc_now()

        with self._write_lock:
            with self.connection() as connection:
                connection.execute(
                    """
                    DELETE FROM temporary_chunks
                    WHERE attachment_id = ?
                    """,
                    (attachment_id,),
                )

                for chunk in chunks:
                    embedding = np.asarray(
                        chunk["embedding"],
                        dtype=np.float32,
                    ).reshape(-1)

                    if embedding.size == 0:
                        raise ValueError(
                            "Temporary chunk embedding is empty."
                        )

                    chunk_id = str(
                        chunk.get("chunk_id")
                        or (
                            f"temp:{attachment_id}:"
                            f"{chunk.get('page_number') or 0}:"
                            f"{uuid4().hex[:8]}"
                        )
                    )

                    connection.execute(
                        """
                        INSERT INTO temporary_chunks (
                            id,
                            conversation_id,
                            attachment_id,
                            title,
                            text,
                            page_number,
                            embedding,
                            embedding_dim,
                            created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            chunk_id,
                            conversation_id,
                            attachment_id,
                            str(chunk.get("title") or ""),
                            str(chunk.get("text") or ""),
                            chunk.get("page_number"),
                            embedding.tobytes(),
                            int(embedding.size),
                            timestamp,
                        ),
                    )

                connection.commit()

    def list_temporary_chunks(
        self,
        owner_id: str,
        conversation_id: str | UUID,
    ) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    tc.id,
                    tc.conversation_id,
                    tc.attachment_id,
                    tc.title,
                    tc.text,
                    tc.page_number,
                    tc.embedding,
                    tc.embedding_dim
                FROM temporary_chunks tc
                JOIN conversations c
                    ON c.id = tc.conversation_id
                JOIN attachments a
                    ON a.id = tc.attachment_id
                WHERE
                    tc.conversation_id = ?
                    AND c.owner_id = ?
                    AND a.status = 'ready'
                ORDER BY tc.created_at ASC
                """,
                (
                    normalize_id(conversation_id),
                    owner_id,
                ),
            ).fetchall()

        chunks: list[dict[str, Any]] = []

        for row in rows:
            item = dict(row)

            embedding = np.frombuffer(
                item.pop("embedding"),
                dtype=np.float32,
            ).copy()

            expected_dim = item.pop("embedding_dim")

            if embedding.size != expected_dim:
                continue

            item["chunk_id"] = item.pop("id")
            item["source_kind"] = "attachment"
            item["embedding"] = embedding
            chunks.append(item)

        return chunks

    def get_conversation_detail(
        self,
        owner_id: str,
        conversation_id: str | UUID,
    ) -> dict[str, Any] | None:
        conversation = self.get_conversation(
            owner_id,
            conversation_id,
        )

        if conversation is None:
            return None

        return {
            "conversation": conversation,
            "messages": self.list_messages(
                owner_id,
                conversation_id,
            ),
            "attachments": self.list_attachments(
                owner_id,
                conversation_id,
            ),
        }