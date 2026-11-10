"""Durable single-host OpenAI quota and concurrency gate."""
import hmac
import sqlite3
import threading
import time
import uuid


class OpenAIAccessGuard:
    def __init__(self, token, hourly, daily, concurrent, *, database=":memory:", lease_seconds=7200):
        self._token = token
        self._hourly, self._daily, self._concurrent = hourly, daily, concurrent
        self._lease_seconds = lease_seconds
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(database), timeout=5, isolation_level=None, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS openai_requests "
                         "(id TEXT PRIMARY KEY, started REAL NOT NULL, expires REAL NOT NULL, active INTEGER NOT NULL)")

    def authorized(self, authorization):
        supplied = authorization.removeprefix("Bearer ") if authorization.startswith("Bearer ") else ""
        return bool(supplied) and hmac.compare_digest(supplied.encode(), self._token.encode())

    def acquire(self):
        # BEGIN IMMEDIATE serializes quota decisions across processes sharing a
        # SQLite database. Leases bound crash recovery.
        with self._lock:
            now = time.time()
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._db.execute("DELETE FROM openai_requests WHERE started <= ? AND (active = 0 OR expires <= ?)",
                                 (now - 86400, now))
                self._db.execute("UPDATE openai_requests SET active = 0 WHERE active = 1 AND expires <= ?", (now,))
                hourly, daily, active = self._db.execute(
                    "SELECT COALESCE(SUM(started > ?), 0), COUNT(*), COALESCE(SUM(active), 0) FROM openai_requests",
                    (now - 3600,)).fetchone()
                if active >= self._concurrent or daily >= self._daily or hourly >= self._hourly:
                    self._db.execute("COMMIT")
                    return None
                lease = uuid.uuid4().hex
                self._db.execute("INSERT INTO openai_requests VALUES (?, ?, ?, 1)",
                                 (lease, now, now + self._lease_seconds))
                self._db.execute("COMMIT")
                return lease
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    def release(self, lease):
        with self._lock:
            self._db.execute("UPDATE openai_requests SET active = 0 WHERE id = ?", (lease,))

    def close(self):
        with self._lock:
            self._db.close()
