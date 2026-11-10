"""Fail-closed, single-owner authentication for the same-origin application."""
from __future__ import annotations

import asyncio
import hmac
import secrets
import sqlite3
import threading
import time
from urllib.parse import urlsplit
from pathlib import Path

from argon2 import PasswordHasher, extract_parameters
from argon2.exceptions import InvalidHashError, VerificationError
from argon2.low_level import Type
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

COOKIE_NAME = "__Host-su-viet-session"
SESSION_SECONDS = 7 * 24 * 60 * 60
LOGIN_WINDOW_SECONDS = 15 * 60
MAX_FAILURES = 5

LOGIN_HTML = Path(__file__).with_name("login.html").read_text(encoding="utf-8")


class AuthManager:
    def __init__(self, settings):
        password_hash = settings.auth_password_hash.get_secret_value() if settings.auth_password_hash else ""
        secret = settings.auth_session_secret.get_secret_value() if settings.auth_session_secret else ""
        try:
            parameters = extract_parameters(password_hash)
            if parameters.type != Type.ID:
                raise ValueError("Argon2id required")
        except (InvalidHashError, ValueError):
            raise RuntimeError("Authentication requires a valid Argon2id password hash") from None
        if len(secret.encode("utf-8")) < 32:
            raise RuntimeError("Authentication requires a session secret of at least 32 bytes")
        self.password_hash = password_hash
        self.hasher = PasswordHasher()
        self.signer = URLSafeTimedSerializer(secret, salt="su-viet-owner-session")
        self.client_hash_key = secret.encode("utf-8")
        self.db_lock = threading.Lock()
        self.db = sqlite3.connect(str(settings.chat_database_path), timeout=5,
                                  isolation_level=None, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS owner_sessions "
                        "(id TEXT PRIMARY KEY, expires REAL NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS owner_login_failures "
                        "(client TEXT NOT NULL, attempted REAL NOT NULL)")
        self.db.execute("CREATE INDEX IF NOT EXISTS owner_login_failures_client "
                        "ON owner_login_failures (client, attempted)")
        self.login_lock = asyncio.Lock()

    def session_id(self, cookie):
        if not cookie or len(cookie) > 4096:
            return None
        try:
            value = self.signer.loads(cookie, max_age=SESSION_SECONDS)
            if not isinstance(value, dict) or not hmac.compare_digest(str(value.get("role", "")).encode(), b"owner"):
                return None
            session_id = value.get("sid")
            if not isinstance(session_id, str) or len(session_id) > 128:
                return None
            return session_id
        except (BadSignature, SignatureExpired, TypeError, ValueError):
            return None

    def valid(self, cookie):
        session_id = self.session_id(cookie)
        if session_id is None:
            return False
        with self.db_lock:
            return self.db.execute("SELECT 1 FROM owner_sessions WHERE id = ? AND expires > ?",
                                   (session_id, time.time())).fetchone() is not None

    def create_session(self):
        session_id = secrets.token_urlsafe(32)
        with self.db_lock:
            self.db.execute("DELETE FROM owner_sessions WHERE expires <= ?", (time.time(),))
            self.db.execute("INSERT INTO owner_sessions VALUES (?, ?)",
                            (session_id, time.time() + SESSION_SECONDS))
        return self.signer.dumps({"role": "owner", "sid": session_id})

    def revoke(self, cookie):
        session_id = self.session_id(cookie)
        if session_id:
            with self.db_lock:
                self.db.execute("DELETE FROM owner_sessions WHERE id = ?", (session_id,))

    def close(self):
        with self.db_lock:
            self.db.close()

    def blocked(self, client):
        now = time.time()
        with self.db_lock:
            self.db.execute("DELETE FROM owner_login_failures WHERE attempted <= ?", (now - LOGIN_WINDOW_SECONDS,))
            count = self.db.execute("SELECT COUNT(*) FROM owner_login_failures WHERE client = ?", (client,)).fetchone()[0]
            total = self.db.execute("SELECT COUNT(*) FROM owner_login_failures").fetchone()[0]
        return count >= MAX_FAILURES or total >= 4096

    async def authenticate(self, client, password):
        async with self.login_lock:
            client = hmac.digest(self.client_hash_key, client.encode("utf-8"), "sha256").hex()
            if self.blocked(client):
                return "limited"
            try:
                valid = await asyncio.to_thread(self.hasher.verify, self.password_hash, password)
            except (VerificationError, InvalidHashError):
                valid = False
            if not valid:
                with self.db_lock:
                    self.db.execute("INSERT INTO owner_login_failures VALUES (?, ?)", (client, time.time()))
                return "invalid"
            with self.db_lock:
                self.db.execute("DELETE FROM owner_login_failures WHERE client = ?", (client,))
            return "ok"


def same_origin(request, production):
    origin = request.headers.get("origin", "")
    try:
        parsed = urlsplit(origin)
        host = request.headers.get("host", "")
        return (parsed.scheme in ({"https"} if production else {"http", "https"})
                and bool(host) and parsed.netloc.lower() == host.lower()
                and not parsed.username and not parsed.password and not parsed.path
                and not parsed.query and not parsed.fragment
                and request.headers.get("sec-fetch-site", "").lower() != "cross-site")
    except ValueError:
        return False


class OwnerAuthMiddleware:
    def __init__(self, app, settings):
        self.app, self.settings = app, settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if not self.settings.auth_enabled:
            if self.settings.app_env.lower() == "production":
                return await JSONResponse({"detail": "Authentication is not configured"}, 503)(scope, receive, send)
            return await self.app(scope, receive, send)
        request = Request(scope, receive)
        manager = getattr(scope["app"].state, "auth_manager", None)
        if manager is None:
            return await JSONResponse({"detail": "Authentication is not configured"}, 503)(scope, receive, send)
        path, method = request.url.path, request.method
        authenticated = manager.valid(request.cookies.get(COOKIE_NAME))
        if path == "/health" and method == "GET":
            return await self.app(scope, receive, send)
        if path == "/login" and method == "GET":
            if authenticated:
                return await RedirectResponse("/", status_code=303)(scope, receive, send)
            response = HTMLResponse(LOGIN_HTML, headers={
                "Cache-Control": "no-store", "X-Frame-Options": "DENY",
                "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; form-action 'self'; frame-ancestors 'none'",
                "X-Content-Type-Options": "nosniff"})
            return await response(scope, receive, send)
        if path == "/auth/login" and method == "POST":
            if not same_origin(request, self.settings.app_env.lower() == "production"):
                return await JSONResponse({"detail": "Forbidden"}, 403)(scope, receive, send)
            body = b""
            async for chunk in request.stream():
                body += chunk
                if len(body) > 4096:
                    return await JSONResponse({"detail": "Invalid request"}, 413)(scope, receive, send)
            try:
                import json
                data = json.loads(body)
                password = data.get("password") if isinstance(data, dict) else None
                if not isinstance(password, str) or not 1 <= len(password) <= 1024:
                    raise ValueError()
            except (ValueError, UnicodeError):
                return await JSONResponse({"detail": "Invalid request"}, 400)(scope, receive, send)
            client = request.client.host if request.client else "unknown"
            result = await manager.authenticate(client, password)
            if result != "ok":
                return await JSONResponse({"detail": "Unable to sign in"}, 429 if result == "limited" else 401,
                                          headers={"Cache-Control": "no-store"})(scope, receive, send)
            response = JSONResponse({"ok": True}, headers={"Cache-Control": "no-store"})
            response.set_cookie(COOKIE_NAME, manager.create_session(), max_age=SESSION_SECONDS,
                                httponly=True, secure=True, samesite="lax", path="/")
            return await response(scope, receive, send)
        if not authenticated:
            if path.startswith(("/api/", "/auth/")) or path in {"/ready", "/docs", "/redoc", "/openapi.json"}:
                return await JSONResponse({"detail": "Authentication required"}, 401,
                                          headers={"Cache-Control": "no-store"})(scope, receive, send)
            return await RedirectResponse("/login", status_code=303)(scope, receive, send)
        if method not in {"GET", "HEAD", "OPTIONS"} and not same_origin(request, self.settings.app_env.lower() == "production"):
            return await JSONResponse({"detail": "Forbidden"}, 403)(scope, receive, send)
        if path == "/auth/logout" and method == "POST":
            manager.revoke(request.cookies.get(COOKIE_NAME))
            response = RedirectResponse("/login", status_code=303, headers={"Cache-Control": "no-store"})
            response.delete_cookie(COOKIE_NAME, path="/", secure=True, httponly=True, samesite="lax")
            return await response(scope, receive, send)
        scope.setdefault("state", {})["owner_authenticated"] = True
        return await self.app(scope, receive, send)
