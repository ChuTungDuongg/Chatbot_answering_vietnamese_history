import secrets
from types import SimpleNamespace

import pytest
from argon2 import PasswordHasher
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from itsdangerous.timed import TimestampSigner
from pydantic import SecretStr

from app.security.auth import AuthManager, COOKIE_NAME, OwnerAuthMiddleware


@pytest.fixture
def auth_app(tmp_path):
    password = secrets.token_urlsafe(24)
    secret = secrets.token_urlsafe(48)
    password_hash = PasswordHasher(time_cost=1, memory_cost=8192, parallelism=1).hash(password)
    settings = SimpleNamespace(auth_enabled=True, app_env="production",
        auth_password_hash=SecretStr(password_hash), auth_session_secret=SecretStr(secret),
        chat_database_path=tmp_path / "auth.sqlite3")
    app = FastAPI()
    app.state.auth_manager = AuthManager(settings)
    app.add_middleware(OwnerAuthMiddleware, settings=settings)

    @app.get("/")
    async def index():
        return {"private": True}

    @app.get("/ready")
    async def ready():
        return {"ready": True}

    @app.get("/api/v1/conversations")
    async def conversations():
        return {"private": True}

    @app.post("/api/v1/chat/stream")
    async def stream():
        return StreamingResponse(iter(["event: answer_delta\ndata: {}\n\n"]), media_type="text/event-stream")

    @app.post("/api/v1/conversations")
    async def create():
        return {"created": True}

    with TestClient(app, base_url="https://testserver") as client:
        yield client, app.state.auth_manager, settings, password
    app.state.auth_manager.close()


def login(client, password):
    return client.post("/auth/login", json={"password": password},
                       headers={"Origin": "https://testserver"}, follow_redirects=False)


def test_fails_closed_for_missing_or_invalid_configuration(tmp_path):
    settings = SimpleNamespace(auth_password_hash=None, auth_session_secret=None,
                               chat_database_path=tmp_path / "db.sqlite")
    with pytest.raises(RuntimeError):
        AuthManager(settings)
    settings.auth_password_hash = SecretStr(PasswordHasher().hash(secrets.token_urlsafe(24)))
    settings.auth_session_secret = SecretStr("short")
    with pytest.raises(RuntimeError):
        AuthManager(settings)
    app = FastAPI()
    app.add_middleware(OwnerAuthMiddleware, settings=SimpleNamespace(auth_enabled=False, app_env="production"))
    with TestClient(app) as client:
        assert client.get("/api/v1/chat/stream").status_code == 503


def test_login_cookie_refresh_logout_and_revocation(auth_app):
    client, manager, _, password = auth_app
    assert client.get("/", follow_redirects=False).headers["location"] == "/login"
    page = client.get("/login")
    assert page.status_code == 200 and "Mật khẩu" in page.text
    response = login(client, password)
    assert response.status_code == 200
    cookie = client.cookies.get(COOKIE_NAME)
    assert cookie and password not in response.text
    header = response.headers["set-cookie"].lower()
    assert "httponly" in header and "secure" in header and "samesite=lax" in header and "path=/" in header
    assert client.get("/").json() == {"private": True}
    assert client.get("/ready").status_code == 200
    logout = client.post("/auth/logout", headers={"Origin": "https://testserver"}, follow_redirects=False)
    assert logout.status_code == 303 and logout.headers["location"] == "/login"
    assert not manager.valid(cookie)
    client.cookies.set(COOKIE_NAME, cookie, domain="testserver.local", path="/")
    assert client.get("/api/v1/conversations").status_code == 401


def test_direct_api_calls_and_fake_client_identity_cannot_bypass_auth(auth_app):
    client, _, _, _ = auth_app
    for method, path in [("get", "/ready"), ("get", "/docs"),
                         ("get", "/openapi.json"), ("get", "/api/v1/conversations"),
                         ("post", "/api/v1/chat/stream")]:
        response = getattr(client, method)(path, headers={"X-Client-ID": "attacker-identity"})
        assert response.status_code == 401
        assert "text/event-stream" not in response.headers.get("content-type", "")


def test_wrong_password_and_rate_limit(auth_app):
    client, _, settings, password = auth_app
    for _ in range(5):
        response = login(client, secrets.token_urlsafe(24))
        assert response.status_code == 401 and response.json()["detail"] == "Unable to sign in"
    assert login(client, secrets.token_urlsafe(24)).status_code == 429
    # A fresh manager/container must not reset the limit.
    restarted = AuthManager(settings)
    try:
        import asyncio
        assert asyncio.run(restarted.authenticate("testclient", password)) == "limited"
    finally:
        restarted.close()


def test_tampered_and_expired_cookies_are_rejected(auth_app, monkeypatch):
    client, manager, _, password = auth_app
    assert login(client, password).status_code == 200
    cookie = client.cookies.get(COOKIE_NAME)
    client.cookies.clear()
    client.cookies.set(COOKIE_NAME, cookie + "x")
    assert client.get("/ready").status_code == 401
    with monkeypatch.context() as patch:
        patch.setattr(TimestampSigner, "get_timestamp", lambda self: 0)
        expired = manager.create_session()
    client.cookies.clear()
    client.cookies.set(COOKIE_NAME, expired)
    assert client.get("/ready").status_code == 401


def test_csrf_rejection_and_authenticated_stream(auth_app):
    client, _, _, password = auth_app
    assert client.post("/auth/login", json={"password": password},
                       headers={"Origin": "https://evil.example"}).status_code == 403
    assert login(client, password).status_code == 200
    assert client.post("/api/v1/conversations").status_code == 403
    assert client.post("/api/v1/conversations", headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.post("/auth/logout", headers={"Origin": "https://evil.example"}).status_code == 403
    response = client.post("/api/v1/chat/stream", headers={"Origin": "https://testserver"})
    assert response.status_code == 200 and "event: answer_delta" in response.text


def test_session_survives_manager_restart(auth_app):
    client, manager, settings, password = auth_app
    assert login(client, password).status_code == 200
    cookie = client.cookies.get(COOKIE_NAME)
    other = AuthManager(settings)
    try:
        assert other.valid(cookie)
        other.revoke(cookie)
        assert not manager.valid(cookie)
    finally:
        other.close()


def test_spa_fallback_never_returns_html_for_unknown_api(monkeypatch, tmp_path):
    from app import main
    from fastapi import HTTPException
    import asyncio
    (tmp_path / "index.html").write_text("<html>spa</html>", encoding="utf-8")
    monkeypatch.setattr(main, "frontend_dist", tmp_path)
    assert asyncio.run(main.spa_fallback("some/client/route")).path == tmp_path / "index.html"
    with pytest.raises(HTTPException) as exc:
        asyncio.run(main.spa_fallback("api/v1/not-found"))
    assert exc.value.status_code == 404
