"""Exercise readiness against real HTTP without Modal, GPU or paid API access."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
from types import SimpleNamespace

import pytest

from scripts import modal_session as session


@pytest.fixture
def health_server(monkeypatch):
    state = SimpleNamespace(responses=[(200, b'{"status":"ok"}')], calls=0, headers=[], paths=[], delay=0,
                            byte_delay=0, received=threading.Event())
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state.calls += 1
            state.headers.append(dict(self.headers))
            state.paths.append(self.path)
            state.received.set()
            status, body = state.responses[min(state.calls - 1, len(state.responses) - 1)]
            if self.path != "/health":
                status, body = 200, b'{"status":"ok"}'
            time.sleep(state.delay)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            if status == 302:
                self.send_header("Location", "/login")
            self.end_headers()
            try:
                for byte in body:
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                    time.sleep(state.byte_delay)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setattr(session, "READY_POLL_SECONDS", .01, raising=False)
    monkeypatch.setattr(session, "READY_REQUEST_TIMEOUT", .1, raising=False)
    yield f"http://127.0.0.1:{server.server_port}", state
    server.shutdown()
    server.server_close()
    worker.join()


def test_cold_start_retries_until_real_health_json_is_ready(health_server):
    url, state = health_server
    state.responses = [(503, b"booting"), (200, b"[]"), (200, b'{"status":"ok"}')]
    assert session.wait_for_ready(url, threading.Event(), timeout=2)
    assert state.calls == 3
    for headers in state.headers:
        assert not any(name.lower() in {"authorization", "cookie", "x-client-id"} for name in headers)


@pytest.mark.parametrize("status,body", [(500, b"private error"), (302, b""),
    (200, b"<html>Login</html>"), (200, b'{"status":"starting"}'), (200, b"null")])
def test_unhealthy_or_redirected_http_never_counts_as_ready(health_server, status, body, capsys):
    url, state = health_server
    state.responses = [(status, body)]
    assert not session.wait_for_ready(url, threading.Event(), timeout=.15)
    assert state.calls >= 1
    assert set(state.paths) == {"/health"}
    assert "private error" not in capsys.readouterr().out


def test_cancelled_probe_does_not_make_an_http_request(health_server):
    url, state = health_server
    stop = threading.Event()
    stop.set()
    assert not session.wait_for_ready(url, stop, timeout=2)
    assert state.calls == 0


def test_slow_http_has_bounded_timeout(health_server):
    url, state = health_server
    state.delay = 1
    started = time.monotonic()
    assert not session.wait_for_ready(url, threading.Event(), timeout=.2)
    assert time.monotonic() - started < 1


def test_cancellation_while_http_waits_returns_without_retry(health_server):
    url, state = health_server
    state.delay = 1
    stop = threading.Event()
    timer = threading.Timer(.05, stop.set)
    timer.start()
    try:
        assert not session.wait_for_ready(url, stop, timeout=5)
    finally:
        timer.join()
    assert state.calls == 1


def test_web_url_is_discovered_in_selected_environment(monkeypatch):
    import modal
    calls = []
    async def web_url():
        return "https://different-region.modal.run/"
    def lookup(name, function_name, *, environment_name):
        calls.append((name, function_name, environment_name))
        return SimpleNamespace(get_web_url=SimpleNamespace(aio=web_url))
    monkeypatch.setattr(modal.Function, "from_name", lookup)
    assert session.lookup_web_url("history", "private") == "https://different-region.modal.run"
    assert calls == [("history", "fastapi_app", "private")]


@pytest.mark.parametrize("url", [None, "http://example.com", "https://user:secret@example.com", "https://example.com/?token=secret"])
def test_missing_or_unsafe_web_url_is_rejected(monkeypatch, url):
    import modal
    async def web_url():
        return url
    monkeypatch.setattr(modal.Function, "from_name", lambda *args, **kwargs: SimpleNamespace(get_web_url=SimpleNamespace(aio=web_url)))
    with pytest.raises(ValueError):
        session.lookup_web_url("history", None)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "not-a-number"])
def test_invalid_readiness_timeout_is_rejected(value):
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        session._positive_seconds(value)


def test_positive_fractional_timeout_is_supported():
    assert session._positive_seconds(".25") == .25


def test_slow_stream_is_bounded_by_total_deadline(health_server):
    url, state = health_server
    state.byte_delay = .04
    started = time.monotonic()
    assert not session.wait_for_ready(url, threading.Event(), timeout=.15)
    assert time.monotonic() - started < .6


def test_cancelling_slow_stream_closes_probe_promptly(health_server):
    url, state = health_server
    state.byte_delay = .04
    stop = threading.Event()
    def cancel():
        state.received.wait(2)
        time.sleep(.05)
        stop.set()
    worker = threading.Thread(target=cancel)
    worker.start()
    started = time.monotonic()
    try:
        assert not session.wait_for_ready(url, stop, timeout=5)
        assert time.monotonic() - started < .6
    finally:
        worker.join()


def test_cancelling_url_discovery_releases_startup_task(monkeypatch):
    import asyncio
    stop = threading.Event()
    closed = threading.Event()
    async def lookup(*args):
        stop.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()
    monkeypatch.setattr(session, "_lookup_web_url", lookup, raising=False)
    started = time.monotonic()
    assert session.prepare_website("history", None, stop, timeout=5, process=SimpleNamespace(poll=lambda: None)) is None
    assert closed.is_set() and time.monotonic() - started < 1


def test_failed_log_process_aborts_startup_before_browser(monkeypatch):
    process = SimpleNamespace(poll=lambda: 7)
    with pytest.raises(RuntimeError, match="log"):
        session.prepare_website("history", None, threading.Event(), timeout=2, process=process)


def test_total_deadline_includes_pending_url_discovery(monkeypatch):
    import asyncio
    closed = threading.Event()
    async def lookup(*args):
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()
    monkeypatch.setattr(session, "_lookup_web_url", lookup)
    started = time.monotonic()
    assert session.prepare_website("history", None, threading.Event(), timeout=.15,
                                   process=SimpleNamespace(poll=lambda: None)) is None
    assert closed.is_set() and time.monotonic() - started < .6


@pytest.mark.parametrize("cancelled", [False, True])
def test_pending_dns_does_not_delay_controller_cleanup(monkeypatch, cancelled):
    import socket
    started_dns = threading.Event()
    release_dns = threading.Event()
    finished_dns = threading.Event()
    stop = threading.Event()
    def slow_dns(*args, **kwargs):
        started_dns.set()
        try:
            release_dns.wait(1.2)
            raise socket.gaierror("private resolver detail")
        finally:
            finished_dns.set()
    monkeypatch.setattr(socket, "getaddrinfo", slow_dns)
    monkeypatch.setenv("NO_PROXY", "*")
    def cancel():
        if started_dns.wait(2):
            stop.set()
    worker = threading.Thread(target=cancel) if cancelled else None
    if worker:
        worker.start()
    started = time.monotonic()
    try:
        assert not session.wait_for_ready("http://pending.invalid", stop, timeout=.3)
        assert started_dns.is_set()
        # The controller can now terminate logs and stop the cloud app, even if
        # the operating system resolver has not returned yet.
        assert time.monotonic() - started < .8
    finally:
        release_dns.set()
        finished_dns.wait(2)
        if worker:
            worker.join()
