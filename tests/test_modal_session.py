"""Run the lifecycle controller against a real child CLI, without Modal access."""
import json
import sys
import threading
import time

import pytest


def controller(tmp_path, monkeypatch, *, failure="", block_deploy=False):
    from scripts import modal_session
    log = tmp_path / "calls.jsonl"
    fake = tmp_path / "fake_cli.py"
    fake.write_text('''import json, os, sys, time
args = sys.argv[1:]
with open(os.environ["SESSION_TEST_LOG"], "a") as output:
    output.write(json.dumps(args) + "\\n")
operation = "deploy" if args[0] == "deploy" else args[1]
if operation == os.environ.get("SESSION_TEST_FAILURE"):
    raise SystemExit(7)
if operation == "logs" or (operation == "deploy" and os.environ.get("SESSION_TEST_BLOCK")):
    while True: time.sleep(.02)
''', encoding="utf-8")
    monkeypatch.setenv("SESSION_TEST_LOG", str(log))
    monkeypatch.setenv("SESSION_TEST_FAILURE", failure)
    if block_deploy:
        monkeypatch.setenv("SESSION_TEST_BLOCK", "1")
    monkeypatch.setattr(modal_session, "MODAL_COMMAND", [sys.executable, str(fake)])
    monkeypatch.setattr(modal_session, "lookup_app_id", lambda name, environment: "ap-owned")
    return modal_session, log


def calls(log):
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def interrupt_when(log, operation, event):
    observed = threading.Event()
    def watch():
        until = time.monotonic() + 5
        while time.monotonic() < until:
            if any(row[:2] == operation for row in calls(log)):
                observed.set()
                break
            time.sleep(.01)
        event.set()
    worker = threading.Thread(target=watch)
    worker.start()
    return worker, observed


@pytest.mark.parametrize("attach", [False, True])
def test_interrupt_stops_exact_app_id_in_selected_environment(tmp_path, monkeypatch, attach):
    session, log = controller(tmp_path, monkeypatch)
    stop = threading.Event()
    watcher, observed = interrupt_when(log, ["app", "logs"], stop)
    result = session.run_session(name="history", environment="private", attach=attach, stop=stop)
    watcher.join()
    assert result == 0 and observed.is_set()
    commands = calls(log)
    assert commands[-1] == ["app", "stop", "ap-owned", "--yes", "--env", "private"]
    assert all(row[-2:] == ["--env", "private"] for row in commands)
    assert sum(row[0] == "deploy" for row in commands) == int(not attach)


def test_interrupt_during_deploy_also_stops_owned_app(tmp_path, monkeypatch):
    session, log = controller(tmp_path, monkeypatch, block_deploy=True)
    stop = threading.Event()
    watcher, observed = interrupt_when(log, ["deploy", "modal_app.py"], stop)
    assert session.run_session(name="history", environment=None, attach=False, stop=stop) == 0
    watcher.join()
    assert observed.is_set()
    assert calls(log)[-1] == ["app", "stop", "ap-owned", "--yes"]
    assert not any(row[:2] == ["app", "logs"] for row in calls(log))


def test_failed_deploy_does_not_stop_existing_app(tmp_path, monkeypatch):
    session, log = controller(tmp_path, monkeypatch, failure="deploy")
    assert session.run_session(name="history", environment=None, attach=False, stop=threading.Event()) != 0
    assert len(calls(log)) == 1 and calls(log)[0][0] == "deploy"


def test_stop_failure_is_reported_without_claiming_success(tmp_path, monkeypatch, capsys):
    session, log = controller(tmp_path, monkeypatch, failure="stop")
    stop = threading.Event()
    watcher, _ = interrupt_when(log, ["app", "logs"], stop)
    assert session.run_session(name="history", environment=None, attach=True, stop=stop) != 0
    watcher.join()
    assert "app stop ap-owned" in capsys.readouterr().err


def test_missing_attach_target_never_stops_another_app(tmp_path, monkeypatch):
    session, log = controller(tmp_path, monkeypatch)
    def missing(*args):
        raise RuntimeError("private upstream detail")
    monkeypatch.setattr(session, "lookup_app_id", missing)
    assert session.run_session(name="history", environment=None, attach=True, stop=threading.Event()) != 0
    assert calls(log) == []


def test_failed_identity_lookup_after_deploy_never_stops_by_mutable_name(tmp_path, monkeypatch, capsys):
    session, log = controller(tmp_path, monkeypatch)
    def missing(*args):
        raise RuntimeError("lookup unavailable")
    monkeypatch.setattr(session, "lookup_app_id", missing)
    assert session.run_session(name="history", environment="private", attach=False, stop=threading.Event()) != 0
    assert len(calls(log)) == 1 and calls(log)[0][0] == "deploy"
    assert "app stop history" in capsys.readouterr().err


@pytest.mark.parametrize("attach", [True, False])
def test_local_process_cleanup_failure_cannot_skip_exact_id_remote_stop(tmp_path, monkeypatch, attach):
    session, log = controller(tmp_path, monkeypatch, block_deploy=not attach)
    def cleanup_failure(process):
        if process is not None:
            process.kill()
            process.wait()
        raise OSError("Cannot terminate local child")
    monkeypatch.setattr(session, "_terminate", cleanup_failure)
    stop = threading.Event()
    watcher, _ = interrupt_when(log, ["app", "logs"] if attach else ["deploy", "modal_app.py"], stop)
    try:
        assert session.run_session(name="history", environment=None, attach=attach, stop=stop) != 0
    finally:
        watcher.join()
    assert calls(log)[-1] == ["app", "stop", "ap-owned", "--yes"]
