"""Deploy (or attach), follow logs, and stop the owned Modal App on Ctrl+C."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading


ROOT = Path(__file__).resolve().parents[1]
MODAL_COMMAND = [sys.executable, "-m", "modal"]


def lookup_app_id(name, environment):
    import modal
    app_id = modal.App.lookup(name, environment_name=environment).app_id
    if not app_id:
        raise RuntimeError("No deployed Modal App")
    return app_id


def _spawn(arguments, environment, name):
    # Keep Ctrl+C in the controller so cleanup is not interrupted along with logs.
    options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
               else {"start_new_session": True})
    child_env = {**os.environ, "MODAL_APP_NAME": name}
    suffix = ["--env", environment] if environment else []
    return subprocess.Popen([*MODAL_COMMAND, *arguments, *suffix], cwd=ROOT, env=child_env, **options)


def _wait(process, stop):
    while not stop.is_set():
        try:
            return process.wait(timeout=.2)
        except subprocess.TimeoutExpired:
            pass
    return None


def _terminate(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def run_session(*, name, environment, attach, stop):
    process, app_id, owns_app, result = None, None, False, 0
    try:
        if not attach:
            print("Deploying Sử Việt. Ctrl+C will stop this Modal App.", flush=True)
            process = _spawn(["deploy", "modal_app.py", "--name", name], environment, name)
            deploy_code = _wait(process, stop)
            if deploy_code is not None and deploy_code != 0:
                print("Deploy failed; existing deployment was not stopped.", file=sys.stderr)
                return 1
            owns_app = True
            try:
                _terminate(process)
                process = None
            except Exception:
                print("Deploy process cleanup failed; resolving the owned App ID for cloud stop.", file=sys.stderr)
                result = 1
        app_id = lookup_app_id(name, environment)
        owns_app = True
        if not stop.is_set():
            print(f"Session attached to {app_id}. Press Ctrl+C to stop the app and its containers.", flush=True)
            process = _spawn(["app", "logs", app_id, "--follow"], environment, name)
            code = _wait(process, stop)
            if code not in (0, None):
                result = 1
    except Exception:
        print("Cannot open the Modal session; check Modal credentials, environment and app name.", file=sys.stderr)
        result = 1
    finally:
        try:
            _terminate(process)
        except Exception:
            print("Local log/deploy process cleanup failed; attempting cloud stop independently.", file=sys.stderr)
            result = 1
    if owns_app and app_id is None:
        print(f"Cannot verify the exact App ID. Check the dashboard, or run: python -m modal app stop {name} --yes"
              + (f" --env {environment}" if environment else ""), file=sys.stderr)
        return 1
    if owns_app:
        target = app_id
        print(f"Stopping Modal App {target}...", flush=True)
        suffix = ["--env", environment] if environment else []
        try:
            stopped = subprocess.run([*MODAL_COMMAND, "app", "stop", target, "--yes", *suffix],
                cwd=ROOT, text=True, capture_output=True, timeout=60)
            if stopped.returncode and "already stopped" not in (stopped.stdout + stopped.stderr).lower():
                raise RuntimeError("Stop was not confirmed")
            print("Modal App stopped. Named Volumes and Secrets were retained.", flush=True)
        except Exception:
            print(f"Stop not confirmed. Run: python -m modal app stop {target} --yes"
                  + (f" --env {environment}" if environment else ""), file=sys.stderr)
            result = 1
    return result


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False, encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attach", action="store_true", help="Own an existing deployment without deploying again")
    parser.add_argument("--name", default=os.getenv("MODAL_APP_NAME", "vn-history-rag-api"))
    parser.add_argument("--env", default=os.getenv("MODAL_ENVIRONMENT"))
    args = parser.parse_args()
    stop = threading.Event()
    previous = {}
    for kind in (signal.SIGINT, signal.SIGTERM, *([signal.SIGBREAK] if hasattr(signal, "SIGBREAK") else [])):
        previous[kind] = signal.signal(kind, lambda *_: stop.set())
    try:
        return run_session(name=args.name, environment=args.env, attach=args.attach, stop=stop)
    finally:
        for kind, handler in previous.items():
            signal.signal(kind, handler)


if __name__ == "__main__":
    raise SystemExit(main())
