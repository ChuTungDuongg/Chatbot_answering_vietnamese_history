"""Deploy, wait for the website, open it, and stop the Modal App on Ctrl+C."""
import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
import webbrowser

import httpx


ROOT = Path(__file__).resolve().parents[1]
MODAL_COMMAND = [sys.executable, "-m", "modal"]
READY_POLL_SECONDS = 2
READY_REQUEST_TIMEOUT = 5


def lookup_app_id(name, environment):
    import modal
    app_id = modal.App.lookup(name, environment_name=environment).app_id
    if not app_id:
        raise RuntimeError("No deployed Modal App")
    return app_id


async def _lookup_web_url(name, environment):
    import modal
    function = modal.Function.from_name(name, "fastapi_app", environment_name=environment)
    url = await function.get_web_url.aio()
    parsed = urlsplit(url or "")
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment):
        raise ValueError("The deployed fastapi_app must have a public HTTPS URL without credentials")
    return url.rstrip("/")


def _run_startup(operation, stop, timeout, process=None):
    """Bound metadata, headers and body reads together; cancellation closes HTTP/RPC resources."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Readiness timeout must be positive and finite")
    async def supervised():
        deadline = time.monotonic() + timeout
        task = asyncio.create_task(operation())
        try:
            while True:
                if stop.is_set():
                    return None
                if process is not None and process.poll() is not None:
                    print("Modal log follower exited during startup; stopping the owned app.", file=sys.stderr)
                    raise RuntimeError("Modal log follower exited during startup")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                if task.done():
                    return task.result()
                await asyncio.wait({task}, timeout=min(.1, remaining))
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(supervised())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        # Unlike asyncio.run(), close() does not join an uncancellable OS DNS
        # lookup in the default executor. Start exact-ID cloud cleanup promptly;
        # the resolver thread can finish independently without owning HTTP state.
        loop.close()


async def _health_probe(url):
    """Warm the ASGI app; no cookies, credentials or inference requests are sent."""
    started, next_report = time.monotonic(), 0
    print("Waiting for Sử Việt startup (GPU allocation, retrieval and web server)...", flush=True)
    async with httpx.AsyncClient(timeout=READY_REQUEST_TIMEOUT, follow_redirects=False) as client:
        while True:
            detail = "Health response is not ready"
            try:
                client.cookies.clear()
                async with asyncio.timeout(READY_REQUEST_TIMEOUT):
                    async with client.stream("GET", url.rstrip("/") + "/health",
                                             headers={"Accept": "application/json"}) as response:
                        if response.status_code == 200:
                            raw = bytearray()
                            async for chunk in response.aiter_bytes(chunk_size=4096):
                                raw.extend(chunk)
                                if len(raw) > 4096:
                                    break
                            body = json.loads(raw) if len(raw) <= 4096 else None
                            if isinstance(body, dict) and body.get("status") == "ok":
                                return True
                        detail = f"HTTP {response.status_code}; health is not ready"
            except (httpx.HTTPError, ValueError, TimeoutError) as exc:
                # Never print upstream bodies, response headers or exception messages.
                detail = type(exc).__name__
            now = time.monotonic()
            if now >= next_report:
                print(f"Still starting ({now - started:.0f}s): {detail}. Ctrl+C cancels and stops the app.", flush=True)
                next_report = now + 15
            await asyncio.sleep(READY_POLL_SECONDS)


def lookup_web_url(name, environment):
    return asyncio.run(_lookup_web_url(name, environment))


def wait_for_ready(url, stop, timeout=960):
    return bool(_run_startup(lambda: _health_probe(url), stop, timeout))


def prepare_website(name, environment, stop, *, timeout, process):
    print("Resolving the deployed website and waiting for startup...", flush=True)
    async def prepare():
        url = await _lookup_web_url(name, environment)
        await _health_probe(url)
        return url
    return _run_startup(prepare, stop, timeout, process)


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


def run_session(*, name, environment, attach, stop, ready_timeout=960, open_browser=True):
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
            url = prepare_website(name, environment, stop, timeout=ready_timeout, process=process)
            if url and not stop.is_set() and process.poll() is None:
                print(f"Sử Việt is ready: {url}", flush=True)
                if open_browser:
                    try:
                        if not webbrowser.open(url):
                            raise OSError("Browser unavailable")
                    except Exception:
                        print("Open the ready URL manually; the browser could not be launched.", file=sys.stderr)
                code = _wait(process, stop)
                if code not in (0, None):
                    result = 1
            elif not stop.is_set():
                print(f"Website did not become ready within {ready_timeout:g}s or log following failed. Check startup logs above.", file=sys.stderr)
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


def _positive_seconds(value):
    try:
        seconds = float(value)
        if math.isfinite(seconds) and seconds > 0:
            return seconds
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("Expected a positive finite number of seconds")


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False, encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attach", action="store_true", help="Own an existing deployment without deploying again")
    parser.add_argument("--name", default=os.getenv("MODAL_APP_NAME", "vn-history-rag-api"))
    parser.add_argument("--env", default=os.getenv("MODAL_ENVIRONMENT"))
    parser.add_argument("--ready-timeout", type=_positive_seconds, default=960,
                        help="Maximum seconds to wait for ASGI startup (default: 960)")
    parser.add_argument("--no-browser", action="store_true", help="Print the ready URL without opening a browser")
    args = parser.parse_args()
    stop = threading.Event()
    previous = {}
    for kind in (signal.SIGINT, signal.SIGTERM, *([signal.SIGBREAK] if hasattr(signal, "SIGBREAK") else [])):
        previous[kind] = signal.signal(kind, lambda *_: stop.set())
    try:
        return run_session(name=args.name, environment=args.env, attach=args.attach, stop=stop,
                           ready_timeout=args.ready_timeout, open_browser=not args.no_browser)
    finally:
        for kind, handler in previous.items():
            signal.signal(kind, handler)


if __name__ == "__main__":
    raise SystemExit(main())
