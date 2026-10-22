"""CPU-only static UI and transparent streaming gateway to the GPU API."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from urllib.parse import urlsplit

import anyio
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import ClientDisconnect

DIST = Path(os.getenv("FRONTEND_DIST_DIR", "/workspace/frontend_dist")).resolve()
UPSTREAM = os.getenv("MODAL_GPU_API_URL", "").strip().rstrip("/")
REQUEST_HEADERS = {"accept", "authorization", "content-type", "content-length",
                   "content-encoding", "x-client-id", "x-request-id"}
RESPONSE_HEADERS = {"content-type", "content-encoding", "content-length", "cache-control",
                    "x-request-id", "content-disposition", "x-accel-buffering",
                    "retry-after", "www-authenticate", "allow", "vary"}
METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]

gateway = FastAPI(title="Vietnamese History LLM frontend gateway")


@gateway.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok", "service": "frontend-gateway", "backend_configured": bool(UPSTREAM)}


class ProxyResponse(StreamingResponse):
    """Own resources even when ASGI send fails before body iteration starts."""
    def __init__(self, upstream, client, headers):
        super().__init__(upstream.aiter_raw(), status_code=upstream.status_code, headers=headers)
        self.upstream, self.client = upstream, client

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await self.upstream.aclose()
                finally:
                    await self.client.aclose()


async def _send(client, req, request, body_sent):
    async def disconnected():
        # Reading disconnect before upload finishes would steal request body frames.
        await body_sent.wait()
        while not await request.is_disconnected():
            await asyncio.sleep(.1)
    sending = asyncio.create_task(client.send(req, stream=True))
    watcher = asyncio.create_task(disconnected())
    try:
        await asyncio.wait({sending, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if sending.done():
            return sending.result()
        # Do not turn a browser abort into a synthetic successful HTTP response.
        raise ClientDisconnect()
    finally:
        with anyio.CancelScope(shield=True):
            for task in (sending, watcher):
                if not task.done():
                    task.cancel()
            await asyncio.gather(sending, watcher, return_exceptions=True)


async def _proxy(request: Request, path: str):
    parsed = urlsplit(UPSTREAM)
    if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise HTTPException(status_code=503, detail="MODAL_GPU_API_URL is not configured")
    headers = {k: v for k, v in request.headers.items() if k.lower() in REQUEST_HEADERS}
    client = httpx.AsyncClient(timeout=httpx.Timeout(connect=30, read=600, write=90, pool=30),
                               follow_redirects=False, trust_env=False)
    upstream = None
    handed_off = False
    body_sent = asyncio.Event()

    async def body():
        async for chunk in request.stream():
            yield chunk
        body_sent.set()

    try:
        req = client.build_request(request.method, f"{UPSTREAM}/{path.lstrip('/')}",
                                   params=request.query_params.multi_items(),
                                   headers=headers, content=body())
        upstream = await _send(client, req, request, body_sent)
        response_headers = {k: v for k, v in upstream.headers.items() if k.lower() in RESPONSE_HEADERS}
        location = upstream.headers.get("location")
        # Keep backend redirects on the browser's same origin.
        if location and location.startswith(UPSTREAM + "/"):
            response_headers["location"] = location[len(UPSTREAM):]
        elif location and location.startswith("/") and not location.startswith("//"):
            response_headers["location"] = location
        if path.startswith("api/v1/ops/"):
            response_headers["cache-control"] = "no-store"
        result = ProxyResponse(upstream, client, response_headers)
        handed_off = True
        return result
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="GPU backend timed out") from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail="GPU backend unavailable") from exc
    finally:
        if not handed_off:
            with anyio.CancelScope(shield=True):
                try:
                    if upstream is not None:
                        await upstream.aclose()
                finally:
                    await client.aclose()


@gateway.api_route("/api/v1/{path:path}", methods=METHODS, include_in_schema=False)
async def api_proxy(request: Request, path: str):
    return await _proxy(request, f"api/v1/{path}")


@gateway.get("/ready", include_in_schema=False)
async def backend_ready(request: Request):
    return await _proxy(request, "ready")


if DIST.is_dir() and (DIST / "assets").is_dir():
    gateway.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@gateway.api_route("/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
async def spa(path: str):
    file_path = (DIST / path).resolve()
    if path and file_path.is_relative_to(DIST) and file_path.is_file():
        cache = "no-cache" if file_path.name == "index.html" else "public, max-age=86400"
        return FileResponse(file_path, headers={"Cache-Control": cache})
    index = DIST / "index.html"
    if index.is_file() and not path.startswith(("api/", "assets/")):
        return FileResponse(index, headers={"Cache-Control": "no-cache"})
    raise HTTPException(status_code=404, detail="Frontend assets are unavailable")
