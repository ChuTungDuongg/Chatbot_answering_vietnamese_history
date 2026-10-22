"""Serve built React assets and stream requests to the independent GPU API."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

DIST = Path(os.getenv("FRONTEND_DIST_DIR", "/workspace/frontend_dist")).resolve()
UPSTREAM = os.getenv("MODAL_GPU_API_URL", "").strip().rstrip("/")
REQUEST_HEADERS = {"accept", "authorization", "content-type", "x-client-id", "x-request-id"}
RESPONSE_HEADERS = {"content-type", "content-encoding", "cache-control", "x-request-id",
                    "content-disposition", "x-accel-buffering"}
METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]

gateway = FastAPI(title="Vietnamese History LLM frontend gateway")


@gateway.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok", "service": "frontend-gateway", "backend_configured": bool(UPSTREAM)}


async def _proxy(request: Request, path: str):
    if not UPSTREAM.startswith(("https://", "http://")):
        raise HTTPException(status_code=503, detail="MODAL_GPU_API_URL is not configured")
    url = f"{UPSTREAM}/{path.lstrip('/')}"
    headers = {k: v for k, v in request.headers.items() if k.lower() in REQUEST_HEADERS}
    client = httpx.AsyncClient(timeout=httpx.Timeout(connect=30, read=None, write=90, pool=30),
                               follow_redirects=False)
    try:
        req = client.build_request(request.method, url, params=request.query_params,
                                   headers=headers, content=request.stream())
        upstream = await client.send(req, stream=True)
    except httpx.RequestError as exc:
        await client.aclose()
        raise HTTPException(status_code=502, detail="GPU backend unavailable") from exc

    response_headers = {k: v for k, v in upstream.headers.items()
                        if k.lower() in RESPONSE_HEADERS}

    async def chunks():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(chunks(), status_code=upstream.status_code, headers=response_headers)


@gateway.api_route("/api/v1/{path:path}", methods=METHODS, include_in_schema=False)
async def api_proxy(request: Request, path: str):
    return await _proxy(request, f"api/v1/{path}")


@gateway.api_route("/ready", methods=["GET"], include_in_schema=False)
async def backend_ready(request: Request):
    return await _proxy(request, "ready")


if DIST.is_dir() and (DIST / "assets").is_dir():
    gateway.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@gateway.get("/{path:path}", include_in_schema=False)
async def spa(path: str):
    file_path = (DIST / path).resolve()
    if path and file_path.is_relative_to(DIST) and file_path.is_file():
        return FileResponse(file_path, headers={"Cache-Control": "public, max-age=86400"})
    index = DIST / "index.html"
    if index.is_file() and not path.startswith(("api/", "assets/")):
        return FileResponse(index, headers={"Cache-Control": "no-cache"})
    raise HTTPException(status_code=404, detail="Frontend assets are unavailable")
