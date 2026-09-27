"""Local-only FastAPI review page for Corpus V1 benchmark annotation."""

from __future__ import annotations

import argparse
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from evaluation.annotation.corpus import CorpusLookup
from evaluation.annotation.report import coverage
from evaluation.annotation.workspace import Workspace


class ReviewRequest(BaseModel):
    action: str
    patch: dict[str, Any] = Field(default_factory=dict)
    expected_revision: int | None = None


def create_app(workspace_path: Path, corpus_path: Path, *, lookup_workspace: Path | None = None) -> FastAPI:
    workspace = Workspace(workspace_path)
    lookup = CorpusLookup(corpus_path, (lookup_workspace or workspace_path) / "corpus_lookup.sqlite3")
    lock = threading.RLock()

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            with lock:
                lookup.close()
                workspace.close()

    app = FastAPI(title="Vietnamese History Annotation", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def home():
        return (Path(__file__).parent / "review.html").read_text(encoding="utf-8")

    @app.get("/api/candidates")
    def candidates(status: str | None = None, limit: int = Query(500, ge=1, le=1000),
                   offset: int = Query(0, ge=0)):
        with lock:
            return workspace.list_candidates(status=status, limit=limit, offset=offset)

    @app.get("/api/candidates/{candidate_id}")
    def candidate(candidate_id: str):
        try:
            with lock:
                item = workspace.get(candidate_id)
        except KeyError:
            raise HTTPException(404, "Candidate not found") from None
        texts: dict[str, dict[str, Any] | None] = {}
        with lock:
            for evidence in item["evidence"]:
                chunk_id = evidence["chunk_id"]
                if chunk_id not in texts:
                    texts[chunk_id] = lookup.get_chunk(chunk_id)
                row = texts[chunk_id]
                if row:
                    evidence.update(title=row.get("title"), text=row.get("text"), url=row.get("url"),
                                    chunk_index=row.get("chunk_index"))
            origin = lookup.get_chunk(item["origin_chunk_id"]) if item.get("origin_chunk_id") else None
            item["origin_evidence"] = ({key: origin.get(key) for key in (
                "chunk_id", "source_id", "document_id", "title", "text", "url", "chunk_index")}
                if origin else None)
        return item

    @app.get("/api/chunks/{chunk_id}/nearby")
    def nearby(chunk_id: str, radius: int = Query(2, ge=0, le=5)):
        with lock:
            if not lookup.has_chunk(chunk_id):
                raise HTTPException(404, "Chunk not found")
            return [{key: row.get(key) for key in ("chunk_id", "source_id", "document_id",
                                                  "title", "url", "text", "chunk_index")}
                    for row in lookup.nearby(chunk_id, radius)]

    @app.post("/api/candidates/{candidate_id}/review")
    def review(candidate_id: str, request: ReviewRequest):
        try:
            with lock:
                return workspace.apply_review(candidate_id, request.patch, request.action, lookup,
                                              expected_revision=request.expected_revision)
        except KeyError:
            raise HTTPException(404, "Candidate not found") from None
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from None

    @app.get("/api/report")
    def report():
        with lock:
            return coverage(workspace.all_candidates())

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("evaluation/annotation/workspace"))
    parser.add_argument("--corpus", type=Path, default=Path("artifacts/corpus_v1/chunks.jsonl"))
    parser.add_argument("--lookup-workspace", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    import uvicorn
    uvicorn.run(create_app(args.workspace, args.corpus, lookup_workspace=args.lookup_workspace), host="127.0.0.1", port=args.port,
                log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
