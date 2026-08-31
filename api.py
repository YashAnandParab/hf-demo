"""HTTP API for the React frontend in ./frontend.

Two endpoints, shaped by what frontend/src/lib/api.ts already expects:

    GET  /models   the RAG versions, offered to the UI as selectable "models"
    POST /chat     SSE — one `sources` frame, then `token` frames, then [DONE]

The version picker IS the model picker. Choosing "Structured RAG" or "Normal
chunking" switches which database the question is answered from, which is the
comparison this whole project exists to make. Which LLM writes the answer is not
a per-request choice — it is GROQ_MODEL in .env, the same for both versions, so
that a difference between two answers is attributable to the structure alone.

Everything downstream is process-wide state — one Postgres connection, one active
version, one resident embedder and reranker — so a global lock serialises
queries. The endpoints are sync `def`, which Starlette runs in a threadpool, so
waiting on that lock never blocks the event loop.

    uvicorn api:app --reload --port 8000
"""
from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import config
import db
import llm
import models
import query
import retrieval
import tracing
import versions

log = logging.getLogger("api")

# The Vite dev server, on whatever port it settled for: it walks upward from 5173
# when that one is busy, so pinning a single origin breaks the moment anything else
# is already running. Localhost on any port, and nothing else — still not "*",
# because this API takes a POST body and spends money on it. Set ALLOWED_ORIGINS
# (comma-separated) to replace the regex with an exact list, which is what you want
# the moment this is served anywhere but your own machine.
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
LOCALHOST_ORIGIN = r"^http://(localhost|127\.0\.0\.1)(:\d+)?$"

# ponytail: one lock serialises the whole pipeline. Per-version connection pools
# are the upgrade if more than one person ever uses this at a time.
_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config.setup_logging()
    db.wait_for_db()
    # Loading the embedder and its reranker takes ~30s. Doing it here means the
    # first question is fast instead of looking like the server hung.
    log.info("warming %s and the reranker…", config.EMBED_MODEL)
    try:
        models.embed_query("warmup")
        if config.RERANKER_BACKEND == "local":
            models.rerank("warmup", [{"chunk_id": 0, "chunk_text": "warmup"}], 1)
    except OSError as exc:
        _abort_if_out_of_memory(exc)
        raise
    log.info("ready — versions: %s", ", ".join(versions.keys()))
    yield
    db.close()
    tracing.flush()


def _abort_if_out_of_memory(exc: OSError) -> None:
    """Turn Windows' commit-limit error into something actionable.

    winerror 1455 surfaces as "The paging file is too small for this operation to
    complete" under forty lines of safetensors traceback, which reads like a
    corrupt download rather than what it is: the machine has no commit charge left.
    The usual cause is a second process already holding the models — an ingest, or
    an earlier server that never exited.
    """
    if getattr(exc, "winerror", None) != 1455:
        return
    raise SystemExit(
        "Out of memory loading the models (Windows commit limit, winerror 1455).\n"
        "  Nothing is wrong with the download. Something else is holding the RAM:\n"
        "  - an ingest still running, or an earlier server that never exited\n"
        "      (Get-Process python | Select Id, WorkingSet, StartTime)\n"
        f"  - or {config.EMBED_MODEL} is simply too big for this machine; the\n"
        "      smaller models are EMBED_MODEL=BAAI/bge-small-en-v1.5 with\n"
        "      EMBED_DIM=384 (needs a re-ingest), and RERANKER_BACKEND=none."
    )


app = FastAPI(title="HFrag RAG API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=None if ALLOWED_ORIGINS else LOCALHOST_ORIGIN,
    allow_methods=["GET", "POST"],
    allow_headers=["content-type"],
)


# ------------------------------------------------------------------- models --


@app.get("/models")
def list_models() -> dict:
    return {
        "models": [
            {
                "id": v.key,
                "name": v.label,
                "provider": f"pgvector · {v.database}",
                "description": v.blurb,
                "capabilities": (
                    ["Knowledge/story split", "Stories attached"]
                    if v.has_stories
                    else ["Flat chunks", "Baseline"]
                ),
            }
            for v in versions.ALL.values()
        ]
    }


# --------------------------------------------------------------------- chat --


class ChatRequest(BaseModel):
    message: str
    # The version key. Empty falls back to RAG_VERSION from .env.
    model: str = ""
    top_k: int = Field(default=config.RERANK_TOP_K, ge=1, le=20)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    # Accepted and ignored: retrieval here is single-turn, so prior messages are
    # not part of the query. Follow-up rewriting would be a feature, not a field.
    conversation_id: str | None = None
    history: list[dict] = []
    stream: bool = True


@app.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    if not req.message.strip():
        raise HTTPException(400, "message is empty")
    try:
        version = versions.resolve(req.model or config.DEFAULT_VERSION)
    except SystemExit as exc:  # resolve() is written for a CLI
        raise HTTPException(400, str(exc)) from None

    return StreamingResponse(
        _events(req, version),
        media_type="text/event-stream",
        # Without these a proxy will happily buffer the whole answer and deliver
        # it in one lump, which looks exactly like streaming being broken.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _events(req: ChatRequest, version: versions.Version) -> Iterator[str]:
    """The SSE body. Errors are streamed as text rather than raised.

    By the time this generator runs the response is already committed with a 200,
    so there is no status code left to return. A failure the user can act on —
    nothing ingested, no API key — is far more useful in the message bubble than
    as a silently dead stream.
    """
    with _lock:
        try:
            db.set_version(version.key)
            _require_ingested(version)
            result = query.run(
                req.message,
                top_k=req.top_k,
                min_score=req.min_score,
                generate=False,
                verbose=False,
            )
            yield _frame({"sources": _sources(result)})

            if not result["knowledge"]:
                yield _frame(
                    {"token": "I could not find anything relevant in the indexed articles."}
                )
            else:
                for delta in llm.stream(version.system_prompt, result["user_prompt"]):
                    yield _frame({"token": delta})
        # SystemExit is what the CLI-oriented helpers raise for a bad key or a
        # missing database, and it is a BaseException — a bare `except Exception`
        # would let it kill the response with nothing written to it.
        except (Exception, SystemExit) as exc:  # noqa: BLE001
            log.exception("chat failed")
            yield _frame({"token": f"\n\n**Error:** {exc}"})
        finally:
            yield "data: [DONE]\n\n"


def _require_ingested(version: versions.Version) -> None:
    hint = f"run: python ingest.py --version {version.key}"
    if not db.database_exists():
        raise RuntimeError(f"database {version.database!r} does not exist yet — {hint}")
    if not db.tables_exist():
        raise RuntimeError(f"{version.label} has no tables yet — {hint}")
    if retrieval.retrievable_count() == 0:
        raise RuntimeError(f"{version.label} has nothing indexed — {hint}")


def _frame(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sources(result: dict) -> list[dict]:
    """Retrieved chunks in the frontend's `Source` shape.

    The `id` is the label the answer will cite — [K1]/[S1] structured, [P1] flat —
    so a citation in the text resolves to the exact card underneath it.
    """
    tag = "K" if versions.active().has_stories else "P"
    out = [_source(h, f"{tag}{i}") for i, h in enumerate(result["knowledge"], start=1)]
    out += [
        _source(h, f"S{i}", arms=["linked story"], story=True)
        for i, h in enumerate(result["stories"], start=1)
    ]
    return out


def _source(
    hit: dict, label: str, arms: list[str] | None = None, story: bool = False
) -> dict:
    return {
        "id": label,
        "document": hit.get("article_name") or "Untitled",
        "url": hit.get("article_url"),
        "chunk": hit.get("source_chunk_id") or hit.get("chunk_index"),
        "content": (hit.get("chunk_text") or "").strip(),
        # A story was attached, not scored — a similarity bar on one would be a
        # number that nothing actually produced.
        "score": None if story else round(query.normalized_score(hit), 3),
        "arms": arms or hit.get("sources") or [],
        # The frontend separates evidence from illustration, and for a story it
        # shows which knowledge chunk pulled it in — the same relation the
        # prompt states as "illustrates K1".
        "kind": "story" if story else "knowledge",
        "illustrates": (hit.get("linked_knowledge_label") or None) if story else None,
    }
