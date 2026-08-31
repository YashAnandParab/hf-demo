"""Checks for the HTTP layer that need neither Postgres nor a Groq key.

What is worth checking here is the mapping, not FastAPI: the score squash, and
the fact that the `id` on a source is exactly the label the answer will cite —
get that wrong and every citation in the UI silently stops resolving.

    pytest test_api.py
"""
from __future__ import annotations

import json

import pytest

import api
import db
import llm
import query
import versions


def test_normalized_score_squashes_the_reranker_logit():
    assert query.normalized_score({"rerank_score": 0.0}) == 0.5
    assert query.normalized_score({"rerank_score": 8.0}) > 0.99
    assert query.normalized_score({"rerank_score": -8.0}) < 0.01
    # Never reranked: the fusion score stands in, and it is already 0..1-ish.
    assert query.normalized_score({"fusion_score": 0.03}) == 0.03
    assert query.normalized_score({}) == 0.0


def _hit(chunk_id: int) -> dict:
    return {
        "chunk_id": chunk_id,
        "source_chunk_id": 100 + chunk_id,
        "article_name": "An Article",
        "article_url": "https://example.com/a",
        "chunk_text": "  some passage  ",
        "rerank_score": 2.0,
        "sources": ["vector", "hq"],
    }


def test_source_ids_are_the_labels_the_prompt_cites():
    versions.use("structured")
    result = {"knowledge": [_hit(1), _hit(2)], "stories": [_hit(3)]}
    sources = api._sources(result)

    assert [s["id"] for s in sources] == ["K1", "K2", "S1"]
    assert sources[0]["chunk"] == 101
    assert sources[0]["content"] == "some passage"  # stripped for the panel
    assert sources[0]["arms"] == ["vector", "hq"]
    # A story was attached, not retrieved and not scored.
    assert sources[2]["score"] is None
    assert sources[2]["arms"] == ["linked story"]

    versions.use("normal")
    assert [s["id"] for s in api._sources({"knowledge": [_hit(1)], "stories": []})] == ["P1"]


def test_frames_are_valid_sse():
    frame = api._frame({"token": "héllo"})
    assert frame.startswith("data: ") and frame.endswith("\n\n")
    assert json.loads(frame[6:].strip()) == {"token": "héllo"}


class _PgError(Exception):
    def __init__(self, message: str, sqlstate: str | None):
        super().__init__(message)
        self.sqlstate = sqlstate


def test_a_rejected_connection_does_not_wait_out_the_timeout():
    # Verbatim from psycopg, sqlstate and all: a connection-time failure happens
    # before there is a session, so there is no sqlstate to key off — matching on
    # one is what made the first version of this guard never fire.
    refused = _PgError(
        'connection failed: connection to server at "127.0.0.1", port 5432 failed: '
        'FATAL:  password authentication failed for user "postgres"',
        None,
    )
    with pytest.raises(SystemExit, match="POSTGRES_PORT"):
        db._abort_if_refused(refused)
    with pytest.raises(SystemExit):
        db._abort_if_refused(_PgError('database "normal_chunking" does not exist', "3D000"))

    # Still coming up, or nothing listening yet: keep retrying.
    db._abort_if_refused(_PgError("connection refused", None))
    db._abort_if_refused(_PgError("connection timeout expired", None))


class _HttpError(Exception):
    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status_code = status


def test_a_retired_model_stops_the_run_instead_of_falling_back():
    # Groq retires models. The fallback is usually retired in the same sweep, so a
    # 404 that only warns lets an ingest finish against a dead model and write zero
    # questions while reporting success — which is exactly what happened here.
    with pytest.raises(SystemExit, match="retired"):
        llm._abort_if_fatal(_HttpError("Error code: 404 - model does not exist", 404))
    with pytest.raises(SystemExit, match="rejected the API key"):
        llm._abort_if_fatal(_HttpError("invalid api key", 401))
    # Transient: these must still reach the retry and fallback paths.
    llm._abort_if_fatal(_HttpError("rate limit", 429))
    llm._abort_if_fatal(_HttpError("bad gateway", 502))


def test_out_of_memory_at_startup_says_what_to_close():
    # winerror 1455 arrives buried under forty lines of safetensors traceback and
    # reads like a corrupt download. It is not — it is the commit limit.
    oom = OSError("The paging file is too small for this operation to complete")
    oom.winerror = 1455
    with pytest.raises(SystemExit, match="commit limit"):
        api._abort_if_out_of_memory(oom)
    # A genuine file problem must keep its own traceback.
    missing = OSError("No such file")
    missing.winerror = 2
    api._abort_if_out_of_memory(missing)
    api._abort_if_out_of_memory(OSError("no winerror at all"))


def test_models_endpoint_lists_every_version():
    listed = api.list_models()["models"]
    assert [m["id"] for m in listed] == versions.keys()
    assert all(m["name"] and m["description"] for m in listed)
