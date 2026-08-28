"""Postgres access. One lazily-opened connection, dict rows, explicit transactions.

Vectors are passed as `'[0.1,0.2,...]'` strings with an explicit `::vector` cast
rather than through an adapter, so nothing beyond psycopg is needed.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import psycopg
from psycopg.rows import dict_row

import config

log = logging.getLogger("db")

_conn: psycopg.Connection | None = None


def connection() -> psycopg.Connection:
    global _conn
    if _conn is None or _conn.closed:
        _conn = psycopg.connect(config.DATABASE_URL, row_factory=dict_row)
    return _conn


def close() -> None:
    global _conn
    if _conn is not None and not _conn.closed:
        _conn.close()
    _conn = None


@contextmanager
def cursor() -> Iterator[psycopg.Cursor]:
    """A transaction. Commits on clean exit, rolls back on any exception."""
    conn = connection()
    try:
        with conn.cursor() as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def fetch_all(sql: str, params: Sequence[Any] = ()) -> list[dict]:
    with cursor() as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def to_pgvector(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(v):.7g}" for v in vector) + "]"


def wait_for_db(timeout: float = 30.0) -> None:
    """Retry until Postgres answers, so a just-started server isn't a hard failure."""
    deadline = time.time() + timeout
    last: Exception | None = None
    while time.time() < deadline:
        try:
            connection().execute("SELECT 1")
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            close()
            time.sleep(1.0)
    raise SystemExit(
        f"could not reach Postgres at {_safe_url()} after {timeout:.0f}s: {last}"
    )


def _safe_url() -> str:
    url = config.DATABASE_URL
    if "@" in url and "//" in url:
        head, tail = url.split("//", 1)
        creds, host = tail.split("@", 1)
        user = creds.split(":", 1)[0]
        return f"{head}//{user}:***@{host}"
    return url


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

SCHEMA_PATH = config.ROOT / "schema.sql"

_DROP_SQL = """
DROP VIEW  IF EXISTS structured_story_links      CASCADE;
DROP TABLE IF EXISTS structured_chunk_questions  CASCADE;
DROP TABLE IF EXISTS structured_chunk_links      CASCADE;
DROP TABLE IF EXISTS structured_chunks           CASCADE;
DROP TABLE IF EXISTS articles                    CASCADE;
"""


def init_schema() -> None:
    sql = SCHEMA_PATH.read_text(encoding="utf-8").replace(
        "__EMBED_DIM__", str(config.EMBED_DIM)
    )
    with cursor() as cur:
        cur.execute(sql)
    log.info("schema ready (vector dimension %d)", config.EMBED_DIM)


def reset_schema() -> None:
    with cursor() as cur:
        cur.execute(_DROP_SQL)
    log.warning("dropped all structured RAG tables")
    init_schema()


def tables_exist() -> bool:
    return bool(fetch_all("SELECT to_regclass('public.structured_chunks') AS t")[0]["t"])


def check_embed_dim() -> None:
    """Fail loudly when the table was built for a different embedding model."""
    rows = fetch_all(
        """
        SELECT a.atttypmod AS dim
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        WHERE c.relname = 'structured_chunks' AND a.attname = 'embedding'
        """
    )
    if not rows:
        return
    actual = rows[0]["dim"]
    if actual and actual > 0 and actual != config.EMBED_DIM:
        raise SystemExit(
            f"embedding column is vector({actual}) but EMBED_DIM is {config.EMBED_DIM}.\n"
            f"The column width is fixed at table creation — re-run with --reset "
            f"(this deletes all ingested chunks)."
        )


# --------------------------------------------------------------------------- #
# Articles
# --------------------------------------------------------------------------- #


def upsert_article(
    *,
    article_name: str,
    full_text: str,
    article_url: str | None = None,
    author: str | None = None,
    published_date: str | None = None,
    site: str | None = None,
) -> int:
    """Insert or update one article, returning its id.

    A URL is the identity when present. Without one, Postgres' UNIQUE constraint
    is no help (NULL never conflicts), so we match on the exact title instead —
    otherwise every re-ingest of an untitled-URL article would duplicate it.
    """
    with cursor() as cur:
        if article_url:
            cur.execute(
                """
                INSERT INTO articles
                    (article_name, article_url, author, published_date, site, full_text)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (article_url) DO UPDATE SET
                    article_name   = EXCLUDED.article_name,
                    author         = EXCLUDED.author,
                    published_date = EXCLUDED.published_date,
                    site           = EXCLUDED.site,
                    full_text      = EXCLUDED.full_text
                RETURNING article_id
                """,
                (article_name, article_url, author, published_date, site, full_text),
            )
            return int(cur.fetchone()["article_id"])

        cur.execute(
            "SELECT article_id FROM articles WHERE article_url IS NULL AND article_name = %s",
            (article_name,),
        )
        row = cur.fetchone()
        if row:
            article_id = int(row["article_id"])
            cur.execute(
                """
                UPDATE articles
                   SET author = %s, published_date = %s, site = %s, full_text = %s
                 WHERE article_id = %s
                """,
                (author, published_date, site, full_text, article_id),
            )
            return article_id

        cur.execute(
            """
            INSERT INTO articles
                (article_name, article_url, author, published_date, site, full_text)
            VALUES (%s, NULL, %s, %s, %s, %s)
            RETURNING article_id
            """,
            (article_name, author, published_date, site, full_text),
        )
        return int(cur.fetchone()["article_id"])
