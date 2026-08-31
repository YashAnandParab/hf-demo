"""Postgres access. One lazily-opened connection, dict rows, explicit transactions.

Vectors are passed as `'[0.1,0.2,...]'` strings with an explicit `::vector` cast
rather than through an adapter, so nothing beyond psycopg is needed.

Which database is connected to depends on the active version (see versions.py):
the two versions live in separate databases, so switching version has to drop the
open connection. `set_version` is the only supported way to switch — assigning
`versions._active` directly would leave this module holding a connection to the
previous version's database.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import psycopg
from psycopg import sql as pgsql
from psycopg.rows import dict_row

import config
import versions

log = logging.getLogger("db")

_conn: psycopg.Connection | None = None


def connection() -> psycopg.Connection:
    global _conn
    if _conn is None or _conn.closed:
        _conn = psycopg.connect(versions.active().database_url, row_factory=dict_row)
    return _conn


def set_version(key: str) -> versions.Version:
    """Switch the active version, closing any connection to the previous one."""
    current = versions.active()
    version = versions.use(key)
    if version.database != current.database:
        close()
    return version


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


def _maintenance_url() -> str:
    return versions.with_database(config.DATABASE_URL, config.POSTGRES_MAINTENANCE_DB)


def wait_for_db(timeout: float = 30.0) -> None:
    """Retry until Postgres answers, so a just-started server isn't a hard failure.

    Polls the MAINTENANCE database rather than the active version's. The normal
    version's database does not exist until the first ingest creates it, and a
    "database does not exist" error is not something waiting will fix — retrying
    it for 30s only delays a clear message by 30s.

    `connect_timeout` is what makes the deadline mean anything. The loop only
    checks the clock BETWEEN attempts, so an attempt that never returns is an
    attempt the timeout cannot interrupt — and that is the common case on Windows
    after Docker Desktop stops: its port proxy can be left holding 5433, so the
    TCP connect succeeds and then waits forever for a server that is gone.
    Without this the wait hangs indefinitely instead of failing in 30 seconds.
    """
    deadline = time.time() + timeout
    last: Exception | None = None
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        try:
            with psycopg.connect(
                _maintenance_url(), connect_timeout=max(2, int(remaining))
            ) as conn:
                conn.execute("SELECT 1")
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.0)
    raise SystemExit(
        f"could not reach Postgres at {_safe_url()} after {timeout:.0f}s: {last}"
    )


def database_exists(name: str | None = None) -> bool:
    name = name or versions.active().database
    with psycopg.connect(_maintenance_url(), row_factory=dict_row) as conn:
        row = conn.execute("SELECT 1 AS ok FROM pg_database WHERE datname = %s", (name,)).fetchone()
    return bool(row)


def ensure_database() -> None:
    """Create the active version's database if it is not there yet.

    Each version owns a database, and the normal-chunking one will not exist on a
    machine that has only ever run the structured version. CREATE DATABASE cannot
    run inside a transaction, hence autocommit on a separate connection to the
    maintenance database.
    """
    name = versions.active().database
    if database_exists(name):
        return
    with psycopg.connect(_maintenance_url(), autocommit=True) as conn:
        # The name comes from config, not from user input, but it still has to be
        # a quoted identifier — a database called `normal-chunking` is otherwise a
        # syntax error rather than a database.
        conn.execute(pgsql.SQL("CREATE DATABASE {}").format(pgsql.Identifier(name)))
    log.warning("created database %r for the %r version", name, versions.active().key)


def _safe_url() -> str:
    url = versions.active().database_url
    if "@" in url and "//" in url:
        head, tail = url.split("//", 1)
        creds, host = tail.split("@", 1)
        user = creds.split(":", 1)[0]
        return f"{head}//{user}:***@{host}"
    return url


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

_DROP_SQL = {
    "structured": """
        DROP VIEW  IF EXISTS structured_story_links      CASCADE;
        DROP TABLE IF EXISTS structured_chunk_questions  CASCADE;
        DROP TABLE IF EXISTS structured_chunk_links      CASCADE;
        DROP TABLE IF EXISTS structured_chunks           CASCADE;
        DROP TABLE IF EXISTS articles                    CASCADE;
    """,
    "normal": """
        DROP TABLE IF EXISTS normal_chunk_questions  CASCADE;
        DROP TABLE IF EXISTS normal_chunks           CASCADE;
        DROP TABLE IF EXISTS articles                CASCADE;
    """,
}


def init_schema() -> None:
    version = versions.active()
    sql = version.schema_path.read_text(encoding="utf-8").replace(
        "__EMBED_DIM__", str(config.EMBED_DIM)
    )
    with cursor() as cur:
        cur.execute(sql)
    log.info(
        "%s schema ready in database %r (vector dimension %d)",
        version.key, version.database, config.EMBED_DIM,
    )


def reset_schema() -> None:
    version = versions.active()
    with cursor() as cur:
        cur.execute(_DROP_SQL[version.key])
    log.warning("dropped all %s tables in database %r", version.key, version.database)
    init_schema()


def tables_exist() -> bool:
    table = versions.active().chunk_table
    return bool(fetch_all("SELECT to_regclass(%s) AS t", (f"public.{table}",))[0]["t"])


def check_schema_version() -> None:
    """Refuse to run against tables built for the story->knowledge link direction.

    `init_schema()` is all CREATE ... IF NOT EXISTS, so it is a no-op against an
    existing database and would leave the old columns in place. Without this the
    first insert fails on a missing column, several statements into a run that has
    already deleted rows.

    Structured only: the normal version has no links to have got backwards, and
    its tables have never had another shape.
    """
    if not versions.active().has_stories or not tables_exist():
        return
    legacy = fetch_all(
        """
        SELECT
          EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'structured_chunks'
                     AND column_name = 'story_summary')       AS has_summary,
          NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name = 'structured_chunk_links'
                         AND column_name = 'knowledge_chunk_id') AS missing_link_column
        """
    )[0]
    if legacy["has_summary"] or legacy["missing_link_column"]:
        raise SystemExit(
            "These tables were built for the old schema (story summaries, and links\n"
            "declared story -> knowledge). The current schema drops story summaries and\n"
            "story embeddings entirely and declares links knowledge -> story, so the two\n"
            "cannot coexist.\n"
            "  Re-run with --reset to drop and rebuild (this deletes all ingested chunks):\n"
            "      python ingest.py data/chunks.json --reset"
        )


def check_embed_dim() -> None:
    """Fail loudly when the table was built for a different embedding model."""
    rows = fetch_all(
        """
        SELECT a.atttypmod AS dim
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        WHERE c.relname = %s AND a.attname = 'embedding'
        """,
        (versions.active().chunk_table,),
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
