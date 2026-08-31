"""The two RAG versions this project can run, side by side.

    structured  the knowledge/story split — data/chunks.json, database `postgres`
    normal      the flat baseline       — data/chunks_normal.json, database `normal_chunking`

They are deliberately identical everywhere except in structure. Same corpus, same
embedding model, same three retrieval arms, same fusion weights, same reranker,
same LLM. What differs is only:

    structured                              normal
    ----------------------------------      --------------------------------------
    chunks carry content_type               no content_type at all
    only knowledge is retrievable           every chunk is retrievable
    stories attach via the link table       no links, no stories
    prompt separates evidence from example  prompt has one kind of passage

So any difference in the answers is attributable to the structure and nothing
else. That is the only reason the two versions share this much code rather than
being two projects.

Each version lives in its OWN Postgres database, not its own schema or its own
table prefix within one. A shared database would mean one `articles` table holding
both corpora, and the chunk counts, orphan warnings and stats of one version would
be polluted by the other's rows.

The active version is process-wide state, set once at startup by `db.set_version`
(which also drops the open connection, since the two versions are different
databases). Nothing here opens a connection or imports `db`, so the registry can
be read by anything without pulling psycopg in.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import config
import prompts


def with_database(url: str, database: str) -> str:
    """Point a Postgres URL at a different database, leaving the rest intact.

    Rewriting the path rather than rebuilding the URL from the POSTGRES_* parts
    means a hand-written DATABASE_URL keeps its credentials, host, and any query
    parameters (sslmode and friends) when the version switches.
    """
    parts = urlsplit(url)
    return urlunsplit(parts._replace(path="/" + database.lstrip("/")))


@dataclass(frozen=True)
class Version:
    key: str
    label: str
    blurb: str
    database: str
    schema_file: str
    chunks_file: str
    chunk_table: str
    question_table: str
    # Whether chunks are typed and stories exist. Everything version-specific in
    # retrieval, ingest and prompting keys off this one flag.
    has_stories: bool
    system_prompt: str

    @property
    def schema_path(self) -> Path:
        return config.ROOT / self.schema_file

    @property
    def chunks_path(self) -> Path:
        return config.DATA_DIR / self.chunks_file

    @property
    def database_url(self) -> str:
        return with_database(config.DATABASE_URL, self.database)


STRUCTURED = Version(
    key="structured",
    label="Structured RAG",
    blurb="knowledge/story split; only knowledge is retrieved, stories are attached",
    database=config.POSTGRES_DB,
    schema_file="schema.sql",
    chunks_file="chunks.json",
    chunk_table="structured_chunks",
    question_table="structured_chunk_questions",
    has_stories=True,
    system_prompt=prompts.STRUCTURED_SYSTEM,
)

NORMAL = Version(
    key="normal",
    label="Normal chunking",
    blurb="flat baseline; every chunk is retrievable, no story/knowledge distinction",
    database=config.NORMAL_POSTGRES_DB,
    schema_file="schema_normal.sql",
    chunks_file="chunks_normal.json",
    chunk_table="normal_chunks",
    question_table="normal_chunk_questions",
    has_stories=False,
    system_prompt=prompts.NORMAL_SYSTEM,
)

ALL: dict[str, Version] = {v.key: v for v in (STRUCTURED, NORMAL)}

_active: Version = ALL.get(config.DEFAULT_VERSION, STRUCTURED)


def keys() -> list[str]:
    return list(ALL)


def resolve(key: str) -> Version:
    """Look a version up by key, or by unambiguous prefix ('s', 'norm', ...)."""
    key = (key or "").strip().lower()
    if key in ALL:
        return ALL[key]
    matches = [v for k, v in ALL.items() if k.startswith(key)] if key else []
    if len(matches) == 1:
        return matches[0]
    raise SystemExit(f"unknown version {key!r} — choose one of: {', '.join(ALL)}")


def active() -> Version:
    return _active


def use(key: str) -> Version:
    """Set the process-wide version. Callers should use `db.set_version` instead,
    which additionally closes the connection to the previous version's database."""
    global _active
    _active = resolve(key)
    return _active
