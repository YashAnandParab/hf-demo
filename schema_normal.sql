-- ---------------------------------------------------------------------------
-- Normal chunking schema: articles, chunks, and hypothetical questions.
--
-- This is the flat baseline the structured version is measured against. It is
-- schema.sql with every structural affordance removed:
--
--   * no content_type column      — a chunk is a chunk; nothing is typed
--   * no link table               — nothing points at anything
--   * no CHECK constraints tying embeddings to a type
--   * every chunk is embedded and every chunk is in the full-text index, so
--     every chunk is a retrieval candidate on its own merits
--
-- Lives in its own database (NORMAL_POSTGRES_DB, default `normal_chunking`), so
-- the `articles` table here holds only this version's corpus.
--
-- __EMBED_DIM__ is substituted by db.init_schema() from config.EMBED_DIM before
-- this file is executed. Changing the embedding model therefore means dropping
-- and recreating the tables (ingest.py --version normal --reset).
-- ---------------------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;


CREATE TABLE IF NOT EXISTS articles (
    article_id      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    article_name    TEXT        NOT NULL,
    article_url     TEXT        UNIQUE,
    author          TEXT,
    published_date  DATE,
    site            TEXT,
    full_text       TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS articles_published_date_idx ON articles (published_date DESC);
CREATE INDEX IF NOT EXISTS articles_name_trgm_idx      ON articles USING gin (article_name gin_trgm_ops);


-- ---------------------------------------------------------------------------
-- One undifferentiated chunk table.
--
-- The structured version's `search_vector` is a CASE that produces NULL for a
-- story, keeping stories out of the FTS arm. Here the expression is
-- unconditional: every row gets a tsvector, so every row can be found lexically.
-- Likewise the HNSW index carries no WHERE clause, because no row is exempt from
-- being embedded.
--
-- This is precisely the behaviour the structured version exists to avoid: story
-- chunks are long, vivid and lexically rich, so under this schema they compete
-- with — and often beat — the terser knowledge chunks in both arms.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS normal_chunks (
    chunk_id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    article_id      BIGINT  NOT NULL REFERENCES articles (article_id) ON DELETE CASCADE,
    -- the chunk_id from the source JSON, kept so a hit can be traced back to the
    -- file even though nothing resolves links against it any more
    source_chunk_id INTEGER,
    chunk_index     INTEGER NOT NULL,
    chunk_text      TEXT    NOT NULL,
    token_count     INTEGER,
    embedding       vector(__EMBED_DIM__),
    search_vector   tsvector GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (article_id, chunk_index),
    UNIQUE (article_id, source_chunk_id)
);

CREATE INDEX IF NOT EXISTS normal_chunks_article_idx ON normal_chunks (article_id, chunk_index);
CREATE INDEX IF NOT EXISTS normal_chunks_fts_idx     ON normal_chunks USING gin (search_vector);

CREATE INDEX IF NOT EXISTS normal_chunks_vec_idx ON normal_chunks
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);


-- Hypothetical questions, generated for every chunk here rather than only for
-- knowledge chunks — the HQ arm is a retrieval arm, and in this version every
-- chunk is a retrieval candidate.
CREATE TABLE IF NOT EXISTS normal_chunk_questions (
    question_id   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    chunk_id      BIGINT NOT NULL REFERENCES normal_chunks (chunk_id) ON DELETE CASCADE,
    question_text TEXT   NOT NULL,
    embedding     vector(__EMBED_DIM__),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS normal_questions_chunk_idx ON normal_chunk_questions (chunk_id);
CREATE INDEX IF NOT EXISTS normal_questions_vec_idx   ON normal_chunk_questions
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
