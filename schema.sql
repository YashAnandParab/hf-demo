-- ---------------------------------------------------------------------------
-- Structured RAG schema: articles, knowledge/story chunks, story->knowledge
-- links, and hypothetical questions.
--
-- __EMBED_DIM__ is substituted by db.init_schema() from config.EMBED_DIM before
-- this file is executed. Changing the embedding model therefore means dropping
-- and recreating the tables (ingest.py --reset).
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
-- Knowledge and story chunks share one table, discriminated by content_type.
--
-- A story may illustrate SEVERAL knowledge chunks (the Rohit story sits between
-- the "five questions" setup and its payoff), so the link is a many-to-many
-- join table rather than a column on the chunk.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS structured_chunks (
    chunk_id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    article_id      BIGINT  NOT NULL REFERENCES articles (article_id) ON DELETE CASCADE,
    -- the chunk_id you assigned by hand in the source JSON, unique per article
    -- and what the story -> knowledge links are resolved against at ingest time
    source_chunk_id INTEGER,
    chunk_index     INTEGER NOT NULL,
    chunk_text      TEXT    NOT NULL,
    content_type    TEXT    NOT NULL CHECK (content_type IN ('knowledge', 'story')),
    -- stories only. When present it is what gets embedded
    story_summary   TEXT,
    token_count     INTEGER,
    embedding       vector(__EMBED_DIM__),
    search_vector   tsvector GENERATED ALWAYS AS (
                        to_tsvector('english', chunk_text || ' ' || coalesce(story_summary, ''))
                    ) STORED,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (article_id, chunk_index),
    UNIQUE (article_id, source_chunk_id),
    CONSTRAINT knowledge_has_no_summary
        CHECK (content_type <> 'knowledge' OR story_summary IS NULL)
);

CREATE INDEX IF NOT EXISTS structured_chunks_article_idx ON structured_chunks (article_id, chunk_index);
CREATE INDEX IF NOT EXISTS structured_chunks_type_idx    ON structured_chunks (content_type);
CREATE INDEX IF NOT EXISTS structured_chunks_fts_idx     ON structured_chunks USING gin (search_vector);

-- Partial HNSW indexes, one per content_type: a knowledge-only query never has
-- to walk story vectors.
CREATE INDEX IF NOT EXISTS structured_chunks_vec_knowledge_idx ON structured_chunks
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    WHERE content_type = 'knowledge';

CREATE INDEX IF NOT EXISTS structured_chunks_vec_story_idx ON structured_chunks
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    WHERE content_type = 'story';


CREATE TABLE IF NOT EXISTS structured_chunk_links (
    story_chunk_id     BIGINT NOT NULL REFERENCES structured_chunks (chunk_id) ON DELETE CASCADE,
    knowledge_chunk_id BIGINT NOT NULL REFERENCES structured_chunks (chunk_id) ON DELETE CASCADE,
    position           INTEGER NOT NULL DEFAULT 0,  -- order given in the source JSON
    PRIMARY KEY (story_chunk_id, knowledge_chunk_id)
);

CREATE INDEX IF NOT EXISTS structured_links_knowledge_idx ON structured_chunk_links (knowledge_chunk_id);
CREATE INDEX IF NOT EXISTS structured_links_story_idx     ON structured_chunk_links (story_chunk_id);


CREATE TABLE IF NOT EXISTS structured_chunk_questions (
    question_id   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    chunk_id      BIGINT NOT NULL REFERENCES structured_chunks (chunk_id) ON DELETE CASCADE,
    question_text TEXT   NOT NULL,
    embedding     vector(__EMBED_DIM__),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS structured_questions_chunk_idx ON structured_chunk_questions (chunk_id);
CREATE INDEX IF NOT EXISTS structured_questions_vec_idx   ON structured_chunk_questions
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);


-- Convenience view: every story with the knowledge chunks it illustrates.
CREATE OR REPLACE VIEW structured_story_links AS
SELECT s.chunk_id      AS story_chunk_id,
       s.article_id,
       s.chunk_text    AS story_text,
       s.story_summary,
       k.chunk_id      AS knowledge_chunk_id,
       k.chunk_text    AS knowledge_text,
       l.position
FROM structured_chunks s
JOIN structured_chunk_links l ON l.story_chunk_id = s.chunk_id
JOIN structured_chunks k      ON k.chunk_id = l.knowledge_chunk_id
WHERE s.content_type = 'story';
