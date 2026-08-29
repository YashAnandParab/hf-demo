-- ---------------------------------------------------------------------------
-- Structured RAG schema: articles, knowledge/story chunks, knowledge->story
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
-- Only knowledge chunks are retrievable. They alone carry an embedding and are
-- alone in the full-text index; a story has neither and can only reach the model
-- by hanging off a knowledge chunk that survived reranking. The two CHECKs below
-- make that an invariant of the table rather than a convention in the ingester.
--
-- A knowledge chunk may cite SEVERAL stories, and one story may illustrate
-- several knowledge chunks (possibly in other articles), so the link is a
-- many-to-many join table rather than a column on the chunk.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS structured_chunks (
    chunk_id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    article_id      BIGINT  NOT NULL REFERENCES articles (article_id) ON DELETE CASCADE,
    -- the chunk_id you assigned by hand in the source JSON, unique per article
    -- and what the knowledge -> story links are resolved against at ingest time
    source_chunk_id INTEGER,
    chunk_index     INTEGER NOT NULL,
    chunk_text      TEXT    NOT NULL,
    content_type    TEXT    NOT NULL CHECK (content_type IN ('knowledge', 'story')),
    token_count     INTEGER,
    -- knowledge only. A story is never a retrieval candidate, so it has no vector.
    embedding       vector(__EMBED_DIM__),
    -- knowledge only, for the same reason: a story must not surface in the FTS arm
    search_vector   tsvector GENERATED ALWAYS AS (
                        CASE WHEN content_type = 'knowledge'
                             THEN to_tsvector('english', chunk_text)
                        END
                    ) STORED,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (article_id, chunk_index),
    UNIQUE (article_id, source_chunk_id),
    CONSTRAINT story_has_no_embedding
        CHECK (content_type <> 'story' OR embedding IS NULL)
);

CREATE INDEX IF NOT EXISTS structured_chunks_article_idx ON structured_chunks (article_id, chunk_index);
CREATE INDEX IF NOT EXISTS structured_chunks_type_idx    ON structured_chunks (content_type);
CREATE INDEX IF NOT EXISTS structured_chunks_fts_idx     ON structured_chunks USING gin (search_vector);

-- Knowledge-only HNSW index: it is the only content_type that has vectors at all.
CREATE INDEX IF NOT EXISTS structured_chunks_vec_knowledge_idx ON structured_chunks
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    WHERE content_type = 'knowledge';


-- `position` is the order the story was listed in on the knowledge chunk's
-- related_story_chunk_ids, so MAX_LINKED_STORIES trims from the author's own
-- ordering rather than arbitrarily.
CREATE TABLE IF NOT EXISTS structured_chunk_links (
    knowledge_chunk_id BIGINT NOT NULL REFERENCES structured_chunks (chunk_id) ON DELETE CASCADE,
    story_chunk_id     BIGINT NOT NULL REFERENCES structured_chunks (chunk_id) ON DELETE CASCADE,
    position           INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (knowledge_chunk_id, story_chunk_id)
);

CREATE INDEX IF NOT EXISTS structured_links_knowledge_idx ON structured_chunk_links (knowledge_chunk_id, position);
CREATE INDEX IF NOT EXISTS structured_links_story_idx     ON structured_chunk_links (story_chunk_id);


-- Hypothetical questions exist only for knowledge chunks: the HQ arm is a
-- retrieval arm, and stories are not retrieval candidates.
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


-- Convenience view: every knowledge chunk with the stories it cites.
CREATE OR REPLACE VIEW structured_story_links AS
SELECT k.chunk_id      AS knowledge_chunk_id,
       k.article_id    AS knowledge_article_id,
       k.chunk_text    AS knowledge_text,
       s.chunk_id      AS story_chunk_id,
       s.article_id    AS story_article_id,
       s.chunk_text    AS story_text,
       l.position
FROM structured_chunks k
JOIN structured_chunk_links l ON l.knowledge_chunk_id = k.chunk_id
JOIN structured_chunks s      ON s.chunk_id = l.story_chunk_id
WHERE k.content_type = 'knowledge';
