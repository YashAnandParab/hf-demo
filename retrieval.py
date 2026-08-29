"""Structured retrieval.

Retrieval is knowledge-only, structurally rather than by convention: story chunks
carry no embedding and are excluded from the full-text index, so no arm can
return one even if it wanted to. Stories are attached afterwards by walking the
knowledge->story link table from whichever knowledge chunks survived reranking.

The reason is that story chunks are long, vivid and lexically rich, so they beat
the terser knowledge chunks in both vector and full-text ranking — exactly the
wrong outcome when the question is asking for the fact. Under this design a story
is never something the retriever chooses; it is context that comes along with the
fact it illustrates.
"""
from __future__ import annotations

import logging

import config
import tracing
from db import fetch_all, to_pgvector

log = logging.getLogger("retrieval")

_SELECT = """
    c.chunk_id,
    c.article_id,
    c.chunk_index,
    c.source_chunk_id,
    c.chunk_text,
    c.content_type,
    a.article_name,
    a.article_url
"""


VECTOR_SQL = f"""
SELECT {_SELECT},
       1 - (c.embedding <=> %s::vector) AS score
FROM structured_chunks c
JOIN articles a USING (article_id)
WHERE c.content_type = 'knowledge' AND c.embedding IS NOT NULL
ORDER BY c.embedding <=> %s::vector
LIMIT %s
"""

FTS_SQL = f"""
SELECT {_SELECT},
       ts_rank_cd(c.search_vector, q.query) AS score
FROM structured_chunks c
JOIN articles a USING (article_id),
     websearch_to_tsquery('english', %s) AS q(query)
WHERE c.content_type = 'knowledge' AND c.search_vector @@ q.query
ORDER BY score DESC
LIMIT %s
"""

# DISTINCT ON collapses a chunk's several questions to its single best match,
# before the outer query re-sorts what survived by score.
HQ_SQL = f"""
SELECT * FROM (
    SELECT DISTINCT ON (c.chunk_id)
           {_SELECT},
           1 - (q.embedding <=> %s::vector) AS score,
           q.question_text AS matched_question
    FROM structured_chunk_questions q
    JOIN structured_chunks c USING (chunk_id)
    JOIN articles a USING (article_id)
    WHERE c.content_type = 'knowledge' AND q.embedding IS NOT NULL
    ORDER BY c.chunk_id, q.embedding <=> %s::vector
) ranked
ORDER BY score DESC
LIMIT %s
"""

# A story may be cited by knowledge chunks in other articles, so this join is not
# constrained to the parent's article.
LINKED_STORIES_SQL = f"""
SELECT {_SELECT},
       l.knowledge_chunk_id,
       l.position
FROM structured_chunk_links l
JOIN structured_chunks c ON c.chunk_id = l.story_chunk_id
JOIN articles a USING (article_id)
WHERE l.knowledge_chunk_id = ANY(%s)
ORDER BY l.knowledge_chunk_id, l.position, c.chunk_index
"""


@tracing.traceable(
    run_type="retriever", name="vector_search", process_inputs=tracing.hide("query_embedding")
)
def vector_search(query_embedding: list[float], top_k: int | None = None) -> list[dict]:
    vec = to_pgvector(query_embedding)
    return fetch_all(VECTOR_SQL, (vec, vec, top_k or config.VECTOR_TOP_K))


@tracing.traceable(run_type="retriever", name="fts_search")
def fts_search(query: str, top_k: int | None = None) -> list[dict]:
    if not query.strip():
        return []
    return fetch_all(FTS_SQL, (query, top_k or config.FTS_TOP_K))


@tracing.traceable(
    run_type="retriever", name="hq_search", process_inputs=tracing.hide("query_embedding")
)
def hq_search(query_embedding: list[float], top_k: int | None = None) -> list[dict]:
    vec = to_pgvector(query_embedding)
    return fetch_all(HQ_SQL, (vec, vec, top_k or config.HQ_TOP_K))


@tracing.traceable(run_type="retriever", name="linked_stories")
def linked_stories(knowledge_chunk_ids: list[int], max_per_chunk: int | None = None) -> list[dict]:
    """Fetch the story chunks cited by the given knowledge chunks.

    One story may be cited by several of the surviving knowledge chunks, so it can
    come back on more than one parent. It is de-duplicated to a single row carrying
    every parent it matched, in `illustrates_chunk_ids` — which is what produces
    the "illustrates K1, K3" label in the context.
    """
    if not knowledge_chunk_ids:
        return []
    max_per_chunk = config.MAX_LINKED_STORIES if max_per_chunk is None else max_per_chunk
    rows = fetch_all(LINKED_STORIES_SQL, (list(knowledge_chunk_ids),))

    per_parent: dict[int, int] = {}
    by_story: dict[int, dict] = {}
    for row in rows:
        parent = row["knowledge_chunk_id"]
        story_id = row["chunk_id"]
        if story_id in by_story:
            by_story[story_id]["illustrates_chunk_ids"].append(parent)
            continue
        if per_parent.get(parent, 0) >= max_per_chunk:
            continue
        per_parent[parent] = per_parent.get(parent, 0) + 1
        row["illustrates_chunk_ids"] = [parent]
        by_story[story_id] = row

    out = list(by_story.values())
    log.debug("attached %d linked stories for %d knowledge chunks", len(out), len(knowledge_chunk_ids))
    return out


def label_stories(stories: list[dict], knowledge: list[dict]) -> list[dict]:
    """Tag each story with the [K*] labels of the knowledge chunks it illustrates."""
    labels = {hit["chunk_id"]: f"K{i}" for i, hit in enumerate(knowledge, start=1)}
    for story in stories:
        marks = [labels[cid] for cid in story.get("illustrates_chunk_ids", []) if cid in labels]
        story["linked_knowledge_label"] = ", ".join(marks)
    return stories


def stats() -> dict:
    return fetch_all(
        """
        SELECT (SELECT count(*) FROM articles)                                          AS articles,
               (SELECT count(*) FROM structured_chunks WHERE content_type='knowledge')   AS knowledge_chunks,
               (SELECT count(*) FROM structured_chunks WHERE content_type='story')       AS story_chunks,
               (SELECT count(*) FROM structured_chunk_links)                             AS story_links,
               (SELECT count(*) FROM structured_chunk_links l
                  JOIN structured_chunks k ON k.chunk_id = l.knowledge_chunk_id
                  JOIN structured_chunks s ON s.chunk_id = l.story_chunk_id
                 WHERE k.article_id <> s.article_id)                                     AS cross_article_links,
               (SELECT count(*) FROM structured_chunks s
                 WHERE s.content_type = 'story'
                   AND NOT EXISTS (SELECT 1 FROM structured_chunk_links l
                                    WHERE l.story_chunk_id = s.chunk_id))                AS orphan_stories,
               (SELECT count(*) FROM structured_chunks
                 WHERE content_type = 'knowledge' AND embedding IS NULL)                 AS missing_embeddings,
               (SELECT count(*) FROM structured_chunk_questions)                         AS questions
        """
    )[0]
