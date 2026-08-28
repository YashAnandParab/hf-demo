"""Structured retrieval.

Retrieval is knowledge-first: all three arms are restricted to knowledge chunks,
and story chunks are attached afterwards through the story->knowledge link table.
This keeps narrative from crowding out the factual passages during fusion, while
still making the story available to the LLM as illustration.

Story chunks are long, vivid and lexically rich, so they beat the terser
knowledge chunks in both vector and full-text ranking — which is exactly the
wrong outcome when the question is asking for the fact. Set
STORY_RETRIEVAL_MODE=include to let them compete anyway.
"""
from __future__ import annotations

import logging

import config
from db import fetch_all, to_pgvector

log = logging.getLogger("retrieval")

_SELECT = """
    c.chunk_id,
    c.article_id,
    c.chunk_index,
    c.source_chunk_id,
    c.chunk_text,
    c.content_type,
    c.story_summary,
    a.article_name,
    a.article_url
"""


def _type_filter() -> str:
    """SQL predicate limiting the arms to knowledge, unless stories are included."""
    if config.STORY_RETRIEVAL_MODE == "include":
        return "TRUE"
    return "c.content_type = 'knowledge'"


VECTOR_SQL = f"""
SELECT {_SELECT},
       1 - (c.embedding <=> %s::vector) AS score
FROM structured_chunks c
JOIN articles a USING (article_id)
WHERE {{type_filter}} AND c.embedding IS NOT NULL
ORDER BY c.embedding <=> %s::vector
LIMIT %s
"""

FTS_SQL = f"""
SELECT {_SELECT},
       ts_rank_cd(c.search_vector, q.query) AS score
FROM structured_chunks c
JOIN articles a USING (article_id),
     websearch_to_tsquery('english', %s) AS q(query)
WHERE {{type_filter}} AND c.search_vector @@ q.query
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
    WHERE {{type_filter}} AND q.embedding IS NOT NULL
    ORDER BY c.chunk_id, q.embedding <=> %s::vector
) ranked
ORDER BY score DESC
LIMIT %s
"""

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

STORY_VECTOR_SQL = f"""
SELECT {_SELECT},
       1 - (c.embedding <=> %s::vector) AS score
FROM structured_chunks c
JOIN articles a USING (article_id)
WHERE c.content_type = 'story' AND c.embedding IS NOT NULL
ORDER BY c.embedding <=> %s::vector
LIMIT %s
"""


def vector_search(query_embedding: list[float], top_k: int | None = None) -> list[dict]:
    vec = to_pgvector(query_embedding)
    sql = VECTOR_SQL.format(type_filter=_type_filter())
    return fetch_all(sql, (vec, vec, top_k or config.VECTOR_TOP_K))


def fts_search(query: str, top_k: int | None = None) -> list[dict]:
    if not query.strip():
        return []
    sql = FTS_SQL.format(type_filter=_type_filter())
    return fetch_all(sql, (query, top_k or config.FTS_TOP_K))


def hq_search(query_embedding: list[float], top_k: int | None = None) -> list[dict]:
    vec = to_pgvector(query_embedding)
    sql = HQ_SQL.format(type_filter=_type_filter())
    return fetch_all(sql, (vec, vec, top_k or config.HQ_TOP_K))


def linked_stories(knowledge_chunk_ids: list[int], max_per_chunk: int | None = None) -> list[dict]:
    """Fetch story chunks attached to the given knowledge chunks.

    A story may illustrate several knowledge chunks, so it can come back on more
    than one parent. It is de-duplicated to a single row carrying every parent it
    matched, in `illustrates_chunk_ids` — which is what produces the
    "illustrates K1, K3" label in the context.
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


def story_search(query_embedding: list[float], top_k: int | None = None) -> list[dict]:
    """Direct similarity search over story embeddings. For inspection only."""
    vec = to_pgvector(query_embedding)
    return fetch_all(STORY_VECTOR_SQL, (vec, vec, top_k or config.VECTOR_TOP_K))


def stats() -> dict:
    return fetch_all(
        """
        SELECT (SELECT count(*) FROM articles)                                       AS articles,
               (SELECT count(*) FROM structured_chunks WHERE content_type='knowledge') AS knowledge_chunks,
               (SELECT count(*) FROM structured_chunks WHERE content_type='story')     AS story_chunks,
               (SELECT count(*) FROM structured_chunk_links)                           AS story_links,
               (SELECT count(*) FROM structured_chunks s
                 WHERE s.content_type = 'story'
                   AND NOT EXISTS (SELECT 1 FROM structured_chunk_links l
                                    WHERE l.story_chunk_id = s.chunk_id))              AS orphan_stories,
               (SELECT count(*) FROM structured_chunks WHERE embedding IS NULL)        AS missing_embeddings,
               (SELECT count(*) FROM structured_chunk_questions)                       AS questions,
               (SELECT count(*) FROM structured_chunk_questions q
                  JOIN structured_chunks c USING (chunk_id)
                 WHERE c.content_type = 'story')                                       AS story_questions
        """
    )[0]
