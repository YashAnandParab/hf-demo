"""Retrieval, for both versions.

The three arms — vector, full-text, hypothetical-question — are identical in both.
What differs is what they are allowed to return, and that difference is in the
schema rather than in this file:

  structured  Story chunks carry no embedding and are excluded from the full-text
              index, so no arm can return one even if it wanted to. Stories are
              attached afterwards by walking the knowledge->story link table from
              whichever knowledge chunks survived reranking.

              The reason is that story chunks are long, vivid and lexically rich,
              so they beat the terser knowledge chunks in both vector and
              full-text ranking — exactly the wrong outcome when the question is
              asking for the fact. Under this design a story is never something
              the retriever chooses; it is context that comes along with the fact
              it illustrates.

  normal      Every chunk is embedded and indexed, so every chunk competes. There
              is nothing to attach afterwards, because nothing points at anything.
              This is the baseline, and the behaviour described above — narrative
              chunks crowding out the terser ones — is what it is there to show.

The SQL is built per version and cached, because the table names differ and the
structured version needs a `content_type = 'knowledge'` filter that would be a
missing-column error against the flat schema.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import config
import tracing
import versions
from db import fetch_all, to_pgvector

log = logging.getLogger("retrieval")


class _Queries:
    """The four statements, resolved for one version's tables."""

    def __init__(self, version: versions.Version):
        chunks = version.chunk_table
        questions = version.question_table

        # `content_type` exists only in the structured schema, so it can be
        # selected and filtered on only there.
        type_column = "\n    c.content_type," if version.has_stories else ""
        knowledge_only = "c.content_type = 'knowledge' AND " if version.has_stories else ""

        select = f"""
    c.chunk_id,
    c.article_id,
    c.chunk_index,
    c.source_chunk_id,
    c.chunk_text,{type_column}
    a.article_name,
    a.article_url
"""

        self.vector = f"""
SELECT {select},
       1 - (c.embedding <=> %s::vector) AS score
FROM {chunks} c
JOIN articles a USING (article_id)
WHERE {knowledge_only}c.embedding IS NOT NULL
ORDER BY c.embedding <=> %s::vector
LIMIT %s
"""

        # `websearch_to_tsquery` ANDs every lexeme, so a natural-language question
        # only matches a chunk containing all of it — "What is the role of the
        # Chief Investment Officer at a mutual fund company?" becomes
        # role & chief & invest & offic & mutual & fund & compani, which nothing
        # in a prose corpus satisfies. Precise when it fires, so it is still tried
        # first; `fts_any` below is the recall fallback.
        self.fts = f"""
SELECT {select},
       ts_rank_cd(c.search_vector, q.query) AS score
FROM {chunks} c
JOIN articles a USING (article_id),
     websearch_to_tsquery('english', %s) AS q(query)
WHERE {knowledge_only}c.search_vector @@ q.query
ORDER BY score DESC
LIMIT %s
"""

        # The same statement with the question's lexemes ORed instead of ANDed.
        # Stemming and stop-word removal come from running the question through
        # to_tsvector first, so the query side is lexeme-for-lexeme what the
        # indexed side holds. ts_rank_cd then does the discriminating: a chunk
        # matching one common term ranks far below one matching several.
        self.fts_any = f"""
SELECT {select},
       ts_rank_cd(c.search_vector, q.query) AS score
FROM {chunks} c
JOIN articles a USING (article_id),
     to_tsquery('english',
                array_to_string(tsvector_to_array(to_tsvector('english', %s)), ' | ')
     ) AS q(query)
WHERE {knowledge_only}c.search_vector @@ q.query
ORDER BY score DESC
LIMIT %s
"""

        # DISTINCT ON collapses a chunk's several questions to its single best
        # match, before the outer query re-sorts what survived by score.
        self.hq = f"""
SELECT * FROM (
    SELECT DISTINCT ON (c.chunk_id)
           {select},
           1 - (q.embedding <=> %s::vector) AS score,
           q.question_text AS matched_question
    FROM {questions} q
    JOIN {chunks} c USING (chunk_id)
    JOIN articles a USING (article_id)
    WHERE {knowledge_only}q.embedding IS NOT NULL
    ORDER BY c.chunk_id, q.embedding <=> %s::vector
) ranked
ORDER BY score DESC
LIMIT %s
"""

        # A story may be cited by knowledge chunks in other articles, so this join
        # is not constrained to the parent's article. Structured only.
        self.linked_stories = f"""
SELECT {select},
       l.knowledge_chunk_id,
       l.position
FROM structured_chunk_links l
JOIN {chunks} c ON c.chunk_id = l.story_chunk_id
JOIN articles a USING (article_id)
WHERE l.knowledge_chunk_id = ANY(%s)
ORDER BY l.knowledge_chunk_id, l.position, c.chunk_index
""" if version.has_stories else ""


@lru_cache(maxsize=len(versions.ALL))
def _queries_for(key: str) -> _Queries:
    return _Queries(versions.ALL[key])


def q() -> _Queries:
    """The statements for whichever version is active right now."""
    return _queries_for(versions.active().key)


@tracing.traceable(
    run_type="retriever", name="vector_search", process_inputs=tracing.hide("query_embedding")
)
def vector_search(query_embedding: list[float], top_k: int | None = None) -> list[dict]:
    vec = to_pgvector(query_embedding)
    return fetch_all(q().vector, (vec, vec, top_k or config.VECTOR_TOP_K))


@tracing.traceable(run_type="retriever", name="fts_search")
def fts_search(query: str, top_k: int | None = None) -> list[dict]:
    """Lexical arm: all-terms first, any-term as a fallback.

    The strict pass is the one worth having — it fires on quoted phrases, proper
    nouns and figures, where every term really does co-occur. It just does not
    fire on questions, which is most of what this arm is asked. Rather than
    loosening it and losing that precision, an empty result falls through to the
    ORed form, so the arm contributes a ranking instead of nothing.
    """
    if not query.strip():
        return []
    top_k = top_k or config.FTS_TOP_K
    hits = fetch_all(q().fts, (query, top_k))
    if hits:
        return hits
    hits = fetch_all(q().fts_any, (query, top_k))
    log.debug("fts: all-terms empty, any-terms returned %d", len(hits))
    return hits


@tracing.traceable(
    run_type="retriever", name="hq_search", process_inputs=tracing.hide("query_embedding")
)
def hq_search(query_embedding: list[float], top_k: int | None = None) -> list[dict]:
    vec = to_pgvector(query_embedding)
    return fetch_all(q().hq, (vec, vec, top_k or config.HQ_TOP_K))


@tracing.traceable(run_type="retriever", name="linked_stories")
def linked_stories(knowledge_chunk_ids: list[int], max_per_chunk: int | None = None) -> list[dict]:
    """Fetch the story chunks cited by the given knowledge chunks.

    One story may be cited by several of the surviving knowledge chunks, so it can
    come back on more than one parent. It is de-duplicated to a single row carrying
    every parent it matched, in `illustrates_chunk_ids` — which is what produces
    the "illustrates K1, K3" label in the context.

    Always empty for the normal version, which has no links.
    """
    if not knowledge_chunk_ids or not versions.active().has_stories:
        return []
    max_per_chunk = config.MAX_LINKED_STORIES if max_per_chunk is None else max_per_chunk
    rows = fetch_all(q().linked_stories, (list(knowledge_chunk_ids),))

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


_STRUCTURED_STATS = """
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

_NORMAL_STATS = """
SELECT (SELECT count(*) FROM articles)                                  AS articles,
       (SELECT count(*) FROM normal_chunks)                             AS chunks,
       (SELECT count(*) FROM normal_chunks WHERE embedding IS NULL)     AS missing_embeddings,
       (SELECT count(*) FROM normal_chunks WHERE search_vector IS NULL) AS missing_search_vectors,
       (SELECT count(*) FROM normal_chunk_questions)                    AS questions
"""


def stats() -> dict:
    """Row counts for the active version. The keys differ between versions —
    'orphan stories' is not a thing that can exist in a schema without links."""
    return fetch_all(_STRUCTURED_STATS if versions.active().has_stories else _NORMAL_STATS)[0]


def retrievable_count(counts: dict | None = None) -> int:
    """How many chunks any arm could return — the one count both versions share."""
    counts = stats() if counts is None else counts
    key = "knowledge_chunks" if versions.active().has_stories else "chunks"
    return int(counts[key])
