"""Ingest hand-written flat chunk JSON into the structured RAG tables.

    python ingest.py data/chunks.json
    python ingest.py data/chunks.json --dry-run     # parse + audit, writes nothing
    python ingest.py data/chunks.json --reset       # drop the tables first

Ordering is two-pass, because a story's links point at knowledge chunks that must
exist before they can be referenced:

    1. insert all knowledge chunks, recording source_chunk_id -> db chunk_id
    2. insert all story chunks
    3. resolve links through that map into structured_chunk_links

Your hand-assigned `chunk_id` is stored as `source_chunk_id`, never as the primary
key — the database assigns its own. So author id 12 becoming DB id 1000 is handled,
and nothing silently points at the wrong row.

Embedding rule:
    knowledge -> embed chunk_text
    story     -> embed story_summary  (generated if absent; falls back to the text)

Re-ingesting an article deletes its existing chunks first, so this is idempotent.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import config
import prompts
from loader import RawChunk, audit, count_tokens, group_by_article, load_chunks

# db / models / llm are imported inside the functions that need them, so that
# `--dry-run` works with nothing installed beyond the standard library.

log = logging.getLogger("ingest")

_INSERT_CHUNK_SQL = """
    INSERT INTO structured_chunks
        (article_id, source_chunk_id, chunk_index, chunk_text,
         content_type, story_summary, token_count, embedding)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
    RETURNING chunk_id
"""


def summarise_story(story_text: str, knowledge_context: str) -> str:
    from llm import chat

    try:
        return chat(
            prompts.SUMMARY_SYSTEM,
            prompts.build_summary_prompt(story_text, knowledge_context),
            max_tokens=300,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("summary generation failed: %s", exc)
        return ""


def ingest_group(chunks: list[RawChunk], *, summarise: bool, with_questions: bool) -> dict:
    import db
    import models

    # keep the author's own ordering; chunks without an id go last
    chunks = sorted(chunks, key=lambda c: (c.source_chunk_id is None, c.source_chunk_id or 0))
    head = chunks[0]

    article_id = db.upsert_article(
        article_name=head.article_name,
        full_text="\n\n".join(c.chunk_text for c in chunks),
        article_url=head.article_url,
        author=head.author,
        published_date=head.published_date,
        site=head.site,
    )
    with db.cursor() as cur:
        cur.execute("DELETE FROM structured_chunks WHERE article_id = %s", (article_id,))

    knowledge = [c for c in chunks if c.content_type == "knowledge"]
    stories = [c for c in chunks if c.content_type == "story"]
    by_source_id = {c.source_chunk_id: c for c in chunks if c.source_chunk_id is not None}

    # ---- story summaries -------------------------------------------------
    # A story's embedding is its summary's embedding. Where no summary was
    # supplied, generate one from the story plus the knowledge it illustrates.
    for story in stories:
        if story.story_summary or not summarise:
            continue
        context = " ".join(by_source_id[t].chunk_text for t in story.links if t in by_source_id)
        story.story_summary = summarise_story(story.chunk_text, context) or None

    without_summary = sum(1 for s in stories if not s.story_summary)
    if without_summary:
        log.warning(
            "%d story chunk(s) have no summary; embedding their full text instead "
            "(retrieval will match on narrative wording, not on the point they make)",
            without_summary,
        )

    # ---- embeddings ------------------------------------------------------
    k_texts = [c.chunk_text for c in knowledge]
    s_texts = [c.story_summary or c.chunk_text for c in stories]
    vectors = models.embed_documents(k_texts + s_texts)
    k_vectors = vectors[: len(k_texts)]
    s_vectors = vectors[len(k_texts) :]

    source_to_db: dict[int, int] = {}
    link_rows: list[tuple[int, int, int]] = []
    chunk_index = 0

    with db.cursor() as cur:
        # ---- pass 1: knowledge -------------------------------------------
        for chunk, vector in zip(knowledge, k_vectors):
            cur.execute(
                _INSERT_CHUNK_SQL,
                (
                    article_id,
                    chunk.source_chunk_id,
                    chunk_index,
                    chunk.chunk_text,
                    "knowledge",
                    None,
                    count_tokens(chunk.chunk_text),
                    db.to_pgvector(vector),
                ),
            )
            db_id = int(cur.fetchone()["chunk_id"])
            if chunk.source_chunk_id is not None:
                source_to_db[chunk.source_chunk_id] = db_id
            chunk_index += 1

        # ---- pass 2: stories, now that knowledge ids exist ----------------
        story_db_ids: list[int] = []
        for chunk, vector in zip(stories, s_vectors):
            cur.execute(
                _INSERT_CHUNK_SQL,
                (
                    article_id,
                    chunk.source_chunk_id,
                    chunk_index,
                    chunk.chunk_text,
                    "story",
                    chunk.story_summary,
                    count_tokens(chunk.chunk_text),
                    db.to_pgvector(vector),
                ),
            )
            db_id = int(cur.fetchone()["chunk_id"])
            story_db_ids.append(db_id)
            if chunk.source_chunk_id is not None:
                source_to_db[chunk.source_chunk_id] = db_id
            chunk_index += 1

        # ---- pass 3: links -----------------------------------------------
        dropped = 0
        for chunk, db_id in zip(stories, story_db_ids):
            for position, target in enumerate(chunk.links):
                target_db = source_to_db.get(target)
                target_raw = by_source_id.get(target)
                if target_db is None or target_raw is None or target_raw.content_type != "knowledge":
                    dropped += 1
                    continue
                link_rows.append((db_id, target_db, position))
        if link_rows:
            cur.executemany(
                """
                INSERT INTO structured_chunk_links (story_chunk_id, knowledge_chunk_id, position)
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                link_rows,
            )
        if dropped:
            log.warning("dropped %d unresolvable link(s) in '%s'", dropped, head.article_name[:50])

    question_count = 0
    if with_questions and config.HQ_ENABLED:
        question_count = ingest_questions(knowledge, stories, source_to_db)

    log.info(
        "article %s '%s': %d knowledge, %d story, %d links, %d questions",
        article_id, head.article_name[:50], len(knowledge), len(stories), len(link_rows), question_count,
    )
    return {
        "article_id": article_id,
        "knowledge": len(knowledge),
        "stories": len(stories),
        "links": len(link_rows),
        "questions": question_count,
    }


def ingest_questions(
    knowledge: list[RawChunk], stories: list[RawChunk], source_to_db: dict[int, int]
) -> int:
    """Generate hypothetical questions for the chunks the HQ arm can actually reach.

    `hq_search` applies the same content_type filter as the other two arms, so under
    the default STORY_RETRIEVAL_MODE=knowledge_only a story's questions could never
    match anything — generating them costs one LLM call plus an embedding per story
    for rows nothing can retrieve. So stories only get questions in `include` mode,
    where they compete in the arms directly.
    """
    import db
    import hq
    import models

    targets: list[tuple[int, str]] = []
    for chunk in knowledge:
        db_id = source_to_db.get(chunk.source_chunk_id)
        if db_id:
            targets.append((db_id, chunk.chunk_text))

    stories_included = config.STORY_RETRIEVAL_MODE == "include"
    if stories_included:
        # a story is indexed under its summary, so that is what it should be asked about
        for chunk in stories:
            db_id = source_to_db.get(chunk.source_chunk_id)
            if db_id:
                targets.append((db_id, chunk.story_summary or chunk.chunk_text))
    elif stories:
        log.info(
            "skipping questions for %d story chunk(s): STORY_RETRIEVAL_MODE=%s means the "
            "HQ arm never sees them",
            len(stories),
            config.STORY_RETRIEVAL_MODE,
        )

    if not targets:
        return 0

    per_chunk = hq.generate_questions_bulk([text for _, text in targets])
    pairs = [
        (db_id, question)
        for (db_id, _), questions in zip(targets, per_chunk)
        for question in questions
    ]
    if not pairs:
        return 0

    vectors = models.embed_documents([q for _, q in pairs])
    with db.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO structured_chunk_questions (chunk_id, question_text, embedding)
            VALUES (%s, %s, %s::vector)
            """,
            [(cid, q, db.to_pgvector(v)) for (cid, q), v in zip(pairs, vectors)],
        )
    return len(pairs)


# --------------------------------------------------------------------------- #
# Dry run
# --------------------------------------------------------------------------- #


def print_dry_run(groups: dict[str, list[RawChunk]], report) -> None:
    print("\n--- dry run, nothing written ---")
    for key, group in groups.items():
        head = group[0]
        print(f"\n{head.article_name}  ({key})")
        for c in sorted(group, key=lambda c: c.source_chunk_id or 0):
            links = f" -> {c.links}" if c.links else ""
            summary = "  [has summary]" if c.story_summary else ""
            print(
                f"  #{str(c.source_chunk_id):<5} {c.content_type:<9} "
                f"{count_tokens(c.chunk_text):>4} tok{links}{summary}"
            )
    if report.errors:
        print(f"\n{len(report.errors)} error(s) — fix these before ingesting.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest hand-written flat chunk JSON")
    parser.add_argument(
        "input",
        nargs="?",
        default=str(config.DATA_DIR / "chunks.json"),
        help="chunk JSON file or a directory of them (default: data/chunks.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="parse, normalise and audit only — no database writes, no model loading",
    )
    parser.add_argument("--reset", action="store_true", help="DROP and recreate all tables first")
    parser.add_argument(
        "--no-summaries",
        action="store_true",
        help="do not generate story summaries; embed the story text directly",
    )
    parser.add_argument("--no-questions", action="store_true", help="skip hypothetical questions")
    args = parser.parse_args()

    config.setup_logging()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"input path not found: {path}")

    chunks, report = load_chunks(path)
    audit(chunks, report)
    report.log()

    if not chunks:
        raise SystemExit("no usable chunks found — see the errors above")

    groups = group_by_article(chunks)
    log.info(
        "%d usable chunk(s) across %d article(s): %d knowledge, %d story",
        len(chunks),
        len(groups),
        sum(1 for c in chunks if c.content_type == "knowledge"),
        sum(1 for c in chunks if c.content_type == "story"),
    )

    if args.dry_run:
        print_dry_run(groups, report)
        return

    import db
    import models
    import retrieval

    db.wait_for_db()
    if args.reset:
        db.reset_schema()
    else:
        db.init_schema()
        db.check_embed_dim()

    # Catch a model/schema mismatch before spending time embedding everything.
    actual_dim = models.embedding_dimension()
    if actual_dim != config.EMBED_DIM:
        raise SystemExit(
            f"{config.EMBED_MODEL} produces {actual_dim}-dim vectors but EMBED_DIM is "
            f"{config.EMBED_DIM}. Set EMBED_DIM={actual_dim} and re-run with --reset."
        )

    totals = {"articles": 0, "knowledge": 0, "stories": 0, "links": 0, "questions": 0}
    for group in groups.values():
        result = ingest_group(
            group,
            summarise=not args.no_summaries,
            with_questions=not args.no_questions,
        )
        totals["articles"] += 1
        for key in ("knowledge", "stories", "links", "questions"):
            totals[key] += result[key]

    log.info(
        "done: %d articles, %d knowledge, %d story, %d links, %d questions",
        totals["articles"], totals["knowledge"], totals["stories"], totals["links"], totals["questions"],
    )

    counts = retrieval.stats()
    print("\n--- database ---")
    for key, value in counts.items():
        print(f"  {key:<20} {value}")
    if counts["orphan_stories"]:
        print(
            "\n  note: orphan stories have no link to any knowledge chunk, so under the\n"
            "  default knowledge-first retrieval nothing will ever pull them in."
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    finally:
        if "db" in sys.modules:
            sys.modules["db"].close()
