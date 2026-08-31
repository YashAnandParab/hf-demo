"""Ingest hand-written flat chunk JSON into one version's tables.

    python ingest.py                                    # structured, data/chunks.json
    python ingest.py --version normal                   # normal, data/chunks_normal.json
    python ingest.py data/chunks.json --dry-run         # parse + audit, writes nothing
    python ingest.py data/chunks.json --reset           # drop the tables first

The input file defaults to whichever file the chosen version owns, so the two are
hard to mix up by accident; passing a path explicitly overrides that.

Each version writes to its OWN database (see versions.py). The normal version's
database is created on first ingest if it does not exist yet.

--- structured ---

Ordering is two-pass and GLOBAL, not per-article:

    1. insert every chunk of every article, recording source_chunk_id -> db chunk_id
    2. resolve all knowledge -> story links through that one map

It has to be global because a knowledge chunk may cite a story in a different
article. Resolving links inside each article group, as an earlier version did,
would silently drop exactly those links.

Your hand-assigned `chunk_id` is stored as `source_chunk_id`, never as the primary
key — the database assigns its own. So author id 12 becoming DB id 1000 is handled,
and nothing silently points at the wrong row.

Embedding rule:
    knowledge -> embed chunk_text
    story     -> no embedding at all

A story is not a retrieval candidate under any setting: it reaches the model only
by being cited by a knowledge chunk that survived reranking. Embedding one would
cost time and space for a vector nothing queries, so the column is left NULL and
the schema enforces it.

Re-ingesting an article deletes its existing chunks first, so this is idempotent.
Note that re-ingesting only PART of a corpus will drop cross-article links whose
story lives in an article that was not part of this run — ingest the whole file.

--- normal ---

One pass, no links: every chunk is embedded, indexed and given hypothetical
questions, because in the flat version every chunk is a retrieval candidate.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import config
import tracing
import versions
from loader import RawChunk, audit, count_tokens, group_by_article, load_chunks

# db / models / llm are imported inside the functions that need them, so that
# `--dry-run` works with nothing installed beyond the standard library.

log = logging.getLogger("ingest")

_INSERT_CHUNK_SQL = """
    INSERT INTO structured_chunks
        (article_id, source_chunk_id, chunk_index, chunk_text,
         content_type, token_count, embedding)
    VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
    RETURNING chunk_id
"""

_INSERT_NORMAL_CHUNK_SQL = """
    INSERT INTO normal_chunks
        (article_id, source_chunk_id, chunk_index, chunk_text, token_count, embedding)
    VALUES (%s, %s, %s, %s, %s, %s::vector)
    RETURNING chunk_id
"""


@tracing.traceable(run_type="chain", name="ingest")
def ingest_all(groups: dict[str, list[RawChunk]], *, with_questions: bool) -> dict:
    """Insert every article, then resolve every link across all of them."""
    import db
    import models

    # ---- articles, and the per-article chunk ordering --------------------
    prepared: list[tuple[int, list[RawChunk]]] = []
    for group in groups.values():
        # keep the author's own ordering; chunks without an id go last
        chunks = sorted(group, key=lambda c: (c.source_chunk_id is None, c.source_chunk_id or 0))
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
        prepared.append((article_id, chunks))

    # ---- embeddings: knowledge only, one batch for the whole corpus ------
    # Iterated in exactly the order the insert loop below will hit them, so the
    # vectors line up without needing a key.
    knowledge = [c for _, chunks in prepared for c in chunks if c.content_type == "knowledge"]
    stories = [c for _, chunks in prepared for c in chunks if c.content_type == "story"]
    log.info("embedding %d knowledge chunk(s); %d story chunk(s) are not embedded",
             len(knowledge), len(stories))
    vectors = iter(models.embed_documents([c.chunk_text for c in knowledge]))

    # ---- pass 1: insert every chunk --------------------------------------
    source_to_db: dict[int, int] = {}
    inserted: list[tuple[RawChunk, int]] = []

    with db.cursor() as cur:
        for article_id, chunks in prepared:
            for chunk_index, chunk in enumerate(chunks):
                is_knowledge = chunk.content_type == "knowledge"
                cur.execute(
                    _INSERT_CHUNK_SQL,
                    (
                        article_id,
                        chunk.source_chunk_id,
                        chunk_index,
                        chunk.chunk_text,
                        chunk.content_type,
                        count_tokens(chunk.chunk_text),
                        db.to_pgvector(next(vectors)) if is_knowledge else None,
                    ),
                )
                db_id = int(cur.fetchone()["chunk_id"])
                inserted.append((chunk, db_id))
                if chunk.source_chunk_id is not None:
                    if chunk.source_chunk_id in source_to_db:
                        log.warning(
                            "chunk_id %s appears more than once across the corpus; links to it "
                            "will resolve to the last one inserted",
                            chunk.source_chunk_id,
                        )
                    source_to_db[chunk.source_chunk_id] = db_id

    # ---- pass 2: links, now that every id in the corpus is known ---------
    by_source = {c.source_chunk_id: c for c, _ in inserted if c.source_chunk_id is not None}
    link_rows: list[tuple[int, int, int]] = []
    dropped = 0

    for chunk, db_id in inserted:
        if chunk.content_type != "knowledge":
            continue
        for position, target in enumerate(chunk.links):
            target_db = source_to_db.get(target)
            target_raw = by_source.get(target)
            if target_db is None or target_raw is None or target_raw.content_type != "story":
                dropped += 1
                continue
            link_rows.append((db_id, target_db, position))

    if link_rows:
        with db.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO structured_chunk_links (knowledge_chunk_id, story_chunk_id, position)
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                link_rows,
            )
    if dropped:
        log.warning("dropped %d unresolvable link(s) — see the loader warnings above", dropped)

    # ---- pass 3: hypothetical questions ----------------------------------
    question_count = 0
    if with_questions and config.HQ_ENABLED:
        question_count = ingest_questions(
            [(db_id, c.chunk_text) for c, db_id in inserted if c.content_type == "knowledge"]
        )

    totals = {
        "articles": len(prepared),
        "knowledge": len(knowledge),
        "stories": len(stories),
        "links": len(link_rows),
        "questions": question_count,
    }
    tracing.add_metadata(**totals, dropped_links=dropped)
    return totals


@tracing.traceable(run_type="chain", name="ingest_normal")
def ingest_all_normal(groups: dict[str, list[RawChunk]], *, with_questions: bool) -> dict:
    """Insert every article and every chunk. No types, no links, one pass.

    Every chunk is embedded — that is the whole difference from the structured
    ingester, which skips stories. Nothing here can decide a chunk is not worth
    retrieving, because in this version nothing knows what kind of chunk it is.
    """
    import db
    import models

    prepared: list[tuple[int, list[RawChunk]]] = []
    for group in groups.values():
        chunks = sorted(group, key=lambda c: (c.source_chunk_id is None, c.source_chunk_id or 0))
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
            cur.execute("DELETE FROM normal_chunks WHERE article_id = %s", (article_id,))
        prepared.append((article_id, chunks))

    all_chunks = [c for _, chunks in prepared for c in chunks]
    log.info("embedding all %d chunk(s) — every chunk is retrievable in this version",
             len(all_chunks))
    vectors = iter(models.embed_documents([c.chunk_text for c in all_chunks]))

    inserted: list[tuple[RawChunk, int]] = []
    with db.cursor() as cur:
        for article_id, chunks in prepared:
            for chunk_index, chunk in enumerate(chunks):
                cur.execute(
                    _INSERT_NORMAL_CHUNK_SQL,
                    (
                        article_id,
                        chunk.source_chunk_id,
                        chunk_index,
                        chunk.chunk_text,
                        count_tokens(chunk.chunk_text),
                        db.to_pgvector(next(vectors)),
                    ),
                )
                inserted.append((chunk, int(cur.fetchone()["chunk_id"])))

    question_count = 0
    if with_questions and config.HQ_ENABLED:
        question_count = ingest_questions([(db_id, c.chunk_text) for c, db_id in inserted])

    totals = {
        "articles": len(prepared),
        "chunks": len(all_chunks),
        "questions": question_count,
    }
    tracing.add_metadata(**totals)
    return totals


@tracing.traceable(run_type="chain", name="hypothetical_questions")
def ingest_questions(targets: list[tuple[int, str]]) -> int:
    """Generate hypothetical questions and index them as the third retrieval arm.

    Structured: only knowledge chunks are passed in by the caller, because the HQ
    arm is a retrieval arm and a story is never a retrieval candidate — questions
    written for one could not match anything, so generating them would cost an LLM
    call plus an embedding per story for rows nothing can reach.

    Normal: every chunk is passed in, for the same reason read the other way —
    every chunk is a retrieval candidate.
    """
    import db
    import hq
    import models

    if not targets:
        return 0

    table = versions.active().question_table
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
            f"""
            INSERT INTO {table} (chunk_id, question_text, embedding)
            VALUES (%s, %s, %s::vector)
            """,
            [(cid, q, db.to_pgvector(v)) for (cid, q), v in zip(pairs, vectors)],
        )
    return len(pairs)


# --------------------------------------------------------------------------- #
# Dry run
# --------------------------------------------------------------------------- #


def print_dry_run_normal(groups: dict[str, list[RawChunk]], report) -> None:
    print("\n--- dry run, nothing written ---")
    for key, group in groups.items():
        head = group[0]
        print(f"\n{head.article_name}  ({key})")
        for c in sorted(group, key=lambda c: c.source_chunk_id or 0):
            print(f"  #{str(c.source_chunk_id):<5} {count_tokens(c.chunk_text):>4} tok")
    total = sum(len(g) for g in groups.values())
    print(f"\n  {total} chunk(s), all embedded, all retrievable — no types, no links")
    if report.errors:
        print(f"\n{len(report.errors)} error(s) — fix these before ingesting.")


def print_dry_run(groups: dict[str, list[RawChunk]], report) -> None:
    print("\n--- dry run, nothing written ---")
    home = {
        c.source_chunk_id: (c.article_url or c.article_name)
        for group in groups.values()
        for c in group
    }
    for key, group in groups.items():
        head = group[0]
        print(f"\n{head.article_name}  ({key})")
        for c in sorted(group, key=lambda c: c.source_chunk_id or 0):
            # mark links whose story lives in another article — they only resolve
            # because link resolution is global
            marks = [
                f"{t}*" if home.get(t) not in (None, c.article_url or c.article_name) else str(t)
                for t in c.links
            ]
            links = f" -> stories [{', '.join(marks)}]" if marks else ""
            print(
                f"  #{str(c.source_chunk_id):<5} {c.content_type:<9} "
                f"{count_tokens(c.chunk_text):>4} tok{links}"
            )
    print("\n  * = story in another article")
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
        default=None,
        help="chunk JSON file or a directory of them (default: the chosen version's own file)",
    )
    parser.add_argument(
        "--version",
        choices=versions.keys(),
        default=config.DEFAULT_VERSION,
        help="which RAG version to ingest into (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="parse, normalise and audit only — no database writes, no model loading",
    )
    parser.add_argument("--reset", action="store_true", help="DROP and recreate all tables first")
    parser.add_argument("--no-questions", action="store_true", help="skip hypothetical questions")
    args = parser.parse_args()

    config.setup_logging()

    # Set before anything reads it: db picks its database from here, hq picks its
    # prompt, retrieval builds its SQL, and the loader is told whether to expect
    # types and links.
    version = versions.use(args.version)
    structured = version.has_stories

    path = Path(args.input) if args.input else version.chunks_path
    if not path.exists():
        hint = ""
        if not args.input and not structured:
            hint = "\n  Generate it with: python tools/make_normal_chunks.py"
        raise SystemExit(f"input path not found: {path}{hint}")

    print(f"version: {version.label}  ({version.blurb})")
    print(f"  file     {path}")
    print(f"  database {version.database}")

    chunks, report = load_chunks(path, structured=structured)
    audit(chunks, report, structured=structured)
    report.log()

    if not chunks:
        raise SystemExit("no usable chunks found — see the errors above")
    if structured and report.legacy_link_keys:
        raise SystemExit(
            f"{report.legacy_link_keys} chunk(s) still declare 'related_knowledge_chunk_ids'. "
            f"Links now live on the knowledge chunk as 'related_story_chunk_ids'; ingesting "
            f"this file would produce no links at all."
        )

    groups = group_by_article(chunks)
    if structured:
        log.info(
            "%d usable chunk(s) across %d article(s): %d knowledge, %d story",
            len(chunks),
            len(groups),
            sum(1 for c in chunks if c.content_type == "knowledge"),
            sum(1 for c in chunks if c.content_type == "story"),
        )
    else:
        log.info("%d usable chunk(s) across %d article(s), all retrievable",
                 len(chunks), len(groups))

    if args.dry_run:
        (print_dry_run if structured else print_dry_run_normal)(groups, report)
        return

    import db
    import models
    import retrieval

    db.wait_for_db()
    # The normal version's database does not exist until the first ingest.
    db.ensure_database()
    if args.reset:
        db.reset_schema()
    else:
        db.check_schema_version()
        db.init_schema()
        db.check_embed_dim()

    # Catch a model/schema mismatch before spending time embedding everything.
    actual_dim = models.embedding_dimension()
    if actual_dim != config.EMBED_DIM:
        raise SystemExit(
            f"{config.EMBED_MODEL} produces {actual_dim}-dim vectors but EMBED_DIM is "
            f"{config.EMBED_DIM}. Set EMBED_DIM={actual_dim} and re-run with --reset."
        )

    if structured:
        totals = ingest_all(groups, with_questions=not args.no_questions)
        log.info(
            "done: %d articles, %d knowledge, %d story, %d links, %d questions",
            totals["articles"], totals["knowledge"], totals["stories"],
            totals["links"], totals["questions"],
        )
    else:
        totals = ingest_all_normal(groups, with_questions=not args.no_questions)
        log.info(
            "done: %d articles, %d chunks, %d questions",
            totals["articles"], totals["chunks"], totals["questions"],
        )

    counts = retrieval.stats()
    print(f"\n--- database {version.database} ({version.key}) ---")
    for key, value in counts.items():
        print(f"  {key:<24} {value}")
    if counts.get("orphan_stories"):
        print(
            "\n  note: orphan stories are cited by no knowledge chunk. Stories are never\n"
            "  retrieved directly, so nothing can ever pull them into an answer."
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    finally:
        tracing.flush()
        if "db" in sys.modules:
            sys.modules["db"].close()
