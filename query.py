"""Ask one of the two RAG versions a question, with every stage printed.

    python query.py "why start investing early"          # asks you which version
    python query.py "..." --version normal               # skip the chooser
    python query.py --repl                               # models stay loaded between questions
    python query.py "..." --fetch-k 30 --top-k 8
    python query.py "..." --retrieval-only               # no LLM call, no token spend

The pipeline is the same in both versions:

    embed ─┬─ vector_search ─┐
           ├─ fts_search   ─┼─ fuse ─ rerank ─ attach_stories ─ generate
           └─ hq_search    ─┘

What the version changes is which database the arms read, what they are allowed
to return, whether stage 4 does anything, and which system prompt stage 5 uses:

  structured  All three arms run over knowledge chunks only — stories have no
              embedding and are not in the full-text index, so they cannot appear
              as candidates. `attach_stories` then walks the knowledge->story link
              table to hand the model the illustration alongside the fact that
              cited it, and the prompt keeps the two roles apart.

  normal      All three arms run over every chunk. Stage 4 is a no-op — there are
              no links — and the prompt has one undifferentiated kind of passage.

In the REPL, `:version normal` switches between them without reloading the models,
which is the fastest way to put the two answers to one question side by side.

With LANGSMITH_API_KEY set, one call to `run` is one trace tree, with each stage
below as a child run.
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import config
import db
import models
import prompts
import retrieval
import tracing
import versions
from fusion import reciprocal_rank_fusion
from llm import chat

RULE = "=" * 78
THIN = "-" * 78


def normalized_score(hit: dict) -> float:
    """One 0..1 relevance number per hit, for the similarity floor and the API.

    The cross-encoder returns a logit (roughly -11..+11), which is neither
    comparable to a fusion score nor meaningful to a slider; a sigmoid squashes it
    into the 0..1 range fusion already lives in. Hits that never reached the
    reranker keep their fusion score, which is already small and positive.
    """
    score = hit.get("rerank_score")
    if score is None:
        return float(hit.get("fusion_score") or 0.0)
    return 1.0 / (1.0 + math.exp(-float(score)))


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


@tracing.traceable(run_type="chain", name="structured_rag")
def run(
    question: str,
    *,
    fetch_k: int | None = None,
    top_k: int | None = None,
    min_score: float = 0.0,
    attach_stories: bool | None = None,
    generate: bool = True,
    verbose: bool = True,
    show_text: bool = False,
) -> dict:
    """Run one question end to end. Returns every stage; prints them if verbose."""
    version = versions.active()
    fetch_k = fetch_k or config.VECTOR_TOP_K
    top_k = top_k or config.RERANK_TOP_K
    # Stories exist only in the structured version, so the setting is irrelevant
    # to the other one — there is nothing in its schema to attach.
    attach = (config.ATTACH_LINKED_STORIES if attach_stories is None else attach_stories) \
        and version.has_stories
    timings: dict[str, float] = {}

    tracing.add_metadata(version=version.key, database=version.database)

    if verbose:
        print(f"\n{RULE}\nQUESTION: {question}\n[{version.label}]\n{RULE}")

    # ---- embed -----------------------------------------------------------
    with _timer("embed", timings):
        query_vector = models.embed_query(question)

    # ---- stage 1: the three arms ----------------------------------------
    with _timer("vector_search", timings):
        vector_hits = retrieval.vector_search(query_vector, fetch_k)
    with _timer("fts_search", timings):
        fts_hits = retrieval.fts_search(question, fetch_k)
    with _timer("hq_search", timings):
        hq_hits = retrieval.hq_search(query_vector, fetch_k)

    arms = {"vector": vector_hits, "fts": fts_hits, "hq": hq_hits}
    if verbose:
        _print_arms(arms, fetch_k, version)

    tracing.add_metadata(**{f"{name}_hits": len(hits) for name, hits in arms.items()})

    # ---- stage 2: fusion -------------------------------------------------
    with _timer("fuse", timings):
        fused = reciprocal_rank_fusion(arms)
    if verbose:
        _print_fusion(fused)

    # ---- stage 3: rerank -------------------------------------------------
    with _timer("rerank", timings):
        knowledge = models.rerank(question, fused, top_k)
    # The similarity floor is applied here, not on the way out, so a passage the
    # caller rejected is not quietly still in the context the answer came from.
    if min_score > 0:
        knowledge = [h for h in knowledge if normalized_score(h) >= min_score]
    if verbose:
        _print_rerank(knowledge, show_text, version)

    # ---- stage 4: attach linked stories ----------------------------------
    stories: list[dict] = []
    if attach and knowledge:
        with _timer("attach_stories", timings):
            stories = retrieval.label_stories(
                retrieval.linked_stories([h["chunk_id"] for h in knowledge]), knowledge
            )
    if verbose:
        _print_stories(stories, attach, show_text, version)

    # ---- stage 5: generate -----------------------------------------------
    # Context format and system prompt are chosen together: the structured
    # prompt's rules are about [K*]/[S*] blocks that the flat formatter does not
    # produce, and the flat prompt's [1], [2] citations are about a numbering the
    # structured formatter does not use.
    if version.has_stories:
        context = prompts.format_context(knowledge, stories)
        user_prompt = prompts.build_user_prompt(question, context)
    else:
        context = prompts.format_context_normal(knowledge)
        user_prompt = prompts.build_user_prompt_normal(question, context)

    answer = ""
    if generate:
        if not knowledge:
            answer = "I could not find anything relevant in the indexed articles."
        else:
            with _timer("generate", timings):
                answer = chat(version.system_prompt, user_prompt)
    if verbose:
        _print_answer(answer, generate, timings)

    tracing.add_metadata(
        knowledge_chunk_ids=[h["chunk_id"] for h in knowledge],
        story_chunk_ids=[h["chunk_id"] for h in stories],
        context_chars=len(context),
        **{f"ms_{k}": v for k, v in timings.items()},
    )

    return {
        "question": question,
        "version": version.key,
        "answer": answer,
        "arms": arms,
        "fused": fused,
        "knowledge": knowledge,
        "stories": stories,
        "context": context,
        "user_prompt": user_prompt,
        "timings_ms": timings,
    }


class _timer:
    def __init__(self, name: str, sink: dict[str, float]):
        self.name, self.sink = name, sink

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.sink[self.name] = round((time.perf_counter() - self.start) * 1000, 1)
        return False


# --------------------------------------------------------------------------- #
# Stage printing
# --------------------------------------------------------------------------- #


def _preview(text: str | None, width: int = 88) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _ref(hit: dict) -> str:
    """Identify a chunk by the id YOU gave it in the JSON, not Postgres' own.

    These drift apart and the gap widens through the file: `source_chunk_id` is the
    hand-assigned `chunk_id` from data/chunks.json, while `chunk_id` is the
    database's identity sequence, which skips nothing and so runs ahead wherever a
    JSON id was absent or a chunk was rejected. Printing only the database id makes
    every stage impossible to trace back to the source file — json#50 and db#49 are
    the same row, and neither number alone tells you that.
    """
    source = hit.get("source_chunk_id")
    db_id = hit.get("chunk_id")
    return f"json#{source} (db {db_id})" if source is not None else f"db#{db_id}"


def _print_arms(arms: dict[str, list[dict]], fetch_k: int, version) -> None:
    scope = "knowledge only" if version.has_stories else "every chunk"
    print(f"\nSTAGE 1 — retrieval arms  (fetch_k={fetch_k}, scope: {scope})")
    for name, hits in arms.items():
        if not hits:
            print(f"  {name:<7} 0 hits   {_empty_arm_hint(name)}")
            continue
        top = hits[0]
        print(f"  {name:<7} {len(hits):>2} hits   best {_ref(top)} "
              f"score={float(top.get('score') or 0):.4f}")
        if name == "hq" and top.get("matched_question"):
            print(f"          via question: \"{_preview(top['matched_question'], 70)}\"")
        print(f"          {_preview(top.get('chunk_text'))}")


def _empty_arm_hint(name: str) -> str:
    return {
        "vector": "(nothing embedded? run ingest.py)",
        "fts": "(no chunk shares even one term with the question)",
        "hq": "(no hypothetical questions indexed; ingest without --no-questions)",
    }.get(name, "")


def _print_fusion(fused: list[dict]) -> None:
    print(
        f"\nSTAGE 2 — reciprocal rank fusion  (k={config.RRF_K}, weights: "
        f"vector={config.WEIGHT_VECTOR} fts={config.WEIGHT_FTS} hq={config.WEIGHT_HQ})"
    )
    if not fused:
        print("  nothing fused — every arm came back empty")
        return
    print(f"  {len(fused)} candidates; top {min(8, len(fused))}:")
    for hit in fused[:8]:
        sources = "+".join(hit.get("sources", []))
        print(
            f"  {hit['fusion_rank']:>2}. {_ref(hit):<22} {hit['fusion_score']:.5f}  "
            f"[{sources:<15}] {_preview(hit.get('chunk_text'), 55)}"
        )


def _print_rerank(knowledge: list[dict], show_text: bool, version) -> None:
    if config.RERANKER_BACKEND == "local":
        backend = config.RERANKER_MODEL
    elif config.RERANKER_BACKEND == "http":
        backend = f"{config.RERANKER_MODEL} via {config.RERANKER_URL}"
    else:
        backend = "none (fusion order)"
    print(f"\nSTAGE 3 — rerank -> top {len(knowledge)}  ({backend})")
    if not knowledge:
        print("  nothing survived")
        return
    # The label is the one the context block will carry, so a citation in the
    # answer can be matched back to a line of this stage.
    tag = "K" if version.has_stories else "P"
    for i, hit in enumerate(knowledge, start=1):
        score = hit.get("rerank_score")
        score_text = f"{score:+.3f}" if score is not None else f"{hit['fusion_score']:.5f}"
        moved = f"was #{hit.get('fusion_rank')}"
        print(f"  {tag}{i}  {_ref(hit):<22} {score_text:>8}  ({moved:<7})  "
              f"{hit.get('article_name', '')[:45]}")
        print(f"      {_preview(hit.get('chunk_text')) if not show_text else hit.get('chunk_text')}")


def _print_stories(stories: list[dict], attached: bool, show_text: bool, version) -> None:
    print("\nSTAGE 4 — linked stories")
    if not version.has_stories:
        print("  n/a — this version has no story/knowledge distinction and no links")
        return
    if not attached:
        print("  disabled (ATTACH_LINKED_STORIES=false)")
        return
    if not stories:
        print("  none — none of the surviving knowledge chunks cites a story")
        return
    for i, hit in enumerate(stories, start=1):
        label = hit.get("linked_knowledge_label") or "?"
        print(f"  S{i}  {_ref(hit):<22} illustrates {label:<12} {hit.get('article_name', '')[:45]}")
        print(f"      {_preview(hit.get('chunk_text')) if not show_text else hit.get('chunk_text')}")


def _print_answer(answer: str, generated: bool, timings: dict[str, float]) -> None:
    print(f"\nSTAGE 5 — answer  ({config.GROQ_MODEL})" if generated else "\nSTAGE 5 — skipped (--retrieval-only)")
    if generated:
        print(THIN)
        print(answer)
        print(THIN)
    total = sum(timings.values())
    parts = "  ".join(f"{k}={v:.0f}ms" for k, v in timings.items())
    print(f"\n  {parts}\n  total={total:.0f}ms")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def choose_version(preselected: str | None) -> versions.Version:
    """Pick the version to talk to: the flag if given, else ask.

    Asked once per process. Falling back to the default on a non-interactive stdin
    keeps `echo "q" | python query.py` and CI usable, where a blocking prompt would
    hang forever with nothing on screen to say why.
    """
    if preselected:
        return db.set_version(preselected)

    if not sys.stdin.isatty():
        return db.set_version(config.DEFAULT_VERSION)

    options = list(versions.ALL.values())
    print("\nWhich version do you want to talk to?")
    for i, v in enumerate(options, start=1):
        default = "  (default)" if v.key == config.DEFAULT_VERSION else ""
        print(f"  {i}. {v.label:<18} {v.blurb}{default}")
        print(f"     database {v.database}")

    while True:
        try:
            raw = input(f"\nversion [1-{len(options)}, Enter for default]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(130)
        if not raw:
            return db.set_version(config.DEFAULT_VERSION)
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return db.set_version(options[int(raw) - 1].key)
        try:
            return db.set_version(raw)
        except SystemExit as exc:
            print(f"  {exc}")


def _preflight() -> None:
    version = versions.active()
    ingest_hint = f"python ingest.py --version {version.key}"

    db.wait_for_db()
    if not db.database_exists():
        raise SystemExit(
            f"database {version.database!r} does not exist yet — run: {ingest_hint}"
        )
    if not db.tables_exist():
        raise SystemExit(f"tables do not exist yet — run: {ingest_hint}")
    db.check_schema_version()
    db.check_embed_dim()

    counts = retrieval.stats()
    if retrieval.retrievable_count(counts) == 0:
        raise SystemExit(f"nothing indexed — run: {ingest_hint}")

    print(f"\n{version.label} — database {version.database}")
    if version.has_stories:
        print(
            f"indexed: {counts['articles']} articles, {counts['knowledge_chunks']} knowledge, "
            f"{counts['story_chunks']} story ({counts['orphan_stories']} orphaned), "
            f"{counts['story_links']} links ({counts['cross_article_links']} cross-article), "
            f"{counts['questions']} questions"
        )
        if counts["orphan_stories"]:
            print(
                f"  note: {counts['orphan_stories']} story chunk(s) are cited by no knowledge "
                f"chunk, so nothing can reach them."
            )
    else:
        print(
            f"indexed: {counts['articles']} articles, {counts['chunks']} chunks "
            f"(all retrievable), {counts['questions']} questions"
        )
    if tracing.enabled():
        print(f"  tracing to LangSmith project {config.LANGSMITH_PROJECT!r}")


def repl(args) -> None:
    _preflight()
    print("loading models…")
    models.embed_query("warmup")
    if config.RERANKER_BACKEND == "local":
        models.rerank("warmup", [{"chunk_id": 0, "chunk_text": "warmup"}], 1)
    elif config.RERANKER_BACKEND == "http" and not models.reranker_healthy():
        print(f"  warning: reranker {config.RERANKER_URL} unreachable — fusion order will be used")
    print(
        "ready. Ask a question, ':version <name>' to switch version, "
        "or Ctrl-C / 'exit' to quit."
    )

    while True:
        try:
            question = input(f"\n[{versions.active().key}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question.lower() in {"exit", "quit", ":q"}:
            return

        # Switching version in-process is the point of having them share a
        # codebase: the models stay resident, so the same question can be put to
        # both databases back to back without a 30-second reload in between.
        if question.lower().startswith((":version", ":v ")) or question.lower() == ":v":
            _switch_version(question.split(maxsplit=1))
            continue

        try:
            run(
                question,
                fetch_k=args.fetch_k,
                top_k=args.top_k,
                attach_stories=not args.no_stories,
                generate=not args.retrieval_only,
                show_text=args.show_text,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  error: {exc}")


def _switch_version(parts: list[str]) -> None:
    if len(parts) < 2:
        print(f"  usage: :version <{'|'.join(versions.keys())}>")
        return
    previous = versions.active()
    try:
        db.set_version(parts[1])
        _preflight()
    except SystemExit as exc:
        print(f"  {exc}")
        db.set_version(previous.key)
        print(f"  staying on {previous.label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query one of the two RAG versions")
    parser.add_argument("question", nargs="?", help="the question to ask")
    parser.add_argument(
        "--version",
        choices=versions.keys(),
        default=None,
        help="which version to talk to; omit and you will be asked",
    )
    parser.add_argument("--repl", action="store_true", help="interactive mode; models stay loaded")
    parser.add_argument("--fetch-k", type=int, default=None, help="candidates per arm before fusion")
    parser.add_argument("--top-k", type=int, default=None, help="knowledge chunks sent to the LLM")
    parser.add_argument("--no-stories", action="store_true", help="do not attach linked stories")
    parser.add_argument("--retrieval-only", action="store_true", help="skip generation entirely")
    parser.add_argument("--show-text", action="store_true", help="print full chunk text, not previews")
    parser.add_argument("--stats", action="store_true", help="print row counts and exit")
    args = parser.parse_args()

    config.setup_logging()

    if args.stats:
        # Report on every version, not just one — the point of --stats is to see
        # at a glance what is loaded where.
        db.wait_for_db()
        for key in versions.keys():
            version = db.set_version(key)
            print(f"\n{version.label}  (database {version.database})")
            if not db.database_exists() or not db.tables_exist():
                print("  not ingested yet")
                continue
            for name, value in retrieval.stats().items():
                print(f"  {name:<24} {value}")
        return

    choose_version(args.version)

    if args.repl:
        repl(args)
        return

    if not args.question:
        parser.error("give a question, or use --repl")

    _preflight()
    run(
        args.question,
        fetch_k=args.fetch_k,
        top_k=args.top_k,
        attach_stories=not args.no_stories,
        generate=not args.retrieval_only,
        show_text=args.show_text,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    finally:
        tracing.flush()
        db.close()
