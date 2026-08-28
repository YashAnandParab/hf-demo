"""Ask the structured RAG a question, with every stage printed.

    python query.py "why start investing early"
    python query.py --repl                        # models stay loaded between questions
    python query.py "..." --fetch-k 30 --top-k 8
    python query.py "..." --retrieval-only        # no LLM call, no token spend

The pipeline:

    embed ─┬─ vector_search ─┐
           ├─ fts_search   ─┼─ fuse ─ rerank ─ attach_stories ─ generate
           └─ hq_search    ─┘

All three arms run over knowledge chunks only; `attach_stories` then walks the
story->knowledge link table to hand the model the illustration alongside the fact.
"""
from __future__ import annotations

import argparse
import sys
import time

import config
import db
import models
import prompts
import retrieval
from fusion import reciprocal_rank_fusion
from llm import chat

RULE = "=" * 78
THIN = "-" * 78


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def run(
    question: str,
    *,
    fetch_k: int | None = None,
    top_k: int | None = None,
    attach_stories: bool | None = None,
    generate: bool = True,
    verbose: bool = True,
    show_text: bool = False,
) -> dict:
    """Run one question end to end. Returns every stage; prints them if verbose."""
    fetch_k = fetch_k or config.VECTOR_TOP_K
    top_k = top_k or config.RERANK_TOP_K
    attach = config.ATTACH_LINKED_STORIES if attach_stories is None else attach_stories
    timings: dict[str, float] = {}

    if verbose:
        print(f"\n{RULE}\nQUESTION: {question}\n{RULE}")

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
        _print_arms(arms, fetch_k)

    # ---- stage 2: fusion -------------------------------------------------
    with _timer("fuse", timings):
        fused = reciprocal_rank_fusion(arms)
    if verbose:
        _print_fusion(fused)

    # ---- stage 3: rerank -------------------------------------------------
    with _timer("rerank", timings):
        knowledge = models.rerank(question, fused, top_k)
    if verbose:
        _print_rerank(knowledge, show_text)

    # ---- stage 4: attach linked stories ----------------------------------
    stories: list[dict] = []
    if attach and knowledge:
        with _timer("attach_stories", timings):
            stories = retrieval.label_stories(
                retrieval.linked_stories([h["chunk_id"] for h in knowledge]), knowledge
            )
    if verbose:
        _print_stories(stories, attach, show_text)

    # ---- stage 5: generate -----------------------------------------------
    context = prompts.format_context(knowledge, stories)
    answer = ""
    if generate:
        if not knowledge:
            answer = "I could not find relevant knowledge in the indexed articles."
        else:
            with _timer("generate", timings):
                answer = chat(prompts.STRUCTURED_SYSTEM, prompts.build_user_prompt(question, context))
    if verbose:
        _print_answer(answer, generate, timings)

    return {
        "question": question,
        "answer": answer,
        "arms": arms,
        "fused": fused,
        "knowledge": knowledge,
        "stories": stories,
        "context": context,
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


def _print_arms(arms: dict[str, list[dict]], fetch_k: int) -> None:
    scope = "knowledge only" if config.STORY_RETRIEVAL_MODE != "include" else "knowledge + stories"
    print(f"\nSTAGE 1 — retrieval arms  (fetch_k={fetch_k}, scope: {scope})")
    for name, hits in arms.items():
        if not hits:
            print(f"  {name:<7} 0 hits   {_empty_arm_hint(name)}")
            continue
        top = hits[0]
        print(f"  {name:<7} {len(hits):>2} hits   best #{top['chunk_id']} "
              f"score={float(top.get('score') or 0):.4f}")
        if name == "hq" and top.get("matched_question"):
            print(f"          via question: \"{_preview(top['matched_question'], 70)}\"")
        print(f"          {_preview(top.get('chunk_text'))}")


def _empty_arm_hint(name: str) -> str:
    return {
        "vector": "(nothing embedded? run ingest.py)",
        "fts": "(no lexical overlap — expected for conceptual questions)",
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
            f"  {hit['fusion_rank']:>2}. #{hit['chunk_id']:<5} {hit['fusion_score']:.5f}  "
            f"[{sources:<15}] {_preview(hit.get('chunk_text'), 60)}"
        )


def _print_rerank(knowledge: list[dict], show_text: bool) -> None:
    backend = config.RERANKER_MODEL if config.RERANKER_BACKEND == "local" else "none (fusion order)"
    print(f"\nSTAGE 3 — rerank -> top {len(knowledge)}  ({backend})")
    if not knowledge:
        print("  nothing survived")
        return
    for i, hit in enumerate(knowledge, start=1):
        score = hit.get("rerank_score")
        score_text = f"{score:+.3f}" if score is not None else f"{hit['fusion_score']:.5f}"
        moved = f"was #{hit.get('fusion_rank')}"
        print(f"  K{i}  #{hit['chunk_id']:<5} {score_text:>8}  ({moved:<7})  "
              f"{hit.get('article_name', '')[:40]}")
        print(f"      {_preview(hit.get('chunk_text')) if not show_text else hit.get('chunk_text')}")


def _print_stories(stories: list[dict], attached: bool, show_text: bool) -> None:
    print("\nSTAGE 4 — linked stories")
    if not attached:
        print("  disabled (ATTACH_LINKED_STORIES=false)")
        return
    if not stories:
        print("  none — no story links to the surviving knowledge chunks")
        return
    for i, hit in enumerate(stories, start=1):
        label = hit.get("linked_knowledge_label") or "?"
        print(f"  S{i}  #{hit['chunk_id']:<5} illustrates {label:<12} {hit.get('article_name', '')[:40]}")
        if hit.get("story_summary"):
            print(f"      summary: {_preview(hit['story_summary'])}")
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


def _preflight() -> None:
    db.wait_for_db()
    if not db.tables_exist():
        raise SystemExit("tables do not exist yet — run: python ingest.py data/chunks.json")
    db.check_embed_dim()
    counts = retrieval.stats()
    if counts["knowledge_chunks"] == 0:
        raise SystemExit("no knowledge chunks indexed — run: python ingest.py data/chunks.json")
    print(
        f"indexed: {counts['articles']} articles, {counts['knowledge_chunks']} knowledge, "
        f"{counts['story_chunks']} story ({counts['orphan_stories']} orphaned), "
        f"{counts['questions']} questions"
    )
    # Questions are generated at ingest time based on the mode in force *then*. Flipping
    # to `include` afterwards leaves the HQ arm blind to stories until a re-ingest.
    if (
        config.STORY_RETRIEVAL_MODE == "include"
        and counts["story_chunks"]
        and not counts["story_questions"]
    ):
        print(
            "  warning: STORY_RETRIEVAL_MODE=include, but no story has hypothetical\n"
            "  questions — they were ingested in knowledge_only mode. The vector and\n"
            "  fts arms still reach stories; the hq arm does not. Re-ingest to fix."
        )


def repl(args) -> None:
    _preflight()
    print("loading models…")
    models.embed_query("warmup")
    if config.RERANKER_BACKEND == "local":
        models.rerank("warmup", [{"chunk_id": 0, "chunk_text": "warmup"}], 1)
    print("ready. Ask a question, or Ctrl-C / 'exit' to quit.")

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question.lower() in {"exit", "quit", ":q"}:
            return
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the structured RAG")
    parser.add_argument("question", nargs="?", help="the question to ask")
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
        db.wait_for_db()
        for key, value in retrieval.stats().items():
            print(f"  {key:<20} {value}")
        return

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
        db.close()
