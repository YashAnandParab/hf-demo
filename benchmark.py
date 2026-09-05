"""Run the benchmark question set against one version and write a Markdown report.

    python benchmark.py --version structured
    python benchmark.py --version normal

Run it once per version. The two write to different files by default, so neither
run clobbers the other and you can diff them afterwards.

    python benchmark.py --version structured --retrieval-only   # no LLM, no quota
    python benchmark.py --version normal --only sf01,st03,xd01
    python benchmark.py --version structured --category cross_document
    python benchmark.py --version structured --delay 10         # slower, gentler on a free key

The report embeds the CLI transcript for each question VERBATIM — this calls the
same `query.run(verbose=True)` the REPL calls, with the same defaults, and only
captures what it prints. Nothing is re-rendered or summarised in the transcript,
so a section of the report and a REPL session for that question are the same
bytes. The per-question header around it (gold fact, gold chunks, whether they
were retrieved) is added above the fence, never inside it.

On output location: under Docker only ./data is bind-mounted to the host, so the
report is written into data/ by default. Anywhere else and it would exist only
inside a container you are about to throw away.

Rate limits: `llm.chat` already waits out a 429 and retries the same model, but a
free key is usually capped per MINUTE as well, and this fires forty-odd questions
back to back. `--delay` puts a gap between questions, and a question that still
fails is recorded as a failure and the run carries on rather than losing the
forty that would have followed.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import logging
import sys
import time
from datetime import datetime, timezone

import config
import db
import models
import query
import retrieval
import tracing
import versions

# --------------------------------------------------------------------------- #
# The question set.
#
# `gold_chunk_ids` are SOURCE chunk ids — the numbers in data/chunks.json, which
# the CLI prints as `json#N`. They are not the database chunk_ids that follow in
# parentheses. The normal corpus is derived from the same file with only
# chunk_type and the links stripped, so one gold id means the same passage in
# both versions and the two reports are directly comparable.
#
# The two id lists are graded differently, because the system treats the two
# kinds of chunk differently:
#
#   gold_chunk_ids    KNOWLEDGE the answer must be built from. The prompt makes
#                     knowledge the only source of truth, so a gold chunk that
#                     does not reach the model is a retrieval failure.
#
#   bonus_story_ids   STORIES that would illustrate the answer well. The prompt
#                     forbids a story being the sole source of a fact, so a
#                     missing story is a weaker answer, not a wrong one. These
#                     are reported in their own column and kept out of recall —
#                     folding them in would score the architecture's own design
#                     rule as a defect.
# --------------------------------------------------------------------------- #

QUESTIONS: list[dict] = [
    # ---- mutual funds: what they are, what happens inside one -------------- #
    {"qid": "mf01", "category": "core_advice",
     "question": "Why should I invest in mutual funds instead of picking stocks myself?",
     "gold_fact": "A mutual fund is a tool for ownership of a diversified portfolio, run by a professional system where ideas are challenged and biases questioned, rather than one person deciding alone.",
     "gold_chunk_ids": [34, 11], "bonus_story_ids": [],
     "notes": "Baseline advice question. Both tracks should tie."},
    {"qid": "mf02", "category": "core_advice",
     "question": "What actually happens to my money after I invest in a mutual fund?",
     "gold_fact": "It enters a disciplined organization: research analysts, a fund manager, a CIO owning firm-wide philosophy, a risk manager, dealers executing trades, plus compliance and operations.",
     "gold_chunk_ids": [5, 6, 7, 8, 9], "bonus_story_ids": [],
     "notes": "Multi-chunk within one article. Tests how much of a 5-part answer fits in a 5-chunk budget."},
    {"qid": "mf03", "category": "objection_handling",
     "question": "I've read a lot and follow markets closely. Can't I just do what a fund manager does?",
     "gold_fact": "The individual decides alone; the professional operates inside a system where ideas are challenged. The best individual investors treated it as a profession, spending years or decades mastering it.",
     "gold_chunk_ids": [11, 52, 51], "bonus_story_ids": [50],
     "notes": "Story 50 (Ashok, thirteen years) is the ideal illustration and links from 51/52. Prime illustration test."},
    {"qid": "mf04", "category": "core_advice",
     "question": "How do I choose from the hundreds of funds and strategies out there?",
     "gold_fact": "More choice does not mean better decisions; the costliest habit is jumping between options. What matters is how you behave with investments, not what you pick.",
     "gold_chunk_ids": [2, 3, 4], "bonus_story_ids": [1],
     "notes": "Story 1 (Cheesecake Factory) links from all three knowledge chunks."},

    # ---- equity: ownership, risk, diversification -------------------------- #
    {"qid": "eq01", "category": "objection_handling",
     "question": "Isn't putting money in the stock market basically gambling?",
     "gold_fact": "Equity is ownership in a business, the same thing a business owner already does. People lose money because they trade and react, not because they invest.",
     "gold_chunk_ids": [23, 24, 25], "bonus_story_ids": [22],
     "notes": "Story 22 (Raj Khanna) links from 23 and 25. Classic objection a real user raises."},
    {"qid": "eq02", "category": "objection_handling",
     "question": "People lose money in equities all the time. Why should I take that risk?",
     "gold_fact": "Risk means uncertainty and is the price of potential reward; losses come from concentration and emotion, and diversification plus discipline addresses both.",
     "gold_chunk_ids": [36, 24, 25], "bonus_story_ids": [],
     "notes": "Spans two articles, knowledge only."},
    {"qid": "eq03", "category": "core_advice",
     "question": "What does diversification actually do for me?",
     "gold_fact": "It spreads investments across assets so no single one dominates the outcome, acting as multiple safety nets.",
     "gold_chunk_ids": [37, 25], "bonus_story_ids": [],
     "notes": "Baseline."},

    # ---- fixed deposits and the meaning of safety -------------------------- #
    {"qid": "fd01", "category": "objection_handling",
     "question": "My money is in fixed deposits and I sleep fine. Why change anything?",
     "gold_fact": "Lifestyle inflation runs well above reported CPI, so risk-free products erode purchasing power; avoiding short-term volatility exposes you to inflation risk instead.",
     "gold_chunk_ids": [75, 72, 71], "bonus_story_ids": [73, 74],
     "notes": "IMPORTANT. Stories 73/74 hold the toll and education-inflation numbers that make this concrete. Tests whether the anecdote reaches the answer without the user asking for it."},
    {"qid": "fd02", "category": "core_advice",
     "question": "Is capital protection the same thing as keeping my money safe?",
     "gold_fact": "No. Preserving purchasing power is what safety means; return of capital alone ignores inflation.",
     "gold_chunk_ids": [78, 76], "bonus_story_ids": [77],
     "notes": "Story 77 (Reema) links from 78."},

    # ---- evaluating what is being sold to you ------------------------------ #
    {"qid": "gd01", "category": "product_evaluation",
     "question": "Everyone's talking about gold right now. Should I be adding some?",
     "gold_fact": "Gold is a shock absorber, not a return engine; fear drives people in and boredom drives them out, and it is meant to stabilize rather than excite.",
     "gold_chunk_ids": [49], "bonus_story_ids": [48],
     "notes": "Story 48 (Rajesh's repeated mistiming) links from 49 and is exactly the cautionary tale this question needs."},
    {"qid": "pm01", "category": "product_evaluation",
     "question": "A big wealth firm is pitching me a PMS. Should I go with them?",
     "gold_fact": "Branded firms face revenue pressure and push high-margin PMS and AIF products, often without suitability or context; your capital deserves a better reason than a pitch.",
     "gold_chunk_ids": [20, 19, 21], "bonus_story_ids": [18],
     "notes": "Story 18 is in a DIFFERENT article. Cross-document link, and the most realistic version of the v1 cross_document test."},
    {"qid": "pm02", "category": "product_evaluation",
     "question": "I've been offered an exclusive product only available to select investors. Is that a good sign?",
     "gold_fact": "The pull of exclusivity is what gets sophisticated investors taken for a ride; returns are often mediocre, exits painful, costs buried. Chase clarity, not exclusivity.",
     "gold_chunk_ids": [55, 54, 56, 57], "bonus_story_ids": [],
     "notes": "Four knowledge chunks, no story. Tests knowledge density directly."},
    {"qid": "pm03", "category": "product_evaluation",
     "question": "Someone suggested consolidating all my investments with one firm. Worth doing?",
     "gold_fact": "Only after a diagnosis. Consolidation without understanding what each holding is for creates confusion, like a pharmacy consolidating medicines nobody reviewed.",
     "gold_chunk_ids": [26, 27], "bonus_story_ids": [],
     "notes": "Baseline, knowledge only."},

    # ---- behaviour under market stress ------------------------------------- #
    {"qid": "bh01", "category": "behavioral",
     "question": "Markets are at an all-time high. Should I book some profits?",
     "gold_fact": "Selling to time the market gives you two or three ways to be wrong; ask what you will do with the proceeds and what you give up by not holding.",
     "gold_chunk_ids": [96, 95, 92], "bonus_story_ids": [93, 94],
     "notes": "Stories 93/94 (Howard Marks and Andrew, Munger) link from 95. This is the realistic version of the v1 Howard Marks trivia question."},
    {"qid": "bh02", "category": "behavioral",
     "question": "Markets just dropped hard and I'm tempted to stop my SIP. What should I do?",
     "gold_fact": "Market behaviour cannot be predicted; past declines look like opportunities in hindsight but felt like risk at the time, and the same applies now.",
     "gold_chunk_ids": [42, 43], "bonus_story_ids": [],
     "notes": "Very common real query. Knowledge only."},
    {"qid": "bh03", "category": "behavioral",
     "question": "My friend's portfolio returned much more than mine this year. Should I switch to what he's doing?",
     "gold_fact": "No. Investments should fit your situation the way clothes fit your body; performance envy and relative deprivation bias push people into strategies that don't suit them.",
     "gold_chunk_ids": [79, 80, 83], "bonus_story_ids": [],
     "notes": "Baseline behavioral."},
    {"qid": "bh04", "category": "behavioral",
     "question": "How do I stop reacting to every market headline?",
     "gold_fact": "Scary headlines are the easiest way to make people sell; the question is whether you would have held through the drawdowns that preceded the big gains.",
     "gold_chunk_ids": [89, 43], "bonus_story_ids": [],
     "notes": "Spans two articles."},
    {"qid": "bh05", "category": "behavioral",
     "question": "I track markets daily and keep moving between funds. Am I actually investing?",
     "gold_fact": "Not necessarily. Reacting to markets, news and peers is not investing; the useful questions are about what your money is for, not which fund is best.",
     "gold_chunk_ids": [12, 14, 16, 17], "bonus_story_ids": [13, 15],
     "notes": "STRONG illustration case. Stories 13 and 15 (Rohit) link from all four knowledge chunks and describe exactly this person."},

    # ---- planning: goals, horizons, benchmarks ----------------------------- #
    {"qid": "pl01", "category": "planning",
     "question": "What rate of return should I be aiming for?",
     "gold_fact": "Only what you need. If 6% hits your number, invest to achieve 7-8%, not more.",
     "gold_chunk_ids": [69], "bonus_story_ids": [],
     "notes": "Baseline planning."},
    {"qid": "pl02", "category": "planning",
     "question": "I need this money for a house down payment in two years. Where should it sit?",
     "gold_fact": "Not in equity. Money needed within three years never sits in equity as an asset class.",
     "gold_chunk_ids": [69, 70], "bonus_story_ids": [],
     "notes": "Practical, very common."},
    {"qid": "pl03", "category": "planning",
     "question": "Should I be trying to beat the Nifty?",
     "gold_fact": "Beating a benchmark someone else chose changes nothing in your life; four investors chasing the same benchmark can all be making the wrong decision because their lives differ.",
     "gold_chunk_ids": [60, 62, 59], "bonus_story_ids": [],
     "notes": "Three knowledge chunks in one article."},
    {"qid": "pl04", "category": "planning",
     "question": "How should I be thinking about what my life will cost decades from now?",
     "gold_fact": "Healthcare and lifestyle costs rise faster than general inflation, so the portfolio has to be built for the future you expect to live in, not the present.",
     "gold_chunk_ids": [64, 66, 67], "bonus_story_ids": [],
     "notes": "Knowledge only, spans one article."},
    {"qid": "pl05", "category": "planning",
     "question": "Where should I even start before picking any investment?",
     "gold_fact": "By answering what you want from life; decision quality comes from understanding your life, not from knowing more products.",
     "gold_chunk_ids": [59, 61, 63], "bonus_story_ids": [],
     "notes": "Knowledge only."},

    # ---- advisors and the people selling certainty -------------------------- #
    {"qid": "ad01", "category": "core_advice",
     "question": "What does a good financial advisor actually do for me?",
     "gold_fact": "Manages risk rather than promising returns: works out your number, targets only the return you need, and diversifies across time so money is there when you need it.",
     "gold_chunk_ids": [68, 69, 70], "bonus_story_ids": [],
     "notes": "Baseline."},
    {"qid": "ad02", "category": "objection_handling",
     "question": "An advisor told me they can get me better returns than the market. Should I believe them?",
     "gold_fact": "Claims built on outperformance and special access are the wrong model; nobody times markets consistently and those who say they do are not being straight with you.",
     "gold_chunk_ids": [68, 96], "bonus_story_ids": [],
     "notes": "Spans two articles."},
    {"qid": "ad03", "category": "objection_handling",
     "question": "A big bank's research says they're cautious right now given the uncertainty. Should I act on that?",
     "gold_fact": "If the statement implies no action, move on. Investors need conviction and alignment, not vague commentary that lets the firm claim it was right either way.",
     "gold_chunk_ids": [100], "bonus_story_ids": [97, 98, 99],
     "notes": "Three stories fan out from one knowledge chunk. Realistic version of the v1 fanout test."},
    {"qid": "ad04", "category": "objection_handling",
     "question": "Analysts on TV seem very confident about where markets are headed. Worth listening to?",
     "gold_fact": "Certainty is the illusion; the wise admit ignorance, focus on what they control, and build portfolios that don't depend on predictions.",
     "gold_chunk_ids": [102, 103], "bonus_story_ids": [101],
     "notes": "Story 101 (Socrates) links from 103. Realistic version of the v1 Socrates trivia question."},

    # ---- performance chasing ------------------------------------------------ #
    {"qid": "rv01", "category": "behavioral",
     "question": "This fund topped the charts last year. Should I move my money into it?",
     "gold_fact": "That is chasing the rearview mirror. Past performance is not the future, and the same fund may have been among the worst a year earlier.",
     "gold_chunk_ids": [46, 47], "bonus_story_ids": [45],
     "notes": "Story 45 is in a DIFFERENT article. Cross-document, and the most realistic framing of this content."},
    {"qid": "rv02", "category": "core_advice",
     "question": "What should I actually be measuring my investments against?",
     "gold_fact": "Whether the plan is on track and the money is funding the life you want, rather than an index someone else chose.",
     "gold_chunk_ids": [47, 63, 60], "bonus_story_ids": [],
     "notes": "Spans three articles, knowledge only."},

    # ---- out of scope: must abstain ----------------------------------------- #
    {"qid": "oos01", "category": "out_of_scope",
     "question": "Should I put some money into Bitcoin?",
     "gold_fact": None, "gold_chunk_ids": [], "bonus_story_ids": [],
     "notes": "HARD. Chunk 2 lists crypto among the too-many-options menu, so it will retrieve. The corpus takes no position on crypto specifically. Must not manufacture one."},
    {"qid": "oos02", "category": "out_of_scope",
     "question": "Which ELSS fund should I pick for my tax saving this year?",
     "gold_fact": None, "gold_chunk_ids": [], "bonus_story_ids": [],
     "notes": "Corpus never names schemes and never discusses ELSS or 80C."},
    {"qid": "oos03", "category": "out_of_scope",
     "question": "What's the LTCG tax rate on equity mutual funds right now?",
     "gold_fact": None, "gold_chunk_ids": [], "bonus_story_ids": [],
     "notes": "HARD. Taxation is mentioned in chunk 56 ('the taxation is messy') with no rates anywhere."},
    {"qid": "oos04", "category": "out_of_scope",
     "question": "Should I buy a house or put the money in equity instead?",
     "gold_fact": None, "gold_chunk_ids": [], "bonus_story_ids": [],
     "notes": "HARD. Real estate appears in chunk 2's list and 'house down payment' framing recurs. No comparison exists in the corpus."},
    {"qid": "oos05", "category": "out_of_scope",
     "question": "How do I open a demat account?",
     "gold_fact": None, "gold_chunk_ids": [], "bonus_story_ids": [],
     "notes": "Purely operational. Corpus is advisory commentary. Easy abstention baseline."},
]


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #


class _Tee:
    """Write to the real console and to the capture buffer at once.

    The console half is what makes a forty-question run watchable; the buffer
    half is what ends up in the report. Both see the same bytes in the same
    order, so neither is a reconstruction of the other.
    """

    def __init__(self, *streams):
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return False


@contextlib.contextmanager
def _capture():
    """Collect stdout AND log records, interleaved as the console shows them.

    The story-drop warnings are the reason the log side matters: they are the
    only signal that a story reached stage 4 and then did not reach the model,
    they arrive on stderr, and a report that captured stdout alone would show a
    story listed under STAGE 4 and silently missing from the answer.
    """
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-7s %(name)-24s %(message)s", datefmt="%H:%M:%S"
        )
    )
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        with contextlib.redirect_stdout(_Tee(sys.stdout, buffer)):
            yield buffer
    finally:
        root.removeHandler(handler)


# --------------------------------------------------------------------------- #
# Scoring
#
# Only retrieval is scored, and only on the one thing that is objectively
# checkable: did the passage the answer needs actually reach the model? Whether
# the generated prose is *right* is a judgement this script deliberately does not
# make up a number for — read the transcript.
# --------------------------------------------------------------------------- #


def _retrieved_labels(result: dict) -> dict[int, str]:
    """Map source_chunk_id -> the label it appears under in the final context."""
    labels: dict[int, str] = {}
    for i, hit in enumerate(result.get("knowledge") or [], start=1):
        source_id = hit.get("source_chunk_id")
        if source_id is not None:
            labels.setdefault(int(source_id), f"K{i}")
    for i, hit in enumerate(result.get("stories") or [], start=1):
        source_id = hit.get("source_chunk_id")
        if source_id is not None:
            labels.setdefault(int(source_id), f"S{i}")
    return labels


def _grade(case: dict, result: dict) -> dict:
    """Score knowledge and stories separately — they are not the same claim.

    Recall counts KNOWLEDGE only. The prompt makes knowledge the sole source of
    truth and forbids a story standing in for a missing fact, so a missing gold
    chunk is a retrieval failure while a missing story is a thinner answer. Adding
    the two together would book the architecture's own rule as a defect.
    """
    labels = _retrieved_labels(result)

    gold = [int(g) for g in case["gold_chunk_ids"]]
    found = {g: labels[g] for g in gold if g in labels}

    bonus = [int(b) for b in case.get("bonus_story_ids") or []]
    bonus_found = {b: labels[b] for b in bonus if b in labels}

    return {
        "gold": gold,
        "found": found,
        "missing": [g for g in gold if g not in found],
        # An out-of-scope case has no gold chunk, so recall is not defined for it.
        "recall": (len(found) / len(gold)) if gold else None,
        "bonus": bonus,
        "bonus_found": bonus_found,
        "bonus_missing": [b for b in bonus if b not in bonus_found],
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _fence(transcript: str) -> str:
    """Wrap the transcript so it survives Markdown untouched.

    A longer fence than anything inside it, so a stray ``` in a retrieved passage
    cannot end the block early and spill the rest of the transcript into the
    document as prose.
    """
    body = transcript.rstrip("\n")
    fence = "`" * max(4, max((len(m) for m in _backtick_runs(body)), default=0) + 1)
    return f"{fence}text\n{body}\n{fence}"


def _backtick_runs(text: str) -> list[str]:
    runs, current = [], ""
    for char in text:
        if char == "`":
            current += char
        elif current:
            runs.append(current)
            current = ""
    if current:
        runs.append(current)
    return runs


def _summary_table(rows: list[dict]) -> list[str]:
    out = [
        "| qid | category | gold knowledge | retrieved as | missing | recall | bonus stories | total ms |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        grade = row["grade"]
        if row["error"]:
            out.append(
                f"| {row['qid']} | {row['category']} | "
                f"{', '.join(map(str, grade['gold'])) or '—'} | "
                f"**error** | — | — | — | — |"
            )
            continue
        found = ", ".join(f"{cid}→{label}" for cid, label in grade["found"].items()) or "—"
        missing = ", ".join(map(str, grade["missing"])) or "—"
        recall = "n/a" if grade["recall"] is None else f"{grade['recall']:.0%}"
        if grade["bonus"]:
            attached = ", ".join(f"{cid}→{label}" for cid, label in grade["bonus_found"].items())
            bonus = f"{len(grade['bonus_found'])}/{len(grade['bonus'])}"
            bonus = f"{bonus} ({attached})" if attached else bonus
        else:
            bonus = "—"
        out.append(
            f"| {row['qid']} | {row['category']} | "
            f"{', '.join(map(str, grade['gold'])) or '—'} | {found} | {missing} | "
            f"{recall} | {bonus} | {row['total_ms']:.0f} |"
        )
    return out


def _category_table(rows: list[dict]) -> list[str]:
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(row["category"], []).append(row)

    out = ["| category | questions | gold knowledge | retrieved | recall | bonus stories attached |",
           "| --- | --- | --- | --- | --- | --- |"]
    for category, group in buckets.items():
        gold = sum(len(r["grade"]["gold"]) for r in group)
        found = sum(len(r["grade"]["found"]) for r in group)
        bonus = sum(len(r["grade"]["bonus"]) for r in group)
        bonus_found = sum(len(r["grade"]["bonus_found"]) for r in group)
        recall = f"{found / gold:.0%}" if gold else "n/a"
        stories = f"{bonus_found}/{bonus} ({bonus_found / bonus:.0%})" if bonus else "—"
        out.append(
            f"| {category} | {len(group)} | {gold} | {found} | {recall} | {stories} |"
        )
    return out


def _header(version, args, started: datetime, counts: dict) -> list[str]:
    if version.has_stories:
        indexed = (
            f"{counts['articles']} articles, {counts['knowledge_chunks']} knowledge, "
            f"{counts['story_chunks']} story ({counts['orphan_stories']} orphaned), "
            f"{counts['story_links']} links ({counts['cross_article_links']} cross-article), "
            f"{counts['questions']} questions"
        )
    else:
        indexed = (
            f"{counts['articles']} articles, {counts['chunks']} chunks "
            f"(all retrievable), {counts['questions']} questions"
        )
    return [
        f"# {version.label} — benchmark run",
        "",
        f"- **version** `{version.key}` — database `{version.database}`",
        f"- **indexed** {indexed}",
        f"- **embedding** `{config.EMBED_MODEL}`",
        f"- **reranker** `{config.RERANKER_MODEL}` ({config.RERANKER_BACKEND})",
        f"- **generation** `{config.GROQ_MODEL}`"
        + ("  _(skipped — `--retrieval-only`)_" if args.retrieval_only else ""),
        f"- **fetch_k** {args.fetch_k or config.VECTOR_TOP_K}"
        f" · **top_k** {args.top_k or config.RERANK_TOP_K}"
        f" · **stories attached** {not args.no_stories and version.has_stories}",
        f"- **started** {started.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
    ]


def _write_report(path, version, args, started, finished, counts, rows) -> None:
    lines: list[str] = []
    lines += _header(version, args, started, counts)
    lines += [
        f"- **finished** {finished.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        f" ({(finished - started).total_seconds():.0f}s for {len(rows)} questions)",
        "",
        "## Retrieval by category",
        "",
    ]
    lines += _category_table(rows)
    lines += [
        "",
        "## Per question",
        "",
    ]
    lines += _summary_table(rows)
    lines += [
        "",
        "`retrieved as` is where each gold chunk landed in the context the model "
        "was given: `K*` for knowledge, `S*` for an attached story. Recall counts "
        "gold KNOWLEDGE only — the passages the answer has to be built from. "
        "`bonus stories` counts illustrations that came along with them; a missing "
        "one is a thinner answer, not a wrong one, so it is kept out of recall. "
        "Neither number says anything about the wording of the answer — read the "
        "transcripts for that.",
        "",
        "---",
        "",
        "## Transcripts",
        "",
        "Each block below is exactly what `python query.py --repl` prints for that "
        "question, captured verbatim.",
        "",
    ]

    for row in rows:
        grade = row["grade"]
        lines += [f"### {row['qid']} — {row['category']}", ""]
        lines += [f"**Question:** {row['question']}", ""]
        if row["gold_fact"]:
            lines += [f"**Gold fact:** {row['gold_fact']}", ""]
        else:
            lines += ["**Gold fact:** _none — the corpus does not contain this; "
                      "the answer should abstain._", ""]
        if grade["gold"]:
            found = ", ".join(f"`{cid}`→`{label}`" for cid, label in grade["found"].items()) or "none"
            missing = ", ".join(f"`{cid}`" for cid in grade["missing"]) or "none"
            lines += [
                f"**Gold knowledge:** {', '.join(f'`{g}`' for g in grade['gold'])} · "
                f"**reached the model:** {found} · **missing:** {missing}",
                "",
            ]
        if grade["bonus"]:
            attached = ", ".join(
                f"`{cid}`→`{label}`" for cid, label in grade["bonus_found"].items()
            ) or "none"
            missing = ", ".join(f"`{cid}`" for cid in grade["bonus_missing"]) or "none"
            lines += [
                f"**Bonus stories:** {', '.join(f'`{b}`' for b in grade['bonus'])} · "
                f"**attached:** {attached} · **missing:** {missing}",
                "",
            ]
        lines += [f"**Notes:** {row['notes']}", ""]
        if row["error"]:
            lines += [f"> **Run failed:** `{row['error']}`", ""]
        lines += [_fence(row["transcript"]), ""]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #


def _select(args) -> list[dict]:
    cases = QUESTIONS
    if args.category:
        wanted = {c.strip() for c in args.category.split(",")}
        cases = [c for c in cases if c["category"] in wanted]
    if args.only:
        wanted = {q.strip() for q in args.only.split(",")}
        cases = [c for c in cases if c["qid"] in wanted]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("no questions matched --only/--category")
    return cases


def _ask(case: dict, args) -> tuple[dict | None, str, str | None]:
    """Run one question, returning (result, transcript, error).

    A failure is caught rather than raised: on a free key the realistic failure
    is one question exhausting a quota, and losing the thirty that would have
    followed it turns a slow run into a wasted one. The error goes in the report
    where the transcript would have been.
    """
    last_error: Exception | None = None
    for attempt in range(1, args.retries + 2):
        with _capture() as buffer:
            try:
                result = query.run(
                    case["question"],
                    fetch_k=args.fetch_k,
                    top_k=args.top_k,
                    attach_stories=not args.no_stories,
                    generate=not args.retrieval_only,
                    show_text=args.show_text,
                )
                return result, buffer.getvalue(), None
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(f"  error: {exc}")
                transcript = buffer.getvalue()

        if attempt <= args.retries:
            backoff = args.delay * (2 ** (attempt - 1)) or 5.0
            print(f"  retry {attempt}/{args.retries} in {backoff:.0f}s…")
            time.sleep(backoff)

    return None, transcript, str(last_error)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the benchmark question set against one version"
    )
    parser.add_argument("--version", choices=versions.keys(), default=config.DEFAULT_VERSION,
                        help="which version to benchmark (default: %(default)s)")
    parser.add_argument("--out", default=None,
                        help="output .md path (default: data/benchmark_<version>.md)")
    parser.add_argument("--delay", type=float, default=5.0,
                        help="seconds between questions, to stay under a free key's "
                             "per-minute limit (default: %(default)s)")
    parser.add_argument("--retries", type=int, default=1,
                        help="retries per question after a failure (default: %(default)s)")
    parser.add_argument("--only", default=None, help="comma-separated qids to run")
    parser.add_argument("--category", default=None, help="comma-separated categories to run")
    parser.add_argument("--limit", type=int, default=None, help="stop after this many questions")
    parser.add_argument("--fetch-k", type=int, default=None, help="candidates per arm before fusion")
    parser.add_argument("--top-k", type=int, default=None, help="knowledge chunks sent to the LLM")
    parser.add_argument("--no-stories", action="store_true", help="do not attach linked stories")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="skip generation entirely — no LLM calls, no quota spent")
    parser.add_argument("--show-text", action="store_true",
                        help="full chunk text in the transcript, not previews")
    args = parser.parse_args()

    config.setup_logging()

    version = db.set_version(args.version)
    query._preflight()

    out_path = config.DATA_DIR / f"benchmark_{version.key}.md"
    if args.out:
        from pathlib import Path
        out_path = Path(args.out)
    counts = retrieval.stats()

    print("loading models…")
    models.embed_query("warmup")
    if config.RERANKER_BACKEND == "local":
        models.rerank("warmup", [{"chunk_id": 0, "chunk_text": "warmup"}], 1)
    elif config.RERANKER_BACKEND == "http" and not models.reranker_healthy():
        # Loud, because a benchmark that silently fell back to fusion order would
        # be reported as a reranked run.
        raise SystemExit(f"reranker {config.RERANKER_URL} is not answering; aborting benchmark")

    cases = _select(args)
    print(f"\nrunning {len(cases)} question(s) against {version.label}")
    print(f"writing {out_path}")
    if not args.retrieval_only and args.delay:
        print(f"  pausing {args.delay:.0f}s between questions for the rate limit")

    started = datetime.now(timezone.utc).astimezone()
    rows: list[dict] = []

    for i, case in enumerate(cases, start=1):
        print(f"\n[{i}/{len(cases)}] {case['qid']}  {case['category']}")
        result, transcript, error = _ask(case, args)
        rows.append({
            **case,
            "transcript": transcript,
            "error": error,
            "grade": _grade(case, result or {}),
            "total_ms": sum((result or {}).get("timings_ms", {}).values()),
        })

        # No point pausing after the last one, and nothing was spent if the LLM
        # was never called.
        if i < len(cases) and args.delay and not args.retrieval_only:
            time.sleep(args.delay)

    finished = datetime.now(timezone.utc).astimezone()
    _write_report(out_path, version, args, started, finished, counts, rows)

    gold = sum(len(r["grade"]["gold"]) for r in rows)
    found = sum(len(r["grade"]["found"]) for r in rows)
    bonus = sum(len(r["grade"]["bonus"]) for r in rows)
    bonus_found = sum(len(r["grade"]["bonus_found"]) for r in rows)
    failed = [r["qid"] for r in rows if r["error"]]

    print(f"\ndone in {(finished - started).total_seconds():.0f}s")
    print(f"  gold knowledge reaching the model: {found}/{gold}"
          + (f" ({found / gold:.0%})" if gold else ""))
    if bonus:
        print(f"  bonus stories attached:            {bonus_found}/{bonus}"
              f" ({bonus_found / bonus:.0%})")
    if failed:
        print(f"  {len(failed)} question(s) failed: {', '.join(failed)}")
    print(f"  report: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    finally:
        tracing.flush()
        db.close()
