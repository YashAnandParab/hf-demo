"""Prompts and context formatting, for both versions.

STRUCTURED_SYSTEM / format_context        -> the knowledge/story split
NORMAL_SYSTEM     / format_context_normal -> the flat baseline

Both system prompts are composed from the same named blocks. Everything that is
not about the knowledge/story split is byte-identical between the two arms, and
is written once below rather than twice. This is deliberate: if the structured
arm also carried a longer prompt, more reasoning steps, and a stronger answer
template, part of any measured win would be prompt engineering rather than
chunking, and there would be no way to separate the two after the fact.

The blocks are headed rather than numbered. Numbered steps would renumber as
soon as the story sections are inserted, and the two prompts would stop being
diffable line-for-line even where the text is the same.

--- domain ---

The corpus is investment and personal-finance articles. The dominant failure
mode there is not invention but recombination: a figure from one passage
attached to a period from another, a percentage silently annualised, a currency
dropped, a 2019 number presented as current. FIGURES AND PERIODS and CONFLICTING
PASSAGES exist for that, and apply to both arms equally.

--- structured ---

Knowledge and stories reach the model as separate, differently-labelled blocks.

KNOWLEDGE is the authoritative source used to derive the answer.
STORY is NOT evidence and is NOT used to determine whether a claim is true.
A STORY may only be used after the knowledge-based answer has been established,
and only as an illustrative example of the relevant knowledge.

The `illustrates K1, K3` header on each story block is the story <-> knowledge
mapping: it tells the model which story may illustrate which knowledge concept.
Those labels are renumbered at format time against the knowledge actually
present in this context (see _resolve_story_links). The IDs stored upstream are
corpus-wide; the [K1], [K2] numbering is per-request, so passing the stored
labels through unchanged would point the model at knowledge blocks that are not
in front of it — silently, and in a way no assertion downstream would catch.

A story is in the context because a retrieved knowledge chunk cited it, never
because it was retrieved — stories carry no embedding and are absent from the
full-text index. That is why the prompt below can state flatly that a story is
never evidence: the retrieval layer already guarantees it never competed as any.

The same property means stories are present by default rather than by
relevance, so the prompt has to push the other way: a story is omitted unless
it earns its place, and it may only illustrate the knowledge named in its own
header. "Or whatever it seems to illustrate" would hand back to the model the
judgement the mapping was built to remove.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared prompt blocks
#
# {TAG} is substituted with the citation prefix for the arm: K for structured,
# P for flat. No other braces appear in these strings.
# ---------------------------------------------------------------------------

_FRAMING = """\
You answer investment and personal-finance questions using retrieved passages \
from a set of articles. Readers range from beginners to experienced investors, \
so explain terms when the passages allow it.
"""

_DERIVE = """\
DERIVE THE ANSWER FROM THE PASSAGES
   Identify which passages bear on the question, and derive every factual,
   financial, and conceptual claim in your answer from them.
"""

_FIGURES = """\
FIGURES AND PERIODS
   Reproduce every number, percentage, currency, and date exactly as the passage
   states it. Do not round, convert between currencies, annualise, extrapolate,
   or otherwise recompute a figure, and do not combine figures from different
   passages into a new one, unless a passage performs that arithmetic itself.

   Every figure keeps the time reference it was given. If a passage says a
   number is as of a particular quarter or year, say so too. If a figure has no
   period attached in the passage, do not supply one, and do not describe it as
   current.
"""

_CONFLICTS = """\
CONFLICTING PASSAGES
   The articles were written at different times and may disagree. If two
   passages give different figures or contradictory claims, present both and
   cite each, noting their dates if the context supplies them. Do not silently
   choose one, and do not average or reconcile them.
"""

_CITATIONS = """\
CITATIONS
   Cite passages inline using [{TAG}1], [{TAG}2], etc.
   Every substantive claim in the answer must be traceable to a cited passage.
"""

_COVERAGE = """\
PARTIAL AND MISSING INFORMATION
   Most questions are only partly covered. Answer the part the passages support,
   then state plainly and specifically what they do not cover. Do not refuse the
   whole question because one part of it is unsupported, and do not present a
   partial answer as a complete one.

   If the passages contain nothing relevant, say so and stop.

   Do not use your own outside knowledge to fill a gap. Ordinary financial
   vocabulary and plain-language explanation are fine — supplying a fact,
   figure, date, or claim that is not in the context is not.
"""

_SCOPE = """\
SCOPE
   Report what the articles say. Do not give the reader a recommendation for
   their own situation, and do not turn a passage's general statement into
   personal advice.
"""

_STYLE = """\
ANSWER STYLE
   Be direct and natural.
   Do not begin with a preamble.
   Do not restate the question.
   Do not explain your retrieval process.
"""


# ---------------------------------------------------------------------------
# Structured-only blocks
# ---------------------------------------------------------------------------

_STRUCTURED_TYPES = """\
The context contains two fundamentally different types of passages:

  KNOWLEDGE [K1], [K2], ...
      The factual, financial, conceptual, and transferable information.
      KNOWLEDGE is the ONLY authoritative evidence for answering the question.

  STORY [S1], [S2], ...
      Narratives and real-world-style examples that illustrate particular
      knowledge passages.
      STORIES are NOT evidence and must NEVER be used to establish, verify, or
      introduce factual claims in the answer.

Answer from KNOWLEDGE first. STORIES exist only to make knowledge easier to
understand.
"""

_STRUCTURED_STORY_USE = """\
USING A STORY
   Each story block names the knowledge it illustrates, as `illustrates K1, K3`.
   A story may only be used to illustrate the knowledge named in its own header.
   Do not use a story to illustrate anything else, however well it seems to fit.

   Most answers do not need a story. Omit it unless it makes an
   already-established knowledge concept meaningfully clearer.

   Never treat events, people, numbers, actions, or outcomes from a STORY as
   independently established facts, and never use one to fill a gap in the
   KNOWLEDGE. If you refer to a story, identify it as an example; the concept it
   illustrates must still carry a KNOWLEDGE citation.

   KNOWLEDGE determines WHAT the answer is.
   STORY determines only HOW the concept may be illustrated.
"""

_STRUCTURED_RULE = """\
The central rule is:

  KNOWLEDGE = SOURCE OF TRUTH
  STORY = EXAMPLE ONLY

A story can explain a concept, but a story can never prove a concept.
"""

_NORMAL_TYPES = """\
The context contains passages [P1], [P2], ... retrieved from those articles.
They are all of one kind: there is no distinction between them, and none is
privileged over any other.
"""

_NORMAL_RULE = """\
The central rule is:

  THE RETRIEVED PASSAGES ARE THE ONLY SOURCE OF TRUTH.

If it is not in the context, it is not in the answer.
"""


def _compose(tag: str, *blocks: str) -> str:
    return "\n".join(b.replace("{TAG}", tag) for b in blocks).strip() + "\n"


STRUCTURED_SYSTEM = _compose(
    "K",
    _FRAMING,
    _STRUCTURED_TYPES,
    _DERIVE,
    _STRUCTURED_STORY_USE,
    _FIGURES,
    _CONFLICTS,
    _CITATIONS,
    _COVERAGE,
    _SCOPE,
    _STYLE,
    _STRUCTURED_RULE,
)

NORMAL_SYSTEM = _compose(
    "P",
    _FRAMING,
    _NORMAL_TYPES,
    _DERIVE,
    _FIGURES,
    _CONFLICTS,
    _CITATIONS,
    _COVERAGE,
    _SCOPE,
    _STYLE,
    _NORMAL_RULE,
)


# ---------------------------------------------------------------------------
# Question generation (HQ arm)
#
# Both HQ prompts forbid lifting the passage's distinctive phrasing. Questions
# that reuse source wording are matched trivially by the full-text half of the
# retrieval stack, which inflates the HQ arm's numbers in a way real user
# queries will not reproduce.
# ---------------------------------------------------------------------------

_HQ_RULES = """\
Write each question the way a user would type it into a search box. Paraphrase:
do not reuse the passage's distinctive phrasing, and do not quote it.

Each question must stand on its own. A reader who has not seen the passage must
be able to understand what is being asked, so no "this article", "the author",
"the above", or similar back-references.

Ask only about what the passage actually contains. Do not invent questions it
cannot answer.

Vary the phrasing and use vocabulary a real user might search for, including
beginner-friendly terminology where appropriate.

One question per line.
No numbering.
No bullets.
No commentary."""


# Only knowledge chunks are ever passed here: the HQ arm is a retrieval arm, and
# stories are not retrieval candidates, so questions written for one could never
# match anything. See ingest.ingest_questions.
HQ_SYSTEM = f"""You write questions that a passage can answer.

Given a KNOWLEDGE passage from a set of investment and personal-finance
articles, generate realistic questions about the financial concepts, principles,
mechanisms, explanations, and claims it contains.

{_HQ_RULES}"""


# The flat version has no typed chunks, so its HQ prompt cannot promise the model
# a "KNOWLEDGE passage" — roughly one chunk in six is a narrative. Asking for
# questions about financial concepts over a story produces questions the story
# does not answer, which pollutes the HQ arm with confident mismatches.
HQ_SYSTEM_NORMAL = f"""You write questions that a passage can answer.

Given a passage from a set of investment and personal-finance articles, generate
realistic questions covering whatever the passage actually contains — a concept,
a mechanism, a claim, or the events of a narrative.

{_HQ_RULES}"""


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

# Emitted instead of an empty string when retrieval returns nothing. A blank
# CONTEXT section is an invitation to answer from parameters; an explicit
# sentinel gives the refusal path something present to trigger on.
NO_CONTEXT = "=== NO PASSAGES RETRIEVED ==="

_DATE_KEYS = ("published_at", "article_date", "published_date", "date")
# `illustrates_chunk_ids` is what retrieval.linked_stories actually writes, and
# it holds db chunk_ids. It leads because the alternatives are fallbacks for
# hits assembled elsewhere.
_LINK_LIST_KEYS = (
    "illustrates_chunk_ids",
    "linked_knowledge_ids",
    "linked_chunk_ids",
    "links",
)


def _source_label(hit: dict) -> str:
    """`article name, 2023-04-12` — the date is the highest-value piece of
    metadata for a finance corpus, since it is what lets the model attach a
    period to an otherwise undated figure and notice that two passages disagree
    because they were written two years apart."""
    name = hit.get("article_name") or "unknown article"
    for key in _DATE_KEYS:
        value = hit.get(key)
        if value:
            return f"{name}, {str(value)[:10]}"
    return name


def _story_link_ids(hit: dict) -> list[str]:
    """IDs of the knowledge this story is linked to.

    Prefers a structured list field, which holds corpus-wide db chunk_ids; falls
    back to scraping the pre-rendered `linked_knowledge_label`, which holds
    per-request labels such as `K1, K3`. The two live in different ID spaces, so
    `_resolve_story_links` keeps a separate lookup for each.
    """
    for key in _LINK_LIST_KEYS:
        value = hit.get(key)
        if isinstance(value, (list, tuple, set)):
            return [str(v) for v in value if v is not None]
        if value is not None and not isinstance(value, (list, tuple, set)):
            return [str(value)]

    label = hit.get("linked_knowledge_label") or ""
    return [m.strip() for m in re.split(r"[,\s]+", label) if m.strip()]


def _resolve_story_links(
    knowledge: list[dict], stories: list[dict]
) -> list[tuple[dict, str]]:
    """Renumber each story's links against this request's [K1], [K2] ordering.

    Stored link IDs are corpus-wide; the K numbering is per-request. Passing the
    stored label through unchanged produces headers like `illustrates K7` in a
    context holding four knowledge blocks — the model then either ignores the
    mapping or attaches the story to the wrong concept, with nothing downstream
    to notice.

    A story whose links all fall outside the current context is dropped: it has
    no knowledge here to illustrate, so its only remaining role would be as
    free-floating narrative the prompt spends its length forbidding.

    The two ID spaces are resolved separately and never aliased into one table.
    A knowledge hit carries both a db `chunk_id` and the `source_chunk_id` it was
    given in the JSON, and those numbers overlap across hits — `json#49 (db 48)`
    registers both 49 and 48 — so a single flat index lets one hit's
    source_chunk_id shadow another hit's db id and silently attach a story to the
    wrong concept.
    """
    by_chunk_id: dict[str, int] = {}
    by_label: dict[str, int] = {}
    for i, hit in enumerate(knowledge, start=1):
        if hit.get("chunk_id") is not None:
            by_chunk_id.setdefault(str(hit["chunk_id"]), i)
        # `linked_knowledge_label` is already numbered against this same
        # enumeration, so K<i> maps to itself rather than to a corpus ID.
        by_label[f"K{i}"] = i

    resolved: list[tuple[dict, str]] = []
    for hit in stories:
        raw = _story_link_ids(hit)
        local = sorted(
            {
                j
                for r in raw
                if (j := by_chunk_id.get(r, by_label.get(r))) is not None
            }
        )

        if not local:
            log.warning(
                "story %s dropped: links %s resolve to no knowledge in context",
                hit.get("source_chunk_id") or hit.get("chunk_id") or "?",
                raw or "(none)",
            )
            continue

        if len(local) < len({r for r in raw if r}):
            log.warning(
                "story %s: %d of %d links fall outside this context",
                hit.get("source_chunk_id") or hit.get("chunk_id") or "?",
                len(raw) - len(local),
                len(raw),
            )

        resolved.append((hit, ", ".join(f"K{i}" for i in local)))

    return resolved


def format_context(knowledge: list[dict], stories: list[dict]) -> str:
    """Render knowledge and story chunks as explicitly separated context blocks.

    Knowledge is authoritative evidence.
    Stories are illustrative references only.
    """
    if not knowledge and not stories:
        return NO_CONTEXT

    blocks: list[str] = []

    if knowledge:
        blocks.append("=== AUTHORITATIVE KNOWLEDGE ===")
        for i, hit in enumerate(knowledge, start=1):
            blocks.append(
                f"[K{i}] ({_source_label(hit)})\n"
                f"{hit.get('chunk_text', '').strip()}"
            )

    linked = _resolve_story_links(knowledge, stories)
    if linked:
        blocks.append("=== ILLUSTRATIVE STORIES (NOT EVIDENCE) ===")
        for i, (hit, illustrates) in enumerate(linked, start=1):
            blocks.append(
                f"[S{i}] ({_source_label(hit)}; illustrates {illustrates})\n"
                f"{hit.get('chunk_text', '').strip()}"
            )

    return "\n\n".join(blocks)


def format_context_normal(chunks: list[dict]) -> str:
    """Render retrieved chunks as one undifferentiated block of numbered passages.

    No roles and no sections — the flat version has nothing to separate. The label
    is `[P1]` rather than a bare `[1]` because a bare bracketed number collides
    with the citation grammar some served models use for their own attached
    files: asked to cite `[1]`, they answer with a file-reference token instead.
    A letter-prefixed tag sits outside that grammar, so the model emits it
    literally — the same reason the structured version cites `[K1]`.

    Worth re-confirming against whichever model is actually serving, rather than
    inheriting the finding from the one it was first observed on.
    """
    if not chunks:
        return NO_CONTEXT

    blocks = ["=== RETRIEVED PASSAGES ==="]
    for i, hit in enumerate(chunks, start=1):
        blocks.append(
            f"[P{i}] ({_source_label(hit)})\n"
            f"{hit.get('chunk_text', '').strip()}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(question: str, context: str) -> str:
    return (
        f"CONTEXT\n"
        f"{context}\n\n"
        f"QUESTION\n"
        f"{question}\n\n"
        f"ANSWER\n"
        f"Use KNOWLEDGE to determine the answer, citing it inline. "
        f"Reproduce figures and their periods exactly. "
        f"Add a mapped STORY only if it makes the concept clearer."
    )


def build_user_prompt_normal(question: str, context: str) -> str:
    return (
        f"CONTEXT\n"
        f"{context}\n\n"
        f"QUESTION\n"
        f"{question}\n\n"
        f"ANSWER\n"
        f"Answer from the retrieved passages only, citing them inline. "
        f"Reproduce figures and their periods exactly."
    )


def build_hq_prompt(text: str, n: int) -> str:
    return (
        f"Write {n} questions this passage can answer.\n\n"
        f"PASSAGE:\n{text}"
    )