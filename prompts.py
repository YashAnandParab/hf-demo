"""Prompts and context formatting for the structured pipeline.

Knowledge and stories reach the model as separate, differently-labelled blocks,
because they carry different authority: knowledge is what may be asserted and
cited, a story is only an illustration of it.
"""
from __future__ import annotations

STRUCTURED_SYSTEM = """You answer questions using retrieved passages from a set of articles.

The context has two kinds of passage:

  KNOWLEDGE [K1], [K2], ...  factual, transferable claims. These are your evidence.
  STORY     [S1], [S2], ...  narrative that illustrates specific knowledge passages.

Rules:
  - Answer only from the context. If it does not contain the answer, say so plainly
    rather than filling the gap from your own knowledge.
  - Cite the knowledge passages you used inline, as [K1] or [K1][K3].
  - You may reference a story to make a point concrete, but never cite a story as
    the source of a fact, and never present its details as general truth.
  - Be direct. No preamble, no restating the question, no summary of what you are
    about to say."""


SUMMARY_SYSTEM = """You write one-paragraph retrieval summaries for narrative passages.

State what happens in the story AND what point it illustrates. The summary is what
gets embedded, so it must contain the vocabulary a reader would search with — not
the story's literary phrasing. Two to three sentences. No preamble."""


HQ_SYSTEM = """You write the questions a passage answers.

Given a passage, output the questions a real reader would type into a search box
that this passage answers well. Vary the phrasing: use the words a newcomer would
reach for, not only the passage's own vocabulary.

One question per line. No numbering, no bullets, no commentary."""


def format_context(knowledge: list[dict], stories: list[dict]) -> str:
    """Render retrieved chunks into the labelled context block the system prompt describes."""
    blocks: list[str] = []

    for i, hit in enumerate(knowledge, start=1):
        source = hit.get("article_name") or "unknown article"
        blocks.append(f"[K{i}] ({source})\n{hit.get('chunk_text', '').strip()}")

    for i, hit in enumerate(stories, start=1):
        source = hit.get("article_name") or "unknown article"
        illustrates = hit.get("linked_knowledge_label") or ""
        header = f"[S{i}] ({source}"
        header += f"; illustrates {illustrates})" if illustrates else ")"
        blocks.append(f"{header}\n{hit.get('chunk_text', '').strip()}")

    return "\n\n".join(blocks)


def build_user_prompt(question: str, context: str) -> str:
    return f"CONTEXT\n{context}\n\nQUESTION\n{question}\n\nANSWER"


def build_summary_prompt(story_text: str, knowledge_context: str) -> str:
    user = f"STORY:\n{story_text}"
    if knowledge_context:
        user += f"\n\nIT ILLUSTRATES THIS IDEA:\n{knowledge_context[:1500]}"
    return user


def build_hq_prompt(text: str, n: int) -> str:
    return f"Write {n} questions this passage answers.\n\nPASSAGE:\n{text}"
