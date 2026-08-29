"""Prompts and context formatting for the structured pipeline.

Knowledge and stories reach the model as separate, differently-labelled blocks.

KNOWLEDGE is the authoritative source used to derive the answer.
STORY is NOT evidence and is NOT used to determine whether a claim is true.
A STORY may only be used after the knowledge-based answer has been established,
and only as an illustrative example of the relevant knowledge.

The `illustrates K1, K3` header on each story block is the story ↔ knowledge
mapping: it tells the model which story may illustrate which knowledge concept.

A story is in the context because a retrieved knowledge chunk cited it, never
because it was retrieved — stories carry no embedding and are absent from the
full-text index. That is why the prompt below can state flatly that a story is
never evidence: the retrieval layer already guarantees it never competed as any.
"""
from __future__ import annotations

STRUCTURED_SYSTEM = """You answer questions using retrieved passages from a set of articles.

The context contains two fundamentally different types of passages:

  KNOWLEDGE [K1], [K2], ...
      These contain the factual, financial, conceptual, and transferable information.
      KNOWLEDGE is the ONLY authoritative evidence for answering the user's question.

  STORY [S1], [S2], ...
      These are narratives or real-world-style examples that illustrate particular
      knowledge passages.
      STORIES are NOT evidence and must NEVER be used to establish, verify, or
      introduce factual claims in the answer.

IMPORTANT:
The user's question must be answered from KNOWLEDGE first.
STORIES exist only to make the knowledge easier to understand.

Follow this reasoning process:

1. IDENTIFY THE RELEVANT KNOWLEDGE
   Determine which KNOWLEDGE passages directly answer the user's question.

2. BUILD THE ANSWER FROM KNOWLEDGE
   Derive every factual, financial, conceptual, or general claim from the
   relevant KNOWLEDGE passages.

   Do NOT use information from a STORY to fill a missing gap in the KNOWLEDGE.

3. IDENTIFY A RELEVANT STORY
   If a STORY is mapped to or clearly illustrates the relevant KNOWLEDGE,
   you may use it as an example to make the explanation more concrete.

4. USE THE STORY ONLY AS AN ILLUSTRATION
   The STORY must support the explanation of an already-established
   KNOWLEDGE concept.

   Never treat events, people, numbers, actions, outcomes, or other details
   from a STORY as independently established facts.

5. KEEP KNOWLEDGE AND STORY ROLES STRICTLY SEPARATE
   KNOWLEDGE → determines WHAT the answer is.
   STORY → determines HOW the concept may be illustrated.

6. CITATIONS
   Cite KNOWLEDGE passages inline using [K1], [K2], etc.

   Never cite a STORY as evidence for a factual claim.
   If you mention a story, you may identify it as an example, but the underlying
   concept must be supported by a KNOWLEDGE citation.

7. MISSING INFORMATION
   If the KNOWLEDGE passages do not contain enough information to answer the
   question, say so plainly.

   Do NOT use a STORY to compensate for missing knowledge.
   Do NOT use your own outside knowledge to complete the answer.

8. ANSWER STYLE
   Be direct and natural.
   Do not begin with a preamble.
   Do not restate the question.
   Do not explain your retrieval process.

   When appropriate, structure the response as:
     - explanation based on KNOWLEDGE
     - followed by a relevant STORY as an example

The central rule is:

  KNOWLEDGE = SOURCE OF TRUTH
  STORY = EXAMPLE ONLY

A story can explain a concept, but a story can never prove a concept.
"""


# Only knowledge chunks are ever passed here: the HQ arm is a retrieval arm, and
# stories are not retrieval candidates, so questions written for one could never
# match anything. See ingest.ingest_questions.
HQ_SYSTEM = """You write questions that a passage can answer.

Given a KNOWLEDGE passage, generate realistic questions that a user might type
into a search system: questions about the financial concepts, principles,
mechanisms, explanations, and claims contained in the passage.

Vary the phrasing and use vocabulary a real user might search for, including
beginner-friendly terminology where appropriate.

One question per line.
No numbering.
No bullets.
No commentary."""


def format_context(knowledge: list[dict], stories: list[dict]) -> str:
    """Render knowledge and story chunks as explicitly separated context blocks.

    Knowledge is authoritative evidence.
    Stories are illustrative references only.
    """
    blocks: list[str] = []

    if knowledge:
        blocks.append("=== AUTHORITATIVE KNOWLEDGE ===")

        for i, hit in enumerate(knowledge, start=1):
            source = hit.get("article_name") or "unknown article"
            blocks.append(
                f"[K{i}] ({source})\n"
                f"{hit.get('chunk_text', '').strip()}"
            )

    if stories:
        blocks.append("=== ILLUSTRATIVE STORIES (NOT EVIDENCE) ===")

        for i, hit in enumerate(stories, start=1):
            source = hit.get("article_name") or "unknown article"
            illustrates = hit.get("linked_knowledge_label") or ""

            header = f"[S{i}] ({source}"
            if illustrates:
                header += f"; illustrates {illustrates}"
            header += ")"

            blocks.append(
                f"{header}\n"
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
        f"Use KNOWLEDGE to determine the answer. "
        f"If useful, use a mapped STORY only as an illustrative example."
    )


def build_hq_prompt(text: str, n: int) -> str:
    return (
        f"Write {n} questions this passage can answer.\n\n"
        f"PASSAGE:\n{text}"
    )
