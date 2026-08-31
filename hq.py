"""Hypothetical questions.

For every chunk, the LLM writes the questions that chunk answers. Those questions
are embedded and searched as a third retrieval arm, which closes the gap between
how a passage is written and how a reader asks about it.

Generation failures are non-fatal: a chunk with no questions is still reachable
through the vector and full-text arms.
"""
from __future__ import annotations

import logging
import re

import config
import prompts
import versions
from llm import chat

log = logging.getLogger("hq")

_LEADING_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def _system_prompt() -> str:
    """The structured version only ever passes knowledge chunks here, so its prompt
    can say so. The flat version passes every chunk, narratives included."""
    return prompts.HQ_SYSTEM if versions.active().has_stories else prompts.HQ_SYSTEM_NORMAL


def generate_questions(text: str, n: int | None = None) -> list[str]:
    n = config.HQ_PER_CHUNK if n is None else n
    if n <= 0 or not text.strip():
        return []
    try:
        raw = chat(_system_prompt(), prompts.build_hq_prompt(text, n), max_tokens=256)
    except Exception as exc:  # noqa: BLE001
        log.warning("question generation failed: %s", exc)
        return []
    return _parse(raw, n)


def generate_questions_bulk(texts: list[str], n: int | None = None) -> list[list[str]]:
    return [generate_questions(text, n) for text in texts]


def _parse(raw: str, n: int) -> list[str]:
    questions: list[str] = []
    for line in raw.splitlines():
        cleaned = _LEADING_MARKER.sub("", line).strip().strip('"')
        if len(cleaned) < 8:
            continue
        questions.append(cleaned)
        if len(questions) >= n:
            break
    return questions
