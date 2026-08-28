"""Loader for hand-written flat chunk files.

The input format is a stream of concatenated JSON objects, one chunk each:

    {"chunk_id": 1, "title": "...", "source": "https://...", "author": "...",
     "published_at": "2025-08-19", "clipped_at": "...", "site": "...",
     "chunk_type": "story", "chunk": "...",
     "related_knowledge_chunk_ids": [12, 14]}

That is neither a JSON array nor strict JSONL (objects span multiple lines), and
hand-written files pick up trailing commas. This module parses it anyway, repairs
what is safely repairable, and reports everything it had to touch.

Normalisation applied:
    title        -> article_name
    source       -> article_url
    published_at -> published_date
    chunk        -> chunk_text
    chunk_type   -> content_type   (lowercased; "Knowledge" -> "knowledge")
    chunk_id     -> source_chunk_id
    related_knowledge_chunk_ids -> links (always a list, even when given as an int)

Nothing here touches the database or an API, so `ingest.py --dry-run` can run the
whole parse-and-audit pass for free.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("loader")

_FIELD_ALIASES = {
    "title": "article_name",
    "article_name": "article_name",
    "source": "article_url",
    "url": "article_url",
    "article_url": "article_url",
    "published_at": "published_date",
    "published_date": "published_date",
    "chunk": "chunk_text",
    "text": "chunk_text",
    "chunk_text": "chunk_text",
    "chunk_type": "content_type",
    "content_type": "content_type",
    "author": "author",
    "site": "site",
    "clipped_at": "clipped_at",
    "summary": "story_summary",
    "story_summary": "story_summary",
}

_LINK_KEYS = (
    "related_knowledge_chunk_ids",
    "related_knowledge_chunk_id",
    "related_chunk_ids",
    "related_chunk_id",
)


@dataclass
class RawChunk:
    source_chunk_id: int | None
    article_name: str
    article_url: str | None
    author: str | None
    published_date: str | None
    site: str | None
    content_type: str
    chunk_text: str
    story_summary: str | None
    links: list[int]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


@dataclass
class LoadReport:
    objects_parsed: int = 0
    trailing_commas_repaired: int = 0
    content_type_normalised: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def log(self) -> None:
        log.info("parsed %d chunk objects", self.objects_parsed)
        if self.trailing_commas_repaired:
            log.warning(
                "repaired %d trailing comma(s) — the source file is not valid JSON",
                self.trailing_commas_repaired,
            )
        if self.content_type_normalised:
            log.warning(
                "lowercased %d chunk_type value(s) (e.g. 'Knowledge' -> 'knowledge')",
                self.content_type_normalised,
            )
        for w in self.warnings:
            log.warning("%s", w)
        for e in self.errors:
            log.error("%s", e)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def _strip_trailing_commas(text: str) -> tuple[str, int]:
    """Remove trailing commas outside of string literals."""
    out: list[str] = []
    in_string = False
    escaped = False
    removed = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue

        if ch == ",":
            j = i + 1
            while j < len(text) and text[j] in " \t\r\n":
                j += 1
            if j < len(text) and text[j] in "}]":
                removed += 1
                i += 1  # drop the comma, keep the whitespace
                continue
        out.append(ch)
        i += 1
    return "".join(out), removed


def parse_objects(text: str, report: LoadReport) -> list[dict]:
    """Parse a stream of concatenated JSON objects, repairing trailing commas."""
    cleaned, removed = _strip_trailing_commas(text)
    report.trailing_commas_repaired += removed

    decoder = json.JSONDecoder()
    objects: list[dict] = []
    idx = 0
    length = len(cleaned)

    while idx < length:
        while idx < length and cleaned[idx] in " \t\r\n":
            idx += 1
        if idx >= length:
            break
        # tolerate a wrapping array or comma-separated objects
        if cleaned[idx] in "[],":
            idx += 1
            continue
        try:
            obj, end = decoder.raw_decode(cleaned, idx)
        except json.JSONDecodeError as exc:
            line = cleaned.count("\n", 0, idx) + 1
            report.errors.append(f"unparseable JSON near line {line}: {exc.msg}")
            # skip to the next line that starts an object and keep going
            nxt = cleaned.find("\n{", idx)
            if nxt == -1:
                break
            idx = nxt + 1
            continue
        if isinstance(obj, dict):
            objects.append(obj)
        elif isinstance(obj, list):
            objects.extend(o for o in obj if isinstance(o, dict))
        idx = end

    return objects


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


def _extract_links(obj: dict, report: LoadReport, chunk_id) -> list[int]:
    for key in _LINK_KEYS:
        if key not in obj:
            continue
        value = obj[key]
        if value is None:
            return []
        if isinstance(value, int):
            return [value]
        if isinstance(value, list):
            out = []
            for v in value:
                try:
                    out.append(int(v))
                except (TypeError, ValueError):
                    report.warnings.append(f"chunk {chunk_id}: ignoring non-integer link {v!r}")
            return out
        report.warnings.append(f"chunk {chunk_id}: unexpected {key} type {type(value).__name__}")
        return []
    return []


def normalise(obj: dict, report: LoadReport) -> RawChunk | None:
    mapped: dict[str, Any] = {}
    for key, value in obj.items():
        target = _FIELD_ALIASES.get(key.strip().lower())
        if target:
            mapped[target] = value

    chunk_id = obj.get("chunk_id")
    text = (mapped.get("chunk_text") or "").strip()
    if not text:
        report.errors.append(f"chunk {chunk_id}: empty chunk text, skipped")
        return None

    name = (mapped.get("article_name") or "").strip()
    if not name:
        report.errors.append(f"chunk {chunk_id}: missing title, skipped")
        return None

    raw_type = str(mapped.get("content_type") or "").strip()
    content_type = raw_type.lower()
    if raw_type and raw_type != content_type:
        report.content_type_normalised += 1
    if content_type not in {"knowledge", "story"}:
        report.errors.append(
            f"chunk {chunk_id}: chunk_type {raw_type!r} is neither 'knowledge' nor 'story', skipped"
        )
        return None

    try:
        source_chunk_id = int(chunk_id) if chunk_id is not None else None
    except (TypeError, ValueError):
        report.warnings.append(f"non-integer chunk_id {chunk_id!r}; links to it will not resolve")
        source_chunk_id = None

    links = _extract_links(obj, report, chunk_id)
    if links and content_type == "knowledge":
        report.warnings.append(
            f"chunk {chunk_id}: knowledge chunk declares links; ignored "
            f"(links are declared on stories)"
        )
        links = []

    return RawChunk(
        source_chunk_id=source_chunk_id,
        article_name=name,
        article_url=(mapped.get("article_url") or None),
        author=(mapped.get("author") or None),
        published_date=(mapped.get("published_date") or None),
        site=(mapped.get("site") or None),
        content_type=content_type,
        chunk_text=text,
        story_summary=(mapped.get("story_summary") or None),
        links=links,
        raw=obj,
    )


def load_chunks(path: Path) -> tuple[list[RawChunk], LoadReport]:
    report = LoadReport()
    files = (
        sorted(p for p in path.rglob("*") if p.suffix in {".json", ".jsonl", ".txt"})
        if path.is_dir()
        else [path]
    )

    chunks: list[RawChunk] = []
    for file in files:
        objects = parse_objects(file.read_text(encoding="utf-8"), report)
        report.objects_parsed += len(objects)
        for obj in objects:
            chunk = normalise(obj, report)
            if chunk:
                chunks.append(chunk)
    return chunks, report


# --------------------------------------------------------------------------- #
# Grouping + link validation
# --------------------------------------------------------------------------- #


def group_by_article(chunks: list[RawChunk]) -> dict[str, list[RawChunk]]:
    """Group chunks into articles, keyed by URL when present, else by title."""
    groups: dict[str, list[RawChunk]] = {}
    for chunk in chunks:
        key = chunk.article_url or f"title::{chunk.article_name}"
        groups.setdefault(key, []).append(chunk)
    return groups


def audit(chunks: list[RawChunk], report: LoadReport) -> None:
    """Warn about links and stories that will not behave as expected."""
    by_id = {c.source_chunk_id: c for c in chunks if c.source_chunk_id is not None}

    seen: set[int] = set()
    for chunk in chunks:
        if chunk.source_chunk_id is None:
            continue
        if chunk.source_chunk_id in seen:
            report.warnings.append(f"duplicate chunk_id {chunk.source_chunk_id}")
        seen.add(chunk.source_chunk_id)

    for chunk in chunks:
        for target in chunk.links:
            other = by_id.get(target)
            if other is None:
                report.warnings.append(
                    f"chunk {chunk.source_chunk_id}: link to {target} — no such chunk_id, dropped"
                )
            elif other.content_type != "knowledge":
                report.warnings.append(
                    f"chunk {chunk.source_chunk_id}: link to {target} which is a "
                    f"'{other.content_type}' chunk, not knowledge — dropped"
                )
            elif (other.article_url or other.article_name) != (
                chunk.article_url or chunk.article_name
            ):
                report.warnings.append(
                    f"chunk {chunk.source_chunk_id}: link to {target} crosses articles — dropped"
                )

    linked_knowledge = {t for c in chunks if c.content_type == "story" for t in c.links}
    for chunk in chunks:
        if chunk.content_type == "story" and not chunk.links:
            report.warnings.append(
                f"chunk {chunk.source_chunk_id} ('{chunk.article_name[:40]}') is a story with no "
                f"links — under knowledge-first retrieval it can never be reached. "
                f"Add related_knowledge_chunk_ids, or set STORY_RETRIEVAL_MODE=include."
            )
    if linked_knowledge:
        log.info("%d knowledge chunk(s) have at least one story attached", len(linked_knowledge))


# --------------------------------------------------------------------------- #
# Token estimate — used only for the token_count column and dry-run output
# --------------------------------------------------------------------------- #


_WORD = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    """Rough token count. Close enough for reporting; nothing depends on exactness."""
    return len(_WORD.findall(text))
