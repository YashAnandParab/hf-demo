"""Loader for hand-written flat chunk files.

The input format is a stream of concatenated JSON objects, one chunk each:

    {"chunk_id": 2, "title": "...", "source": "https://...", "author": "...",
     "published_at": "2025-08-19", "clipped_at": "...", "site": "...",
     "chunk_type": "knowledge", "chunk": "...",
     "related_story_chunk_ids": [1]}

That is neither a JSON array nor strict JSONL (objects span multiple lines), and
hand-written files pick up trailing commas. This module parses it anyway, repairs
what is safely repairable, and reports everything it had to touch.

Links are declared on the KNOWLEDGE chunk and point at the stories that
illustrate it. A story never declares anything and is never retrieved on its own
merits — it reaches the model only by being attached to a knowledge chunk that
survived reranking. Files written the other way round (a story declaring
`related_knowledge_chunk_ids`) are detected and rejected with a migration note
rather than silently losing every link.

A link may cross articles: a knowledge chunk in one article can point at a story
in another, so link resolution at ingest time is global rather than per-article.

Normalisation applied:
    title        -> article_name
    source       -> article_url
    published_at -> published_date
    chunk        -> chunk_text
    chunk_type   -> content_type   (lowercased; "Knowledge" -> "knowledge")
    chunk_id     -> source_chunk_id
    related_story_chunk_ids -> links (always a list, even when given as an int)

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
}

_LINK_KEYS = (
    "related_story_chunk_ids",
    "related_story_chunk_id",
    "related_story_ids",
    "related_stories",
)

# The pre-inversion format. Recognised only so the failure is a clear message
# instead of an ingest that quietly produces zero links.
_LEGACY_LINK_KEYS = (
    "related_knowledge_chunk_ids",
    "related_knowledge_chunk_id",
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
    links: list[int]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


@dataclass
class LoadReport:
    objects_parsed: int = 0
    trailing_commas_repaired: int = 0
    content_type_normalised: int = 0
    legacy_link_keys: int = 0
    cross_article_links: int = 0
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
        if self.cross_article_links:
            log.info(
                "%d link(s) point at a story in a different article — allowed, resolved globally",
                self.cross_article_links,
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
    for key in _LEGACY_LINK_KEYS:
        if key in obj and obj[key]:
            report.legacy_link_keys += 1
            report.errors.append(
                f"chunk {chunk_id}: uses '{key}'. Links are now declared on the KNOWLEDGE "
                f"chunk as 'related_story_chunk_ids' pointing at its stories. This file is "
                f"in the old story->knowledge format and would ingest with no links at all."
            )
            return []

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
    if links and content_type == "story":
        report.warnings.append(
            f"chunk {chunk_id}: story chunk declares links; ignored "
            f"(links are declared on knowledge chunks and point at stories)"
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


def _article_key(chunk: RawChunk) -> str:
    return chunk.article_url or f"title::{chunk.article_name}"


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
            elif other.content_type != "story":
                report.warnings.append(
                    f"chunk {chunk.source_chunk_id}: link to {target} which is a "
                    f"'{other.content_type}' chunk, not a story — dropped"
                )
            elif _article_key(other) != _article_key(chunk):
                # Allowed: a knowledge chunk may borrow a story from another article.
                report.cross_article_links += 1

    # A story is reachable only by being pointed at. Nothing else can retrieve it.
    referenced = {
        t for c in chunks if c.content_type == "knowledge" for t in c.links
    }
    for chunk in chunks:
        if chunk.content_type != "story":
            continue
        if chunk.source_chunk_id is None or chunk.source_chunk_id not in referenced:
            report.warnings.append(
                f"chunk {chunk.source_chunk_id} ('{chunk.article_name[:40]}') is a story that no "
                f"knowledge chunk points at. Stories are never retrieved directly, so nothing "
                f"can ever reach it — add its id to a knowledge chunk's related_story_chunk_ids."
            )

    with_stories = sum(1 for c in chunks if c.content_type == "knowledge" and c.links)
    if with_stories:
        log.info("%d knowledge chunk(s) have at least one story attached", with_stories)


# --------------------------------------------------------------------------- #
# Token estimate — used only for the token_count column and dry-run output
# --------------------------------------------------------------------------- #


_WORD = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    """Rough token count. Close enough for reporting; nothing depends on exactness."""
    return len(_WORD.findall(text))
