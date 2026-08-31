"""Derive the flat-baseline chunk file from the structured one.

    python tools/make_normal_chunks.py                     # data/chunks.json -> data/chunks_normal.json
    python tools/make_normal_chunks.py in.json out.json

The "normal chunking" version is the same corpus with the structure taken away:

    chunk_type               -> dropped   (no knowledge/story distinction)
    related_story_chunk_ids  -> dropped   (no knowledge -> story links)

Everything else — article metadata and the chunk text itself — is copied through
byte-for-byte, so the only variable between the two versions is the structure.
That is the whole point of the comparison: same text, same embedding model, same
retrieval arms, same reranker; only the structure differs.

Regenerate this file whenever data/chunks.json changes, otherwise the two
versions are answering from different corpora and nothing they disagree about
means anything.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from loader import LoadReport, parse_objects  # noqa: E402

# Every spelling loader.py accepts for the two structural fields. Dropping only
# the canonical name would leave an alias behind for the loader to pick up.
_DROP = {
    "chunk_type",
    "content_type",
    "related_story_chunk_ids",
    "related_story_chunk_id",
    "related_story_ids",
    "related_stories",
    # the pre-inversion spellings, in case an old file is passed in
    "related_knowledge_chunk_ids",
    "related_knowledge_chunk_id",
}


def strip(obj: dict) -> dict:
    """Drop the structural fields, keeping the original key order."""
    return {k: v for k, v in obj.items() if k.strip().lower() not in _DROP}


def main() -> None:
    args = sys.argv[1:]
    src = Path(args[0]) if args else config.DATA_DIR / "chunks.json"
    dst = Path(args[1]) if len(args) > 1 else config.DATA_DIR / "chunks_normal.json"

    if not src.exists():
        raise SystemExit(f"input not found: {src}")

    report = LoadReport()
    objects = parse_objects(src.read_text(encoding="utf-8"), report)
    if not objects:
        raise SystemExit(f"no chunk objects parsed from {src}")

    stripped = [strip(obj) for obj in objects]

    # Written as a real JSON array — the loader accepts one, and unlike the
    # hand-written source this file is generated, so there is no reason for it not
    # to be valid JSON.
    dst.write_text(json.dumps(stripped, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    types = {}
    for obj in objects:
        key = str(obj.get("chunk_type") or obj.get("content_type") or "?").lower()
        types[key] = types.get(key, 0) + 1
    links = sum(len(obj.get("related_story_chunk_ids") or []) for obj in objects)

    print(f"read   {src}")
    print(f"  {len(objects)} chunks: " + ", ".join(f"{n} {t}" for t, n in sorted(types.items())))
    print(f"  {links} knowledge -> story link(s)")
    print(f"wrote  {dst}")
    print(f"  {len(stripped)} chunks, no chunk_type, no links — every chunk is retrievable")
    if report.errors:
        print(f"\n  {len(report.errors)} parse error(s) in the source file:")
        for err in report.errors:
            print(f"    {err}")


if __name__ == "__main__":
    main()
