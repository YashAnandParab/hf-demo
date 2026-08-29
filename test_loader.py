"""Offline tests: chunk parsing, normalisation, link auditing, RRF fusion.

No database, no models, no API key. `pytest -q` from the project root.

The link-resolution test uses a fake cursor that mimics `RETURNING chunk_id`, so
the two-pass id mapping — the part most likely to silently corrupt data — is
exercised for real, including the cross-article case that a per-article map used
to drop.
"""
from __future__ import annotations

from pathlib import Path

from fusion import reciprocal_rank_fusion
from loader import (
    LoadReport,
    _strip_trailing_commas,
    audit,
    group_by_article,
    load_chunks,
    normalise,
    parse_objects,
)

# Mirrors the real file: concatenated multi-line objects, trailing commas, mixed
# chunk_type casing, list-valued links declared on the KNOWLEDGE chunk, and one
# link that reaches into another article.
SAMPLE = """
{"chunk_id": 12,
  "title": "The Shift Most Investors Need To Make",
  "source": "https://example.com/shift",
  "author": "Amar Pandit",
  "published_at": "2026-06-02",
  "site": "Happyrich Investor",
  "chunk_type": "Knowledge",
  "chunk":"Investing is about answering five simple questions.",
  "related_story_chunk_ids": [13, 1],
}
{"chunk_id": 13,
  "title": "The Shift Most Investors Need To Make",
  "source": "https://example.com/shift",
  "chunk_type": "story",
  "chunk":"Rohit is 42. When I asked how much he will need, he paused."
}
{"chunk_id": 14,
  "title": "The Shift Most Investors Need To Make",
  "source": "https://example.com/shift",
  "chunk_type": "Knowledge",
  "chunk":"The five questions are about saving, risk, need, timing and legacy.",
  "related_story_chunk_ids": 13
}
{"chunk_id": 1,
  "title": "The Cheesecake Factory Menu",
  "source": "https://example.com/cheesecake",
  "chunk_type": "story",
  "chunk":"The menu is thick, glossy and endless."
}
"""


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def test_trailing_comma_stripper_ignores_commas_inside_strings():
    import json

    text = '{"a": "value, with comma", "b": 1,}'
    cleaned, removed = _strip_trailing_commas(text)
    assert removed == 1
    assert "value, with comma" in cleaned
    assert json.loads(cleaned) == {"a": "value, with comma", "b": 1}


def test_trailing_comma_stripper_leaves_valid_json_alone():
    text = '{"a": [1, 2, 3], "b": {"c": 4}}'
    cleaned, removed = _strip_trailing_commas(text)
    assert removed == 0
    assert cleaned == text


def test_parses_concatenated_multiline_objects_with_trailing_commas():
    report = LoadReport()
    objects = parse_objects(SAMPLE, report)
    assert len(objects) == 4
    assert report.trailing_commas_repaired == 1
    assert [o["chunk_id"] for o in objects] == [12, 13, 14, 1]


def test_parses_a_wrapping_array_too():
    report = LoadReport()
    objects = parse_objects('[{"chunk_id": 1}, {"chunk_id": 2}]', report)
    assert [o["chunk_id"] for o in objects] == [1, 2]


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


def test_field_aliases_and_casing():
    report = LoadReport()
    chunk = normalise(
        {
            "chunk_id": 5,
            "title": "T",
            "source": "https://e.com/a",
            "published_at": "2026-07-21",
            "chunk_type": "Knowledge",
            "chunk": "body",
            "site": "S",
        },
        report,
    )
    assert chunk.article_name == "T"
    assert chunk.article_url == "https://e.com/a"
    assert chunk.published_date == "2026-07-21"
    assert chunk.chunk_text == "body"
    assert chunk.content_type == "knowledge"
    assert chunk.source_chunk_id == 5
    assert report.content_type_normalised == 1


def test_scalar_link_is_coerced_to_a_list():
    report = LoadReport()
    chunk = normalise(
        {"chunk_id": 2, "title": "T", "chunk_type": "knowledge", "chunk": "x",
         "related_story_chunk_id": 7},
        report,
    )
    assert chunk.links == [7]


def test_links_declared_on_a_story_are_dropped():
    """Only knowledge chunks declare links now; the reverse is a leftover."""
    report = LoadReport()
    chunk = normalise(
        {"chunk_id": 2, "title": "T", "chunk_type": "story", "chunk": "x",
         "related_story_chunk_ids": [7]},
        report,
    )
    assert chunk.links == []
    assert any("story chunk declares links" in w for w in report.warnings)


def test_old_link_direction_is_rejected_with_a_migration_note():
    """A file in the pre-inversion format must fail loudly, not ingest linkless."""
    report = LoadReport()
    normalise(
        {"chunk_id": 2, "title": "T", "chunk_type": "story", "chunk": "x",
         "related_knowledge_chunk_ids": [12, 14]},
        report,
    )
    assert report.legacy_link_keys == 1
    assert any("related_story_chunk_ids" in e for e in report.errors)


def test_unknown_chunk_type_is_rejected():
    report = LoadReport()
    assert normalise({"chunk_id": 1, "title": "T", "chunk_type": "opinion", "chunk": "x"}, report) is None
    assert report.errors


def test_empty_chunk_text_is_rejected():
    report = LoadReport()
    assert normalise({"chunk_id": 1, "title": "T", "chunk_type": "story", "chunk": "   "}, report) is None
    assert report.errors


# --------------------------------------------------------------------------- #
# Grouping + audit
# --------------------------------------------------------------------------- #


def _sample_chunks():
    report = LoadReport()
    return [c for c in (normalise(o, report) for o in parse_objects(SAMPLE, report)) if c], report


def test_groups_by_source_url():
    chunks, _ = _sample_chunks()
    groups = group_by_article(chunks)
    assert len(groups) == 2
    assert {len(g) for g in groups.values()} == {3, 1}


def test_audit_flags_a_story_nothing_points_at():
    report = LoadReport()
    chunks = [
        normalise({"chunk_id": 1, "title": "T", "chunk_type": "knowledge", "chunk": "k"}, report),
        normalise({"chunk_id": 2, "title": "T", "chunk_type": "story", "chunk": "s"}, report),
    ]
    audit(chunks, report)
    assert any("no knowledge chunk points at" in w for w in report.warnings)


def test_audit_is_quiet_when_every_story_is_cited():
    chunks, report = _sample_chunks()
    audit(chunks, report)
    assert not any("points at" in w for w in report.warnings)


def test_audit_counts_a_cross_article_link_without_warning():
    """Knowledge 12 cites story 1 in another article. That is allowed now."""
    chunks, report = _sample_chunks()
    audit(chunks, report)
    assert report.cross_article_links == 1
    assert not any("crosses articles" in w for w in report.warnings)


def test_audit_flags_dangling_link():
    report = LoadReport()
    chunks = [
        normalise({"chunk_id": 1, "title": "T", "chunk_type": "knowledge", "chunk": "k",
                   "related_story_chunk_ids": [99]}, report),
    ]
    audit(chunks, report)
    assert any("no such chunk_id" in w for w in report.warnings)


def test_audit_flags_knowledge_linked_to_knowledge():
    report = LoadReport()
    chunks = [
        normalise({"chunk_id": 1, "title": "T", "chunk_type": "knowledge", "chunk": "a"}, report),
        normalise({"chunk_id": 2, "title": "T", "chunk_type": "knowledge", "chunk": "b",
                   "related_story_chunk_ids": [1]}, report),
    ]
    audit(chunks, report)
    assert any("not a story" in w for w in report.warnings)


def test_loads_from_disk(tmp_path: Path):
    f = tmp_path / "chunks.json"
    f.write_text(SAMPLE, encoding="utf-8")
    chunks, report = load_chunks(f)
    assert len(chunks) == 4
    assert report.trailing_commas_repaired == 1


# --------------------------------------------------------------------------- #
# Two-pass link resolution
# --------------------------------------------------------------------------- #


class FakeCursor:
    """Mimics psycopg's RETURNING chunk_id with DB-assigned ids starting at 1000."""

    def __init__(self):
        self.next_id = 1000
        self.inserted: list[tuple] = []
        self.links: list[tuple] = []
        self._last = None

    def execute(self, sql, params=()):
        if "INSERT INTO structured_chunks" in sql and "RETURNING" in sql:
            self.inserted.append(params)
            self._last = {"chunk_id": self.next_id}
            self.next_id += 1
        elif "INSERT INTO articles" in sql:
            self._last = {"article_id": 1}

    def executemany(self, sql, rows):
        if "structured_chunk_links" in sql:
            self.links.extend(rows)

    def fetchone(self):
        return self._last


def _resolve(chunks):
    """The global two-pass resolution ingest_all performs, minus the database."""
    cur = FakeCursor()
    source_to_db: dict[int, int] = {}
    inserted: list[tuple] = []

    for chunk in chunks:
        cur.execute("INSERT INTO structured_chunks ... RETURNING chunk_id", (chunk.source_chunk_id,))
        db_id = cur.fetchone()["chunk_id"]
        inserted.append((chunk, db_id))
        if chunk.source_chunk_id is not None:
            source_to_db[chunk.source_chunk_id] = db_id

    by_source = {c.source_chunk_id: c for c, _ in inserted}
    rows = []
    for chunk, db_id in inserted:
        if chunk.content_type != "knowledge":
            continue
        for pos, target in enumerate(chunk.links):
            tdb, traw = source_to_db.get(target), by_source.get(target)
            if tdb and traw and traw.content_type == "story":
                rows.append((db_id, tdb, pos))
    return source_to_db, rows


def test_two_pass_resolution_maps_author_ids_to_db_ids():
    """Author ids 12/13/14/1 must resolve to DB ids, not be inserted verbatim."""
    chunks, _ = _sample_chunks()
    source_to_db, rows = _resolve(chunks)

    assert source_to_db == {12: 1000, 13: 1001, 14: 1002, 1: 1003}
    # knowledge 12 cites stories 13 and 1, in the order given
    assert (1000, 1001, 0) in rows
    assert (1000, 1003, 1) in rows
    # knowledge 14's scalar link resolved too
    assert (1002, 1001, 0) in rows
    assert len(rows) == 3


def test_cross_article_link_resolves_because_the_map_is_global():
    """Story 1 lives in the cheesecake article; knowledge 12 lives in shift."""
    chunks, _ = _sample_chunks()
    twelve = next(c for c in chunks if c.source_chunk_id == 12)
    one = next(c for c in chunks if c.source_chunk_id == 1)
    assert twelve.article_url != one.article_url

    _, rows = _resolve(chunks)
    assert (1000, 1003, 1) in rows, "cross-article link was dropped"


def test_link_to_a_knowledge_chunk_is_not_written():
    report = LoadReport()
    chunks = [
        normalise({"chunk_id": 1, "title": "T", "chunk_type": "knowledge", "chunk": "a"}, report),
        normalise({"chunk_id": 2, "title": "T", "chunk_type": "knowledge", "chunk": "b",
                   "related_story_chunk_ids": [1]}, report),
    ]
    _, rows = _resolve(chunks)
    assert rows == []


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #


def _hit(chunk_id: int, score: float = 0.5) -> dict:
    return {"chunk_id": chunk_id, "chunk_text": f"chunk {chunk_id}", "score": score}


def test_rrf_rewards_a_chunk_found_by_several_arms():
    fused = reciprocal_rank_fusion(
        {
            "vector": [_hit(1), _hit(2)],
            "fts": [_hit(2), _hit(3)],
            "hq": [_hit(2)],
        }
    )
    assert fused[0]["chunk_id"] == 2          # only chunk in all three arms
    assert set(fused[0]["sources"]) == {"vector", "fts", "hq"}
    assert fused[0]["fusion_rank"] == 1


def test_rrf_records_per_arm_ranks_and_carries_the_matched_question():
    hq_hit = _hit(7)
    hq_hit["matched_question"] = "how does compounding work?"
    fused = reciprocal_rank_fusion({"vector": [_hit(9), _hit(7)], "hq": [hq_hit]})
    seven = next(h for h in fused if h["chunk_id"] == 7)
    assert seven["arm_ranks"] == {"vector": 2, "hq": 1}
    assert seven["matched_question"] == "how does compounding work?"


def test_rrf_on_empty_arms_returns_nothing():
    assert reciprocal_rank_fusion({"vector": [], "fts": [], "hq": []}) == []
