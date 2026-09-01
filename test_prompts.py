"""Checks for story->knowledge link resolution, which needs no Postgres.

This is the seam where the two ID spaces meet. `retrieval.linked_stories` hands
over db `chunk_id`s in `illustrates_chunk_ids`, `retrieval.label_stories` writes
a per-request `K<n>` label for the CLI to print, and `prompts` has to turn one of
those into the `illustrates K1, K3` header the model reads.

It is worth checking because getting it wrong fails silently in the direction
that looks like success: the CLI prints the story under STAGE 4 either way, and
only a WARNING on the way into the prompt says it never reached the model. The
answer then reads as a clean "not in the provided passages".

    pytest test_prompts.py
"""
from __future__ import annotations

import prompts
import retrieval


def _knowledge(source_chunk_id: int, chunk_id: int) -> dict:
    """A reranked knowledge hit, carrying both IDs the way a real row does."""
    return {
        "source_chunk_id": source_chunk_id,
        "chunk_id": chunk_id,
        "article_name": "An Article",
        "chunk_text": "a knowledge passage",
    }


def _story(source_chunk_id: int, chunk_id: int, parents: list[int]) -> dict:
    return {
        "source_chunk_id": source_chunk_id,
        "chunk_id": chunk_id,
        "illustrates_chunk_ids": list(parents),
        "article_name": "An Article",
        "chunk_text": "a story passage",
    }


def _labels(knowledge: list[dict], stories: list[dict]) -> list[str]:
    return [label for _, label in prompts._resolve_story_links(knowledge, stories)]


def test_a_story_resolves_to_the_position_of_the_knowledge_it_illustrates():
    # "The Wisdom of I Don't Know": the Socrates story hangs off db chunk 102,
    # which reranked second.
    knowledge = [_knowledge(47, 45), _knowledge(103, 102)]
    assert _labels(knowledge, [_story(101, 100, [102])]) == ["K2"]


def test_several_parents_collapse_to_one_sorted_label():
    knowledge = [_knowledge(47, 45), _knowledge(51, 50), _knowledge(46, 44)]
    assert _labels(knowledge, [_story(45, 46, [44, 45])]) == ["K1, K3"]


def test_a_story_whose_parents_all_missed_the_cut_is_dropped():
    knowledge = [_knowledge(47, 45)]
    assert _labels(knowledge, [_story(101, 100, [102])]) == []


def test_a_source_chunk_id_never_stands_in_for_another_hits_db_id():
    """The collision that made this resolve to the wrong concept rather than fail.

    `json#49 (db 48)` and `json#48 (db 47)` are both plausible neighbours in one
    context, and 48 means a different chunk in each ID space. A story linked to
    db 48 must land on the first hit, not on the second because its
    source_chunk_id happens to read 48.
    """
    knowledge = [_knowledge(49, 48), _knowledge(48, 47)]
    assert _labels(knowledge, [_story(1, 1, [48])]) == ["K1"]
    assert _labels(knowledge, [_story(1, 1, [47])]) == ["K2"]


def test_the_label_written_for_the_cli_is_read_back_as_the_same_position():
    """`linked_knowledge_label` is already per-request, not a corpus ID.

    Re-resolving it against corpus IDs is what dropped every story: `K1` was
    looked up among db chunk_ids, matched nothing, and the story was discarded.
    """
    knowledge = [_knowledge(47, 45), _knowledge(103, 102)]
    stories = [_story(101, 100, [102])]
    retrieval.label_stories(stories, knowledge)
    assert stories[0]["linked_knowledge_label"] == "K2"

    # Same story, structured field stripped: the fallback path must agree.
    del stories[0]["illustrates_chunk_ids"]
    assert _labels(knowledge, stories) == ["K2"]


def test_a_resolved_story_reaches_the_rendered_context():
    knowledge = [_knowledge(103, 102)]
    context = prompts.format_context(knowledge, [_story(101, 100, [102])])
    assert "a story passage" in context
    assert "K1" in context
