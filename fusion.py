"""Reciprocal Rank Fusion over the retrieval arms.

RRF combines rankings, not scores, so a cosine similarity and a `ts_rank_cd`
value never have to be made comparable — only their positions matter:

    score(chunk) = Σ  weight(arm) / (RRF_K + rank_in_arm)

`RRF_K` (default 60) controls how sharply position matters: lower makes the top
of each arm dominate. The per-arm weights exist because the arms are not equally
trustworthy — FTS starts at 0.8 because `ts_rank_cd` rewards term repetition,
which over-favours long chunks.
"""
from __future__ import annotations

from typing import Mapping

import config

_WEIGHTS = {
    "vector": lambda: config.WEIGHT_VECTOR,
    "fts": lambda: config.WEIGHT_FTS,
    "hq": lambda: config.WEIGHT_HQ,
}


def reciprocal_rank_fusion(
    arms: Mapping[str, list[dict]], top_k: int | None = None
) -> list[dict]:
    """Fuse per-arm hit lists into one ranked list.

    Each surviving hit carries `fusion_score`, `sources` (which arms found it,
    best-contribution first), `arm_ranks`, and `fusion_rank`.
    """
    top_k = config.FUSION_TOP_K if top_k is None else top_k
    merged: dict[int, dict] = {}
    contributions: dict[int, dict[str, float]] = {}

    for arm, hits in arms.items():
        weight = _WEIGHTS.get(arm, lambda: 1.0)()
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit["chunk_id"]
            entry = merged.get(chunk_id)
            if entry is None:
                entry = dict(hit)
                entry["arm_scores"] = {}
                entry["arm_ranks"] = {}
                merged[chunk_id] = entry
                contributions[chunk_id] = {}
            # a chunk found by HQ carries the question that matched it
            if hit.get("matched_question") and not entry.get("matched_question"):
                entry["matched_question"] = hit["matched_question"]

            entry["arm_scores"][arm] = hit.get("score")
            entry["arm_ranks"][arm] = rank
            contributions[chunk_id][arm] = weight / (config.RRF_K + rank)

    for chunk_id, entry in merged.items():
        parts = contributions[chunk_id]
        entry["fusion_score"] = sum(parts.values())
        entry["sources"] = [arm for arm, _ in sorted(parts.items(), key=lambda kv: -kv[1])]
        entry.pop("score", None)

    ranked = sorted(merged.values(), key=lambda h: h["fusion_score"], reverse=True)[:top_k]
    for position, entry in enumerate(ranked, start=1):
        entry["fusion_rank"] = position
    return ranked
