"""Local embedding and reranking models.

Both are loaded lazily and cached, so importing this module is cheap and a
`--dry-run` never pulls ~4.6GB of weights. In the REPL the models stay resident
between questions, which is why `query.py --repl` is much faster than repeated
one-shot invocations.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Sequence

import config
import tracing

log = logging.getLogger("models")


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    log.info("loading embedding model %s (first run downloads weights)", config.EMBED_MODEL)
    return SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)


def embed_documents(texts: Sequence[str], batch_size: int | None = None) -> list[list[float]]:
    """Embed passages. bge documents take no instruction prefix."""
    if not texts:
        return []
    vectors = _embedder().encode(
        list(texts),
        batch_size=batch_size or config.EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 64,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vectors]


@tracing.traceable(
    run_type="embedding",
    name="embed_query",
    process_outputs=tracing.vector_output,
)
def embed_query(text: str) -> list[float]:
    """Embed a question. bge asks for a retrieval instruction on the query side."""
    vector = _embedder().encode(
        config.EMBED_QUERY_PREFIX + text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vector.tolist()


def embedding_dimension() -> int:
    model = _embedder()
    # renamed in sentence-transformers 6.0; support both
    getter = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    return int(getter())


# --------------------------------------------------------------------------- #
# Reranking
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _cross_encoder():
    from sentence_transformers import CrossEncoder

    log.info("loading reranker %s", config.RERANKER_MODEL)
    return CrossEncoder(config.RERANKER_MODEL, device=config.EMBED_DEVICE)


@tracing.traceable(run_type="chain", name="rerank")
def rerank(question: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
    """Cross-encoder rerank over the fused candidates.

    Every candidate is a knowledge chunk scored on its own text — the text it was
    embedded and indexed under. If the reranker is disabled or fails, fusion order
    is kept rather than failing the query: a worse ordering still answers the
    question.
    """
    top_k = config.RERANK_TOP_K if top_k is None else top_k
    if not candidates:
        return []

    if config.RERANKER_BACKEND != "local":
        return candidates[:top_k]

    try:
        pairs = [(question, hit.get("chunk_text") or "") for hit in candidates]
        scores = _cross_encoder().predict(pairs, show_progress_bar=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("reranker failed (%s); keeping fusion order", exc)
        return candidates[:top_k]

    for hit, score in zip(candidates, scores):
        hit["rerank_score"] = float(score)
    ranked = sorted(candidates, key=lambda h: h["rerank_score"], reverse=True)
    tracing.add_metadata(
        reranker=config.RERANKER_MODEL,
        candidates_in=len(candidates),
        kept=len(ranked[:top_k]),
    )
    return ranked[:top_k]
