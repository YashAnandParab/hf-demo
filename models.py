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


def _tei_scores(question: str, texts: list[str]) -> list[float]:
    """Score every passage against the question on a remote TEI server.

    raw_scores=true is deliberate. TEI's default applies a sigmoid server-side,
    but query.normalized_score() applies its own to what it assumes is a logit;
    taking the default would squash every score into 0.5..0.73 and quietly
    disable the similarity floor. Asking for logits keeps the two backends on
    one scale.

    Returns scores in the caller's passage order, not TEI's ranked order.
    """
    import json
    import urllib.request

    scores = [0.0] * len(texts)
    batch = max(1, config.RERANKER_MAX_BATCH)

    for start in range(0, len(texts), batch):
        window = texts[start : start + batch]
        payload = json.dumps(
            {
                "query": question,
                "texts": window,
                "raw_scores": True,
                "return_text": False,
                # A chunk longer than the server's max_input_length is truncated
                # rather than 413'd. Right-truncation keeps the head, which is
                # where a chunk states its subject.
                "truncate": True,
                "truncation_direction": "Right",
            }
        ).encode()
        request = urllib.request.Request(
            config.RERANKER_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=config.RERANKER_TIMEOUT) as response:
            results = json.loads(response.read())

        # TEI answers sorted by score, so the index is the only link back to the
        # passage that was sent — and it is relative to this batch.
        for item in results:
            scores[start + int(item["index"])] = float(item["score"])

    return scores


def reranker_healthy() -> bool:
    """Whether the remote reranker answers. Always True for non-http backends."""
    if config.RERANKER_BACKEND != "http":
        return True

    import urllib.parse
    import urllib.request

    health = urllib.parse.urljoin(config.RERANKER_URL, "/health")
    try:
        with urllib.request.urlopen(health, timeout=config.RERANKER_TIMEOUT) as response:
            return 200 <= response.status < 300
    except Exception as exc:  # noqa: BLE001
        log.warning("reranker at %s is not answering (%s)", health, exc)
        return False


@tracing.traceable(run_type="chain", name="rerank")
def rerank(question: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
    """Cross-encoder rerank over the fused candidates.

    Every candidate is a knowledge chunk scored on its own text — the text it was
    embedded and indexed under. If the reranker is disabled or fails, fusion order
    is kept rather than failing the query: a worse ordering still answers the
    question. That fallback covers the http backend too, so an unreachable
    reranker server degrades the ranking instead of breaking the pipeline.
    """
    top_k = config.RERANK_TOP_K if top_k is None else top_k
    if not candidates:
        return []

    if config.RERANKER_BACKEND not in {"local", "http"}:
        return candidates[:top_k]

    texts = [hit.get("chunk_text") or "" for hit in candidates]
    try:
        if config.RERANKER_BACKEND == "http":
            scores = _tei_scores(question, texts)
        else:
            scores = _cross_encoder().predict(
                [(question, text) for text in texts], show_progress_bar=False
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("reranker failed (%s); keeping fusion order", exc)
        return candidates[:top_k]

    for hit, score in zip(candidates, scores):
        hit["rerank_score"] = float(score)
    ranked = sorted(candidates, key=lambda h: h["rerank_score"], reverse=True)
    tracing.add_metadata(
        reranker=config.RERANKER_MODEL,
        backend=config.RERANKER_BACKEND,
        candidates_in=len(candidates),
        kept=len(ranked[:top_k]),
    )
    return ranked[:top_k]
