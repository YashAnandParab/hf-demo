"""Central configuration.

Every value is env-driven. A `.env` sitting next to this file is loaded
automatically, so the scripts behave the same whether you export the variables
yourself or keep them in the file.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # optional: exported env vars work just as well
    pass


def _int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, "") or default)
    except ValueError:
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, "") or default)
    except ValueError:
        return default


def _bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------- database ---
# A full URL wins; otherwise it is assembled from the parts, which is friendlier
# when you already have a Postgres running somewhere.
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = _int("POSTGRES_PORT", 5432)
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_DB = os.getenv("POSTGRES_DB", "postgres")

DATABASE_URL = os.getenv("DATABASE_URL") or (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# -------------------------------------------------------------- embeddings ---
# Local sentence-transformers model. bge-m3 is 1024-dim with an 8192-token window,
# so it embeds a long chunk whole where bge-large-en-v1.5 truncated at 512.
# Both are 1024-dim, so moving between them needs no schema change — but it does
# need `--reset` and a re-ingest, because vectors from one model are meaningless
# in the other's space. bge-base (768) or bge-small (384) additionally require
# changing EMBED_DIM, since the vector() column width is fixed at table creation.
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
EMBED_DIM = _int("EMBED_DIM", 1024)
EMBED_BATCH_SIZE = _int("EMBED_BATCH_SIZE", 32)
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "") or None  # None -> auto (cuda if present)


def _default_query_prefix(model: str) -> str:
    """The retrieval instruction the model was trained to expect on queries.

    This differs BETWEEN bge generations, and getting it wrong fails silently —
    retrieval simply gets worse, with nothing in the logs. The v1.5 English models
    were trained with an instruction on the query side; bge-m3 was not, and
    prepending one to every query just adds noise the model never saw in training.
    Deriving it from the model name means switching EMBED_MODEL cannot leave the
    wrong prefix behind. Set EMBED_QUERY_PREFIX to override.
    """
    name = model.lower()
    if "bge-m3" in name:
        return ""
    if "bge" in name and "-en" in name:
        return "Represent this sentence for searching relevant passages: "
    return ""


# An empty value is a legitimate setting, so os.getenv's default is not enough:
# EMBED_QUERY_PREFIX= in .env must mean "no prefix", not "fall back to the default".
_raw_prefix = os.getenv("EMBED_QUERY_PREFIX")
EMBED_QUERY_PREFIX = _default_query_prefix(EMBED_MODEL) if _raw_prefix is None else _raw_prefix

# ---------------------------------------------------------------- reranker ---
# local -> sentence-transformers CrossEncoder
# none  -> keep fusion order (no torch needed at query time)
RERANKER_BACKEND = os.getenv("RERANKER_BACKEND", "local").lower()
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# -------------------------------------------------------------- generation ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")
LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.1)
LLM_MAX_TOKENS = _int("LLM_MAX_TOKENS", 1024)
# gpt-oss models only. They burn the token budget on a hidden reasoning channel
# before writing any content, so "low" leaves room for an actual answer.
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low").lower()
LLM_MAX_RETRIES = _int("LLM_MAX_RETRIES", 4)

# --------------------------------------------- hypothetical questions (HQ) ---
HQ_ENABLED = _bool("HQ_ENABLED", True)
HQ_PER_CHUNK = _int("HQ_PER_CHUNK", 3)

# --------------------------------------------------------------- retrieval ---
VECTOR_TOP_K = _int("VECTOR_TOP_K", 20)
FTS_TOP_K = _int("FTS_TOP_K", 20)
HQ_TOP_K = _int("HQ_TOP_K", 20)
FUSION_TOP_K = _int("FUSION_TOP_K", 30)
RERANK_TOP_K = _int("RERANK_TOP_K", 5)
RRF_K = _int("RRF_K", 60)
WEIGHT_VECTOR = _float("WEIGHT_VECTOR", 1.0)
WEIGHT_FTS = _float("WEIGHT_FTS", 0.8)
WEIGHT_HQ = _float("WEIGHT_HQ", 1.0)

# ------------------------------------------------------- structured RAG -----
# Stories are never retrieval candidates: they carry no embedding and are absent
# from the full-text index. They reach the model only by being attached, through
# the link table, to a knowledge chunk that survived reranking.
ATTACH_LINKED_STORIES = _bool("ATTACH_LINKED_STORIES", True)
MAX_LINKED_STORIES = _int("MAX_LINKED_STORIES", 2)

# --------------------------------------------------------------- langsmith ---
# Tracing is opt-in and degrades to a no-op: with no key set, the decorators in
# tracing.py pass the function through untouched and nothing leaves the machine.
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "") or os.getenv("LANGCHAIN_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "hf-demo-rag")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
# explicit off-switch that works even when a key is present
LANGSMITH_TRACING = _bool("LANGSMITH_TRACING", True)

# -------------------------------------------------------------------- misc ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def setup_logging() -> None:
    import logging

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s  %(levelname)-7s %(name)-24s %(message)s",
        datefmt="%H:%M:%S",
    )
    # sentence-transformers is chatty on load
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
