FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hfcache

WORKDIR /app

# CPU-only torch, installed first and on its own. It is by far the largest layer
# and it never changes, so it stays cached through every later rebuild. It also
# pre-satisfies sentence-transformers' torch dependency, which would otherwise
# pull the ~2.5GB CUDA build from PyPI.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Both schemas, not just schema.sql — each version reads its own at init, so a
# missing one fails the normal version at the first ingest rather than at build.
COPY *.py *.sql ./
COPY tools/*.py ./tools/

# Model weights land in HF_HOME, which compose bind-mounts to the host's own
# HuggingFace cache — so the ~4.6GB for bge-m3 and its reranker is downloaded once
# and shared with the local venv, rather than once per environment.

EXPOSE 8000

# The HTTP API the React frontend talks to. The CLIs are still here and are run by
# exec-ing into this container (or `docker compose run --rm cli …`, which does not
# hold a second copy of the models):
#     docker compose exec app python ingest.py
#     docker compose exec app python ingest.py --version normal
#     docker compose exec -it app python query.py --repl
#
# ONE worker, deliberately. Everything downstream of api.py is process-wide state —
# one Postgres connection, one active version, one resident embedder and reranker —
# serialised by a single lock. A second worker would be a second 4.6GB copy of the
# models guarding a lock the first one cannot see.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
