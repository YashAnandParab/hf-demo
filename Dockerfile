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

COPY *.py schema.sql ./

# Model weights land in HF_HOME, which compose bind-mounts to the host's own
# HuggingFace cache — so the ~4.6GB for bge-m3 and its reranker is downloaded once
# and shared with the local venv, rather than once per environment.
#
# The container idles rather than running anything: this is a CLI tool, so you
# exec into it.
#     docker compose exec app python ingest.py data/chunks.json
#     docker compose exec -it app python query.py --repl
CMD ["sleep", "infinity"]
