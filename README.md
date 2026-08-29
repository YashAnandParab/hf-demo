# Structured RAG — knowledge vs story, on Postgres + pgvector

Chunks are split into two kinds. **Knowledge** is the transferable claim; **story**
is the narrative that illustrates it. Only knowledge competes in retrieval — the
stories that illustrate the surviving knowledge are attached afterwards through a
link table, and handed to the model as illustration rather than as evidence.

Plain Python scripts and a Postgres you already have. No Docker, no web server.

```
embed ─┬─ vector_search ─┐
       ├─ fts_search   ─┼─ fuse (RRF) ─ rerank ─ attach_stories ─ generate
       └─ hq_search    ─┘
```

Every stage prints, so you can see which chunks were retrieved by which arm, which
survived reranking, and what the answer was actually built from.

## Layout

```
config.py        every setting, env-driven
schema.sql       articles, chunks, knowledge→story links, questions
tracing.py       optional LangSmith tracing; a no-op without a key
db.py            connection, transactions, schema init, article upsert
loader.py        parses your hand-written chunk JSON; normalises and audits it
models.py        local bge embeddings + cross-encoder reranker (lazy-loaded)
llm.py           Groq chat, with 429 handling that retries the SAME model
hq.py            hypothetical-question generation
fusion.py        reciprocal rank fusion
retrieval.py     the three arms, the story link walk, /stats
prompts.py       system prompts and context formatting
ingest.py        JSON  → database
query.py         question → answer, every stage printed
test_loader.py   offline tests: no db, no models, no API key
data/chunks.json your chunks
```

---

## 1. Setup

Two ways to run it. Docker is the one to use if you already have a pgvector
container; the local venv is faster for editing code and running the tests.

`ingest.py` creates the extensions, tables and indexes itself either way — there is
no separate migration step.

### Docker

The compose file does **not** run its own Postgres. It joins the network of an
existing pgvector container, because a second server would fight over the published
port and split your data in two. Set the network name and the database host to
match yours:

```yaml
# docker-compose.yml
    environment:
      POSTGRES_HOST: rag-postgres     # the container's name
      POSTGRES_PORT: 5432             # its INTERNAL port, not the host mapping
networks:
  postgres:
    external: true
    name: postgress_default           # docker inspect <container> to find this
```

Then:

```bash
cp .env.example .env                  # put your GROQ_API_KEY in
docker compose up -d --build          # ~3 min: CPU torch + transformers
docker compose exec app python ingest.py data/chunks.json --dry-run
```

The container idles rather than running anything — this is a CLI tool, so you exec
into it. `./data` is bind-mounted, so editing `data/chunks.json` on the host needs
no rebuild; editing a `.py` file does (`docker compose build`, ~2s from cache).

Model weights are **not** in a container-private volume. `HF_HOME` is bind-mounted
to the host's own HuggingFace cache (`%USERPROFILE%\.cache\huggingface`, or set
`HF_CACHE_DIR`), so the ~4.6GB for `bge-m3` and its reranker is downloaded once and
shared with the local venv rather than once per environment. It survives
`docker compose down` by living outside Docker entirely.

**If you'd rather have a self-contained Postgres**, drop the `networks:` block and
add a service:

```yaml
  db:
    image: pgvector/pgvector:pg17
    environment: {POSTGRES_PASSWORD: postgres}
    ports: ["5434:5432"]              # 5434 to avoid colliding with an existing one
    volumes: [pgdata:/var/lib/postgresql/data]
```

…then point `POSTGRES_HOST` at `db` and add `pgdata:` under `volumes:`.

### Local venv

```bash
python -m venv .venv                  # on Python 3.12 or 3.13, see the note below
.venv/Scripts/activate                # Windows;  source .venv/bin/activate elsewhere
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env                  # then put your GROQ_API_KEY in
```

Point it at a Postgres with the `vector` extension available. Set `POSTGRES_HOST` /
`POSTGRES_PORT` / credentials in `.env`, or give it a full `DATABASE_URL`. When the
database is in Docker, `POSTGRES_PORT` is the **host** mapping (often 5433), not
5432 — that difference is the single most common cause of "connection refused".

## 2. Check your chunk file before spending anything

```bash
python ingest.py data/chunks.json --dry-run
```

This parses, normalises and audits without touching the database, loading a model,
or making an API call. Run it first, every time.

```
WARNING loader  lowercased 84 chunk_type value(s) (e.g. 'Knowledge' -> 'knowledge')
INFO    loader  7 link(s) point at a story in a different article — allowed,
                resolved globally
WARNING loader  chunk 94 ('On Selling Out — Part 3') is a story that no knowledge
                chunk points at. Stories are never retrieved directly, so nothing
                can ever reach it.

The Shift Most Investors Need To Make  (https://…/the-shift-most-investors-need-to-make/)
  #12    knowledge   29 tok -> stories [13]
  #13    story       59 tok
  #14    knowledge  365 tok -> stories [13]
  #19    knowledge  438 tok -> stories [18*]

  * = story in another article
```

## 3. Ingest

```bash
python ingest.py data/chunks.json
python ingest.py data/chunks.json --reset          # drop the tables and start over
python ingest.py data/chunks.json --no-questions   # skip HQ generation (no LLM cost)
```

Re-ingesting an article deletes its chunks first, so this is idempotent.

## 4. Ask

```bash
python query.py "why start investing early"
python query.py --repl                             # models stay resident between questions
python query.py "..." --fetch-k 30 --top-k 8
python query.py "..." --retrieval-only             # no LLM call at all
python query.py --stats
```

- `--fetch-k` — candidates each arm pulls before fusion
- `--top-k` — knowledge chunks that reach the LLM
- `--show-text` — full chunk text instead of previews

Use `--repl` for anything more than one question: the embedding model and reranker
take ~30s to load and stay loaded between questions.

`--retrieval-only` is the one to reach for when tuning. It shows which arm found
each chunk, so you can tell whether a miss is an embedding problem, a keyword
problem, or an HQ coverage problem, before touching the weights.

## 5. Inspect the database

```bash
psql -h localhost -U postgres -d postgres

\d structured_chunks
SELECT content_type, count(*) FROM structured_chunks GROUP BY 1;
SELECT * FROM structured_story_links;          -- every story with what it illustrates
```

---

## The input format

A stream of concatenated JSON objects, one chunk each. Not an array, not strict
JSONL — objects span multiple lines:

```json
{"chunk_id": 2, "title": "...", "source": "https://...", "author": "...",
 "published_at": "2026-06-02", "site": "...", "chunk_type": "knowledge",
 "chunk": "...", "related_story_chunk_ids": [1]}
```

**Links are declared on the knowledge chunk** and point at the stories that
illustrate it. A story declares nothing: it is passive, named by whatever cites it.

The loader is deliberately forgiving, because hand-written files drift:

| It accepts | It does |
|---|---|
| Concatenated multi-line objects | Streams them with `raw_decode` |
| A wrapping `[ … ]` array | Handles that too |
| Trailing commas before `}` | Strips them (outside string literals) and warns |
| `"Knowledge"` vs `"knowledge"` | Lowercases and warns |
| `title` / `source` / `published_at` / `chunk` / `chunk_type` | Maps to schema names |
| `related_story_chunk_ids: 7` or `[7, 9]` | Coerces to a list either way |
| `related_knowledge_chunk_ids` (the old direction) | **Rejects the file**, with a migration note |

That last row is deliberate. Silently ignoring the old key would ingest the whole
corpus with zero links and no story would ever reach an answer — a failure that
looks exactly like a retrieval-quality problem.

Your `chunk_id` values are stored as `source_chunk_id`, **not** as the primary key —
the database assigns its own. So author id 12 becoming DB id 1000 is handled;
nothing silently points at the wrong row.

**Link resolution is global, not per-article.** Every chunk in the run is inserted
first, then all links are resolved against one corpus-wide id map, because a
knowledge chunk may cite a story in a *different* article. The corollary is that
ingesting only part of a corpus drops any link whose story lives in an article that
was not in the run — ingest the whole file.

Links are dropped, with a warning naming the chunk id, when the target does not
exist or is a knowledge chunk rather than a story.

---

## Why retrieval is knowledge-only

Story chunks are long, vivid and lexically rich, so they beat the terser knowledge
chunks in both vector and full-text ranking — exactly the wrong outcome when the
question asks for the fact.

So a story is not a retrieval candidate at all. This is structural rather than a
setting:

- a story row's `embedding` is `NULL`, and a `CHECK` constraint keeps it that way
- a story row's `search_vector` is `NULL`, so the FTS arm cannot match its text
- the HQ arm only ever indexes knowledge, so there is nothing to match either

All three arms therefore return knowledge and only knowledge. `attach_stories`
then walks the link table from whichever knowledge chunks survived reranking and
hands the model their stories as illustration. The prompt tells the model to cite
knowledge as `[K1]` and never to cite a story as the source of a fact.

**The consequence: a story nothing cites is unreachable.** There is no mode in
which it competes on its own merits. The dry-run audit warns about these by chunk
id; the fix is to add its id to some knowledge chunk's `related_story_chunk_ids`.

Set `ATTACH_LINKED_STORIES=false` to see what pure knowledge retrieval does alone.

### Links are many-to-many, and may cross articles

One knowledge chunk may cite several stories, and one story may be cited by several
knowledge chunks — including ones in other articles, which is how a single vivid
anecdote gets reused across a series. `structured_chunk_links` is therefore a join
table.

When one story is cited by several of the retrieved knowledge chunks it is
de-duplicated to a single context block labelled with every chunk it illustrates
(`illustrates K1, K3`). `MAX_LINKED_STORIES` caps how many stories each surviving
knowledge chunk contributes, trimming from the order your JSON listed them in. The
`structured_story_links` view flattens all of this for psql.

---

## Configuration

Everything is env-driven; `.env.example` has the full list with commentary.

**`EMBED_DIM` must match your embedding model.** It sets the `vector(n)` column
width at table-creation time. `bge-m3` and `bge-large-en-v1.5` are both 1024;
`bge-base` is 768, `bge-small` is 384. Changing it means `--reset` and a re-ingest
— and note that swapping between two models of the *same* dimension still needs a
re-ingest, because the stored vectors live in the old model's space. `ingest.py`
checks the model's real output dimension against `EMBED_DIM` before embedding
anything, and `query.py` checks the column against `EMBED_DIM` at startup, so a
mismatch fails immediately rather than producing quiet nonsense.

**`EMBED_QUERY_PREFIX` is derived from the model, not defaulted.** bge's v1.5
English models were trained with a retrieval instruction on the query side
(`"Represent this sentence for searching relevant passages: "`); **bge-m3 was not**.
Prepending one anyway adds text the model never saw in training, and the failure is
silent — retrieval just gets worse, with nothing in the logs. So `config.py` picks
the prefix from `EMBED_MODEL` and switching models cannot leave the wrong one
behind. Set `EMBED_QUERY_PREFIX` to override; an empty value is respected as "no
prefix" rather than falling back to the default.

**Fusion weights.** `WEIGHT_VECTOR` / `WEIGHT_FTS` / `WEIGHT_HQ` scale each arm's
RRF contribution. FTS starts at 0.8 because `ts_rank_cd` rewards term repetition,
which over-favours long chunks. `RRF_K` (default 60) controls how sharply rank
position matters — lower makes the top of each arm dominate.

**Reranker.** `RERANKER_BACKEND=local` runs a cross-encoder over the fused
candidates. Set it to `none` to keep fusion order and skip loading a second model.
If the reranker fails at query time it logs a warning and falls back to fusion
order rather than failing the request.

**Reasoning models.** `openai/gpt-oss-*` spends the token budget on a hidden
reasoning channel before writing any content. At the default effort it exhausts
`LLM_MAX_TOKENS` and returns `content=''` with `finish_reason='length'` — a blank
answer or a chunk with no questions, with nothing saying why. `GROQ_REASONING_EFFORT=low`
(the default here, and applied only to `gpt-oss` models) leaves room for an answer,
and `llm.py` raises a specific error rather than returning the empty string if it
still happens. A non-reasoning model like `llama-3.3-70b-versatile` sidesteps this
entirely and is plenty for summaries and question generation.

**Auth failures stop the run.** A 401/403 fails identically on every model, so
falling back would just double the doomed requests and bury the cause under a wall
of warnings. `llm.py` raises `SystemExit` — deliberately a `BaseException`, so it
escapes the per-chunk `except Exception` handlers that exist to keep ingestion going
through a single bad chunk.

**In Docker, `.env` is read when the container is created**, not on `exec`. After
editing it: `docker compose up -d --force-recreate`.

**Rate limits.** A free Groq key will rate-limit during ingestion, since HQ
generation is one call per chunk. `llm.py` waits out a 429 (honouring `retry-after`,
or the wait Groq puts in the message body) and retries **the same model** — a
non-429 failure is what triggers the fallback model, so a rate limit never silently
changes which model wrote your data.

---

## Tracing with LangSmith

Optional, and off unless you set a key. There is no LangChain here — `tracing.py`
wraps the bare `langsmith` SDK and resolves each decorator **once at import time**:
with no key, `@traceable` is the identity function and the pipeline pays nothing at
call time.

```bash
pip install langsmith
# .env
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=hf-demo-rag
```

One call to `query.run` becomes one trace tree:

```
structured_rag                       chain
├── embed_query                      embedding   (logs dimensions, not 1024 floats)
├── vector_search                    retriever
├── fts_search                       retriever
├── hq_search                        retriever
├── rerank                           chain       candidates_in / kept
├── linked_stories                   retriever
└── groq.chat                        llm         model, attempts, tokens, finish_reason
```

The root run carries the numbers worth filtering on later: hits per arm, the
surviving `knowledge_chunk_ids` and `story_chunk_ids`, context size, and per-stage
milliseconds. `groq.chat` records the model that **actually** answered, which is
not always the one asked for — a non-429 failure falls back.

Ingestion traces too, as `ingest` with a `hypothetical_questions` child.

Tracing can never fail a query. A missing package, a bad key or an unreachable
endpoint degrades to warnings in the log; the answer still comes back. Set
`LANGSMITH_TRACING=false` to switch it off while leaving the key in place.

Because CLI runs are short-lived and traces are batched on a background thread,
`ingest.py` and `query.py` both call `tracing.flush()` on the way out.

---

## Tuning notes

- **HQ coverage is the usual weak spot.** `HQ_PER_CHUNK=3` is thin for dense
  chunks. Check `questions / knowledge_chunks` in `--stats`; if the `hq` arm
  contributes near nothing on real questions, your generated questions aren't
  covering the angles users actually ask from — raise `HQ_PER_CHUNK` or rewrite
  `HQ_SYSTEM` in `prompts.py`.
- **An empty `fts` arm is not a bug** on conceptual questions — there is simply no
  lexical overlap. If it's empty on most of your questions, drop `WEIGHT_FTS` so it
  stops adding noise when it does fire.
- **HNSW recall.** The indexes use `m=16, ef_construction=64`. If recall looks low
  at scale, raise it per session: `SET hnsw.ef_search = 100;`.

## Tests

```bash
pytest -q test_loader.py
```

18 tests covering the parser, the field normalisation, the link audit, the two-pass
id resolution, and RRF fusion. No database, no models, no API key.

## Notes

**Python version.** `sentence-transformers` pulls in `torch`, which lags new Python
releases. If `pip install -r requirements.txt` cannot find a torch wheel, create the
venv on Python 3.12 or 3.13.

**CPU-only torch** is much smaller than the default CUDA build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

On CPU, embedding a few hundred chunks takes well under a minute; the reranker adds
roughly a second per query. For GPU, install the CUDA torch build — nothing else
changes, `EMBED_DEVICE` is auto-detected.

**Connecting to Postgres in Docker from the host** — the container's 5432 is
usually mapped to something else (5433 is the common choice, to avoid colliding
with a local install). Set `POSTGRES_PORT` to the *host* port.

**`_archive/`** holds the Docker Compose, Makefile, FastAPI and LangGraph files
from the previous version of this project. Nothing imports them; delete when you're
confident you don't want them back.
