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
schema.sql       articles, chunks, story→knowledge links, questions
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

Model weights live in the `hfcache` named volume, so the ~1.3GB download survives
`docker compose down` and every rebuild.

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
WARNING loader  repaired 2 trailing comma(s) — the source file is not valid JSON
WARNING loader  lowercased 5 chunk_type value(s) (e.g. 'Knowledge' -> 'knowledge')
WARNING loader  chunk 1 ('The Cheesecake Factory Menu') is a story with no links —
                under knowledge-first retrieval it can never be reached.

The Shift Most Investors Need To Make  (https://…/the-shift-most-investors-need-to-make/)
  #12    knowledge   22 tok
  #13    story       34 tok -> [12, 14]
  #14    knowledge   38 tok
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
{"chunk_id": 13, "title": "...", "source": "https://...", "author": "...",
 "published_at": "2026-06-02", "site": "...", "chunk_type": "story",
 "chunk": "...", "related_knowledge_chunk_ids": [12, 14]}
```

The loader is deliberately forgiving, because hand-written files drift:

| It accepts | It does |
|---|---|
| Concatenated multi-line objects | Streams them with `raw_decode` |
| A wrapping `[ … ]` array | Handles that too |
| Trailing commas before `}` | Strips them (outside string literals) and warns |
| `"Knowledge"` vs `"knowledge"` | Lowercases and warns |
| `title` / `source` / `published_at` / `chunk` / `chunk_type` | Maps to schema names |
| `related_knowledge_chunk_ids: 7` or `[7, 9]` | Coerces to a list either way |

Your `chunk_id` values are stored as `source_chunk_id`, **not** as the primary key —
the database assigns its own. Links resolve through a two-pass insert: knowledge
first, then stories, then links resolved through the id map. So author id 12
becoming DB id 1000 is handled; nothing silently points at the wrong row.

Links are dropped, with a warning, when the target does not exist, is a story
rather than knowledge, or lives in a different article.

**Story summaries.** A story's embedding is its *summary's* embedding, not its
text's. Supply `summary` yourself, or the ingester generates one (this is what
`--no-summaries` turns off, falling back to embedding the story text — at which
point retrieval matches on narrative wording rather than on the point being made).

---

## Why retrieval is knowledge-only

Story chunks are long, vivid and lexically rich, so they beat the terser knowledge
chunks in both vector and full-text ranking — exactly the wrong outcome when the
question asks for the fact. Restricting all three arms to `content_type='knowledge'`
keeps narrative out of the fusion contest; `attach_stories` then walks the link
table and hands the model the illustration alongside the fact. The prompt tells the
model to cite knowledge as `[K1]` and treat stories as illustration only.

**The trade-off: an unlinked story is unreachable.** It never competes in the arms
and nothing pulls it in, so it may as well not be indexed. The dry-run audit warns
about these by chunk id. Two fixes: add `related_knowledge_chunk_ids` to the story,
or set `STORY_RETRIEVAL_MODE=include` so stories compete directly — which costs some
knowledge precision, exactly the trade the default avoids.

Set `ATTACH_LINKED_STORIES=false` to see what pure knowledge retrieval does alone.

### Stories get no hypothetical questions by default

`hq_search` applies the same `content_type` filter as the other two arms, so in
`knowledge_only` mode a story's questions could never match anything. Generating
them anyway costs an LLM call plus an embedding per story for permanently
unreachable rows, so `ingest.py` skips them and says so:

```
INFO ingest  skipping questions for 3 story chunk(s): STORY_RETRIEVAL_MODE=knowledge_only
             means the HQ arm never sees them
```

Under `STORY_RETRIEVAL_MODE=include` they are generated, from the story's summary
rather than its text — the summary is what the story is indexed under.

This is decided at **ingest** time. If you flip to `include` afterwards, the vector
and FTS arms start reaching stories immediately but the HQ arm stays blind to them
until you re-ingest; `query.py` prints a warning at startup when it sees that state.

### Links are many-to-many

A story often sits between a setup and its payoff and illustrates both, so
`structured_chunk_links` is a join table: `"related_knowledge_chunk_ids": [12, 14]`
stores two rows. When one story attaches to several retrieved knowledge chunks it is
de-duplicated to a single context block labelled with every chunk it illustrates
(`illustrates K1, K3`). The `structured_story_links` view flattens this for psql.

---

## Configuration

Everything is env-driven; `.env.example` has the full list with commentary.

**`EMBED_DIM` must match your embedding model.** It sets the `vector(n)` column
width at table-creation time. `bge-large-en-v1.5` is 1024; `bge-base` is 768,
`bge-small` is 384. Changing it means `--reset` and a re-ingest — `ingest.py`
checks the model's real output dimension against `EMBED_DIM` before embedding
anything, and `query.py` checks the column against `EMBED_DIM` at startup, so a
mismatch fails immediately rather than producing quiet nonsense.

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
`LLM_MAX_TOKENS` and returns `content=''` with `finish_reason='length'` — an empty
summary or a blank answer, with nothing saying why. `GROQ_REASONING_EFFORT=low`
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

## Tuning notes

- **HQ coverage is the usual weak spot.** `HQ_PER_CHUNK=3` is thin for dense
  chunks. Check `questions / knowledge_chunks` in `--stats` (`story_questions` is
  0 unless you ingested in `include` mode); if the `hq` arm
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
