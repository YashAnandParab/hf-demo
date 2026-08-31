# HFrag — chat frontend

React + Vite + TypeScript + Tailwind client for the RAG backend in the parent
directory. It talks to `api.py` over HTTP; start that first.

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # type-check + production bundle
npm run check    # SSE stream-parser self-check
```

## Configuration

`.env` (copied from `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:8000` | Where `api.py` is listening |
| `VITE_APP_NAME` | `HFrag` | Name in the sidebar and empty state |

## The "model" picker is the version picker

The dropdown in the composer lists what `GET /models` returns, and what the
backend returns there are the two **RAG versions**, not two LLMs:

- **Structured RAG** — knowledge/story split; only knowledge chunks are
  retrievable, and the stories they cite are attached afterwards.
- **Normal chunking** — the flat baseline; every chunk competes.

Picking one switches the Postgres database the answer is retrieved from, which
is the comparison this project exists to make. Which LLM writes the answer is
`GROQ_MODEL` in the backend's `.env`, deliberately the same for both — otherwise
a difference between two answers could be the model rather than the structure.

## The wire

`src/lib/api.ts` is the only module that knows a network exists.

**`GET /models`** — `{ "models": [...] }` or a bare array:

```json
{ "id": "structured", "name": "Structured RAG",
  "provider": "pgvector · postgres",
  "description": "knowledge/story split; …",
  "capabilities": ["Knowledge/story split", "Stories attached"] }
```

**`POST /chat`** — request:

```json
{ "conversation_id": "…", "message": "…", "model": "structured",
  "history": [{ "role": "user", "content": "…" }],
  "top_k": 3, "min_score": 0, "stream": true }
```

Response is `text/event-stream`:

```
data: {"sources": [ … ]}

data: {"token": "The "}

data: [DONE]
```

(`delta` and `content` are accepted as aliases for `token`; a bare non-JSON
frame is treated as a token. A backend that answers with a plain
`{"answer": "…", "sources": [...]}` JSON body instead is rendered in one shot,
with no code change.)

A source maps onto what `query.run()` already returns:

| Wire field | Backend origin |
|---|---|
| `id` | the citation label: `K1`/`S1` structured, `P1` flat |
| `document` | `article_name` |
| `url` | `article_url` |
| `chunk` | `source_chunk_id` |
| `content` | `chunk_text` |
| `score` | `rerank_score`, squashed to 0..1 |
| `arms` | `sources` (`vector` / `fts` / `hq`), or `linked story` |

Citation markers in the answer (`[K1]`, `[S1]`, `[P1]`, or a bare `[1]`) resolve
against the `sources` array by `id` and become clickable references. Markers with
no matching source stay as plain text, so a hallucinated `[7]` never offers a
dead link.

## Layout

```
src/
  types.ts              wire + domain types
  store.ts              zustand: chat / models / ui, persisted to localStorage
  lib/api.ts            the only module that fetches
  lib/sse.ts            pure SSE framing (checked by sse.check.ts)
  components/           Sidebar · Composer · Message · SourcePanel · Settings · ui
```

Conversations live in `localStorage` under `hfrag.*`. Nothing leaves the
browser except calls to your own backend.
