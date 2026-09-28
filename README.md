# Elimam Backend — Islamic Scholar API

FastAPI backend for **Elimam**, a source-grounded Islamic research assistant.
It exposes a pydantic-ai chat agent over turath (Qur'an, Hadith, Tafsir, books,
athar) indexed in Qdrant, with Postgres for users/sessions/keys, plus a thin
aiogram Telegram bot that talks to it over HTTP.

Monorepo layout:

- `server.py` — FastAPI app (run from repo root)
- `api/` — routers: `chat`, `users`, `sessions`, `keys`, `messages`,
  `providers`, `me_providers`, `search`
- `search.py` — the only search layer (dense + BM25 sparse + RRF hybrid, TEI +
  reranker + Qdrant)
- `identity.py` — auth resolution (pure module, unit-testable)
- `db/connection.py` — psycopg2 pool (only file in `db/`)
- `skills/turath-index-skill.md` — LLM system prompt + collection/filter docs
- `telegram_bot/` — aiogram bot (own venv/pyproject, HTTP-only, never touches DB)
- `server_mcp.py` — MCP server exposing the same search tools
- `notebooks/` — ingestion (`update_db.ipynb` populates Qdrant)

## Stack

| Layer | Tech |
| --- | --- |
| API | FastAPI + uvicorn (`server:app`) |
| Agent | pydantic-ai (+ `pydantic-ai-harness`: compaction), multi-provider (OpenAI, Anthropic, Groq, Google, DeepSeek, xAI, OpenRouter, Mistral, local omniroute) |
| Vector search | Qdrant (`:6333`, external) + TEI embedding (`GATE-AraBert-v1`, `:8080`) + TEI reranker (`bge-reranker-v2-m3`, `:8081`) + fastembed BM25 |
| DB | Postgres (`:5432`, external), raw psycopg2 SQL — **no ORM/migrations here** |
| Bot | aiogram, polling |
| Observability | stdlib JSON-lines logging (`request_id`), Logfire + Opik OTLP spans |
| Proxy | nginx (`:80/:443`), omniroute (`127.0.0.1:20128`) |

Schema source of truth is **`../frontend/src/db/schema.ts`** (Drizzle,
migrations in `../frontend/drizzle/`). Any schema change must be mirrored **by
hand** in `api/*.py` raw SQL in the same change. This service never runs DDL.

## Prerequisites

- Python 3.12 (Docker) / 3.13 (uv `pyproject.toml`) — see
  [Dependency manifests](#dependency-manifests-have-drifted)
- Docker Compose
- External Postgres + Qdrant running (not in `docker-compose.yaml` — configured
  via `.env`: `QDRANT_URL`, `DATABASE_*`)
- HuggingFace cache with `GATE-AraBert-v1` for local runs (or use the `embedding`
  TEI service)
- `.env` secrets: `TELEGRAM_BOT_TOKEN`, `BOT_SHARED_SECRET`,
  `ENCRYPTION_MASTER_KEY` (plus `USER_SHARED_SECRET`, `FRONTEND_ORIGIN` for web)

## Quickstart

### 1. Configure

```bash
cp .env.example .env   # then fill in secrets + QDRANT_URL + DATABASE_*
```

Required env (see `docker-compose.yaml`, `db/connection.py`, `identity.py`):

```dotenv
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
DATABASE_HOST=localhost
DATABASE_USER=
DATABASE_NAME=
DATABASE_PASSWORD=
DATABASE_PORT=5432
DATABASE_SSLMODE=disable   # local docker Postgres; prod stays "require"
BOT_SHARED_SECRET=          # must match telegram service
USER_SHARED_SECRET=         # HMAC secret for Next.js proxy X-User-* headers
ENCRYPTION_MASTER_KEY=      # Fernet key for api/keys.py encrypt/decrypt
TELEGRAM_BOT_TOKEN=
FRONTEND_ORIGIN=            # empty = no direct browser CORS (desired: Next proxy only)
AUTH_MODE=permissive        # "permissive" (rollout aid) or "enforce"
EMBEDDING_URL=http://embedding:80
RERANKER_URL=http://reranker:80
OTEL_EXPORTER_OTLP_ENDPOINT=
OTEL_EXPORTER_OTLP_HEADERS=
LOG_LEVEL=INFO
```

DB migrations are applied from the frontend, never here:

```bash
cd ../frontend && bun run db:generate && bun run db:migrate
```

### 2a. Local dev (needs Postgres + Qdrant + TEI up)

```bash
uvicorn server:app --reload   # from repo root (./skills/... relative path matters)
```

### 2b. Docker Compose (backend + bot + nginx + omniroute + TEI)

```bash
docker compose up            # backend reachable as http://backend:8000 on the compose net
docker compose logs backend  # JSON-lines logs, greppable by request_id
```

Health: `GET /chat/health`. Backend has no published port by design — the
Next.js proxy is the only browser entry; nginx/bot reach it via service name.

### 3. Populate Qdrant

Collections (`quran`, `hadith`, `tafsir`, `books`, `sunnah`, ~148k book points)
live in gitignored `qdrant_storage/`:

```bash
# open notebooks/update_db.ipynb (see notebooks/ for per-corpus builders)
```

## API reference

Base: `http://localhost:8000`. Interactive docs: `/docs`, `/openapi.json`.

Auth: every non-public route goes through `identity_middleware`
(`identity.py`). Callers are either the Next.js proxy
(`X-User-Id` + `X-User-Timestamp` + `X-User-Signature =
hex(hmac_sha256(USER_SHARED_SECRET, "{id}:{ts}"))`, 300 s window) or the bot
(`X-Bot-Secret: BOT_SHARED_SECRET` + `X-Telegram-Id`). **Never send user
identity in the body.** Public paths: `/chat`, `/chat/health`, `/providers`,
`/docs`, `/openapi.json`, `/redoc`.

| Router | Endpoints |
| --- | --- |
| `POST /chat` | bot chat, body `{message}` only; SSE stream; resolves user/active session/model/key server-side |
| `POST /chat/web` | web chat `{message, model_name, model_provider, model_variant, session_id, user_provider_id?}`; SSE; persists pydantic-ai messages to `sessions.pydantic_message` |
| `GET /chat/health` | liveness (used by compose healthcheck) |
| `POST /search/dense_search`, `/sparse_search`, `/hybrid_search` | `{collection, query_text (Arabic), top_k ≤ 100, filters?, rerank_pool ≤ 500}` |
| `GET /providers` | public catalog `{slug: {url, models}}` (DB-backed, cached in-process) |
| `GET /me/connections` `POST /me/connections` `POST /me/connections/custom` `PATCH/DELETE /me/connections/{id}` | per-user provider connections (`user_providers`) |
| `POST /me/connections/{id}/sync` `POST .../models` `PATCH/DELETE .../models/{model_id}` | model discovery / manual add / enable-disable |
| `GET /users/{id}`, `GET /users/telegram/{id}`, `POST /users/add`, `PUT /users/telegram/link` | bot-only writes (`require_bot`); reads `require_owner` |
| `GET /sessions/{id}`, `GET /sessions/active/user/{user_id}`, `POST /sessions/add`, `PUT /sessions/{id}/model`, `POST /sessions/reset/{user_id}` | one active session per user (partial unique index) |
| `GET /messages/{session_id}`, `POST /messages/add` | conversation history |
| `GET /keys/{user_id}/{provider}/exists`, `GET /keys/{user_id}`, `POST /keys/add`, `PUT /keys/update` | legacy per-provider keys; Fernet-encrypted (`ENCRYPTION_MASTER_KEY`); `api_keys` table is frozen — do not read/write it |

SSE framing: `event: <type>\ndata: <json>\n\n` (`api/chat.py::sse`).

## Search

`search.py` is the only search layer. All query text is Arabic-normalized
(`camel-tools` dediac + alef normalize).

- **Dense:** TEI `GATE-AraBert-v1` via `EMBEDDING_URL`, Qdrant vector search
- **Sparse:** fastembed `Qdrant/bm25`, Qdrant sparse index
- **Hybrid:** RRF fusion of dense + sparse, then `bge-reranker-v2-m3`
  cross-encoder rerank via `RERANKER_URL` (batch size 32)
- **Direct getters:** `get_quran` (`1:1` ids), `get_tafsir`
  (`saadi|katheer|moyassar|tabary|baghawy`), `get_hadith`
  (`bukhari|muslim…` — 9 books), `get_book` / `get_books_*`

Per-collection filters: `quran{surah_number, surah}`,
`hadith{book, grade}`, `tafsir{surah_number, surah, ayah_number}`,
`books/sunnah{book_id, book_name, category_name, all_authors, author_death,
book_date (+ athar_number)}`.

Known gaps (see `skills/turath-index-skill.md`): no Sahih Muslim;
`get_book` numbering is unverifiable. The agent answers only from retrieved
sources, else *"لم أجد معلومات كافية في المصادر المتاحة."*

`POST /chat` tools are wired in `api/chat.py`; the same functions are exposed
over MCP in `server_mcp.py` (`El Imam`: `get_quran`, `get_tafsir`,
`dense/sparse/hybrid_search`, …).

## Telegram bot (`telegram_bot/`)

Thin aiogram interface — HTTP to the backend only, never the DB. Own
venv/pyproject; `app/config.py::Settings.from_env()` requires
`TELEGRAM_BOT_TOKEN` + `BOT_SHARED_SECRET` (paths/URLs overridable via
`BACKEND_*` env).

Flow: `/start` → lookup user → new users pick provider → enter API key → pick
model → pick variant (`low|high|max`) → first session created. Messages →
`POST /chat` (`{message}`) as SSE. `/reset` deactivates + recreates session
with the same model config. Routers in `app/handlers/`: `start, model, reset,
chat, help`.

## Testing

```bash
pytest tests/test_providers_config.py  # standalone, self-sufficient
pytest tests/test_identity.py tests/test_route_coverage.py  # no stack needed
pytest tests/  # rest need docker compose up + populated Qdrant + models
```

- `tests/test_providers_config.py` — standalone catalog test
- `tests/test_identity.py`, `test_route_coverage.py` — auth/route coverage
  (pure `identity.py`, no DB)
- `temp/test_search.py` — integration test, needs populated Qdrant (~148k
  points) + models; fails on fresh checkout
- `tests/test_dahl_*.py` — ad-hoc debug scripts, not pytest tests

## Project structure

```text
server.py              FastAPI app, middleware (request-id log, identity, CORS), routers
identity.py            pure auth module (HMAC user sig / bot secret, guards)
api/chat.py            pydantic-ai agent + SSE + session persistence
api/search.py          thin caps (top_k/rerank_pool) over search.py
api/providers.py       DB-backed catalog (cached) + resolve_model_config
api/me_providers.py    user connections + model sync (user-facing)
api/users|sessions|messages|keys.py   raw-SQL CRUD
search.py              embeddings/sparse/RRF/rerank/Qdrant + filters + getters
db/connection.py       psycopg2 ThreadedConnectionPool + get_conn/close_pool
skills/turath-index-skill.md  agent system prompt + corpus docs
server_mcp.py          FastMCP "El Imam" over search tools
telegram_bot/app/      aiogram bot (handlers/, backend.py, config.py, states.py)
notebooks/             ingestion (update_db.ipynb + per-corpus builders)
nginx/nginx.conf       reverse proxy; certbot/ for TLS
Dockerfile             py3.12 multi-stage (torch CPU + requirements.txt)
docker-compose.yaml    backend + embedding/reranker TEI + telegram + nginx + omniroute
tests/  temp/  diagrams/  vectordb/  docs/
```
