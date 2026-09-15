# Elimam — Islamic scholar app (monorepo)

FastAPI backend + Qdrant vector search + Postgres + aiogram Telegram bot. The Next.js app lives in `ui/` (out of scope; it has its own AGENTS.md).

## Stack layout

- FastAPI app is `server.py` at the **repo root** (not `backend/server`); routers in `api/` (chat, users, sessions, keys, messages, providers). Run: `uvicorn server:app --reload`.
- Qdrant on `:6333`; `search.py` is the only search layer (dense GATE-AraBert-v1 + BM25 sparse + RRF hybrid). The 5 collections (`quran`, `hadith`, `tafsir`, `books`, `sunnah`) live in gitignored `qdrant_storage/`, populated via `notebooks/update_db.ipynb`.
- Postgres on `:5432`; backend uses raw psycopg2 SQL in `api/*.py`. Schema source of truth is **`../frontend/src/db/schema.ts`** (Drizzle) with migrations in `../frontend/drizzle/`. This service has no schema of its own and no migration step: `api/*.py` uses raw psycopg2 SQL against the same database, and any schema change must be mirrored here **by hand** in the same change.
- Telegram bot in `telegram_bot/` (own venv/pyproject). It only talks to the backend over HTTP, never the DB.

## The big gotcha: import-time side effects

Importing `server`, `api.chat`, or `search` requires the whole stack and the models:

- `search.py` loads SentenceTransformer + fastembed models and connects to Qdrant at import.
- `api/chat.py` calls `setup_indexes()` and reads `./skills/turath-index-skill.md` (relative path — run from repo root).
- `api/providers.py` loads `config/providers.yaml` at import; default path is `/app/config/providers.yaml` (docker). Set `PROVIDERS_CONFIG_PATH` locally or imports fail.
- `db/connection.py` builds a psycopg2 pool from `DATABASE_*` env at import; needs `.env` + running Postgres.

So unit tests that avoid importing these modules fail fast; anything touching the search/chat layer needs `docker compose up` first.

## Running

- `docker compose up` = backend (`:8000`) + telegram bot + nginx + omniroute. Qdrant and Postgres are **external**: `QDRANT_URL` and the `DATABASE_*` vars come from `.env`. Needs `.env` secrets (`TELEGRAM_BOT_TOKEN`, `BOT_SHARED_SECRET`, `ENCRYPTION_MASTER_KEY`).
- DB migrations are applied from the frontend (`cd ../frontend && bun run db:generate && bun run db:migrate`); this service never runs DDL. `db/` contains exactly one file, `connection.py` (the psycopg2 pool) — do not add Drizzle artifacts here.
- Tests: `temp/test_search.py` is an integration test — requires a *populated* Qdrant (~148k book points) and the models, so it fails on a fresh checkout. `tests/test_providers_config.py` is standalone and self-sufficient. `tests/test_dahl_*.py` are ad-hoc debug scripts, not pytest tests.

## Dependency manifests have drifted

Two root manifests disagree: `pyproject.toml` (uv, `uv.lock`, requires-python >=3.13) and `requirements.txt` (used by the dockerfiles, Python 3.12). Code imports `pydantic_ai_harness` (only in `requirements.txt`); the pyproject entry `pydantic-harness` is stale. Edit the manifest the target environment actually uses.

## Auth model

- Backend endpoints take `X-Bot-Secret` (must equal `BOT_SHARED_SECRET`, shared with the bot) and `X-Telegram-Id` headers. Never accept user identity in the request body.
- API keys are stored per (user, provider) in `api_keys.encrypted_key`; `encrypt()`/`decrypt()` in `api/keys.py` are **not** identity functions — they use Fernet with `os.environ["ENCRYPTION_MASTER_KEY"]` (the module docstring still names the stale seam). Never log or return keys.

## Agent wiring

- `skills/turath-index-skill.md` is the LLM system prompt and documents collections, filters, and known gaps (e.g. no Sahih Muslim; `get_book` numbering is unverifiable). The chat agent's tools are wired in `api/chat.py`.
- `POST /chat` returns SSE; conversation history is persisted as pydantic-ai messages in `sessions.pydantic_message` (jsonb).

## Legacy — don't extend

`cli/` (old chromadb stack), `web.py` (references a missing `retrieve_function`, contains a hardcoded API key), and `get_books.py`/`get_headers.py` (JPype/shameladb ingestion) are not part of the active stack.