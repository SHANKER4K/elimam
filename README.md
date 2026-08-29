# Elimam

Islamic scholar agent. FastAPI backend + Qdrant vector search + Postgres +
Telegram bot.

## Layout

| Path | What's there |
|------|--------------|
| `server.py` | FastAPI app entrypoint. |
| `api/` | HTTP routers (chat, sessions, users, messages, keys, providers). |
| `search.py` | Hybrid Qdrant search (dense + sparse + RRF). |
| `db/` | Postgres connection + schema. |
| `skills/` | LLM system prompts (turath, citations). |
| `telegram_bot/` | Aiogram bot; talks to the backend over HTTP. |
| `ui/` | Next.js app (separate `AGENTS.md`). |

## Running locally

```bash
docker compose up
# Backend:    http://localhost:8000
# Swagger UI: http://localhost:8000/docs
```

## API documentation

- Live interactive docs: <http://localhost:8000/docs>
- Hand-written reference: [`docs/api.md`](docs/api.md) (auth model, SSE
  protocol, citation hydration, error codes)

## Tests

```bash
python -m pytest tests/ -v
```

Note: `test_search.py` is an integration test that requires a populated
Qdrant (~148k book points) and the embedding models; it is not a regression
gate on a fresh checkout.
