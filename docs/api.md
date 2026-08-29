# Elimam API

FastAPI backend served at `:8000`. Auto-generated docs:

- Swagger UI: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>
- OpenAPI 3.1: <http://localhost:8000/openapi.json>

This file covers what the auto-generated docs do **not**: the auth model,
the SSE event protocol, and behavioral notes verified against the source.

## Authentication

Only three routes are authenticated (verified in `api/*.py` — no other
route declares `Header` params):

| Route | Headers required |
|-------|------------------|
| `POST /chat` | `X-Bot-Secret`, `X-Telegram-Id` |
| `POST /chat/structured` | `X-Bot-Secret`, `X-Telegram-Id` |
| `GET /providers` | `X-Bot-Secret` |

- `X-Bot-Secret` must equal the backend's `BOT_SHARED_SECRET` (from `.env`).
  Any client that knows the secret can set `X-Telegram-Id`; without the
  secret the request is rejected with 401. The trust boundary is the secret
  itself — keep it out of source control.
- `X-Telegram-Id` identifies the user the bot is acting for. It is never
  accepted from the request body.
- All other routes (`/sessions`, `/users`, `/messages`, `/keys`, the four
  `/chat/*_search` routes, `/chat/health`) take no headers. They are
  intended for internal calls and the Next.js UI.

API keys are stored per `(user_id, provider)`, Fernet-encrypted at write
time (`ENCRYPTION_MASTER_KEY`). The server never returns the raw key
(verified: `GET /keys/*` returns metadata only) and never logs it.

## Chat: streaming vs. structured

`POST /chat` streams the agent's output as Server-Sent Events:

| Event | When | Payload |
|-------|------|---------|
| `message_start` | First text part begins | `{"index": int, "text": str}` |
| `text_delta` | Each text chunk | `{"text": str}` |
| `tool` | A tool is called | `{"text": str, "tool_call_id": str, "args": dict}` |
| `tool_result` | The tool returns | `{"text": str, "tool_call_id": str}` |
| `message_end` | Current part ends | `{}` |
| `done` | The whole run finished | `{"output": ChatResponse}` |

`POST /chat/structured` returns the same `ChatResponse` synchronously.

## Citation hydration

The LLM emits inline markers `{arabic_fragment|surah_number:ayah_number}`.
The server replaces them with `[surah:ayah]`, verifies each fragment
against the authoritative Quran record, and builds `ChatResponse`:

- `raw_response_text` — text with markers replaced by canonical citations.
- `detected_citations` — one entry per marker in left-to-right order;
  `order` is 1-indexed; `verse_fragment_text` is the **verified** Quran
  text (falls back to the full ayah when the marker fragment is a
  tashkīl-drifted substring — verified in `_resolve_fragment`).
- `resources` — one entry per unique (source_type, surah, ayah, tafsir_book)
  tuple actually cited. `tafsir_book` is set only for `source_type="tafsir"`.
  Unknown tafsir slugs are dropped (verified in `_tafsir_url` — no URL is
  fabricated).

Malformed markers (missing pipe, unclosed brace) raise `ModelRetry` in the
output validator and the LLM retries up to `retries={"output": 3}` (verified
in `build_agent`). Persistent failure surfaces as a 422 on `/chat/structured`
or as a partial stream on `/chat`.

## Errors

All error responses are `{"detail": str}` (verified: every route raises
`HTTPException(status, detail=...)`; the global handler returns
`{"detail": "Internal Server Error"}` for 500s and logs the traceback in the
JSON logs, correlated by `request_id` via `X-Request-ID`).

| Status | When |
|--------|------|
| 400 | Bad input, missing Telegram identity, missing API key, or a domain `psycopg2` error surfaced as 400. |
| 401 | Missing or wrong `X-Bot-Secret`. |
| 404 | Resource not found. |
| 409 | User has no active session. |
| 422 | Pydantic body validation (auto) or a Quran record missing for a marker (domain). |
| 500 | Unhandled exception; body is always `{"detail": "Internal Server Error"}`. |

## Quickstart

```bash
# Requires BOT_SHARED_SECRET in your environment. Never commit it.
# 1. Stream a chat turn (SSE)
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Bot-Secret: $BOT_SHARED_SECRET" \
  -H "X-Telegram-Id: 123456789" \
  -d '{"message": "ما هي آية الكرسي؟"}'

# 2. Same call, structured response
curl -s -X POST http://localhost:8000/chat/structured \
  -H "Content-Type: application/json" \
  -H "X-Bot-Secret: $BOT_SHARED_SECRET" \
  -H "X-Telegram-Id: 123456789" \
  -d '{"message": "ما هي آية الكرسي؟"}' | jq
```

> Never paste real API keys, bot secrets, or Telegram credentials into
> Swagger examples or repo files.

## Local development

```bash
uvicorn server:app --reload
# Swagger UI: http://localhost:8000/docs
```
