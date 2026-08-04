# Telegram Bot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Telegram bot that lets users chat with the Islamic scholar agent served by `server.py`, streaming answers as they are generated.

**Architecture:** The bot runs as a separate process from the server. It long-polls the Telegram Bot API, and for every user text message it POSTs to the server's `POST /chat` SSE endpoint with `session_id = chat_id:generation`, consumes the SSE stream (`text_delta` events), accumulates the answer, splits it into Telegram-size chunks (≤4096 chars), and sends them. History is kept server-side per session_id; `/reset` bumps the generation to start a fresh conversation.

**Tech Stack:** `python-telegram-bot` ≥21 (polling, update loop, rate-limit handling), `httpx` (already installed, SSE consumption), pytest (already installed, sync tests with `asyncio.run` — no pytest-asyncio needed).

## Global Constraints

- Target server: root `server.py` (`POST /chat`, body `{"message": str, "session_id": str}`, SSE events `message_start`/`text_delta`/`tool`/`tool_result`/`message_end`/`done`). Do not touch `server.py` or `backend/server.py`.
- Bot must be a separate process: `python telegram_bot.py`.
- No new test dependencies. Tests are sync, using `asyncio.run()`.
- Responses are Arabic → always `reply_text` without `parse_mode` (avoids HTML escaping bugs entirely).
- Config via env vars, no config file: `TELEGRAM_BOT_TOKEN` (required), `ISLAM_SERVER_URL` (default `http://localhost:8000`).
- `python-telegram-bot>=21` is already installed.
- Work on a new branch `feat/telegram-bot`. Commit style matches repo (`feat:` prefix).

## File Structure

- `telegram_bot.py` — SSE chat client (`stream_chat`), text splitter (`split_text`), Telegram handlers + `main()`. Importing the module must not start the bot (guarded by `if __name__ == "__main__"`).
- `test_telegram_bot.py` — tests for `stream_chat` (via `httpx.MockTransport`) and `split_text`.
- `pyproject.toml` — dependency added.

---

### Task 1: SSE chat client

**Files:**

- Create: `telegram_bot.py`
- Test: `test_telegram_bot.py`

**Interfaces:**

- Consumes: server `POST /chat` SSE stream.
- Produces: `async def stream_chat(message: str, session_id: str, *, base_url: str = "", client: httpx.AsyncClient | None = None) -> AsyncIterator[str]` — yields each non-empty text chunk (from `message_start`/`text_delta` events). `client` is injectable for tests; when None a client is created/closed internally. Returns nothing special on `done`.
- Produces: `def split_text(text: str, limit: int = 4096) -> list[str]` — chunks on `\n` boundaries; a single over-long paragraph is hard-sliced at `limit`.

- [ ] **Step 1: Write the failing test**

`test_telegram_bot.py`:

```python
import asyncio
import httpx

from telegram_bot import split_text, stream_chat

SSE_BODY = (
    'event: message_start\n'
    'data: {"index":0,"text":""}\n\n'
    'event: text_delta\n'
    'data: {"text":"السلام"}\n\n'
    'event: text_delta\n'
    'data: {"text":" عليكم"}\n\n'
    'event: tool\n'
    'data: {"text":"get_quran","tool_call_id":"c1"}\n\n'
    'event: done\n'
    'data: {"output":"السلام عليكم"}\n\n'
)


def test_stream_chat_joins_text_deltas():
    async def run():
        def fake_sse(request: httpx.Request) -> httpx.Response:
            assert request.json() == {"message": "مرحبا", "session_id": "1:0"}
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=SSE_BODY.encode(),
            )

        transport = httpx.MockTransport(fake_sse)
        async with httpx.AsyncClient(transport=transport) as client:
            chunks = [c async for c in stream_chat("مرحبا", "1:0", client=client)]
        return chunks

    assert asyncio.run(run()) == ["السلام", " عليكم"]


def test_stream_chat_raises_on_http_error():
    async def run():
        def fake_sse(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        transport = httpx.MockTransport(fake_sse)
        async with httpx.AsyncClient(transport=transport) as client:
            async for _ in stream_chat("مرحبا", "1:0", client=client):
                pass

    try:
        asyncio.run(run())
        assert False, "expected HTTPStatusError"
    except httpx.HTTPStatusError:
        pass


def test_split_text_respects_limit():
    assert split_text("short") == ["short"]
    parts = split_text("أ" * 9000, limit=4096)
    assert len(parts) == 3
    assert all(len(p) <= 4096 for p in parts)
    assert "".join(parts) == "أ" * 9000


def test_split_text_breaks_on_newlines():
    text = "خط\n" * 3000  # 4 chars per line, 12000 total
    parts = split_text(text, limit=4096)
    assert len(parts) > 1
    assert all("\n" in p or len(p) <= 4096 for p in parts)
    assert "".join(parts) == text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test_telegram_bot.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'telegram_bot'`

- [ ] **Step 3: Write minimal implementation**

`telegram_bot.py`:

```python
import asyncio
import json
import os
from collections.abc import AsyncIterator

import httpx


async def stream_chat(
    message: str,
    session_id: str,
    *,
    base_url: str = "",
    client: httpx.AsyncClient | None = None,
) -> AsyncIterator[str]:
    """Stream the agent's reply from the server's /chat SSE endpoint."""
    base_url = base_url or os.environ.get("ISLAM_SERVER_URL", "http://localhost:8000")
    own_client = client is None
    client = client or httpx.AsyncClient(timeout=60.0)
    try:
        async with client.stream(
            "POST",
            f"{base_url}/chat",
            json={"message": message, "session_id": session_id},
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line[6:])
                text = payload.get("text")
                if text:
                    yield text
    finally:
        if own_client:
            await client.aclose()


def split_text(text: str, limit: int = 4096) -> list[str]:
    """Split on newlines; hard-slice any single paragraph longer than limit."""
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    buf: list[str] = []
    size = 0
    for para in text.split("\n"):
        if len(para) > limit:  # one paragraph alone exceeds the limit
            if buf:
                parts.append("\n".join(buf))
                buf, size = [], 0
            parts.extend(para[i : i + limit] for i in range(0, len(para), limit))
        elif size + len(para) + 1 > limit:
            parts.append("\n".join(buf))
            buf, size = [para], len(para)
        else:
            buf.append(para)
            size += len(para) + 1
    if buf:
        parts.append("\n".join(buf))
    return parts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test_telegram_bot.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add telegram_bot.py test_telegram_bot.py
git commit -m "feat: SSE chat client and text splitter for telegram bot"
```

---

### Task 2: Telegram bot wiring

**Files:**

- Modify: `telegram_bot.py` (append handlers + `main`)
- Modify: `pyproject.toml` (dependency via `uv add`)

**Interfaces:**

- Consumes: `stream_chat`, `split_text` from Task 1; env var `TELEGRAM_BOT_TOKEN`.
- Produces: runnable bot — `uv run python telegram_bot.py`.

- [ ] **Step 1: Add dependency**

Run: `uv add python-telegram-bot`
Expected: `python-telegram-bot` appears in `pyproject.toml` dependencies and `uv.lock`.

- [ ] **Step 2: Write the failing test (module import must not start the bot)**

Add to `test_telegram_bot.py`:

```python
def test_import_does_not_start_bot():
    import subprocess
    import sys

    # Importing the module must not block or require a token.
    result = subprocess.run(
        [sys.executable, "-c", "import telegram_bot; print('ok')"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0 and result.stdout.strip() == "ok"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest test_telegram_bot.py::test_import_does_not_start_bot -v`
Expected: FAIL with `KeyError: 'TELEGRAM_BOT_TOKEN'` or a blocking main loop (timeout) — whichever the stub raises first. (If the file doesn't exist yet, `ModuleNotFoundError` also counts.)

- [ ] **Step 4: Write the bot**

Append to `telegram_bot.py`:

```python
from telegram import ChatAction, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

sessions: dict[int, int] = {}  # chat_id -> generation, bumped by /reset


def session_id(chat_id: int) -> str:
    return f"{chat_id}:{sessions.get(chat_id, 0)}"


async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    chat_id = update.message.chat_id
    await ctx.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    try:
        text = "".join(
            [
                chunk
                async for chunk in stream_chat(update.message.text, session_id(chat_id))
            ]
        )
    except Exception:
        await update.message.reply_text("حدث خطأ في الاتصال بالخادم، حاول مرة أخرى.")
        return
    if not text:
        text = "لم أستطع توليد رد."
    for part in split_text(text):
        await update.message.reply_text(part)


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "السلام عليكم، أنا مساعدك في الأسئلة الإسلامية.\n"
        "اسألني عن القرآن والحديث والتفسير والعقيدة، وسأجيب مع ذكر المصادر.\n"
        "استخدم /reset لبدء محادثة جديدة."
    )


async def cmd_reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    sessions[chat_id] = sessions.get(chat_id, 0) + 1
    await update.message.reply_text("تم تصفير المحادثة، ابدأ سؤالاً جديداً.")


def main() -> None:
    app = Application.builder().token(os.environ["TELEGRAM_BOT_TOKEN"]).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.run_polling()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest test_telegram_bot.py -v`
Expected: 5 passed

- [ ] **Step 6: Manual integration check**

1. Terminal 1 (from project root): `uv run uvicorn server:app --port 8000`
2. Terminal 2: `TELEGRAM_BOT_TOKEN=<token> uv run python telegram_bot.py`
3. In Telegram: `/start`, then a question in Arabic. Expected: typing indicator, then the full answer (may arrive in several messages if >4096 chars).
4. `/reset` then another question. Expected: answer without memory of the first exchange.

- [ ] **Step 7: Commit**

```bash
git add telegram_bot.py test_telegram_bot.py pyproject.toml uv.lock
git commit -m "feat: telegram bot streaming answers from islamic scholar server"
```

---

## Self-Review

**Spec coverage:**

- "Telegram bot that works with that server" → Task 2 (bot) + Task 1 (server client). Bot talks to `server.py`'s `/chat` SSE; session history handled server-side per session_id.
- Streaming → `stream_chat` consumes `text_delta` events; typing indicator shown while waiting.
- Nothing modifies the server; both files coexist.

**Placeholder scan:** No TBDs; every step has concrete code or commands. Test code is complete, not described.

**Type consistency:** `stream_chat(message, session_id, *, base_url="", client=None) -> AsyncIterator[str]`, `split_text(text, limit=4096) -> list[str]` — same signatures in Task 1 tests, Task 1 implementation, and Task 2 usage. `session_id(chat_id)` returns `f"{chat_id}:{generation}"` everywhere.

**Known simplifications (deliberate):**

- `ponytail:` no live message editing — final answer is sent once at `done` (typing action covers perceived latency). Add `editMessageText` per delta if real-time streaming is wanted.
- `ponytail:` polling, not webhook — works without a public URL. Switch to `Application.run_webhook` if the server is ever publicly reachable.
- Plain-text replies (no `parse_mode`) — Arabic text is sent as-is, no escaping bugs.
