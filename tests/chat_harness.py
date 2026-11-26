"""Shared helpers for tests that drive `api.chat.stream` end to end.

Importing `api.chat` runs `retrieval.setup_indexes()` against Qdrant (see
AGENTS.md), so that call is stubbed out *before* the import. Everything else is
real: the Postgres schema is the one the frontend's Drizzle schema generates,
and the LLM is pydantic-ai's own `TestModel` / `FunctionModel`, so no network
and no Qdrant tool calls are involved.
"""

import json
import uuid

import retrieval

retrieval.retrieval.setup_indexes = lambda: None

import api.chat as chat  # noqa: E402
from db.connection import get_conn  # noqa: E402


def make_session() -> str:
    """Insert a throwaway user + session and return the session id."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO users DEFAULT VALUES RETURNING id")
            user_id = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO sessions (user_id, source) VALUES (%s, 'web') RETURNING id",
                (user_id,),
            )
            session_id = str(cur.fetchone()[0])
        conn.commit()
    return session_id


def message_rows(session_id: str) -> list[tuple]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role::text, content, sequence FROM messages "
                "WHERE session_id = %s ORDER BY sequence",
                (session_id,),
            )
            return cur.fetchall()


def parse_sse(chunks: list[str]) -> list[tuple[str, object]]:
    events = []
    for chunk in chunks:
        lines = chunk.strip().splitlines()
        event = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        events.append((event, data))
    return events


async def run_turn(session_id: str, prompt: str, **kwargs) -> list[tuple[str, object]]:
    params = dict(
        provider_slug="openai",
        provider_url="",
        api_key="sk-test",
        model_name="test",
        variant="none",
    )
    params.update(kwargs)
    chunks = [c async for c in chat.stream(prompt, session_id=session_id, **params)]
    return parse_sse(chunks)


def random_uuid() -> str:
    return str(uuid.uuid4())
