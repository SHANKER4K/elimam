"""#24: the SSE stream always ends with `error`/`done`, and concurrent turns
on one session can't overwrite each other."""

import asyncio
import datetime

from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from db.connection import get_conn
from tests.chat_harness import chat, make_session, run_turn
from tests.fake_llm_server import FakeLLM


def _use(monkeypatch, agent_factory):
    monkeypatch.setattr(chat, "_build_agent", lambda **_kw: agent_factory())


def _names(events):
    return [e for e, _ in events]


def test_provider_5xx_ends_with_error_then_done_instead_of_a_dropped_stream():
    server = FakeLLM(status=500)
    try:
        session_id = make_session()
        events = asyncio.run(
            run_turn(session_id, "q", provider_slug="custom", provider_url=server.url)
        )
    finally:
        server.close()
    assert _names(events)[-2:] == ["error", "done"]
    error = dict(events)["error"]
    assert error["code"] == "model_http_error"
    assert error["status"] == 500
    assert error["message"]


def test_unexpected_exception_is_reported_without_leaking_internals(monkeypatch):
    async def boom(messages, info):
        raise RuntimeError("password=hunter2 at db.internal:5432")
        yield  # pragma: no cover - makes this an async generator

    _use(monkeypatch, lambda: Agent(FunctionModel(stream_function=boom)))
    events = asyncio.run(run_turn(make_session(), "q"))
    assert _names(events)[-2:] == ["error", "done"]
    error = dict(events)["error"]
    assert error["code"] == "internal_error"
    assert "hunter2" not in error["message"]


def test_non_json_tool_result_does_not_kill_the_stream(monkeypatch):
    def factory():
        agent = Agent(TestModel(custom_output_text="ok"))

        @agent.tool_plain
        def when() -> object:
            return {"at": datetime.datetime(2026, 1, 1), "ids": {1}}

        return agent

    _use(monkeypatch, factory)
    events = asyncio.run(run_turn(make_session(), "q"))
    assert "tool_result" in _names(events)
    assert events[-1] == ("done", {"output": "ok"})


def test_second_turn_on_a_busy_session_is_refused(monkeypatch):
    release = asyncio.Event()

    async def stream_fn(messages, info):
        await release.wait()
        yield "first"

    _use(monkeypatch, lambda: Agent(FunctionModel(stream_function=stream_fn)))
    session_id = make_session()

    async def both():
        first = asyncio.create_task(run_turn(session_id, "one"))
        await asyncio.sleep(0.3)  # first turn is now waiting on the model
        try:
            second = await asyncio.wait_for(run_turn(session_id, "two"), 5)
        finally:
            release.set()
        return await first, second

    first, second = asyncio.run(both())
    assert first[-1] == ("done", {"output": "first"})
    assert _names(second) == ["error", "done"]
    assert second[0][1]["code"] == "session_busy"
    # Only the first turn is stored.
    assert [r for r in _rows(session_id)] == [("user", "one"), ("assistant", "first")]


def test_turn_does_not_overwrite_a_write_that_landed_meanwhile(monkeypatch):
    """Another process (second worker, the bot) saved this session mid-turn."""
    session_id = make_session()

    async def stream_fn(messages, info):
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE sessions SET pydantic_message = '[{\"other\": 1}]'::jsonb WHERE id = %s",
                    (session_id,),
                )
            conn.commit()
        yield "mine"

    _use(monkeypatch, lambda: Agent(FunctionModel(stream_function=stream_fn)))
    events = asyncio.run(run_turn(session_id, "q"))

    assert dict(events)["error"]["code"] == "session_conflict"
    assert events[-1][0] == "done"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT pydantic_message FROM sessions WHERE id = %s", (session_id,))
            assert cur.fetchone()[0] == [{"other": 1}]
    assert _rows(session_id) == []


def test_busy_guard_is_released_after_an_error(monkeypatch):
    async def boom(messages, info):
        raise RuntimeError("x")
        yield  # pragma: no cover

    _use(monkeypatch, lambda: Agent(FunctionModel(stream_function=boom)))
    session_id = make_session()
    asyncio.run(run_turn(session_id, "q"))
    _use(monkeypatch, lambda: Agent(TestModel(call_tools=[], custom_output_text="ok")))
    assert asyncio.run(run_turn(session_id, "q"))[-1] == ("done", {"output": "ok"})


def _rows(session_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role::text, content FROM messages WHERE session_id = %s ORDER BY sequence",
                (session_id,),
            )
            return cur.fetchall()


def test_client_disconnect_closes_the_turn_promptly(monkeypatch):
    """Starlette aclose()s the generator when the client goes away; the inner
    run must be closed right then (not whenever GC gets to it), so the partial
    turn is persisted and the busy guard released before aclose() returns."""

    async def stream_fn(messages, info):
        yield "partial "
        await asyncio.sleep(30)
        yield "never"

    _use(monkeypatch, lambda: Agent(FunctionModel(stream_function=stream_fn)))
    session_id = make_session()

    async def disconnect_after_first_delta():
        gen = chat.stream(
            "q", session_id=session_id, provider_slug="openai", provider_url="",
            api_key="k", model_name="m", variant="none",
        )
        async for chunk in gen:
            if "text_delta" in chunk or "message_start" in chunk:
                break
        await gen.aclose()
        return session_id in chat._active_turns, chat.load_session(session_id)

    still_busy, history = asyncio.run(disconnect_after_first_delta())
    assert not still_busy
    assert history and history[0].parts[-1].content == "q"
