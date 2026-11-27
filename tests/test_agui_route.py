"""POST /chat/agui end to end, with Postgres and Qdrant stubbed out.

The handler, the AG-UI adapter and the agent run for real; only the database
helpers and the provider lookup are replaced, so what is asserted here is the
route's own logic: identity, session ownership, model resolution, and what is
persisted after a run.
"""

import importlib
import sys
import types

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

THREAD = "11111111-1111-4111-8111-111111111111"
ME, OTHER = "user-me", "user-other"
PRIOR = [
    ModelRequest(parts=[UserPromptPart(content="سلام")]),
    ModelResponse(parts=[TextPart(content="وعليكم السلام")]),
]
CONFIG = {
    "model": "m",
    "url": "https://p.example/v1",
    "variant": "low",
    "api_key": "k",
    "requires_key": True,
    "user_provider_id": "up-1",
}
PROPS = {"model_provider": "openai", "model_name": "m", "model_variant": "low"}


def body(**overrides):
    payload = {
        "threadId": THREAD,
        "runId": "r1",
        "state": {},
        "messages": [{"id": "m1", "role": "user", "content": "ما حكم الصيام؟"}],
        "tools": [],
        "context": [],
        "forwardedProps": PROPS,
    }
    payload.update(overrides)
    return payload


async def answer(messages, info):
    yield "الحمد لله"


@pytest.fixture
def route(monkeypatch):
    """api.agui imported against a stubbed db.connection and a no-op index setup."""
    stub = types.ModuleType("db.connection")
    stub.get_conn = lambda: pytest.fail("the route must not reach the database here")
    stub.close_pool = lambda: None
    monkeypatch.setitem(sys.modules, "db.connection", stub)

    import retrieval as retrieval_pkg

    monkeypatch.setattr(retrieval_pkg.retrieval, "setup_indexes", lambda: None)

    before = set(sys.modules)
    for name in ("api.chat", "api.agui", "api.sessions", "api.users", "api.providers", "api.keys"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    agui = importlib.import_module("api.agui")

    calls = {"created": [], "saved": [], "appended": []}
    monkeypatch.setattr(agui, "get_session", lambda sid: pytest.fail("set per test"))
    monkeypatch.setattr(agui, "resolve_model_config", lambda *a: dict(CONFIG))
    monkeypatch.setattr(agui, "create_web_session", lambda sid, req: calls["created"].append((sid, req)))
    monkeypatch.setattr(agui, "load_session", lambda sid: list(PRIOR))
    monkeypatch.setattr(agui, "save_session_messages", lambda sid, msgs: calls["saved"].append(list(msgs)))
    monkeypatch.setattr(agui, "append_session_messages", lambda sid, msgs: calls["appended"].append(list(msgs)))
    monkeypatch.setattr(agui, "_build_agent", lambda **kw: Agent(FunctionModel(stream_function=answer)))

    app = FastAPI()

    @app.middleware("http")
    async def identity(request, call_next):
        request.state.user_id = request.headers.get("x-test-user")
        return await call_next(request)

    app.include_router(agui.router)
    yield agui, TestClient(app), calls

    for name in set(sys.modules) - before:
        if name.startswith("api."):
            sys.modules.pop(name, None)


def no_session(agui, monkeypatch):
    def missing(sid):
        raise HTTPException(status_code=404, detail="Session not found")

    monkeypatch.setattr(agui, "get_session", missing)


def post(client, payload=None, user=ME):
    headers = {"x-test-user": user} if user else {}
    return client.post("/chat/agui", json=payload or body(), headers=headers)


def test_a_new_thread_creates_its_session_streams_and_saves_only_the_new_turn(route, monkeypatch):
    agui, client, calls = route
    no_session(agui, monkeypatch)

    response = post(client)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "RUN_STARTED" in response.text and "RUN_FINISHED" in response.text
    assert "الحمد لله" in response.text

    (sid, created), = calls["created"]
    assert sid == THREAD and created.user_id == ME and created.source == "web"
    assert created.user_provider_id == "up-1", "the session is pinned to the connection that answered"

    # The stored blob is the whole conversation; the log gets this turn only.
    assert len(calls["saved"][0]) == len(PRIOR) + 2
    assert len(calls["appended"][0]) == 2


def test_an_existing_session_of_the_caller_is_not_recreated(route, monkeypatch):
    agui, client, calls = route
    monkeypatch.setattr(
        agui, "get_session", lambda sid: {"user_id": ME, "user_provider_id": "up-1"}
    )
    assert post(client).status_code == 200
    assert calls["created"] == []


def test_another_users_session_is_a_404_that_does_not_reveal_it(route, monkeypatch):
    agui, client, calls = route
    monkeypatch.setattr(
        agui, "get_session", lambda sid: {"user_id": OTHER, "user_provider_id": None}
    )
    response = post(client)
    assert response.status_code == 404 and response.json()["detail"] == "Session not found"
    assert calls["saved"] == [] and calls["appended"] == []


def test_no_identity_is_401(route):
    _agui, client, calls = route
    assert post(client, user=None).status_code == 401
    assert calls["created"] == []


def test_missing_model_config_is_409_before_anything_is_created(route, monkeypatch):
    agui, client, calls = route
    no_session(agui, monkeypatch)
    response = post(client, body(forwardedProps={"model_name": "m"}))
    assert response.status_code == 409
    assert calls["created"] == []


def test_a_thread_id_that_is_not_a_uuid_is_400(route):
    _agui, client, _calls = route
    assert post(client, body(threadId="not-a-uuid")).status_code == 400


def test_a_malformed_run_input_is_422(route):
    _agui, client, _calls = route
    assert post(client, {"threadId": THREAD}).status_code == 422


def test_a_provider_that_needs_a_key_but_has_none_is_400(route, monkeypatch):
    agui, client, calls = route
    no_session(agui, monkeypatch)
    monkeypatch.setattr(agui, "resolve_model_config", lambda *a: {**CONFIG, "api_key": None})
    response = post(client)
    assert response.status_code == 400 and calls["created"] == []
