"""#32: small correctness cleanups, checked through the real app + Postgres."""

import uuid

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import tests.chat_harness  # noqa: F401  (stubs Qdrant index setup before api.chat loads)
import api.sessions as sessions
import server
from retrieval import Retrieval


@pytest.fixture(scope="module")
def client():
    return TestClient(server.app, raise_server_exceptions=False)


def _bot(telegram_id: str) -> dict:
    return {"X-Bot-Secret": "botsecret", "X-Telegram-Id": telegram_id}


def test_get_session_bad_row_is_a_500_not_a_typeerror(monkeypatch):
    def broken(row):
        raise IndexError("tuple index out of range")

    monkeypatch.setattr(sessions, "_row_to_session", broken)
    with pytest.raises(HTTPException) as exc:
        sessions.get_session(str(uuid.uuid4()))
    assert exc.value.status_code == 500


def test_database_errors_are_not_echoed_to_the_client(client):
    # A session for a user_provider_id that doesn't exist violates a foreign key.
    headers = _bot(f"tg-{uuid.uuid4().hex[:8]}")
    response = client.post(
        "/sessions/add",
        json={"source": "telegram", "user_provider_id": str(uuid.uuid4())},
        headers=headers,
    )
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "violates" not in detail and "constraint" not in detail and "sessions" not in detail


def test_add_session_returns_an_object_not_a_one_element_list(client):
    response = client.post("/sessions/add", json={"source": "telegram"}, headers=_bot(f"tg-{uuid.uuid4().hex[:8]}"))
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, dict)
    assert body["source"] == "telegram" and body["is_active"] is True


def test_identity_401s_are_logged_and_carry_a_request_id(client, monkeypatch):
    monkeypatch.setattr(server, "AUTH_MODE", "enforce")
    response = client.get("/sessions/" + str(uuid.uuid4()), headers={"X-Request-ID": "req-123"})
    assert response.status_code == 401
    assert response.headers.get("X-Request-ID") == "req-123"


def test_web_chat_does_not_treat_a_db_outage_as_a_new_session(client, monkeypatch):
    import api.chat as chat

    def outage(session_id):
        raise HTTPException(status_code=500, detail="Database error")

    created = []
    monkeypatch.setattr(chat, "get_session", outage)
    monkeypatch.setattr(chat, "create_web_session", lambda *a, **k: created.append(a))
    response = client.post(
        "/chat/web",
        json={
            "message": "q",
            "model_name": "m",
            "model_provider": "openai",
            "model_variant": "low",
            "session_id": str(uuid.uuid4()),
        },
        headers=_bot(f"tg-{uuid.uuid4().hex[:8]}"),
    )
    assert response.status_code == 500
    assert created == []


def test_lookup_misses_return_a_typed_not_found_for_the_llm():
    class _Empty:
        def scroll(self, **kwargs):
            return [], None

    r = Retrieval(url="http://127.0.0.1:1")
    r.__dict__["client"] = _Empty()
    assert r.by_id("quran", "999:1") == {"error": "not_found", "collection": "quran", "id": "999:1"}
    assert r.by_hadith("bukhari", 99999) == {
        "error": "not_found",
        "collection": "hadith",
        "id": "bukhari:99999.0",
    }
