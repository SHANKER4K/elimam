"""#23: custom connections must honour `api_style` and `extra_headers` at chat time."""

import asyncio
import functools
import json

import pytest

from api.keys import encrypt
from api.providers import resolve_model_config
from db.connection import get_conn
from tests.chat_harness import chat, make_session, run_turn
from tests.fake_llm_server import FakeLLM


@pytest.fixture
def llm():
    server = FakeLLM()
    yield server
    server.close()


def _custom_connection(api_style: str, base_url: str, headers: dict) -> tuple[str, str]:
    """A user with one custom connection serving model `m1`."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO users DEFAULT VALUES RETURNING id")
            user_id = str(cur.fetchone()[0])
            cur.execute(
                """
                INSERT INTO user_providers
                    (user_id, is_custom, custom_name, base_url, api_style,
                     encrypted_key, extra_headers)
                VALUES (%s, true, 'gw', %s, %s, %s, %s::jsonb)
                RETURNING id
                """,
                (user_id, base_url, api_style, encrypt("sk-custom"), json.dumps(headers)),
            )
            connection_id = str(cur.fetchone()[0])
            cur.execute(
                "INSERT INTO provider_models (user_provider_id, model_id, variants) "
                "VALUES (%s, 'm1', ARRAY['low'])",
                (connection_id,),
            )
        conn.commit()
    return user_id, connection_id


def test_resolver_returns_api_style_and_headers_of_the_connection():
    user_id, connection_id = _custom_connection(
        "anthropic_compatible", "https://gw.example/v1", {"X-Org": "acme"}
    )
    config = resolve_model_config(connection_id, "custom", "m1", "low", user_id)
    assert config["api_style"] == "anthropic_compatible"
    assert config["extra_headers"] == {"X-Org": "acme"}
    assert config["api_key"] == "sk-custom"


def test_resolver_rejects_headers_that_would_not_pass_validation_today():
    # Rows written before header validation existed must not be sent blindly.
    user_id, connection_id = _custom_connection(
        "openai_compatible", "https://gw.example/v1", {"Host": "evil"}
    )
    with pytest.raises(ValueError):
        resolve_model_config(connection_id, "custom", "m1", "low", user_id)


def test_anthropic_compatible_custom_endpoint_speaks_the_messages_api(llm):
    session_id = make_session()
    events = asyncio.run(
        run_turn(
            session_id,
            "سؤال",
            provider_slug="custom",
            provider_url=llm.url + "/v1",  # stored like the catalog: .../v1
            api_key="sk-custom",
            model_name="m1",
            variant="low",
            api_style="anthropic_compatible",
            extra_headers={"X-Org": "acme"},
        )
    )
    request = llm.last
    assert request["path"] == "/v1/messages"  # not /v1/v1/messages, not /chat/completions
    # The Anthropic SDK decides how to send the key (x-api-key or Bearer).
    assert "sk-custom" in (request["headers"].get("x-api-key", "") + request["headers"].get("authorization", ""))
    assert request["headers"]["x-org"] == "acme"
    assert ("done", {"output": "ok"}) in events


def test_openai_compatible_custom_endpoint_gets_extra_headers(llm):
    session_id = make_session()
    asyncio.run(
        run_turn(
            session_id,
            "سؤال",
            provider_slug="custom",
            provider_url=llm.url + "/v1",
            model_name="m1",
            variant="low",
            api_style="openai_compatible",
            extra_headers={"HTTP-Referer": "https://elimam.app"},
        )
    )
    assert llm.last["path"] == "/v1/chat/completions"
    assert llm.last["headers"]["http-referer"] == "https://elimam.app"


def test_builtin_providers_also_get_the_connections_extra_headers(llm, monkeypatch):
    monkeypatch.setattr(
        chat, "OpenAIProvider", functools.partial(chat.OpenAIProvider, base_url=llm.url)
    )
    agent = chat._build_agent(
        provider_slug="openai",
        model_name="gpt-5",
        provider_url="",
        api_key="k",
        variant="low",
        extra_headers={"OpenAI-Organization": "org-1"},
    )
    asyncio.run(agent.run("q"))
    assert llm.last["headers"]["openai-organization"] == "org-1"
