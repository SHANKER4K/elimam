"""The compaction summarizer must reach OpenCode with a session id.

`SummarizingCompaction` writes its summary with a *nested* Agent built from
`ctx.model` and the capability's own `model_settings`, so a header that lives
only in the run's settings never reaches it -- OpenCode then answers the
summary request with "Request is missing x-opencode-session".
"""

import asyncio

import pytest
from pydantic_ai_harness.compaction import SummarizingCompaction

from tests.chat_harness import chat, make_session, run_turn
from tests.fake_llm_server import FakeLLM


@pytest.fixture
def llm():
    server = FakeLLM()
    yield server
    server.close()


def test_the_summary_request_carries_the_session_header(llm, monkeypatch):
    # The fake server is not on opencode.ai, so hand the header to every URL.
    monkeypatch.setattr(
        chat, "opencode_session_headers", lambda *_: {"x-opencode-session": "elimam-s1"}
    )
    # Compact on the first turn that has any history at all.
    monkeypatch.setattr(
        chat, "capabilities", [SummarizingCompaction(max_tokens=1, keep_messages=0)]
    )
    turn = dict(
        provider_slug="custom",
        provider_url=llm.url + "/v1",
        api_key="sk",
        model_name="m1",
        variant="low",
    )
    session_id = make_session()
    asyncio.run(run_turn(session_id, "سؤال أول", **turn))
    before = len(llm.requests)
    asyncio.run(run_turn(session_id, "سؤال ثان", **turn))

    after = llm.requests[before:]
    assert len(after) == 2, "expected the summary request, then the turn"
    assert [r["headers"].get("x-opencode-session") for r in after] == [
        "elimam-s1",
        "elimam-s1",
    ]


def test_the_free_provider_client_carries_the_header(monkeypatch):
    """`deepseek-v4.1-flash` is served by the free provider on opencode.ai."""
    monkeypatch.setenv("FREE_PROVIDER_URL", "https://opencode.ai/zen/go/v1")
    agent = chat._build_agent(
        provider_slug="free",
        model_name="deepseek-v4.1-flash",
        provider_url="",
        api_key=None,
        variant="low",
        session_id="sess-1",
    )
    client = agent.model.provider.client
    assert client.default_headers["x-opencode-session"] == "elimam-sess-1"


def test_other_hosts_get_no_session_header():
    agent = chat._build_agent(
        provider_slug="custom",
        model_name="m1",
        provider_url="https://api.example.com/v1",
        api_key="sk",
        variant="low",
        session_id="sess-1",
    )
    assert "x-opencode-session" not in agent.model.provider.client.default_headers
