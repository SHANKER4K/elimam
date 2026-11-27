"""#22: variants go through pydantic-ai's unified `thinking` setting.

The real adapters are driven against a local fake server, so these assert on
the request bodies that would reach OpenAI / Anthropic.
"""

import asyncio
import functools

import pytest

from tests.chat_harness import chat
from tests.fake_llm_server import FakeLLM


@pytest.fixture
def llm(monkeypatch):
    server = FakeLLM()
    # Builtin providers take no base_url in _build_agent; aim them at the fake.
    monkeypatch.setattr(
        chat, "OpenAIProvider", functools.partial(chat.OpenAIProvider, base_url=server.url)
    )
    monkeypatch.setattr(
        chat, "AnthropicProvider", functools.partial(chat.AnthropicProvider, base_url=server.url)
    )
    yield server
    server.close()


def _run(agent):
    return asyncio.run(agent.run("سؤال")).output


def _build(slug, model, variant, url=""):
    return chat._build_agent(
        provider_slug=slug, model_name=model, provider_url=url, api_key="k", variant=variant
    )


def test_new_claude_gets_adaptive_thinking_not_a_budget(llm):
    # Opus 4.7+ return 400 for {"type": "enabled", "budget_tokens": ...}; the old
    # code sent exactly that for any variant (pydantic-ai raised UserError first).
    assert _run(_build("anthropic", "claude-opus-4-7", "high")) == "ok"
    body = llm.last["body"]
    assert body["thinking"]["type"] == "adaptive"
    assert "budget_tokens" not in body["thinking"]
    assert body["output_config"]["effort"] == "high"


def test_older_claude_still_gets_a_budget_scaled_to_the_variant(llm):
    _run(_build("anthropic", "claude-sonnet-4-5", "low"))
    low = llm.last["body"]["thinking"]
    _run(_build("anthropic", "claude-sonnet-4-5", "high"))
    high = llm.last["body"]["thinking"]
    assert low["type"] == high["type"] == "enabled"
    assert low["budget_tokens"] < high["budget_tokens"]


def test_openai_is_not_forced_onto_the_flex_tier(llm):
    _run(_build("openai", "gpt-5", "low"))
    body = llm.last["body"]
    assert "service_tier" not in body
    assert body["reasoning_effort"] == "low"


def test_variant_none_turns_thinking_off(llm):
    _run(_build("anthropic", "claude-sonnet-4-5", "none"))
    assert "thinking" not in llm.last["body"] or llm.last["body"]["thinking"]["type"] == "disabled"


def test_custom_openai_compatible_model_without_reasoning_is_left_alone(llm):
    # A custom endpoint serving a non-reasoning model must not get reasoning_effort.
    _run(_build("my-gateway", "llama-3-70b", "high", url=llm.url))
    body = llm.last["body"]
    assert "reasoning_effort" not in body
    assert "service_tier" not in body
    assert body["temperature"] == 0.5


@pytest.mark.parametrize(
    "variant, expected",
    [
        (None, None),
        ("", None),
        ("none", False),
        ("low", "low"),
        ("high", "high"),
        ("max", "max"),
        ("thinking", True),  # unknown labels mean "on, provider default"
    ],
)
def test_variant_mapping(variant, expected):
    settings = chat._build_settings(variant)
    if expected is None:
        assert settings is None
    else:
        assert settings["thinking"] == expected
