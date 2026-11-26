"""#21: each turn must add only *its own* rows to `messages`.

Needs the Postgres from `.env` (like every test that imports `api.*`).
"""

import asyncio

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel

from tests.chat_harness import chat, make_session, message_rows, run_turn


@pytest.fixture
def test_agent(monkeypatch):
    def build(**_kwargs):
        return Agent(
            TestModel(call_tools=[], custom_output_text="جواب"),
            system_prompt="system",
        )

    monkeypatch.setattr(chat, "_build_agent", build)


def test_two_turns_store_four_rows_not_six(test_agent):
    session_id = make_session()

    asyncio.run(run_turn(session_id, "سؤال 1"))
    asyncio.run(run_turn(session_id, "سؤال 2"))

    assert message_rows(session_id) == [
        ("user", "سؤال 1", 1),
        ("assistant", "جواب", 2),
        ("user", "سؤال 2", 3),
        ("assistant", "جواب", 4),
    ]

    # The full pydantic-ai history is still kept for the next turn.
    history = chat.load_session(session_id)
    assert len(history) == 4


def test_projection_skips_tool_only_and_system_only_messages():
    tool_return = ModelRequest(
        parts=[ToolReturnPart(tool_name="search", content={"x": 1}, tool_call_id="c1")]
    )
    system_only = ModelRequest(parts=[SystemPromptPart(content="sys")])
    tool_call = ModelResponse(
        parts=[ThinkingPart(content="..."), ToolCallPart(tool_name="search", args={})]
    )

    assert chat._project_message(tool_return) is None
    assert chat._project_message(system_only) is None
    assert chat._project_message(tool_call) is None


def test_projection_keeps_only_text_of_multimodal_prompts():
    prompt = ModelRequest(
        parts=[
            SystemPromptPart(content="sys"),
            UserPromptPart(
                content=[
                    "ما حكم ",
                    BinaryContent(data=b"\x89PNG", media_type="image/png"),
                    "هذا؟",
                ]
            ),
        ]
    )
    assert chat._project_message(prompt) == ("user", "ما حكم هذا؟")

    answer = ModelResponse(parts=[TextPart(content="أ"), TextPart(content="ب")])
    assert chat._project_message(answer) == ("assistant", "أب")


def test_append_skips_unprojectable_messages():
    session_id = make_session()
    chat.append_session_messages(
        session_id,
        [
            ModelRequest(parts=[UserPromptPart(content="q")]),
            ModelResponse(parts=[ToolCallPart(tool_name="search", args={})]),
            ModelRequest(
                parts=[ToolReturnPart(tool_name="search", content="r", tool_call_id="c")]
            ),
            ModelResponse(parts=[TextPart(content="a")]),
        ],
    )
    assert message_rows(session_id) == [("user", "q", 1), ("assistant", "a", 2)]


def test_tool_calling_turn_logs_only_the_question_and_the_answer(monkeypatch):
    def build(**_kwargs):
        agent = Agent(TestModel(custom_output_text="الجواب"), system_prompt="system")

        @agent.tool_plain
        def lookup(q: str) -> str:
            return "نتيجة"

        return agent

    monkeypatch.setattr(chat, "_build_agent", build)
    session_id = make_session()

    events = asyncio.run(run_turn(session_id, "ابحث"))

    assert "tool_result" in {e for e, _ in events}
    assert message_rows(session_id) == [("user", "ابحث", 1), ("assistant", "الجواب", 2)]
    # tool call + tool return are kept in the pydantic-ai history.
    assert len(chat.load_session(session_id)) == 4
