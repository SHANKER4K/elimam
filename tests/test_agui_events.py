"""AG-UI event stream: what the Elimam UI reads, no Postgres and no Qdrant.

A scripted model drives a real pydantic-ai agent through the real AGUIAdapter,
so these tests fail if the adapter's event shapes change under us.
"""

import asyncio

from ag_ui.core import RunAgentInput, UserMessage
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from api.agui_events import ElimamAGUIAdapter

VERSE = "اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ"
PAYLOADS = {("quran", "2:255"): {"ids": "2:255", "text": VERSE, "surah": "البقرة", "ayah_number": 255}}


def lookup(collection: str, ids: str):
    return PAYLOADS.get((collection, ids), [])


def run_input(text: str = "آية الكرسي") -> RunAgentInput:
    return RunAgentInput(
        thread_id="11111111-1111-4111-8111-111111111111",
        run_id="run-1",
        state=None,
        messages=[UserMessage(id="m1", role="user", content=text)],
        tools=[],
        context=[],
        forwarded_props={},
    )


def collect(model: FunctionModel, *, tools=()):
    agent = Agent(model, tools=list(tools))
    adapter = ElimamAGUIAdapter(
        agent=agent, run_input=run_input(), manage_message_history="server"
    )
    adapter.citation_lookup = lookup

    async def go():
        return [e async for e in adapter.run_stream(message_history=[])]

    return asyncio.run(go())


def types(events) -> list[str]:
    return [getattr(e.type, "value", e.type) for e in events]


async def search_then_answer(messages: list[ModelMessage], info: AgentInfo):
    if len(messages) == 1:  # first request: call the tool, args arrive in pieces
        yield {0: DeltaToolCall(name="search", json_args='{"query_text": "آية ', tool_call_id="t1")}
        yield {0: DeltaToolCall(json_args='الكرسي", "collection": "quran"}')}
    else:  # second request: answer, with a citation tag split mid-tag
        yield "قال تعالى: <quran id="
        yield '"2:255">لا إله إلا هو</qu'
        yield "ran> انتهى."


def search(query_text: str, collection: str) -> list[dict]:
    """stand-in for the retrieval tool"""
    return [{"ids": "2:255", "preview": VERSE}]


def test_tool_events_carry_the_metadata_the_ui_reads():
    events = collect(FunctionModel(stream_function=search_then_answer), tools=[search])

    start = next(e for e in events if types([e]) == ["TOOL_CALL_START"])
    assert start.metadata == {"title": "بحث في المصادر", "icon": "search", "subtitle": ""}

    ready = next(e for e in events if getattr(e, "name", "") == "tool_call_ready")
    assert ready.value["tool_call_id"] == "t1"
    assert "آية الكرسي" in ready.value["subtitle"], "subtitle comes from the finished arguments"

    result = next(e for e in events if types([e]) == ["TOOL_CALL_RESULT"])
    assert result.metadata["status"] == "ok"
    assert result.metadata["count"] == 1
    assert isinstance(result.metadata["duration_ms"], int)
    assert result.metadata["preview"][0]["ids"] == "2:255"


def test_citation_tags_are_rewritten_and_emitted_as_cards():
    events = collect(FunctionModel(stream_function=search_then_answer), tools=[search])

    text = "".join(e.delta for e in events if types([e]) == ["TEXT_MESSAGE_CONTENT"])
    # The model wrote the anchor without tashkeel, split across chunks; the text
    # stream carries one canonical tag whose body is the source's own spelling.
    assert text.startswith('قال تعالى: <quran id="2:255" match="exact"')
    assert ">لَا إِلَٰهَ إِلَّا هُو</quran>" in text
    assert "لا إله إلا هو" not in text, "the model's unmarked spelling is replaced"
    assert text.count("<quran") == 1 and text.endswith("</quran> انتهى.")

    cards = [e.value for e in events if getattr(e, "name", "") == "citation"]
    assert [c["ids"] for c in cards] == ["2:255"]
    assert cards[0]["text"] == "لَا إِلَٰهَ إِلَّا هُو"
    assert cards[0]["match"] == "exact"


def test_run_is_bracketed_by_started_and_finished():
    events = collect(FunctionModel(stream_function=search_then_answer), tools=[search])
    kinds = types(events)
    assert kinds[0] == "RUN_STARTED" and kinds[-1] == "RUN_FINISHED"


def test_a_failing_model_ends_the_stream_with_run_error_not_an_exception():
    async def boom(messages, info):
        raise RuntimeError("provider is down")
        yield  # pragma: no cover

    events = collect(FunctionModel(stream_function=boom))
    assert types(events)[-1] == "RUN_ERROR"
    assert "provider is down" in events[-1].message
