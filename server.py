"""
FastAPI server for the Islamic scholar agent.
Run: uvicorn backend.server:app --reload
"""

from pathlib import Path


import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartStartEvent,
    PartDeltaEvent,
    PartEndEvent,
    AgentRunResultEvent,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.messages import TextPart, TextPartDelta
import json

from pydantic_ai.capabilities import Capability
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai_harness.compaction import (
    ClearToolResults,
    SummarizingCompaction,
    ReportContextUsage,
)

from search import (
    get_quran,
    get_hadith,
    get_tafsir,
    get_book,
    get_books_hadith,
    get_books_tafsir,
    get_books_books,
    setup_indexes,
    dense_search,
    sparse_search,
    hybrid_search,
    hybrid_search_weighted,
)

setup_indexes()

# ── Agent setup ──────────────────────────────────────────────────────────────

provider = OpenAIProvider(
    base_url="https://opencode.ai/zen/v1",
    api_key="sk-KEzLkDC9IkYiDlRRr4KjX0tvoaUsQDNw2gg0b88PgUJTVemSFGGNSOpc9ABZWNqO",
)
model = OpenAIChatModel("deepseek-v4-flash-free", provider=provider)

system_prompt = ""

with open("./skills/turath-index-skill.md") as file:
    system_prompt = file.read()

hooks = Hooks()

compact_tools = ClearToolResults(max_tokens=70_000)
compact_summary = SummarizingCompaction(max_fraction=0.5, keep_messages=30)
context_report = ReportContextUsage(
    on_usage=lambda usage: print(f"{usage.fraction:.0%}")
)


@hooks.on.before_tool_execute
async def log_tool_call(ctx, *, call, tool_def, args):
    print(f"🔧 Calling {call.tool_name}({args})")
    return args


@hooks.on.before_model_request
async def log_compaction(ctx, request_context):
    before = len(request_context.messages)
    print(f"📨 {before} messages entering model")
    return request_context


agent = Agent(
    model,
    name="islamic_scholar_agent",
    model_settings={"thinking": "high"},
    system_prompt=system_prompt,
    capabilities=[hooks, compact_tools, compact_summary, context_report],
    tools=[
        get_quran,
        get_hadith,
        get_tafsir,
        get_book,
        get_books_hadith,
        get_books_tafsir,
        get_books_books,
        dense_search,
        sparse_search,
        hybrid_search,
        hybrid_search_weighted,
    ],
)

# ── FastAPI app ──────────────────────────────────────────────────────────────
messages: dict = {}

app = FastAPI(title="Islamic Scholar API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ponytail: lock to your Next.js origin in prod
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str


class ChatResponse(BaseModel):
    response: str
    # ponytail: add tool_calls list here if the UI needs to show them


# @app.post("/chat")
# async def chat(req: ChatRequest) -> ChatResponse:
#     global messages
#     if not messages.get(req.session_id):
#         messages[req.session_id] = []
#
#     result = await agent.run(req.message, message_history=messages[req.session_id])
#     messages[req.session_id] = [*messages[req.session_id], *result.all_messages()]
#     return ChatResponse(response=result.output)


def sse(event: str, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream(prompt: str, session_id):
    global messages
    async with agent.run_stream_events(
        prompt, message_history=messages[session_id]
    ) as events:
        async for event in events:
            match event:
                case PartStartEvent(part=TextPart() as part):
                    yield sse(
                        "message_start",
                        {
                            "index": event.index,
                            "text": part.content,
                        },
                    )

                case PartDeltaEvent(delta=TextPartDelta() as delta):
                    yield sse(
                        "text_delta",
                        {
                            "text": delta.content_delta,
                        },
                    )

                case FunctionToolCallEvent(part=ToolCallPart() as part):
                    yield sse(
                        "tool",
                        {"text": part.tool_name, "tool_call_id": part.tool_call_id},
                    )

                case FunctionToolResultEvent(part=ToolReturnPart() as part):
                    yield sse(
                        "tool_result",
                        {"text": part.content, "tool_call_id": part.tool_call_id},
                    )

                case PartEndEvent(part=TextPart()):
                    yield sse("message_end", {})

                case AgentRunResultEvent(result=result):
                    messages[session_id] = [
                        *messages[session_id],
                        *result.all_messages(),
                    ]
                    yield sse(
                        "done",
                        {
                            "output": result.output,
                        },
                    )


@app.post("/chat")
async def chat_stream(req: ChatRequest):
    global messages
    if not messages.get(req.session_id):
        messages[req.session_id] = []

    prompt = req.message
    return StreamingResponse(
        stream(prompt, session_id=req.session_id),
        media_type="text/event-stream",
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/dense_search")
def dense_search_(
    collection: str, query_text: str, top_k: int = 10, filters: dict | None = None
) -> list:
    return dense_search(collection, query_text, top_k, filters)


@app.post("/sparse_search")
def sparse_search_(
    collection: str, query_text: str, top_k: int = 10, filters: dict | None = None
) -> list:
    return sparse_search(collection, query_text, top_k, filters)


@app.post("/hybrid_search")
def hybrid_search_(
    collection: str,
    query_text: str,
    top_k: int = 10,
    pool: int = 50,
    filters: dict | None = None,
) -> list:
    return hybrid_search(collection, query_text, top_k, pool, filters)


# @app.post('/compact')
# def compact(session_id:str):
#     if not messages['session_id']:
#         return {"status":'failed','message':'The chat is empty'}
#
#     await compac
