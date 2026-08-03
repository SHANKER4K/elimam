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

system_prompt = """
You are a Salafi Sunni Islamic scholar assistant. You answer Islamic questions using the provided retrieval tools — Quran, hadith, tafsir, and aqeedah books.

Core principles:
- The Salaf (first three generations) are the authoritative reference for understanding Islam.
- You follow their aqeedah and defend it against opposing views.
- You always respond in Arabic unless asked otherwise.
- Never fabricate quotes or knowledge. Use the tools to retrieve verified texts.
- Every claim must have a citation from the tool results.
"""


def load_skill(path: Path) -> Capability | None:
    parts = path.read_text().split("---", 2)
    if len(parts) < 3:
        return None
    _, header, body = parts
    meta = yaml.safe_load(header)
    if not meta or "description" not in meta:
        return None
    return Capability(
        id=meta["id"],
        description=meta["description"],
        instructions=body.strip(),
    )


skills_dir = Path("skills")
skills = [s for p in skills_dir.glob("*.md") if (s := load_skill(p)) is not None]

hooks = Hooks()


@hooks.on.before_tool_execute
async def log_tool_call(ctx, *, call, tool_def, args):
    print(f"🔧 Calling {call.tool_name}({args})")
    return args


agent = Agent(
    model,
    name="islamic_scholar_agent",
    model_settings={"thinking": "high"},
    system_prompt=system_prompt,
    capabilities=[*skills, hooks],
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
