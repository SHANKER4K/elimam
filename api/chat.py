import json
import logging
import os

from fastapi import APIRouter, Header, HTTPException
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
from pydantic_ai.messages import TextPart, TextPartDelta, ModelMessagesTypeAdapter
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai_harness.compaction import (
    ClearToolResults,
    SummarizingCompaction,
    ReportContextUsage,
)

from db.connection import get_conn

logger = logging.getLogger("api.chat")
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
from api.users import get_or_create_user_by_telegram_id
from api.keys import get_decrypted_key
from api.providers import resolve_model_config

router = APIRouter(prefix="/chat", tags=["Chat"])

# ---------------------------------------------------------------------------
# NOTE on authentication (design rule #5 in the spec):
# The Telegram bot only sends {"message": "..."}. It cannot put telegram_id
# in the body (that would let any client impersonate any user). Instead the
# bot authenticates itself to the backend with a shared bot secret, and
# forwards the caller's Telegram ID in a header that only the trusted bot
# process can set. This keeps API keys and session IDs out of the request
# body entirely, per rules #4/#5.
# ---------------------------------------------------------------------------
BOT_SHARED_SECRET = os.environ.get("BOT_SHARED_SECRET", "")


class WebChatRequest(BaseModel):
    message: str
    model_name: str
    model_provider: str
    model_variant: str
    session_id: str
    api_key: str
    web: bool = False


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


def sse(event: str, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def load_session(session_id: str) -> list:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pydantic_message FROM sessions WHERE id = %s",
                (session_id,),
            )
            row = cur.fetchone()
        if row is None or row[0] is None:
            return []
        return ModelMessagesTypeAdapter.validate_python(row[0])


def save_session_messages(session_id: str, msgs) -> None:
    payload = ModelMessagesTypeAdapter.dump_json(msgs).decode()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET pydantic_message = %s, updated_at = now() WHERE id = %s",
                (payload, session_id),
            )
        conn.commit()


def get_active_session_row(user_id: str) -> dict | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, model_provider, model_name, model_variant
                FROM sessions
                WHERE user_id = %s AND is_active IS TRUE
                """,
                (user_id,),
            )
            row = cur.fetchone()
    if row is None:
        return None
    return {
        "id": str(row[0]),
        "model_provider": row[1],
        "model_name": row[2],
        "model_variant": row[3],
    }


def resolve_telegram_caller(
    x_bot_secret: str | None,
    x_telegram_id: str | None,
) -> str:
    """authenticated Telegram user -> users.id, per the spec's resolution
    chain. Only the bot (which knows BOT_SHARED_SECRET) may set the
    X-Telegram-Id header; arbitrary clients cannot impersonate a user."""
    if not BOT_SHARED_SECRET or x_bot_secret != BOT_SHARED_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not x_telegram_id:
        raise HTTPException(status_code=400, detail="Missing Telegram identity")

    user = get_or_create_user_by_telegram_id(
        telegram_id=x_telegram_id,
        username=None,
        display_name=None,
    )
    return user["id"]


async def stream(
    prompt: str,
    session_id: str,
    provider_url: str,
    api_key: str,
    model_name: str,
    variant: str,
    web: bool,
):
    global sessions
    provider = OpenAIProvider(base_url=provider_url, api_key=api_key)
    model = OpenAIChatModel(model_name, provider=provider)
    model_settings = OpenAIResponsesModelSettings(
        temperature=0.5, service_tier="flex", thinking=str(variant)
    )

    agent = Agent(
        model,
        name="islamic_scholar_agent",
        model_settings=model_settings,
        system_prompt=system_prompt,
        capabilities=capabilities,
        tools=tools,
    )

    history = load_session(session_id) if not web else sessions.get(session_id, [])
    async with agent.run_stream_events(prompt, message_history=history) as events:
        async for event in events:
            match event:
                case PartStartEvent(part=TextPart() as part):
                    yield sse(
                        "message_start", {"index": event.index, "text": part.content}
                    )
                case PartDeltaEvent(delta=TextPartDelta() as delta):
                    yield sse("text_delta", {"text": delta.content_delta})

                case FunctionToolCallEvent(part=ToolCallPart() as part):
                    yield sse(
                        "tool",
                        {
                            "text": part.tool_name,  # tool name
                            "tool_call_id": part.tool_call_id,  # id to match results
                            "args": part.args_as_dict(),  # the parsed arguments (dict)
                        },
                    )

                case FunctionToolResultEvent(part=ToolReturnPart() as part):
                    yield sse(
                        "tool_result",
                        {"text": part.content, "tool_call_id": part.tool_call_id},
                    )

                case PartEndEvent(part=TextPart()):
                    yield sse("message_end", {})

                case AgentRunResultEvent(result=result):
                    if not web:
                        save_session_messages(session_id, result.all_messages())
                    else:
                        sessions[session_id] = result.all_messages()
                    yield sse("done", {"output": result.output})


setup_indexes()

system_prompt = ""
with open("./skills/turath-index-skill.md") as file:
    system_prompt = file.read()

compact_tools = ClearToolResults(max_tokens=70_000)
compact_summary = SummarizingCompaction(max_fraction=0.5, keep_messages=30)
context_report = ReportContextUsage(
    on_usage=lambda usage: logger.info(
        "context_usage", extra={"fraction": round(usage.fraction * 100)}
    )
)

hooks = Hooks()
capabilities = [hooks, compact_tools, compact_summary, context_report]
tools = [
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
]


@hooks.on.before_tool_execute
async def log_tool_call(ctx, *, call, tool_def, args):
    logger.info(
        "tool_call",
        extra={"tool": call.tool_name, "tool_call_id": call.tool_call_id},
    )
    return args


@hooks.on.before_model_request
async def log_compaction(ctx, request_context):
    before = len(request_context.messages)
    logger.debug("messages_entering_model", extra={"n_messages": before})
    return request_context


@router.post("")
async def chat_stream(
    req: ChatRequest,
    x_bot_secret: str | None = Header(default=None, alias="X-Bot-Secret"),
    x_telegram_id: str | None = Header(default=None, alias="X-Telegram-Id"),
):
    user_id = resolve_telegram_caller(x_bot_secret, x_telegram_id)

    session = get_active_session_row(user_id)
    if session is None:
        raise HTTPException(
            status_code=409,
            detail="User has no active session",
        )

    model_provider = session["model_provider"]
    model_name = session["model_name"]
    model_variant = session["model_variant"]

    if not model_provider or not model_name or not model_variant:
        raise HTTPException(
            status_code=409,
            detail="Active session is missing model configuration",
        )

    try:
        model_config = resolve_model_config(
            provider=model_provider,
            model=model_name,
            variant=model_variant,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    api_key = get_decrypted_key(user_id, model_provider)
    if not api_key:
        raise HTTPException(
            status_code=400, detail="No API key on file for this provider"
        )

    return StreamingResponse(
        stream(
            req.message,
            session_id=session["id"],
            provider_url=model_config["url"],
            api_key=api_key,
            model_name=model_config["model"],
            variant=model_config["variant"],
            web=False,
        ),
        media_type="text/event-stream",
    )


sessions = {}


@router.post("/web")
async def chat_web_stream(req: WebChatRequest):

    model_provider = req.model_provider
    model_name = req.model_name
    model_variant = req.model_variant

    if not model_provider or not model_name or not model_variant:
        raise HTTPException(
            status_code=409,
            detail="Active session is missing model configuration",
        )

    try:
        model_config = resolve_model_config(
            provider=model_provider,
            model=model_name,
            variant=model_variant,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    api_key = req.api_key
    if not api_key:
        raise HTTPException(
            status_code=400, detail="No API key on file for this provider"
        )

    return StreamingResponse(
        stream(
            req.message,
            session_id=req.session_id,
            provider_url=model_config["url"],
            api_key=api_key,
            model_name=model_config["model"],
            variant=model_config["variant"],
            web=req.web,
        ),
        media_type="text/event-stream",
    )


@router.post("/dense_search")
def dense_search_(
    collection: str, query_text: str, top_k: int = 10, filters: dict | None = None
) -> list:
    return dense_search(collection, query_text, top_k, filters)


@router.post("/sparse_search")
def sparse_search_(
    collection: str, query_text: str, top_k: int = 10, filters: dict | None = None
) -> list:
    return sparse_search(collection, query_text, top_k, filters)


@router.post("/hybrid_search")
def hybrid_search_(
    collection: str,
    query_text: str,
    top_k: int = 10,
    pool: int = 50,
    filters: dict | None = None,
) -> list:
    return hybrid_search(collection, query_text, top_k, pool, filters)


@router.post("/hybrid_search_weighted")
def hybrid_search_weighted_(
    collection: str,
    query_text: str,
    top_k: int = 10,
    pool: int = 50,
    weights: tuple[float, float] = (0.7, 0.3),
    filters: dict | None = None,
) -> list:
    return hybrid_search_weighted(collection, query_text, top_k, pool, weights, filters)


@router.get("/health")
async def health():
    return {"status": "ok"}
