import json
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
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai_harness.compaction import (
    ClearToolResults,
    SummarizingCompaction,
    ReportContextUsage,
)

from db.connection import get_conn
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


def create_active_session(user_id: str, source: str = "telegram") -> dict:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sessions (user_id, source, pydantic_message, is_active)
                VALUES (%s, %s, %s, true)
                RETURNING id, model_provider, model_name, model_variant
                """,
                (user_id, source, json.dumps([])),
            )
            row = cur.fetchone()
    conn.commit()
    return {
        "id": str(row[0]),
        "model_provider": row[1],
        "model_name": row[2],
        "model_variant": row[3],
    }


def resolve_telegram_caller(
    x_bot_secret: str | None,
    x_telegram_id: str | None,
    x_telegram_username: str | None,
    x_telegram_display_name: str | None,
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
        username=x_telegram_username,
        display_name=x_telegram_display_name,
    )
    return user["id"]


async def stream(
    prompt: str,
    session_id: str,
    provider_url: str,
    api_key: str,
    model_name: str,
    variant: str,
):
    provider = OpenAIProvider(base_url=provider_url, api_key=api_key)
    model = OpenAIChatModel(model_name, provider=provider)

    agent = Agent(
        model,
        name="islamic_scholar_agent",
        model_settings={"thinking": str(variant)},
        system_prompt=system_prompt,
        capabilities=capabilities,
        tools=tools,
    )

    history = load_session(session_id)
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
                    save_session_messages(session_id, result.all_messages())
                    yield sse("done", {"output": result.output})


setup_indexes()

system_prompt = ""
with open("./skills/turath-index-skill.md") as file:
    system_prompt = file.read()

compact_tools = ClearToolResults(max_tokens=70_000)
compact_summary = SummarizingCompaction(max_fraction=0.5, keep_messages=30)
context_report = ReportContextUsage(
    on_usage=lambda usage: print(f"{usage.fraction:.0%}")
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
    print(f"🔧 Calling {call.tool_name}({args})")
    return args


@hooks.on.before_model_request
async def log_compaction(ctx, request_context):
    before = len(request_context.messages)
    print(f"📨 {before} messages entering model")
    return request_context


# Fallback config from env, used only if a session has no provider/model set
# (should not normally happen once /start always creates a full session).
DEFAULT_MODEL_NAME = os.environ.get("MODEL", "deepseek-v4-flash-free")
DEFAULT_MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "opencode")
DEFAULT_PROVIDER_URL = os.environ.get(
    "MODEL_PROVIDER_URL", "https://opencode.ai/zen/v1"
)
DEFAULT_VARIANT = os.environ.get("MODEL_VARIANT", "low")
DEFAULT_API_KEY = os.environ.get("API_KEY", "")

# provider -> base_url. Extend as PROVIDERS grows; kept out of the bot.
PROVIDER_BASE_URLS: dict[str, str] = {
    DEFAULT_MODEL_PROVIDER: DEFAULT_PROVIDER_URL,
}


@router.post("")
async def chat_stream(
    req: ChatRequest,
    x_bot_secret: str | None = Header(default=None, alias="X-Bot-Secret"),
    x_telegram_id: str | None = Header(default=None, alias="X-Telegram-Id"),
    x_telegram_username: str | None = Header(default=None, alias="X-Telegram-Username"),
    x_telegram_display_name: str | None = Header(
        default=None, alias="X-Telegram-Display-Name"
    ),
):
    user_id = resolve_telegram_caller(
        x_bot_secret, x_telegram_id, x_telegram_username, x_telegram_display_name
    )

    session = get_active_session_row(user_id)
    if session is None:
        # Should not normally happen (/start always creates one), but keep
        # the endpoint self-healing rather than erroring the user out.
        session = create_active_session(user_id)

    model_provider = session["model_provider"] or DEFAULT_MODEL_PROVIDER
    model_name = session["model_name"] or DEFAULT_MODEL_NAME
    model_variant = session["model_variant"] or DEFAULT_VARIANT

    api_key = get_decrypted_key(user_id, model_provider) or DEFAULT_API_KEY
    if not api_key:
        raise HTTPException(
            status_code=400, detail="No API key on file for this provider"
        )

    provider_url = PROVIDER_BASE_URLS.get(model_provider, DEFAULT_PROVIDER_URL)

    return StreamingResponse(
        stream(
            req.message,
            session_id=session["id"],
            provider_url=provider_url,
            api_key=api_key,
            model_name=model_name,
            variant=model_variant,
        ),
        media_type="text/event-stream",
    )


@router.get("/health")
async def health():
    return {"status": "ok"}


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
