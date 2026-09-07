import asyncio
import json
import logging
import os
from typing import Any
import uuid

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic_ai import (
    Agent,
    CallToolsNode,
    FunctionToolset,
    ModelRequestNode,
    ToolCallPart,
    ToolReturnPart,
    ThinkingPart,
    ModelRequestNode,
    PartStartEvent,
    PartDeltaEvent,
    PartEndEvent,
)
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    UserPromptPart,
    TextPart,
    TextPartDelta,
    ModelMessagesTypeAdapter,
)
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai_harness.compaction import (
    ClearToolResults,
    SummarizingCompaction,
    ReportContextUsage,
)
from pydantic_graph import End

from api.sessions import SessionCreate, create_web_session, get_session
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
from api.providers import resolve_model_config

logger = logging.getLogger("api.chat")
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
    user_id: str
    model_name: str
    model_provider: str
    model_variant: str
    session_id: str


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


def sse(event: str, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


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

tools = FunctionToolset(
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
    ]
)


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


def _project_message(m) -> tuple[str, str | None, dict]:
    kind = type(m).__name__
    role = "assistant" if kind == "ModelResponse" else "user"
    text = "".join(
        p.content for p in m.parts if isinstance(p, (TextPart, UserPromptPart))
    )
    metadata = ModelMessagesTypeAdapter.dump_python([m], mode="json")[0]
    return role, text or None, metadata


def append_session_messages(session_id: str, msgs) -> None:
    if not msgs:
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(MAX(sequence), 0) FROM messages WHERE session_id = %s",
                (session_id,),
            )
            (start,) = cur.fetchone()
            for i, m in enumerate(msgs, start=start + 1):
                role, content, metadata = _project_message(m)
                cur.execute(
                    "INSERT INTO messages (session_id, role, content, metadata, sequence) "
                    "VALUES (%s, %s, %s, %s::jsonb, %s)",
                    (session_id, role, content, json.dumps(metadata), i),
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
):
    provider = OpenAIProvider(base_url=provider_url, api_key=api_key)
    model = OpenAIChatModel(model_name, provider=provider)
    model_settings = OpenAIChatModelSettings(
        temperature=0.5,
        openai_service_tier="flex",
        openai_reasoning_effort=variant,
    )

    agent = Agent(
        model,
        name="islamic_scholar_agent",
        model_settings=model_settings,
        system_prompt=system_prompt,
        capabilities=capabilities,
        toolsets=[tools],
        retries=3,
    )

    history = await asyncio.to_thread(load_session, session_id)
    last_output: Any = None

    try:
        async with agent.iter(prompt, message_history=history) as agent_run:
            try:
                async for node in agent_run:
                    match node:
                        case ModelRequestNode(request=req):
                            for part in req.parts:
                                if isinstance(part, ToolReturnPart):
                                    yield sse(
                                        "tool_result",
                                        {
                                            "tool_name": part.tool_name,
                                            "tool_call_id": part.tool_call_id,
                                            "content": part.content,
                                        },
                                    )
                            async with node.stream(agent_run.ctx) as events:
                                async for event in events:
                                    if isinstance(event, PartStartEvent) and isinstance(
                                        event.part, TextPart
                                    ):
                                        yield sse(
                                            "message_start",
                                            {
                                                "index": event.index,
                                                "text": event.part.content,
                                            },
                                        )
                                    elif isinstance(
                                        event, PartDeltaEvent
                                    ) and isinstance(event.delta, TextPartDelta):
                                        yield sse(
                                            "text_delta",
                                            {
                                                "index": event.index,
                                                "text": event.delta.content_delta,
                                            },
                                        )
                                    elif isinstance(event, PartEndEvent):
                                        yield sse("message_end", {"index": event.index})

                        case CallToolsNode(model_response=resp):
                            for part in resp.parts:
                                if isinstance(part, ToolCallPart):
                                    yield sse(
                                        "tool",
                                        {
                                            "text": part.tool_name,
                                            "tool_call_id": part.tool_call_id,
                                            "args": part.args_as_dict(),
                                        },
                                    )
                                elif isinstance(part, ThinkingPart):
                                    yield sse(
                                        "thinking",
                                        {
                                            "text": part.content,
                                            "id": part.id,
                                            "provider_name": part.provider_name,
                                        },
                                    )

                        case End(data=result):
                            last_output = result.output

            except StopAsyncIteration:
                # الـ stream انتهى بشكل طبيعي - لا ترفع الخطأ
                logger.debug("Stream completed normally")
                pass
            finally:
                # احفظ الـ messages بس إذا الـ agent_run لسه موجود
                try:
                    await asyncio.to_thread(
                        save_session_messages, session_id, agent_run.all_messages()
                    )
                    await asyncio.to_thread(
                        append_session_messages, session_id, agent_run.all_messages()
                    )
                except Exception as e:
                    logger.error(f"Failed to save session: {e}")

    except ModelHTTPError as e:
        message = e.body.get("message") if isinstance(e.body, dict) else str(e)
        yield sse("text_delta", {"text": message})
        raise
    except asyncio.CancelledError:
        logger.info("agent_run_cancelled", extra={"session_id": session_id})
        raise

    yield sse("done", {"output": last_output})


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
def chat_stream(
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
        ),
        media_type="text/event-stream",
    )


@router.post("/web")
async def chat_web_stream(req: WebChatRequest):

    model_provider = req.model_provider
    model_name = req.model_name
    model_variant = req.model_variant

    try:
        uuid.UUID(req.session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="session_id must be a UUID")

    try:
        session = await asyncio.to_thread(get_session, req.session_id)
    except HTTPException:
        session = None

    if session is None:
        await asyncio.to_thread(
            create_web_session,
            req.session_id,
            SessionCreate(
                user_id=req.user_id,
                source="web",
                model_provider=model_provider,
                model_name=model_name,
                model_variant=model_variant,
            ),
        )

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

    api_key = get_decrypted_key(req.user_id, req.model_provider)
    if api_key is None and req.model_provider != "free":
        raise HTTPException(400, "API key not set for this provider")

    return StreamingResponse(
        stream(
            req.message,
            session_id=req.session_id,
            provider_url=model_config["url"],
            api_key=api_key,
            model_name=model_config["model"],
            variant=model_config["variant"],
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
