import asyncio
import json
from contextlib import aclosing
import logging
import os
import time
from typing import Any, get_args
import uuid

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
import logfire
from pydantic import BaseModel
from pydantic_ai import (
    Agent,
    CallToolsNode,
    FunctionToolset,
    ModelRequestNode,
    ModelSettings,
    ToolCallPart,
    ToolReturnPart,
    ThinkingPart,
    PartStartEvent,
    PartDeltaEvent,
    PartEndEvent,
)
from pydantic_ai.exceptions import AgentRunError, ModelHTTPError
from pydantic_ai.settings import ThinkingEffort, ThinkingLevel
from pydantic_ai.messages import (
    ModelResponse,
    UserPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPartDelta,
    ModelMessagesTypeAdapter,
)
from openai import AsyncOpenAI
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.google import GoogleProvider

from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

from pydantic_ai.models.xai import XaiModel
from pydantic_ai.providers.xai import XaiProvider

from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.providers.mistral import MistralProvider

from pydantic_ai_harness.compaction import (
    ClearToolResults,
    SummarizingCompaction,
    ReportContextUsage,
)
from pydantic_graph import End

from api.sessions import SessionCreate, create_web_session, get_session
from db.connection import get_conn

from retrieval import (
    find_book,
    find_category,
    get_book,
    get_books_books,
    get_books_categories,
    get_books_sunnah,
    get_chunk,
    get_hadith,
    get_quran,
    get_tafsir,
    retrieval,
    search,
)
from api.tool_presenter import describe_call, describe_result
from retrieval.citations import CitationStream, render_citations
from api.users import get_or_create_user_by_telegram_id
from identity import Unauthorized, current_user_id, resolve_identity
from api.providers import resolve_model_config
from provider_url import opencode_session_headers

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

retrieval.setup_indexes()


class WebChatRequest(BaseModel):
    message: str
    model_name: str
    model_provider: str
    model_variant: str
    session_id: str
    # The caller's connection to `model_provider`. Builtin providers can also be
    # resolved by slug, but a custom connection has no slug, so a brand-new
    # web session must name it explicitly.
    user_provider_id: str | None = None


class ChatRequest(BaseModel):
    message: str


def sse(event: str, data):
    # default=str: a tool result holding a datetime/set/... must not kill the stream.
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


system_prompt = ""
with open("./skills/turath-index-skill.md") as file:
    system_prompt = file.read()

# ponytail: trigger on a fraction of the model's real context window, not an
# absolute 70k that a 64k model never reaches. Unknown/custom models fall back to
# 32k (compacting early costs a summary; overflowing loses the request).
compact_tools = ClearToolResults(max_fraction=0.35, fallback_context_window=32_000)
compact_summary = SummarizingCompaction(
    max_fraction=0.5, keep_messages=20, fallback_context_window=32_000
)
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
        get_books_books,
        get_books_sunnah,
        get_books_categories,
        get_chunk,
        find_book,
        find_category,
        search,
    ],
    # defer_loading=True,
)


_THINKING_EFFORTS = set(get_args(ThinkingEffort))
_THINKING_OFF = {"none", "off", "false"}


def _thinking_level(variant: str | None) -> ThinkingLevel | None:
    """A stored variant -> pydantic-ai's unified `thinking` value.

    None means "send nothing, keep the model's default". Catalog variants
    already use the unified vocabulary (low/high/max...); any other label
    means "thinking on at the provider's default effort".
    """
    if not variant:
        return None
    v = variant.strip().lower()
    if v in _THINKING_OFF:
        return False
    if v in _THINKING_EFFORTS:
        return v  # type: ignore[return-value]
    return True


def _build_settings(variant: str | None, **extra: Any) -> ModelSettings | None:
    """Model settings for a variant, via the unified `thinking` setting.

    Each adapter translates `thinking` for its model: adaptive vs budget
    thinking on Anthropic, `reasoning_effort` on OpenAI, `thinking_config` on
    Google, and so on. Models whose profile has no thinking support ignore it.
    """
    settings = ModelSettings(**extra)
    level = _thinking_level(variant)
    if level is not None:
        settings["thinking"] = level
    return settings or None


def _headers(*layers: dict[str, str] | None) -> dict[str, Any]:
    """`extra_headers=` kwargs merging the given header dicts (later wins)."""
    merged: dict[str, str] = {}
    for layer in layers:
        merged.update(layer or {})
    return {"extra_headers": merged} if merged else {}


def _anthropic_base_url(url: str) -> str:
    """Connections store the URL their `/models` lives under (`.../v1`, like
    the catalog's https://api.anthropic.com/v1), but the Anthropic SDK appends
    `/v1/messages` itself."""
    url = url.rstrip("/")
    return url[: -len("/v1")] if url.endswith("/v1") else url


def _openai_provider(
    base_url: str | None, api_key: str | None, headers: dict[str, str]
) -> OpenAIProvider:
    """An OpenAI-compatible provider whose client sends `headers` on every request.

    `extra_headers` in `ModelSettings` only reach the run's own requests:
    SummarizingCompaction writes its summary with a nested Agent built from
    `ctx.model` and its own settings, so the OpenCode session id has to ride on
    the client -- otherwise the summary request comes back "Request is missing
    x-opencode-session".
    """
    if not headers:
        return OpenAIProvider(base_url=base_url, api_key=api_key)
    return OpenAIProvider(
        openai_client=AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or os.environ.get("OPENAI_API_KEY") or "api-key-not-set",
            default_headers=headers,
        )
    )


def _build_agent(
    provider_slug: str,
    model_name: str,
    provider_url: str,
    api_key: str | None,
    variant: str,
    session_id: str | None = None,
    api_style: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> Agent:
    """Build a PydanticAI Agent for OpenAI, Anthropic, OpenRouter, DeepSeek, or other compatible providers."""

    builders = {
        "openai": lambda: OpenAIChatModel(
            model_name, provider=OpenAIProvider(api_key=api_key)
        ),
        "anthropic": lambda: AnthropicModel(
            model_name, provider=AnthropicProvider(api_key=api_key)
        ),
        "groq": lambda: GroqModel(model_name, provider=GroqProvider(api_key=api_key)),
        "mistral": lambda: MistralModel(
            model_name, provider=MistralProvider(api_key=api_key)
        ),
        "deepseek": lambda: OpenAIChatModel(
            model_name, provider=DeepSeekProvider(api_key=api_key)
        ),
        "openrouter": lambda: OpenRouterModel(
            model_name, provider=OpenRouterProvider(api_key=api_key)
        ),
        "google": lambda: GoogleModel(
            model_name, provider=GoogleProvider(api_key=api_key)
        ),
        "xai": lambda: XaiModel(model_name, provider=XaiProvider(api_key=api_key)),
    }
    if provider_slug in builders:
        model = builders[provider_slug]()
        model_settings = _build_settings(variant, **_headers(extra_headers))
    elif provider_slug == "free":
        # The "free" provider is a special case: it doesn't require an API key,
        # and the user must select a model from the available options.
        free_url = os.environ.get("FREE_PROVIDER_URL")
        provider = _openai_provider(
            free_url,
            os.environ.get("FREE_PROVIDER_API_KEY"),
            opencode_session_headers(free_url, session_id),
        )
        model = OpenAIChatModel(model_name, provider=provider)
        model_settings = _build_settings(variant, temperature=0.5)
    else:
        # Custom connections and catalog providers without a dedicated
        # adapter: the connection's api_style says which wire protocol it speaks.
        if api_style == "anthropic_compatible":
            model = AnthropicModel(
                model_name,
                provider=AnthropicProvider(
                    api_key=api_key, base_url=_anthropic_base_url(provider_url)
                ),
            )
        else:
            provider = _openai_provider(
                provider_url, api_key, opencode_session_headers(provider_url, session_id)
            )
            model = OpenAIChatModel(model_name, provider=provider)

        # OpenCode Zen/Go refuse to route a request with no session id. An
        # OpenAI connection gets it from the client built above; an
        # anthropic-compatible one only from here.
        model_settings = _build_settings(
            variant,
            temperature=0.5,
            **_headers(extra_headers, opencode_session_headers(provider_url, session_id)),
        )

    return Agent(
        model,
        name="islamic_scholar_agent",
        model_settings=model_settings,
        system_prompt=system_prompt,
        capabilities=capabilities,
        toolsets=[tools],
        retries=3,
    )


class SessionConflict(Exception):
    """The session's history changed while this turn was running."""


def load_session_state(session_id: str) -> tuple[list, str | None]:
    """(history, digest). The digest identifies the stored history so the save
    at the end of the turn can tell whether someone else wrote in between."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pydantic_message, md5(pydantic_message::text) "
                "FROM sessions WHERE id = %s",
                (session_id,),
            )
            row = cur.fetchone()
    if row is None:
        return [], None
    history = ModelMessagesTypeAdapter.validate_python(row[0]) if row[0] else []
    return history, row[1]


def load_session(session_id: str) -> list:
    return load_session_state(session_id)[0]


def save_session_messages(session_id: str, msgs) -> None:
    payload = ModelMessagesTypeAdapter.dump_json(msgs).decode()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET pydantic_message = %s, updated_at = now() WHERE id = %s",
                (payload, session_id),
            )
        conn.commit()


def persist_turn(session_id: str, all_msgs, new_msgs, expected_digest: str | None) -> None:
    """Save the history and append this turn's chat-log rows, atomically, and
    only if the stored history is still the one the turn started from.

    Raises SessionConflict (and writes nothing) if another turn -- another
    worker, the bot, a second tab -- saved this session in the meantime.
    Compare-and-swap rather than a lock: a lock would pin a pooled connection
    for the whole (minutes-long) model run.
    """
    payload = ModelMessagesTypeAdapter.dump_json(all_msgs).decode()
    with get_conn() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE sessions SET pydantic_message = %s, updated_at = now() "
                    "WHERE id = %s AND md5(pydantic_message::text) IS NOT DISTINCT FROM %s",
                    (payload, session_id, expected_digest),
                )
                if cur.rowcount == 0:
                    raise SessionConflict(session_id)
                _insert_chat_rows(cur, session_id, new_msgs)
            conn.commit()
        except BaseException:
            conn.rollback()
            raise


def _project_message(m) -> tuple[str, str] | None:
    """The chat-log row for one pydantic-ai message, or None if it has none.

    Only the user's own prompt and the model's visible answer are logged:
    tool returns, system prompts, tool calls and thinking stay in
    `sessions.pydantic_message`. Multimodal prompts keep their text items only.
    """
    if isinstance(m, ModelResponse):
        role = "assistant"
        texts = [p.content for p in m.parts if isinstance(p, TextPart)]
    else:
        role = "user"
        texts = []
        for p in m.parts:
            if not isinstance(p, UserPromptPart):
                continue
            items = [p.content] if isinstance(p.content, str) else p.content
            texts.extend(item for item in items if isinstance(item, str))
    text = "".join(texts)
    if not text:
        return None

    # Persist the canonical citation tags, so cards re-render after a reload
    # without persisting tool results. Guarded by "<" to avoid Qdrant lookups
    # on the (usually tag-free) messages.
    if role == "assistant" and "<" in text:
        text = render_citations(text)

    return role, text


def _insert_chat_rows(cur, session_id: str, msgs) -> None:
    # Each INSERT's subquery sees the rows inserted before it in this
    # transaction, so +1 is the next free sequence every time.
    for role, content in filter(None, map(_project_message, msgs)):
        cur.execute(
            """
            INSERT INTO messages (session_id, role, content, sequence)
            VALUES (
                %s, %s, %s,
                (SELECT COALESCE(MAX(sequence), 0) + 1
                 FROM messages WHERE session_id = %s)
            )
            """,
            (session_id, role, content, session_id),
        )


def append_session_messages(session_id: str, msgs) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            _insert_chat_rows(cur, session_id, msgs)
        conn.commit()


def get_active_session_row(user_id: str) -> dict | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, model_provider, model_name, model_variant, user_provider_id
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
        "user_provider_id": str(row[4]) if row[4] else None,
    }


def resolve_telegram_caller(
    x_bot_secret: str | None,
    x_telegram_id: str | None,
) -> str:
    """authenticated Telegram user -> users.id. Only the bot (which knows
    BOT_SHARED_SECRET) may set the X-Telegram-Id header; arbitrary clients
    cannot impersonate a user. Delegates to the shared resolver so there is
    exactly one implementation of the bot branch."""
    try:
        identity = resolve_identity(
            {
                "X-Bot-Secret": x_bot_secret or "",
                "X-Telegram-Id": x_telegram_id or "",
            },
            now=time.time(),
            user_secret="",
            bot_secret=BOT_SHARED_SECRET,
            lookup_telegram_user=lambda telegram_id: get_or_create_user_by_telegram_id(
                telegram_id=telegram_id
            )["id"],
        )
    except Unauthorized as exc:
        raise HTTPException(status_code=401, detail="Unauthorized") from exc
    return identity.user_id


# Sessions with a turn in flight in this process. Async generators run on the
# event loop, so a plain set is race-free here; persist_turn's compare-and-swap
# covers other processes.
_active_turns: set[str] = set()

GENERIC_ERROR = "حدث خطأ أثناء التوليد"


def _error_payload(exc: BaseException) -> dict[str, Any]:
    """What the client may see about a failed run. Errors from the user's own
    model provider are passed on; anything else is logged, not shown."""
    if isinstance(exc, SessionConflict):
        return {
            "code": "session_conflict",
            "message": "تم تحديث هذه المحادثة من مكان آخر؛ لم يُحفظ هذا الرد.",
        }
    if isinstance(exc, ModelHTTPError):
        body = exc.body
        message = body.get("message") if isinstance(body, dict) else None
        return {
            "code": "model_http_error",
            "status": exc.status_code,
            "message": message or str(exc),
        }
    if isinstance(exc, AgentRunError):
        return {"code": type(exc).__name__, "message": str(exc)}
    return {"code": "internal_error", "message": GENERIC_ERROR}


async def stream(
    prompt: str,
    session_id: str,
    provider_slug: str,
    provider_url: str,
    api_key: str | None,
    model_name: str,
    variant: str,
    api_style: str | None = None,
    extra_headers: dict[str, str] | None = None,
):
    """SSE for one turn. Always ends with `done`, preceded by `error` if the
    turn failed -- once the response has started, raising would only drop the
    connection and leave the client guessing."""
    if session_id in _active_turns:
        yield sse(
            "error",
            {"code": "session_busy", "message": "هناك رد قيد التوليد في هذه المحادثة."},
        )
        yield sse("done", {"output": None})
        return

    _active_turns.add(session_id)
    outcome: dict[str, Any] = {"output": None}
    try:
        # aclosing: when the client disconnects, Starlette aclose()s this
        # generator; the inner run must be closed right here (persisting the
        # partial turn) rather than whenever the GC finalises it.
        turn = _run_turn(
            outcome,
            prompt,
            session_id=session_id,
            provider_slug=provider_slug,
            provider_url=provider_url,
            api_key=api_key,
            model_name=model_name,
            variant=variant,
            api_style=api_style,
            extra_headers=extra_headers,
        )
        async with aclosing(turn):
            async for chunk in turn:
                yield chunk
    except asyncio.CancelledError:
        logger.info("agent_run_cancelled", extra={"session_id": session_id})
        raise
    except Exception as exc:
        if isinstance(exc, SessionConflict):
            logger.warning("session_conflict", extra={"session_id": session_id})
        else:
            logger.exception("agent_run_failed", extra={"session_id": session_id})
        yield sse("error", _error_payload(exc))
        outcome["output"] = None
    finally:
        _active_turns.discard(session_id)

    yield sse("done", {"output": outcome["output"]})


async def _run_turn(
    outcome: dict,
    prompt: str,
    session_id: str,
    provider_slug: str,
    provider_url: str,
    api_key: str | None,
    model_name: str,
    variant: str,
    api_style: str | None = None,
    extra_headers: dict[str, str] | None = None,
):
    agent = _build_agent(
        provider_slug=provider_slug,
        model_name=model_name,
        provider_url=provider_url,
        api_key=api_key,
        variant=variant,
        session_id=session_id,
        api_style=api_style,
        extra_headers=extra_headers,
    )

    history, digest = await asyncio.to_thread(load_session_state, session_id)
    citations = CitationStream()

    async with agent.iter(prompt, message_history=history) as agent_run:
        # Whatever the run produced is persisted, even if it stopped early: the
        # full history for the next turn, and only this turn's messages for the
        # chat log (all_messages() includes the history passed in).
        try:
            async for node in agent_run:
                match node:
                    case ModelRequestNode(request=req):
                        for part in req.parts:
                            if isinstance(part, ToolReturnPart):
                                started = _TOOL_STARTS.pop(part.tool_call_id, None)
                                duration_ms = (
                                    round((time.monotonic() - started) * 1000)
                                    if started is not None
                                    else None
                                )
                                yield sse(
                                    "tool_result",
                                    {
                                        "tool_name": part.tool_name,
                                        "tool_call_id": part.tool_call_id,
                                        "content": part.content,
                                        **describe_result(
                                            part.tool_name,
                                            part.content,
                                            duration_ms,
                                        ),
                                    },
                                )
                        async with node.stream(agent_run.ctx) as events:
                            async for event in events:
                                if isinstance(event, PartStartEvent):
                                    if isinstance(event.part, TextPart):
                                        text, cards = citations.feed(
                                            event.part.content
                                        )
                                        yield sse(
                                            "message_start",
                                            {
                                                "index": event.index,
                                                "text": text,
                                            },
                                        )
                                        for card in cards:
                                            yield sse("citation", card)
                                    elif isinstance(event.part, ThinkingPart):
                                        yield sse(
                                            "thinking_start",
                                            {
                                                "index": event.index,
                                                "id": event.part.id,
                                                "provider_name": event.part.provider_name,
                                            },
                                        )
                                elif isinstance(event, PartDeltaEvent):
                                    if isinstance(event.delta, TextPartDelta):
                                        text, cards = citations.feed(
                                            event.delta.content_delta or ""
                                        )
                                        if text:
                                            yield sse(
                                                "text_delta",
                                                {
                                                    "index": event.index,
                                                    "text": text,
                                                },
                                            )
                                        for card in cards:
                                            yield sse("citation", card)
                                    elif isinstance(event.delta, ThinkingPartDelta):
                                        yield sse(
                                            "thinking_delta",
                                            {
                                                "index": event.index,
                                                "text": event.delta.content_delta,
                                                "provider_name": event.delta.provider_name,
                                            },
                                        )
                                elif isinstance(event, PartEndEvent):
                                    # index + kind let a client close the block
                                    # it opened, whether text or thinking.
                                    if isinstance(event.part, ThinkingPart):
                                        yield sse(
                                            "thinking_end", {"index": event.index}
                                        )
                                    else:
                                        # Anything an unterminated tag was
                                        # holding back is model output: emit it.
                                        tail = citations.flush()
                                        if tail:
                                            yield sse(
                                                "text_delta",
                                                {"index": event.index, "text": tail},
                                            )
                                        yield sse(
                                            "message_end", {"index": event.index}
                                        )

                    case CallToolsNode(model_response=resp):
                        for part in resp.parts:
                            if isinstance(part, ToolCallPart):
                                args = part.args_as_dict()
                                yield sse(
                                    "tool",
                                    {
                                        "text": part.tool_name,
                                        "tool_name": part.tool_name,
                                        "tool_call_id": part.tool_call_id,
                                        "args": args,
                                        "status": "running",
                                        **describe_call(part.tool_name, args),
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
                        outcome["output"] = result.output

        except BaseException:
            try:
                await asyncio.to_thread(
                    persist_turn,
                    session_id,
                    agent_run.all_messages(),
                    agent_run.new_messages(),
                    digest,
                )
            except Exception:
                # Never let a failed save hide why the run failed.
                logger.exception("session_save_failed", extra={"session_id": session_id})
            raise
        else:
            # The stream carried canonical tags; `output` must agree with it, or
            # a client that renders on `done` would show the model's raw markup.
            output = outcome["output"]
            if isinstance(output, str) and "<" in output:
                outcome["output"] = await asyncio.to_thread(render_citations, output)
            await asyncio.to_thread(
                persist_turn,
                session_id,
                agent_run.all_messages(),
                agent_run.new_messages(),
                digest,
            )


# Start times keyed by tool_call_id, so tool_result can report a duration.
# ponytail: a tool that raises and exhausts retries never produces a
# ToolReturnPart, so its entry is never popped — bounded by the tool calls in one
# request; switch to an after_tool_execute hook if that ever grows.
_TOOL_STARTS: dict[str, float] = {}


@hooks.on.before_tool_execute
async def log_tool_call(ctx, *, call, tool_def, args):
    _TOOL_STARTS[call.tool_call_id] = time.monotonic()
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
            session["user_provider_id"],
            model_provider,
            model_name,
            model_variant,
            user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if model_config["api_key"] is None and model_config["requires_key"]:
        raise HTTPException(
            status_code=400, detail="No API key on file for this provider"
        )

    return StreamingResponse(
        stream(
            req.message,
            session_id=session["id"],
            provider_slug=model_provider,
            provider_url=model_config["url"],
            api_key=model_config["api_key"],
            model_name=model_config["model"],
            variant=model_config["variant"],
            api_style=model_config["api_style"],
            extra_headers=model_config["extra_headers"],
        ),
        media_type="text/event-stream",
    )


@router.post("/web")
async def chat_web_stream(req: WebChatRequest, request: Request):

    user_id = current_user_id(request)
    model_provider = req.model_provider
    model_name = req.model_name
    model_variant = req.model_variant

    try:
        uuid.UUID(req.session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="session_id must be a UUID")

    try:
        session = await asyncio.to_thread(get_session, req.session_id)
    except HTTPException as exc:
        # Only "not found" means "start a new session". A DB outage must not
        # fall through to create_web_session and collide on the primary key.
        if exc.status_code != 404:
            raise
        session = None

    if session is not None and str(session["user_id"]) != user_id:
        # Do not leak whether the session exists for someone else.
        raise HTTPException(status_code=404, detail="Session not found")

    if not model_provider or not model_name or not model_variant:
        raise HTTPException(
            status_code=409,
            detail="Active session is missing model configuration",
        )

    # The connection is resolved before the insert so a brand-new session is
    # pinned to the connection that answered this request.
    try:
        model_config = resolve_model_config(
            session["user_provider_id"] if session else req.user_provider_id,
            model_provider,
            model_name,
            model_variant,
            user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if model_config["api_key"] is None and model_config["requires_key"]:
        raise HTTPException(400, "API key not set for this provider")

    if session is None:
        await asyncio.to_thread(
            create_web_session,
            req.session_id,
            SessionCreate(
                user_id=user_id,
                source="web",
                model_provider=model_provider,
                model_name=model_name,
                model_variant=model_variant,
                user_provider_id=model_config["user_provider_id"],
            ),
        )

    return StreamingResponse(
        stream(
            req.message,
            session_id=req.session_id,
            provider_slug=model_provider,
            provider_url=model_config["url"],
            api_key=model_config["api_key"],
            model_name=model_config["model"],
            variant=model_config["variant"],
            api_style=model_config["api_style"],
            extra_headers=model_config["extra_headers"],
        ),
        media_type="text/event-stream",
    )


@router.get("/health")
async def health():
    return {"status": "ok"}
