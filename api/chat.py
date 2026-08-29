import asyncio
import json
import logging
import os
import re
from typing import Literal

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, HttpUrl, model_validator
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
from pydantic_ai.messages import (
    TextPart,
    TextPartDelta,
    ModelMessagesTypeAdapter,
    ModelMessage,
)
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai_harness.compaction import (
    ClearToolResults,
    SummarizingCompaction,
    ReportContextUsage,
)

from db.connection import get_conn

from search import (
    get_quran,
    get_tafsir,
    get_books_tafsir,
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
    model_name: str
    model_provider: str
    model_variant: str
    session_id: str
    api_key: str


class ChatRequest(BaseModel):
    message: str


class TelegramChatResponse(BaseModel):
    response: str


CITATION_PATTERN = re.compile(r"\[(\d{1,3}):(\d{1,3})\]")

# Inline marker the LLM emits: {exact_arabic_fragment|surah:ayah}. The server
# replaces it with [surah:ayah] and validates the fragment against the record.
RAW_CITATION_PATTERN = re.compile(
    r"\{(?P<fragment>[^{}|]+)\|(?P<surah>\d{1,3}):(?P<ayah>\d{1,3})\}"
)

# Module-level tashkīl-fallback counter for production observability.
_fragment_fallback_count = 0

TAFSIR_BOOK_TO_SLUG = {
    "saadi": "ar-tafseer-al-saddi",
    "katheer": "ar-tafsir-ibn-kathir",
    "moyassar": "ar-tafsir-muyassar",
    "tabary": "ar-tafsir-al-tabari",
    "baghawy": "ar-tafsir-al-baghawi",
}

TAFSIR_BOOK_TO_NAME = {
    "saadi": "السعدي",
    "katheer": "ابن كثير",
    "moyassar": "الميسر",
    "tabary": "الطبري",
    "baghawy": "البغوي",
}

RETRIEVAL_TOOLS = {
    "get_quran",
    "get_tafsir",
    "dense_search",
    "sparse_search",
    "hybrid_search",
    "hybrid_search_weighted",
}


class DetectedCitation(BaseModel):
    """
    One Quran citation occurrence in the final API response.

    Fully server-constructed; the LLM never authors these fields. `order`
    follows the left-to-right occurrence order in `raw_response_text`.
    """

    order: int = Field(ge=1)

    surah_number: int = Field(ge=1, le=114)
    surah_name: str = Field(min_length=1)
    ayah_number: int = Field(ge=1)

    citation: str = Field(description="Canonical Quran citation, e.g. [2:153].")

    verse_fragment_text: str = Field(
        min_length=1,
        description="Verified Quran text extracted from the Quran collection.",
    )

    @model_validator(mode="after")
    def validate_consistency(self) -> "DetectedCitation":
        expected = f"[{self.surah_number}:{self.ayah_number}]"
        if self.citation != expected:
            raise ValueError(
                f"citation must equal {expected!r} based on surah_number "
                "and ayah_number"
            )
        return self


class Resource(BaseModel):
    """A unique source actually used for the final answer."""

    source_type: Literal["quran", "tafsir"]

    title: str = Field(min_length=1)
    url: HttpUrl

    surah_number: int = Field(ge=1, le=114)
    ayah_number: int = Field(ge=1)

    tafsir_book: str | None = None

    @model_validator(mode="after")
    def validate_source_type(self) -> "Resource":
        if self.source_type == "tafsir" and not self.tafsir_book:
            raise ValueError("tafsir_book is required for a tafsir resource")
        if self.source_type == "quran" and self.tafsir_book is not None:
            raise ValueError("tafsir_book must be None for a quran resource")
        return self


class ChatResponse(BaseModel):
    """
    Final API response returned to the client after server-side hydration.
    """

    raw_response_text: str = Field(min_length=1)

    detected_citations: list[DetectedCitation] = Field(default_factory=list)
    resources: list[Resource] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_citation_occurrences(self) -> "ChatResponse":
        orders = [item.order for item in self.detected_citations]
        if orders != list(range(1, len(orders) + 1)):
            raise ValueError("Citation orders must be 1, 2, 3, ... in list order")
        for item in self.detected_citations:
            if item.citation not in self.raw_response_text:
                raise ValueError(f"{item.citation} does not occur in raw_response_text")
        return self


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


def _iter_tool_returns(messages: list[ModelMessage]):
    """Yield every ToolReturnPart from the researcher's message history."""
    for msg in messages:
        for part in getattr(msg, "parts", []):
            if isinstance(part, ToolReturnPart):
                yield part


def _records_from_tool_return(part: ToolReturnPart) -> list[dict]:
    """Unpack a tool return's content into a list of record payload dicts.

    pydantic-ai serializes user-defined tool returns to JSON, so `content`
    may be a JSON string. `get_quran`/`get_tafsir` return a single payload
    dict; the search tools return Qdrant points, each with a `payload` key.
    """
    content = part.content
    if isinstance(content, str):
        try:
            data = json.loads(content)
        except (ValueError, TypeError):
            return []
    elif isinstance(content, (dict, list)):
        data = content
    else:
        return []

    records: list[dict] = []
    if isinstance(data, dict):
        if data.get("error"):
            return []
        records.append(
            data.get("payload", data) if isinstance(data.get("payload"), dict) else data
        )
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict) or item.get("error"):
                continue
            payload = item.get("payload")
            if isinstance(payload, dict):
                records.append(payload)
            else:
                records.append(item)
    return records


def _quran_url(surah_number: int, ayah_number: int) -> str:
    return f"https://quran.com/{surah_number}/{ayah_number}"


def _tafsir_url(surah_number: int, ayah_number: int, tafsir_book: str) -> str | None:
    slug = TAFSIR_BOOK_TO_SLUG.get(tafsir_book)
    if not slug:
        # ponytail: never fabricate a URL for an unknown slug — the resource
        # is dropped instead of mislabeling a Quran URL as a Tafsir source.
        return None
    return f"https://quran.com/{surah_number}/{ayah_number}/tafsirs/{slug}"


def _quran_resource(record: dict) -> Resource | None:
    surah_number = int(record.get("surah_number", 0))
    ayah_number = int(record.get("ayah_number", 0))
    if not surah_number or not ayah_number:
        return None
    surah_name = str(record.get("surah", ""))
    return Resource(
        source_type="quran",
        title=f"سورة {surah_name}: {ayah_number}",
        url=_quran_url(surah_number, ayah_number),
        surah_number=surah_number,
        ayah_number=ayah_number,
        tafsir_book=None,
    )


def _tafsir_resource(record: dict) -> Resource | None:
    tafsir_book = record.get("tafsir_book")
    if not tafsir_book:
        return None
    surah_number = int(record.get("surah_number", 0))
    ayah_number = int(record.get("ayah_number", 0))
    if not surah_number or not ayah_number:
        return None
    surah_name = str(record.get("surah", ""))
    display_name = TAFSIR_BOOK_TO_NAME.get(tafsir_book, tafsir_book)
    url = _tafsir_url(surah_number, ayah_number, tafsir_book)
    if not url:
        return None
    return Resource(
        source_type="tafsir",
        title=f"تفسير {display_name} - {surah_name}: {ayah_number}",
        url=url,
        surah_number=surah_number,
        ayah_number=ayah_number,
        tafsir_book=tafsir_book,
    )


def _collect_resources(messages: list[ModelMessage]) -> list[Resource]:
    """Walk the tool returns; build one Resource per unique retrieval.

    Reads the records the researcher actually received (no re-running of
    searches). `get_quran`/search-on-quran yield `quran` resources; anything
    carrying a `tafsir_book` payload key yields a `tafsir` resource.
    """
    seen: set[tuple] = set()
    resources: list[Resource] = []

    def _add(resource: Resource | None) -> None:
        if resource is None:
            return
        key = (
            resource.source_type,
            resource.surah_number,
            resource.ayah_number,
            resource.tafsir_book,
        )
        if key in seen:
            return
        seen.add(key)
        resources.append(resource)

    for part in _iter_tool_returns(messages):
        if part.tool_name not in RETRIEVAL_TOOLS:
            continue
        for record in _records_from_tool_return(part):
            if "tafsir_book" in record:
                _add(_tafsir_resource(record))
            else:
                _add(_quran_resource(record))

    return resources


def _extract_marker_requests(raw_text: str) -> tuple[str, list[dict]]:
    """Replace every {fragment|surah:ayah} marker with a [surah:ayah] token.

    Returns the cleaned text and an ordered list of fragment requests (one
    per occurrence, duplicates allowed).
    """
    requests: list[dict] = []

    def _replace(match: re.Match) -> str:
        fragment = match.group("fragment").strip()
        surah_number = int(match.group("surah"))
        ayah_number = int(match.group("ayah"))
        citation = f"[{surah_number}:{ayah_number}]"

        if not fragment:
            raise ValueError(f"Empty Quran fragment in marker for {citation}")

        requests.append(
            {
                "citation": citation,
                "surah_number": surah_number,
                "ayah_number": ayah_number,
                "requested_fragment_text": fragment,
            }
        )
        return citation

    cleaned = RAW_CITATION_PATTERN.sub(_replace, raw_text)
    return cleaned, requests


def _resolve_fragment(
    ayah_text: str,
    requested_fragment_text: str,
) -> str:
    """Return the verified fragment inside the authoritative ayah.

    Falls back to the full verified ayah on a substring miss and logs the
    fallback for observability (tashkīl drift is the common cause).
    """
    global _fragment_fallback_count

    start = ayah_text.find(requested_fragment_text)
    if start != -1:
        return ayah_text[start : start + len(requested_fragment_text)]

    _fragment_fallback_count += 1
    logger.warning(
        "fragment_not_found_fallback_to_full_ayah",
        extra={"requested_fragment": requested_fragment_text},
    )
    return ayah_text


def _validate_marker_syntax(data: str) -> str:
    """Output validator: reject malformed citation markers so the LLM retries.

    Runs inside pydantic-ai's output-validation loop (text path), so raising
    `ModelRetry` feeds the error back to the model up to the output-retry
    budget instead of surfacing a hard 422.
    """
    stripped = RAW_CITATION_PATTERN.sub("", data)
    if "{" in stripped or "}" in stripped:
        raise ModelRetry(
            "Your response contains a malformed citation marker. Every Quran "
            "reference must use exactly the form {fragment_text|surah_number:"
            "ayah_number} with the fragment copied from the retrieved record. "
            "Fix or remove the malformed marker and return the corrected text."
        )
    return data


async def hydrate_chat_response(
    raw_text: str,
    messages: list[ModelMessage],
) -> ChatResponse:
    """
    Build the public API response from the LLM's raw text and the
    researcher's tool history.

    The LLM emits `{fragment|surah:ayah}` markers; the server replaces them
    with `[surah:ayah]` tokens, validates each fragment against the
    authoritative Quran record, and constructs the final response.
    """
    cleaned_text, fragment_requests = _extract_marker_requests(raw_text)

    quran_cache: dict[str, dict] = {}
    detected_citations: list[DetectedCitation] = []
    resource_by_key: dict[tuple, Resource] = {}

    for order, request in enumerate(fragment_requests, start=1):
        citation = request["citation"]
        surah_number = request["surah_number"]
        ayah_number = request["ayah_number"]

        if citation not in quran_cache:
            record = await asyncio.to_thread(
                get_quran, id=f"{surah_number}:{ayah_number}"
            )
            # get_quran returns a bare payload dict (search.py:348). If it
            # ever starts wrapping payloads, unwrap here.
            if not isinstance(record, dict) or not record.get("text"):
                raise HTTPException(
                    status_code=422,
                    detail=f"Quran record not found for {citation}",
                )
            quran_cache[citation] = record

        record = quran_cache[citation]
        ayah_text = str(record["text"])

        verified_fragment = _resolve_fragment(
            ayah_text, request["requested_fragment_text"]
        )

        detected_citations.append(
            DetectedCitation(
                order=order,
                surah_number=surah_number,
                surah_name=str(record.get("surah", "")),
                ayah_number=ayah_number,
                citation=citation,
                verse_fragment_text=verified_fragment,
            )
        )

        quran_resource = _quran_resource(record)
        if quran_resource:
            key = (
                quran_resource.source_type,
                quran_resource.surah_number,
                quran_resource.ayah_number,
                quran_resource.tafsir_book,
            )
            resource_by_key[key] = quran_resource

    for resource in _collect_resources(messages):
        key = (
            resource.source_type,
            resource.surah_number,
            resource.ayah_number,
            resource.tafsir_book,
        )
        resource_by_key[key] = resource

    # ponytail: catch prompt drift where the LLM emits a malformed marker
    # (missing pipe, unclosed brace) that the regex silently skipped. The
    # output validator normally catches these; this is a last-resort log.
    if "{" in cleaned_text or "}" in cleaned_text:
        logger.warning(
            "unprocessed_citation_marker_leaked",
            extra={"cleaned_text": cleaned_text},
        )

    return ChatResponse(
        raw_response_text=cleaned_text,
        detected_citations=detected_citations,
        resources=list(resource_by_key.values()),
    )


async def stream(
    request_agent: Agent,
    prompt: str,
    session_id: str,
):

    history = await asyncio.to_thread(load_session, session_id)
    saved = False
    try:
        async with request_agent.run_stream_events(
            prompt, message_history=history
        ) as events:
            async for event in events:
                match event:
                    case PartStartEvent(part=TextPart() as part):
                        yield sse(
                            "message_start",
                            {"index": event.index, "text": part.content},
                        )
                    case PartDeltaEvent(delta=TextPartDelta() as delta):
                        yield sse("text_delta", {"text": delta.content_delta})

                    case FunctionToolCallEvent(part=ToolCallPart() as part):
                        yield sse(
                            "tool",
                            {
                                "text": part.tool_name,
                                "tool_call_id": part.tool_call_id,
                                "args": part.args_as_dict(),
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
                        await asyncio.to_thread(
                            save_session_messages, session_id, result.all_messages()
                        )
                        saved = True
                        # `request_agent` outputs plain text; hydrate the
                        # server-constructed fields before emitting.
                        hydrated = await hydrate_chat_response(
                            result.output,
                            result.all_messages(),
                        )
                        yield sse(
                            "done",
                            {"output": hydrated.model_dump(mode="json")},
                        )
    finally:
        if not saved:
            # ponytail: partial history is lost on disconnect; log it until we
            # persist accumulated deltas on abort.
            logger.warning("stream_aborted_no_save")


setup_indexes()

with open("./skills/turath-index-skill.md") as file:
    turath = file.read()

with open("./skills/citations.md") as file:
    citations = file.read()

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
    get_tafsir,
    get_books_tafsir,
    dense_search,
    sparse_search,
    hybrid_search,
    hybrid_search_weighted,
]


def build_agent(
    *,
    api_key: str,
    model_name: str = "deepseek-v4-flash",
    variant: str = "low",
    system_prompt: str,
) -> Agent:
    """Build a request-local agent.

    DeepSeek is the default provider; the per-user API key is passed in
    explicitly instead of leaking from a module-level global.
    """
    model = OpenAIChatModel(
        model_name,
        provider=DeepSeekProvider(api_key=api_key),
    )
    model_settings = OpenAIResponsesModelSettings(
        temperature=0.5,
        service_tier="flex",
        thinking=str(variant),
    )
    agent = Agent(
        model,
        name="islamic_scholar_agent",
        model_settings=model_settings,
        system_prompt=system_prompt,
        output_type=str,
        retries={"output": 3},
        capabilities=capabilities,
        tools=tools,
    )
    agent.output_validator(_validate_marker_syntax)
    return agent


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

    model_name = "deepseek-v4-flash"
    model_variant = "low"

    api_key = os.getenv("API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400, detail="No API key on file for this provider"
        )

    request_agent = build_agent(
        api_key=api_key,
        model_name=model_name,
        variant=model_variant,
        system_prompt=turath,
    )

    return StreamingResponse(
        stream(
            request_agent,
            req.message,
            session_id=session["id"],
        ),
        media_type="text/event-stream",
    )


@router.post("/structured")
async def chat(
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

    # DeepSeek is the default provider for this endpoint; the session's
    # model config is intentionally ignored (single-model deployment).
    model_name = "deepseek-v4-flash"
    model_variant = "low"

    api_key = os.getenv("API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="No API key on file for this provider",
        )

    request_agent = build_agent(
        api_key=api_key,
        model_name=model_name,
        variant=model_variant,
        system_prompt=citations,
    )

    history = await asyncio.to_thread(load_session, session["id"])

    result = await request_agent.run(req.message, message_history=history)

    hydrated = await hydrate_chat_response(
        result.output,
        result.all_messages(),
    )

    await asyncio.to_thread(
        save_session_messages,
        session["id"],
        result.all_messages(),
    )

    return hydrated


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
