"""AG-UI route: POST /chat/agui (issue #20).

The same scholar agent as /chat/web, spoken over the AG-UI protocol through
pydantic-ai's AGUIAdapter instead of the hand-written SSE in api/chat.py.

- Identity, model resolution and session ownership are exactly /chat/web's.
- `threadId` is the session UUID; model/provider/variant/connection ride in
  `forwardedProps`. History stays server-side (`manage_message_history="server"`):
  the client sends only its new message.
- Errors and aborts are protocol events (RUN_ERROR) the adapter emits itself,
  so there is no half-finished stream (issue #24 for the old route).
- The frontend's extras (`metadata` on tool events, citation cards) are added
  by `ElimamEventStream`, on top of the standard events.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from ag_ui.core import CustomEvent
from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError
from pydantic_ai.messages import ModelRequest

from api.agui_events import ElimamAGUIAdapter
from api.chat import (
    SessionConflict,
    _active_turns,
    _build_agent,
    load_session_state,
    persist_turn,
)
from api.providers import resolve_model_config
from api.sessions import SessionCreate, create_web_session, get_session
from identity import current_user_id

logger = logging.getLogger("api.agui")
router = APIRouter(prefix="/chat", tags=["Chat"])


def _prop(props: Any, key: str) -> str | None:
    value = props.get(key) if isinstance(props, dict) else None
    return value if isinstance(value, str) and value else None


@router.post("/agui")
async def chat_agui(request: Request):
    user_id = current_user_id(request)

    try:
        run_input = ElimamAGUIAdapter.build_run_input(await request.body())
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail="Invalid AG-UI run input") from exc

    session_id = run_input.thread_id
    try:
        uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="threadId must be a UUID")

    props = run_input.forwarded_props
    model_provider = _prop(props, "model_provider")
    model_name = _prop(props, "model_name")
    model_variant = _prop(props, "model_variant")
    if not model_provider or not model_name or not model_variant:
        raise HTTPException(
            status_code=409, detail="Active session is missing model configuration"
        )

    try:
        session = await asyncio.to_thread(get_session, session_id)
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
        session = None

    if session is not None and str(session["user_id"]) != user_id:
        # Do not leak whether the session exists for someone else.
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        model_config = resolve_model_config(
            session["user_provider_id"] if session else _prop(props, "user_provider_id"),
            model_provider,
            model_name,
            model_variant,
            user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if model_config["api_key"] is None and model_config["requires_key"]:
        raise HTTPException(status_code=400, detail="API key not set for this provider")

    if session is None:
        await asyncio.to_thread(
            create_web_session,
            session_id,
            SessionCreate(
                user_id=user_id,
                source="web",
                model_provider=model_provider,
                model_name=model_name,
                model_variant=model_variant,
                user_provider_id=model_config["user_provider_id"],
            ),
        )

    # One turn per session at a time, as on /chat/web. Check and add happen with
    # no await between them, so a second request sees the first. Answered with
    # a plain 409 because the response has not started yet.
    if session_id in _active_turns:
        raise HTTPException(
            status_code=409, detail="هناك رد قيد التوليد في هذه المحادثة."
        )
    _active_turns.add(session_id)
    try:
        agent = _build_agent(
            provider_slug=model_provider,
            model_name=model_config["model"],
            provider_url=model_config["url"],
            api_key=model_config["api_key"],
            variant=model_config["variant"],
            session_id=session_id,
            api_style=model_config["api_style"],
            extra_headers=model_config["extra_headers"],
        )
        history, digest = await asyncio.to_thread(load_session_state, session_id)

        try:
            adapter = await ElimamAGUIAdapter.from_request(
                request,
                agent=agent,
                manage_system_prompt="server",
                manage_message_history="server",
            )
        except ValidationError as exc:
            raise HTTPException(
                status_code=422, detail="Invalid AG-UI run input"
            ) from exc
    except BaseException:
        _active_turns.discard(session_id)
        raise

    # With server-held history the client's new message is part of the run's
    # *input*, so result.new_messages() holds only what the agent produced. The
    # log needs the user's turn too: take it from the adapter (the last one,
    # in case a client resends the whole thread).
    user_turn = [m for m in adapter.messages if isinstance(m, ModelRequest)][-1:]

    # ponytail: persisted on a completed run only. A run the user aborts
    # (closing the connection) or that fails is not written, so that turn is
    # simply absent from the history instead of half-recorded; the old /chat/web
    # saved partial turns in a `finally`. Upgrade path: save RunCancelled
    # .all_messages() from on_cancel once compaction can be told apart from new.
    async def on_complete(result):
        try:
            # Atomic, and only if the stored history is still the one this turn
            # started from (compare-and-swap on `digest`, see persist_turn).
            await asyncio.to_thread(
                persist_turn,
                session_id,
                result.all_messages(),
                [*user_turn, *result.new_messages()],
                digest,
            )
        except SessionConflict:
            logger.warning("session_conflict", extra={"session_id": session_id})
            yield CustomEvent(
                name="session_conflict",
                value={"message": "تم تحديث هذه المحادثة من مكان آخر؛ لم يُحفظ هذا الرد."},
            )
        except Exception:
            logger.exception("agui_save_failed", extra={"session_id": session_id})

    async def guarded():
        # ponytail: released when the stream is iterated to its end or closed. A
        # response whose client vanishes before the first byte never starts
        # this generator, so its `finally` cannot run; /chat/web's stream() has
        # the same edge. Upgrade path: a TTL on the entry.
        try:
            async for event in adapter.run_stream(
                message_history=history, on_complete=on_complete
            ):
                yield event
        finally:
            _active_turns.discard(session_id)

    return adapter.streaming_response(guarded())
