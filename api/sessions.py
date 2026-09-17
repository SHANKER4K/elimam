import json
import logging

import psycopg2
from pydantic_core.core_schema import float_schema
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from db.connection import get_conn
from identity import can_access, current_user_id, require_owner

router = APIRouter(prefix="/sessions", tags=["Sessions"])
logger = logging.getLogger("api.providers")


class SessionCreate(BaseModel):
    # Ignored when an identity was resolved; permissive-rollout fallback only.
    user_id: str | None = None
    source: str
    model_provider: str | None = None
    model_name: str | None = None
    model_variant: str | None = None
    user_provider_id: str | None = None


class SessionModelUpdate(BaseModel):
    model_provider: str
    model_name: str
    model_variant: str


def _row_to_session(row) -> dict | None:
    if row is None:
        return None
    return {
        "id": str(row[0]),
        "user_id": str(row[1]),
        "source": row[2],
        "model_provider": row[3],
        "model_name": row[4],
        "model_variant": row[5],
        "is_active": row[6],
        "user_provider_id": str(row[7]) if len(row) > 7 and row[7] else None,
    }


SESSION_COLUMNS = "id, user_id, source, model_provider, model_name, model_variant, is_active, user_provider_id"


def get_session(session_id: str):
    """Internal helper (no auth): sync so internal callers can use
    asyncio.to_thread() and get a real result instead of an unawaited
    coroutine. The route below adds the ownership check."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT {SESSION_COLUMNS}, pydantic_message FROM sessions WHERE id = %s",
                    (session_id,),
                )
                row = cur.fetchone()
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    try:
        session = _row_to_session(row)
    except Exception as e:
        print("=" * 20, flush=True)
        print(row, flush=True)
        print("=" * 20, flush=True)
        raise f"{e}, Here is the problem ,{row}"
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("/{session_id}")
def get_session_route(session_id: str, request: Request):
    session = get_session(session_id)
    if not can_access(request, session["user_id"]):
        # Same answer as an unknown id: do not confirm it exists.
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("/active/user/{user_id}")
async def get_active_session(user_id: str, request: Request):
    """The one source of truth for "what session is this user currently in"."""
    require_owner(request, user_id)
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT {SESSION_COLUMNS} FROM sessions WHERE user_id = %s AND is_active IS TRUE",
                    (user_id,),
                )
                row = cur.fetchone()
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    session = _row_to_session(row)
    if session is None:
        raise HTTPException(status_code=404, detail="No active session")
    return session


@router.post("/add")
def add_session(req: SessionCreate, request: Request):
    """Creates a new active session. Caller (backend logic, not the bot) is
    responsible for deactivating any prior active session first when that
    matters (see reset_session / create_user_with_session for atomic paths).
    """
    user_id = current_user_id(request, req.user_id)
    with get_conn() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO sessions (user_id, source, model_provider, model_name, model_variant, pydantic_message, is_active)
                    VALUES (%s, %s, %s, %s, %s, %s, true)
                    RETURNING {SESSION_COLUMNS}
                    """,
                    (
                        user_id,
                        req.source,
                        req.model_provider,
                        req.model_name,
                        req.model_variant,
                        json.dumps([]),
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        except psycopg2.Error as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {e}")
    return (_row_to_session(row),)


def create_web_session(session_id: str, req: SessionCreate) -> None:
    """Insert a web session keyed by the client-generated id, never active.

    The web client owns the conversation id (its URL path segment); we insert
    under that exact id so message history FKs resolve. is_active=false keeps
    web conversations out of the one-active-per-user unique index.
    """
    with get_conn() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sessions (id, user_id, source, model_provider, model_name, model_variant, user_provider_id, pydantic_message, is_active)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, false)
                    """,
                    (
                        session_id,
                        req.user_id,
                        req.source,
                        req.model_provider,
                        req.model_name,
                        req.model_variant,
                        req.user_provider_id,
                        json.dumps([]),
                    ),
                )
            conn.commit()
        except psycopg2.Error as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {e}")


@router.put("/{session_id}/model")
def update_session_model(session_id: str, req: SessionModelUpdate, request: Request):
    """Used by /model. Updates the active session's config WITHOUT creating
    a new session and WITHOUT touching is_active."""
    user_id = current_user_id(request)
    with get_conn() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE sessions
                    SET model_provider = %s,
                        model_name = %s,
                        model_variant = %s,
                        updated_at = now()
                    WHERE id = %s AND user_id = %s
                    RETURNING {SESSION_COLUMNS}
                    """,
                    (
                        req.model_provider,
                        req.model_name,
                        req.model_variant,
                        session_id,
                        user_id,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        except psycopg2.Error as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {e}")
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "status": True,
        "message": "Updated session model",
        "session": _row_to_session(row),
    }


def reset_session(user_id: str) -> dict:
    """Atomic /reset: deactivate the current active session and create a new
    one carrying over the same model configuration. The old session and its
    messages remain untouched in the database."""
    with get_conn() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT {SESSION_COLUMNS} FROM sessions WHERE user_id = %s AND is_active IS TRUE FOR UPDATE",
                    (user_id,),
                )
                current = cur.fetchone()
                if current is None:
                    raise HTTPException(
                        status_code=404, detail="No active session to reset"
                    )

                current_session = _row_to_session(current)

                cur.execute(
                    "UPDATE sessions SET is_active = false, updated_at = now() WHERE id = %s",
                    (current_session["id"],),
                )

                cur.execute(
                    f"""
                    INSERT INTO sessions (user_id, source, model_provider, model_name, model_variant, pydantic_message, is_active)
                    VALUES (%s, %s, %s, %s, %s, %s, true)
                    RETURNING {SESSION_COLUMNS}
                    """,
                    (
                        user_id,
                        current_session["source"],
                        current_session["model_provider"],
                        current_session["model_name"],
                        current_session["model_variant"],
                        json.dumps([]),
                    ),
                )
                new_row = cur.fetchone()
            conn.commit()
        except psycopg2.Error as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {e}")
    return _row_to_session(new_row)


@router.post("/reset/{user_id}")
async def reset_session_route(user_id: str, request: Request):
    require_owner(request, user_id)
    new_session = reset_session(user_id)
    return {"status": True, "message": "Session reset", "session": new_session}
