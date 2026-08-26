import json

import psycopg2
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from db.connection import get_conn

router = APIRouter(prefix="/sessions", tags=["Sessions"])


class SessionCreate(BaseModel):
    user_id: str
    source: str
    model_provider: str | None = None
    model_name: str | None = None
    model_variant: str | None = None


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
    }


SESSION_COLUMNS = (
    "id, user_id, source, model_provider, model_name, model_variant, is_active"
)


@router.get("/{session_id}")
async def get_session(session_id: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {SESSION_COLUMNS}, pydantic_message FROM sessions WHERE id = %s",
                (session_id,),
            )
            row = cur.fetchone()
    session = _row_to_session(row)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("/active/user/{user_id}")
async def get_active_session(user_id: str):
    """The one source of truth for "what session is this user currently in"."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {SESSION_COLUMNS} FROM sessions WHERE user_id = %s AND is_active IS TRUE",
                (user_id,),
            )
            row = cur.fetchone()
    session = _row_to_session(row)
    if session is None:
        raise HTTPException(status_code=404, detail="No active session")
    return session


@router.post("/add")
def add_session(req: SessionCreate):
    """Creates a new active session. Caller (backend logic, not the bot) is
    responsible for deactivating any prior active session first when that
    matters (see reset_session / create_user_with_session for atomic paths).
    """
    with get_conn() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sessions (user_id, source, model_provider, model_name, model_variant, pydantic_message, is_active)
                    VALUES (%s, %s, %s, %s, %s, %s, true)
                    RETURNING id, user_id, source, model_provider, model_name, model_variant, is_active
                    """,
                    (
                        req.user_id,
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
            raise HTTPException(status_code=400, detail=str(e))
    return (_row_to_session(row),)


@router.put("/{session_id}/model")
def update_session_model(session_id: str, req: SessionModelUpdate):
    """Used by /model. Updates the active session's config WITHOUT creating
    a new session and WITHOUT touching is_active."""
    with get_conn() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE sessions
                    SET model_provider = %s,
                        model_name = %s,
                        model_variant = %s,
                        updated_at = now()
                    WHERE id = %s
                    RETURNING id, user_id, source, model_provider, model_name, model_variant, is_active
                    """,
                    (req.model_provider, req.model_name, req.model_variant, session_id),
                )
                row = cur.fetchone()
            conn.commit()
        except psycopg2.Error as e:
            conn.rollback()
            raise HTTPException(status_code=400, detail=str(e))
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
                    """
                    INSERT INTO sessions (user_id, source, model_provider, model_name, model_variant, pydantic_message, is_active)
                    VALUES (%s, %s, %s, %s, %s, %s, true)
                    RETURNING id, user_id, source, model_provider, model_name, model_variant, is_active
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
            raise HTTPException(status_code=400, detail=str(e))
    return _row_to_session(new_row)


@router.post("/reset/{user_id}")
async def reset_session_route(user_id: str):
    new_session = reset_session(user_id)
    return {"status": True, "message": "Session reset", "session": new_session}
