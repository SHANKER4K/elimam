import psycopg2
from pydantic import BaseModel
from db.connection import get_conn
from fastapi import APIRouter, HTTPException, Request

from identity import current_user_id, optional_user_id

router = APIRouter(prefix="/messages", tags=["Messages"])


class Message(BaseModel):
    session_id: str
    role: str
    content: str
    sequence: int
    # Ignored when an identity was resolved; permissive-rollout fallback only.
    user_id: str | None = None


@router.get("/{session_id}")
async def get_session_messages(session_id: str, request: Request):
    # COALESCE(NULL, s.user_id) keeps the pre-migration unfiltered read
    # working under AUTH_MODE=permissive; enforce never gets here without it.
    user_id = optional_user_id(request)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT m.id, m.session_id, m.role, m.content, m.sequence, m.created_at "
                "FROM messages m JOIN sessions s ON s.id = m.session_id "
                "WHERE m.session_id = %s AND s.user_id = COALESCE(%s, s.user_id) "
                "ORDER BY m.sequence ASC",
                (session_id, user_id),
            )
            rows = cur.fetchall()
    return [
        {
            "id": str(r[0]),
            "session_id": str(r[1]),
            "role": r[2],
            "content": r[3],
            "sequence": r[4],
            "created_at": r[5].isoformat() if r[5] else None,
        }
        for r in rows
    ]


@router.post("/add")
def add_message(req: Message, request: Request):
    user_id = current_user_id(request, req.user_id)
    with get_conn() as conn:
        try:
            with conn.cursor() as cur:
                # One statement, so the ownership check cannot race the insert.
                # role is cast because migration 0007 made it an enum.
                cur.execute(
                    """
                    INSERT INTO messages (session_id, role, content, sequence)
                    SELECT %s, %s::message_role, %s, %s
                    WHERE EXISTS (
                        SELECT 1 FROM sessions s
                        WHERE s.id = %s AND s.user_id = %s
                    )
                    RETURNING id, session_id, role, content, sequence, created_at
                    """,
                    (
                        req.session_id,
                        req.role,
                        req.content,
                        req.sequence,
                        req.session_id,
                        user_id,
                    ),
                )
                row = cur.fetchone()
                if row is None:
                    conn.rollback()
                    raise HTTPException(status_code=404, detail="Session not found")
            conn.commit()
        except psycopg2.Error as e:
            conn.rollback()
            raise HTTPException(status_code=400, detail=str(e))
    return {
        "id": str(row[0]),
        "session_id": str(row[1]),
        "role": row[2],
        "content": row[3],
        "sequence": row[4],
    }
