import psycopg2
from pydantic import BaseModel
from api._docs import COMMON_ERROR_RESPONSES
from db.connection import get_conn
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/messages", tags=["Messages"])


class Message(BaseModel):
    session_id: str
    role: str
    content: str
    metadata: dict = {}
    sequence: int


class MessageOut(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    metadata: dict = {}
    sequence: int
    created_at: str | None = None


@router.get(
    "/{session_id}",
    summary="List messages in a session, oldest first",
    description=(
        "Returns the persisted message history ordered by `sequence`. The "
        "live `pydantic_message` blob used by the agent lives on the session "
        "row, not here."
    ),
    response_model=list[MessageOut],
)
async def get_session_messages(session_id: str) -> list[MessageOut]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, session_id, role, content, metadata, sequence, created_at "
                "FROM messages WHERE session_id = %s ORDER BY sequence ASC",
                (session_id,),
            )
            rows = cur.fetchall()
    return [
        {
            "id": str(r[0]),
            "session_id": str(r[1]),
            "role": r[2],
            "content": r[3],
            "metadata": r[4],
            "sequence": r[5],
            "created_at": r[6].isoformat() if r[6] else None,
        }
        for r in rows
    ]


@router.post(
    "/add",
    summary="Persist a single message",
    description=(
        "Internal helper used by tests and migrations. The chat agent writes "
        "its own history to `sessions.pydantic_message` directly."
    ),
    response_model=MessageOut,
    responses={400: COMMON_ERROR_RESPONSES[400]},
)
def add_message(req: Message) -> MessageOut:
    with get_conn() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO messages (session_id, role, content, metadata, sequence)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id, session_id, role, content, metadata, sequence, created_at
                    """,
                    (req.session_id, req.role, req.content, req.metadata, req.sequence),
                )
                row = cur.fetchone()
            conn.commit()
        except psycopg2.Error as e:
            conn.rollback()
            raise HTTPException(status_code=400, detail=str(e))
    return {
        "id": str(row[0]),
        "session_id": str(row[1]),
        "role": row[2],
        "content": row[3],
        "metadata": row[4],
        "sequence": row[5],
    }
