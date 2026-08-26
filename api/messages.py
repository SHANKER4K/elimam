import psycopg2
from pydantic import BaseModel
from db.connection import get_conn
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/messages", tags=["Messages"])


class Message(BaseModel):
    session_id: str
    role: str
    content: str
    metadata: dict = {}
    sequence: int


@router.get("/{session_id}")
async def get_session_messages(session_id: str):
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


@router.post("/add")
def add_message(req: Message):
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
