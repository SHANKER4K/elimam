import psycopg2
from pydantic import BaseModel
from db.connection import get_conn
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/users", tags=["Users"])


class User(BaseModel):
    username: str | None = None
    display_name: str | None = None
    telegram_id: str | None = None
    email: str | None = None


class UserOut(BaseModel):
    id: str
    username: str | None
    display_name: str | None
    telegram_id: str | None
    email: str | None


def _row_to_user(row) -> dict | None:
    if row is None:
        return None
    return {
        "id": str(row[0]),
        "username": row[1],
        "display_name": row[2],
        "telegram_id": row[3],
        "email": row[4],
    }


@router.get("/{user_id}")
async def get_user(user_id: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, display_name, telegram_id, email FROM users WHERE id = %s",
                (user_id,),
            )
            row = cur.fetchone()
    user = _row_to_user(row)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.get("/telegram/{telegram_id}")
async def get_user_by_telegram_id(telegram_id: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, display_name, telegram_id, email FROM users WHERE telegram_id = %s",
                (telegram_id,),
            )
            row = cur.fetchone()
    user = _row_to_user(row)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.post("/add")
def add_user(req: User):
    with get_conn() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (username, display_name, telegram_id, email)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (telegram_id) DO UPDATE
                        SET username = COALESCE(EXCLUDED.username, users.username)
                    RETURNING id, username, display_name, telegram_id, email
                    """,
                    (req.username, req.display_name, req.telegram_id, req.email),
                )
                row = cur.fetchone()
            conn.commit()
        except psycopg2.Error as e:
            conn.rollback()
            raise HTTPException(status_code=400, detail=str(e))
    return _row_to_user(row)


def get_or_create_user_by_telegram_id(
    telegram_id: str,
    username: str | None = None,
    display_name: str | None = None,
) -> dict:
    """Shared helper so /chat and other internal callers don't duplicate
    the lookup-or-create logic. Not exposed directly as a route."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, display_name, telegram_id, email FROM users WHERE telegram_id = %s",
                (telegram_id,),
            )
            row = cur.fetchone()
    existing = _row_to_user(row)
    if existing is not None:
        return existing

    with get_conn() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (username, display_name, telegram_id)
                    VALUES (%s, %s, %s)
                    RETURNING id, username, display_name, telegram_id, email
                    """,
                    (username, display_name, telegram_id),
                )
                row = cur.fetchone()
            conn.commit()
        except psycopg2.Error as e:
            conn.rollback()
            raise HTTPException(status_code=400, detail=str(e))
    return _row_to_user(row)
