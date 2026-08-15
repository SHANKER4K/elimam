"""API key storage for the api_keys table.

Design notes (per the refactor spec):
- API keys belong to USERS, not sessions. One key per (user, provider).
- Keys are never returned in responses and never logged.
- `encrypted_key` is a single isolated column so a real encryption/decryption
  step (e.g. Fernet/AES-GCM via a KMS-managed secret) can be added later
  without touching callers. `encrypt()`/`decrypt()` below are the seam:
  today they are the identity function, but every write/read of a raw key
  goes through them so upgrading is a one-file change.
"""

import psycopg2
from pydantic import BaseModel
from db.connection import get_conn
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/keys", tags=["API Keys"])


def encrypt(raw_key: str) -> str:
    # ponytail: plug real encryption here (e.g. Fernet.encrypt) when a KMS
    # key/secret is available. Keeping this isolated per design rule #3.
    return raw_key


def decrypt(stored_value: str) -> str:
    # ponytail: plug real decryption here to match encrypt() above.
    return stored_value


class ApiKeyIn(BaseModel):
    user_id: str
    provider: str
    api_key: str


class ApiKeyExists(BaseModel):
    provider: str
    has_key: bool


def _row_to_key_meta(row) -> dict | None:
    if row is None:
        return None
    return {
        "id": str(row[0]),
        "user_id": str(row[1]),
        "provider": row[2],
        "created_at": row[3].isoformat() if row[3] else None,
        "updated_at": row[4].isoformat() if row[4] else None,
    }


@router.get("/{user_id}/{provider}/exists")
async def has_key(user_id: str, provider: str):
    """Used by /model to decide whether to ask for a new API key."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM api_keys WHERE user_id = %s AND provider = %s",
                (user_id, provider),
            )
            row = cur.fetchone()
    return {"provider": provider, "has_key": row is not None}


@router.get("/{user_id}")
async def list_keys(user_id: str):
    """Metadata only. Never returns the actual key value."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, user_id, provider, created_at, updated_at FROM api_keys WHERE user_id = %s",
                (user_id,),
            )
            rows = cur.fetchall()
    return [_row_to_key_meta(r) for r in rows]


@router.post("/add")
def add_or_update_key(req: ApiKeyIn):
    """Upsert: one key per (user_id, provider). Never logs or echoes the key."""
    with get_conn() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO api_keys (user_id, provider, encrypted_key)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (user_id, provider) DO UPDATE
                        SET encrypted_key = EXCLUDED.encrypted_key,
                            updated_at = now()
                    RETURNING id, user_id, provider, created_at, updated_at
                    """,
                    (req.user_id, req.provider, encrypt(req.api_key)),
                )
                row = cur.fetchone()
            conn.commit()
        except psycopg2.Error as e:
            conn.rollback()
            raise HTTPException(status_code=400, detail=str(e))
    return _row_to_key_meta(row)


def get_decrypted_key(user_id: str, provider: str) -> str | None:
    """Internal helper for the backend (e.g. /chat) to fetch the raw key.
    NOT exposed as a route - never return this value to a client."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT encrypted_key FROM api_keys WHERE user_id = %s AND provider = %s",
                (user_id, provider),
            )
            row = cur.fetchone()
    if row is None:
        return None
    return decrypt(row[0])
