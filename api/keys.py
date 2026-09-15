"""API key storage, now backed by `user_providers`.

Design notes (per the refactor spec):
- API keys belong to USERS, not sessions. One connection per (user, provider).
- Keys are never returned in responses and never logged.
- `encrypted_key` is a single isolated column so a real encryption/decryption
  step (e.g. Fernet/AES-GCM via a KMS-managed secret) can be added later
  without touching callers. `encrypt()`/`decrypt()` below are the seam.
- DEPRECATED ROUTES: this whole `/keys` router is a shim for the legacy
  Telegram bot web flow; the authenticated connection CRUD replaces it
  (Phase D deletes the web callers). The `api_keys` table itself is frozen:
  it is the rollback snapshot for this feature and is never read or written
  here any more.
"""

import psycopg2
import os
from pydantic import BaseModel
from db.connection import get_conn
from fastapi import APIRouter, HTTPException, Request
from cryptography.fernet import Fernet

from identity import current_user_id, require_owner

router = APIRouter(prefix="/keys", tags=["API Keys"])


def encrypt(raw_key: str) -> str:
    master_key = os.environ["ENCRYPTION_MASTER_KEY"]
    fernet = Fernet(master_key)

    encrypted_key = fernet.encrypt(raw_key.encode())
    return encrypted_key.decode()


def decrypt(stored_value: str) -> str:
    master_key = os.environ["ENCRYPTION_MASTER_KEY"]
    fernet = Fernet(master_key)

    return fernet.decrypt(stored_value).decode()


class ApiKeyIn(BaseModel):
    # Ignored when an identity was resolved; permissive-rollout fallback only.
    user_id: str | None = None
    provider: str
    api_key: str


class ApiKeyExists(BaseModel):
    provider: str
    has_key: bool


def _row_to_key_meta(row) -> dict | None:
    """(id, user_id, provider_slug, created_at, updated_at) -> response dict."""
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
async def has_key(user_id: str, provider: str, request: Request):
    """DEPRECATED (bot shim): used by /model to decide whether to ask for a
    new API key. Reads the user's connection instead of `api_keys`."""
    require_owner(request, user_id)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM user_providers up "
                "JOIN providers p ON p.id = up.provider_id "
                "WHERE up.user_id = %s AND p.slug = %s",
                (user_id, provider),
            )
            row = cur.fetchone()
    return {"provider": provider, "has_key": row is not None}


@router.get("/{user_id}")
async def list_keys(user_id: str, request: Request):
    """DEPRECATED (bot shim): metadata only. Never returns the actual key."""
    require_owner(request, user_id)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT up.id, up.user_id, p.slug, up.created_at, up.updated_at "
                "FROM user_providers up "
                "LEFT JOIN providers p ON p.id = up.provider_id "
                "WHERE up.user_id = %s ORDER BY p.slug",
                (user_id,),
            )
            rows = cur.fetchall()
    return [_row_to_key_meta(r) for r in rows]


@router.post("/add")
def add_or_update_key(req: ApiKeyIn, request: Request):
    """DEPRECATED (bot shim): upserts the user's connection for this provider
    (one per provider) and seeds its `provider_models` from the provider
    catalog, so a key added today still yields a model list. Never logs or
    echoes the key."""
    user_id = current_user_id(request, req.user_id)
    with get_conn() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, default_base_url, api_style, models "
                    "FROM providers WHERE slug = %s",
                    (req.provider,),
                )
                provider = cur.fetchone()

                if provider is None:
                    conn.rollback()
                    raise HTTPException(
                        status_code=404,
                        detail=f"Unknown provider: '{req.provider}'",
                    )

                provider_id, base_url, api_style, models = provider

                cur.execute(
                    """
                    INSERT INTO user_providers
                        (user_id, provider_id, is_custom, base_url, api_style, encrypted_key)
                    VALUES (%s, %s, false, %s, %s, %s)
                    ON CONFLICT (user_id, provider_id) DO UPDATE
                        SET encrypted_key = EXCLUDED.encrypted_key,
                            base_url = EXCLUDED.base_url,
                            api_style = EXCLUDED.api_style,
                            updated_at = now()
                    RETURNING id, user_id, created_at, updated_at
                    """,
                    (user_id, provider_id, base_url, api_style, encrypt(req.api_key)),
                )
                row = cur.fetchone()
                connection_id = row[0]

                for model_id, model_config in (models or {}).items():
                    cur.execute(
                        """
                        INSERT INTO provider_models
                            (user_provider_id, model_id, display_name, variants,
                             is_custom, enabled)
                        VALUES (%s, %s, %s, %s, false, true)
                        ON CONFLICT (user_provider_id, model_id) DO UPDATE
                            SET variants = EXCLUDED.variants,
                                enabled = true
                        """,
                        (
                            connection_id,
                            model_id,
                            model_id,
                            (model_config or {}).get("variants") or [],
                        ),
                    )
            conn.commit()
        except psycopg2.Error as e:
            conn.rollback()
            raise HTTPException(status_code=400, detail=str(e))

    return _row_to_key_meta((row[0], row[1], req.provider, row[2], row[3]))


@router.put("/update")
def update_key(req: ApiKeyIn, request: Request):
    """DEPRECATED (bot shim): replace the key on an existing connection.
    404 if the user has no connection for this provider."""
    user_id = current_user_id(request, req.user_id)
    with get_conn() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE user_providers up
                    SET encrypted_key = %s,
                        updated_at = now()
                    FROM providers p
                    WHERE p.id = up.provider_id
                      AND p.slug = %s
                      AND up.user_id = %s
                    RETURNING up.id, up.user_id, up.created_at, up.updated_at
                    """,
                    (encrypt(req.api_key), req.provider, user_id),
                )
                row = cur.fetchone()

            if row is None:
                conn.rollback()
                raise HTTPException(
                    status_code=404,
                    detail=f"API key for provider '{req.provider}' not found.",
                )

            conn.commit()
        except psycopg2.Error as e:
            conn.rollback()
            raise HTTPException(status_code=400, detail=str(e))

    return _row_to_key_meta((row[0], row[1], req.provider, row[2], row[3]))
