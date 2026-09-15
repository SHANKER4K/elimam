"""Provider catalog + per-connection model resolution, read from the DB.

The `providers` table is the source of truth (`default_base_url` for the URL,
`models` jsonb for the catalog `{model_id: {"variants": [...]}}`). The YAML
file this module used to read is gone.

Two public names are imported elsewhere and must keep working:
- `load_catalog()` — the shape the Telegram bot builds its keyboards from:
  `{slug: {"url": ..., "models": {model: {"variants": [...]}}}}`.
- `resolve_model_config(...)` — resolves a model against the *connection*
  that will pay for the request, and raises `ValueError` for anything a
  client got wrong (both `chat.py` call sites catch it).
"""

from __future__ import annotations

import hmac
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Header

from api.keys import decrypt
from db.connection import get_conn

router = APIRouter(prefix="/providers", tags=["Providers"])


BOT_SHARED_SECRET = os.environ.get("BOT_SHARED_SECRET", "")

# ponytail: process-local catalog cache; if the backend ever runs multi-worker,
# move it to the write-invalidate pattern or drop it. One worker today, so the
# worst case is one stale read after a Phase C write.
_catalog_cache: dict[str, dict[str, Any]] | None = None


def _provider_rows() -> dict[str, dict[str, Any]]:
    global _catalog_cache
    if _catalog_cache is None:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT slug, id, default_base_url, api_style, requires_key, models "
                    "FROM providers ORDER BY id"
                )
                rows = cur.fetchall()
        _catalog_cache = {
            slug: {
                "id": provider_id,
                "url": url,
                "api_style": api_style,
                "requires_key": bool(requires_key),
                "models": models or {},
            }
            for slug, provider_id, url, api_style, requires_key, models in rows
        }
    return _catalog_cache


def invalidate_catalog_cache() -> None:
    """Called by the write routes after they change `providers`."""
    global _catalog_cache
    _catalog_cache = None


def load_catalog() -> dict[str, dict[str, Any]]:
    """The bot contract: {slug: {"url": ..., "models": {model: {"variants": [...]}}}}."""
    return {
        slug: {"url": row["url"], "models": row["models"]}
        for slug, row in _provider_rows().items()
    }


def _load_connection(
    user_provider_id: str | None,
    provider_slug: str,
    model: str,
    user_id: str,
) -> tuple | None:
    """The caller's own connection to this provider, or None.

    `up.user_id = %s` is the ownership predicate and it is the only thing
    standing between one user's request and another user's funded connection:
    a foreign `user_provider_id` matches no row here instead of spending it.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            if user_provider_id is not None:
                cur.execute(
                    "SELECT up.id, up.user_id, up.base_url, up.api_style, "
                    "       up.encrypted_key, pm.variants, p.requires_key "
                    "FROM user_providers up "
                    "JOIN providers p ON p.id = up.provider_id "
                    "LEFT JOIN provider_models pm "
                    "       ON pm.user_provider_id = up.id AND pm.model_id = %s "
                    "WHERE up.id = %s AND up.user_id = %s",
                    (model, user_provider_id, user_id),
                )
            else:
                # Legacy session rows predate sessions.user_provider_id.
                cur.execute(
                    "SELECT up.id, up.user_id, up.base_url, up.api_style, "
                    "       up.encrypted_key, pm.variants, p.requires_key "
                    "FROM user_providers up "
                    "JOIN providers p ON p.id = up.provider_id "
                    "LEFT JOIN provider_models pm "
                    "       ON pm.user_provider_id = up.id AND pm.model_id = %s "
                    "WHERE p.slug = %s AND up.user_id = %s",
                    (model, provider_slug, user_id),
                )
            return cur.fetchone()


def resolve_model_config(
    user_provider_id: str | None,
    provider_slug: str,
    model: str,
    variant: str,
    user_id: str,
) -> dict[str, Any]:
    """Resolve (connection, model, variant) -> everything `stream()` needs.

    Falls back to the provider's defaults with **no key** when the user has no
    connection for it; a keyless provider (`providers.requires_key` false) can
    still be used that way, which is the old free-provider behaviour.
    """
    provider = _provider_rows().get(provider_slug)

    if provider is None:
        raise ValueError(f"Unknown model provider: {provider_slug}")

    row = _load_connection(user_provider_id, provider_slug, model, user_id)

    if row is None:
        url = provider["url"]
        api_style = provider["api_style"]
        api_key = None
        requires_key = provider["requires_key"]
        resolved_user_provider_id = None
        variants = None
    else:
        (
            resolved_user_provider_id,
            connection_user_id,
            url,
            api_style,
            encrypted_key,
            variants,
            requires_key,
        ) = row
        if str(connection_user_id) != str(user_id):
            # Unreachable: the WHERE clause above already filters by user_id.
            raise ValueError(f"Unknown model provider: {provider_slug}")
        api_key = decrypt(encrypted_key) if encrypted_key else None

    model_config = provider["models"].get(model)

    if not isinstance(model_config, dict):
        raise ValueError(f"Unknown model: {provider_slug}/{model}")

    if variants is None:
        variants = model_config.get("variants", [])

    if variant not in variants:
        raise ValueError(f"Invalid variant '{variant}' for {provider_slug}/{model}")

    return {
        "provider": provider_slug,
        "url": url,
        "model": model,
        "variant": variant,
        "user_provider_id": (
            str(resolved_user_provider_id) if resolved_user_provider_id else None
        ),
        "api_style": api_style,
        "api_key": api_key,
        "requires_key": bool(requires_key),
    }


@router.get("")
async def list_providers(
    x_bot_secret: str | None = Header(default=None, alias="X-Bot-Secret"),
) -> dict[str, Any]:
    """Return the full validated provider configuration.
    The Telegram bot calls this once at startup to build keyboards."""
    # /providers stays public to the middleware (the bot fetches the catalog
    # with X-Bot-Secret only, no X-Telegram-Id), so it keeps its own check.
    if not BOT_SHARED_SECRET or not x_bot_secret or not hmac.compare_digest(
        x_bot_secret, BOT_SHARED_SECRET
    ):
        raise HTTPException(status_code=401, detail="Unauthorized")

    return {"providers": load_catalog()}
