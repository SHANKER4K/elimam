"""Authenticated per-user provider connections and SSRF-safe model discovery.

Ownership: the caller is always the authenticated principal
(``identity.current_user_id``, resolved by the middleware), never a body or
path user id, and every ``user_providers`` lookup/write carries
``AND user_id = %s``. A connection id belonging to somebody else therefore
matches no row and becomes a **404, not a 403** (a 403 would confirm that the
row exists).

``base_url`` semantics: stored exactly as typed -- the provider's API root,
which normally already ends in ``/v1`` -- and model discovery appends
``/models``. The four seeded providers' URLs already include ``/v1``, which is
why the path is ``/models`` and not ``/v1/models``.

Keys: ``encrypted_key`` is read (under the ownership predicate) only to build
an upstream request and is never serialized; responses expose ``hasKey``.

The DB pool is imported lazily inside the data-access functions below, so this
module imports (and is monkeypatchable) without ``DATABASE_*`` env or a live
Postgres. Routes call only those functions -- never ``get_conn`` directly --
which is what keeps the live database out of the test path.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from identity import current_user_id
from provider_url import (
    UnsafeProviderUrl,
    UpstreamBodyTooLarge,
    allow_insecure_for,
    assert_safe_provider_url,
    safe_get_json,
    sanitize_extra_headers,
)

logger = logging.getLogger("me_providers")

router = APIRouter(prefix="/me/providers", tags=["My Providers"])

API_STYLES = ("openai_compatible", "anthropic_compatible")
ANTHROPIC_VERSION = "2023-06-01"

MSG_UNKNOWN_PROVIDER = "المزوّد غير موجود."
MSG_CONNECTION_NOT_FOUND = "الاتصال غير موجود."
MSG_MODEL_NOT_FOUND = "النموذج غير موجود."
MSG_KEY_REQUIRED = "هذا المزوّد يتطلب مفتاح API."
MSG_INVALID_KEY = "المفتاح غير صالح أو غير مصرّح به."
MSG_DISCOVERY_FAILED = "تعذّر جلب قائمة النماذج من المزوّد."
MSG_UNREACHABLE = "تعذّر الاتصال بمزوّد الخدمة."
MSG_BAD_API_STYLE = "نوع واجهة المزوّد غير مدعوم."
MSG_NOTHING_TO_UPDATE = "لا يوجد أي تغيير في الطلب."


def require_principal(request: Request) -> str:
    """The authenticated user id, never a body/path value.

    ``current_user_id`` without a legacy argument raises 401 when no identity
    was resolved even while ``AUTH_MODE=permissive``, which is what keeps the
    rollout exemption away from this new surface.
    """
    return current_user_id(request)


# ── Data access ──────────────────────────────────────────────────────────────
# Everything the routes touch. Tests monkeypatch these functions, so no real
# connection object is ever constructed outside production.

_CONNECTION_COLUMNS = (
    "up.id, up.provider_id, up.is_custom, up.custom_name, up.base_url, "
    "up.api_style, up.encrypted_key, up.extra_headers, up.last_validated_at, "
    "up.created_at, up.updated_at, p.slug, p.name, p.default_base_url, "
    "p.requires_key, p.default_variants"
)

_MODEL_COLUMNS = (
    "pm.user_provider_id, pm.model_id, pm.display_name, pm.variants, pm.is_custom, "
    "pm.enabled, pm.context_window, pm.supports_tools, pm.supports_vision, "
    "pm.last_synced_at"
)

_UPDATE_COLUMNS = {
    "base_url": "base_url = %s",
    "api_style": "api_style = %s",
    "custom_name": "custom_name = %s",
    "encrypted_key": "encrypted_key = %s",
    "extra_headers": "extra_headers = %s",
}


def _conn():
    from db.connection import get_conn

    return get_conn()


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return {}
    return {}


def _as_json(value: Any) -> str:
    return json.dumps(value or {})


def _row_to_connection(row) -> dict[str, Any]:
    return {
        "id": str(row[0]),
        "provider_id": row[1],
        "is_custom": bool(row[2]),
        "custom_name": row[3],
        "base_url": row[4],
        "api_style": row[5],
        "has_key": row[6] is not None,
        "extra_headers": _as_dict(row[7]),
        "last_validated_at": row[8],
        "created_at": row[9],
        "updated_at": row[10],
        "slug": row[11],
        "provider_name": row[12],
        "default_base_url": row[13],
        "requires_key": None if row[14] is None else bool(row[14]),
        "default_variants": list(row[15] or []),
    }


def _row_to_model(row) -> dict[str, Any]:
    return {
        "model_id": row[1],
        "display_name": row[2],
        "variants": list(row[3] or []),
        "is_custom": bool(row[4]),
        "enabled": bool(row[5]),
        "context_window": row[6],
        "supports_tools": row[7],
        "supports_vision": row[8],
        "last_synced_at": row[9],
    }


def _catalog_rows() -> list[dict[str, Any]]:
    """The provider catalog, read straight from `providers` (read-only here)."""
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, slug, name, api_style, default_base_url, logo_url, "
                "default_variants, requires_key, docs_url, models "
                "FROM providers ORDER BY id"
            )
            rows = cur.fetchall()
        conn.commit()
    return [
        {
            "id": r[0],
            "slug": r[1],
            "name": r[2],
            "api_style": r[3],
            "default_base_url": r[4],
            "logo_url": r[5],
            "default_variants": list(r[6] or []),
            "requires_key": bool(r[7]),
            "docs_url": r[8],
            "models": _as_dict(r[9]),
        }
        for r in rows
    ]


def _catalog_by_id() -> dict[int, dict[str, Any]]:
    return {row["id"]: row for row in _catalog_rows()}


def _list_connections(user_id: str) -> list[dict[str, Any]]:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {_CONNECTION_COLUMNS} FROM user_providers up "
                "LEFT JOIN providers p ON p.id = up.provider_id "
                "WHERE up.user_id = %s ORDER BY up.created_at",
                (user_id,),
            )
            rows = cur.fetchall()
        conn.commit()
    return [_row_to_connection(row) for row in rows]


def _get_connection(user_id: str, connection_id: str) -> dict[str, Any] | None:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {_CONNECTION_COLUMNS} FROM user_providers up "
                "LEFT JOIN providers p ON p.id = up.provider_id "
                "WHERE up.id = %s AND up.user_id = %s",
                (connection_id, user_id),
            )
            row = cur.fetchone()
        conn.commit()
    return _row_to_connection(row) if row else None


def _encrypted_key(user_id: str, connection_id: str) -> str | None:
    """The stored ciphertext, read under the ownership predicate.

    Kept out of the connection dict on purpose: nothing that is not explicitly
    serialized can leak into a response.
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT encrypted_key FROM user_providers "
                "WHERE id = %s AND user_id = %s",
                (connection_id, user_id),
            )
            row = cur.fetchone()
        conn.commit()
    return row[0] if row else None


def _provider_models(
    user_id: str, connection_id: str | None = None
) -> dict[str, list[dict[str, Any]]]:
    """`{connection_id: [models]}` for this user (one connection when given)."""
    query = (
        f"SELECT {_MODEL_COLUMNS} FROM provider_models pm "
        "JOIN user_providers up ON up.id = pm.user_provider_id "
        "WHERE up.user_id = %s"
    )
    params: list[Any] = [user_id]
    if connection_id is not None:
        query += " AND up.id = %s"
        params.append(connection_id)
    query += " ORDER BY pm.model_id"

    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
        conn.commit()

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[0]), []).append(_row_to_model(row))
    return grouped


def _upsert_connection(
    user_id: str,
    provider_id: int | None,
    *,
    is_custom: bool,
    custom_name: str | None,
    base_url: str,
    api_style: str,
    encrypted_key: str | None,
    extra_headers: dict[str, Any],
) -> str:
    """Connect (or re-connect) the user's single row for a builtin provider.

    A custom row has ``provider_id IS NULL``, so the unique index never
    conflicts for it and the same statement covers both cases.
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_providers
                    (user_id, provider_id, is_custom, custom_name, base_url,
                     api_style, encrypted_key, extra_headers)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, provider_id) DO UPDATE
                    SET base_url = EXCLUDED.base_url,
                        api_style = EXCLUDED.api_style,
                        custom_name = COALESCE(EXCLUDED.custom_name, user_providers.custom_name),
                        encrypted_key = COALESCE(EXCLUDED.encrypted_key, user_providers.encrypted_key),
                        extra_headers = EXCLUDED.extra_headers,
                        updated_at = now()
                RETURNING id
                """,
                (
                    user_id,
                    provider_id,
                    is_custom,
                    custom_name,
                    base_url,
                    api_style,
                    encrypted_key,
                    _as_json(extra_headers),
                ),
            )
            row = cur.fetchone()
        conn.commit()
    return str(row[0])


def _update_connection(user_id: str, connection_id: str, fields: dict[str, Any]) -> bool:
    """Patch whitelisted columns; the ownership predicate is in the WHERE."""
    assignments = [_UPDATE_COLUMNS[key] for key in fields]
    values = [
        _as_json(fields[key]) if key == "extra_headers" else fields[key]
        for key in fields
    ]
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE user_providers SET "
                + ", ".join(assignments)
                + ", updated_at = now() WHERE id = %s AND user_id = %s",
                tuple(values) + (connection_id, user_id),
            )
            changed = cur.rowcount
        conn.commit()
    return bool(changed)


def _discard_connection(user_id: str, connection_id: str) -> None:
    """Undo a create whose discovery step failed.

    A 400 never carries the connection id, so a row saved before the failure
    can never be cleaned up by the client: it just sits in the settings page as
    a provider that does not work. Models cascade away with it.

    ponytail: reuse ``_delete_connection`` (ownership predicate included)
    rather than a second DELETE statement; a failure here is logged and leaves
    the row, which is no worse than the old behaviour.
    """
    try:
        _delete_connection(user_id, connection_id)
    except Exception:
        logger.exception("failed to discard connection after a failed create")


def _delete_connection(user_id: str, connection_id: str) -> bool:
    """Delete the user's own connection; provider_models cascade."""
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM user_providers WHERE id = %s AND user_id = %s",
                (connection_id, user_id),
            )
            changed = cur.rowcount
        conn.commit()
    return bool(changed)


def _sync_models(
    user_id: str,
    connection_id: str,
    model_ids: list[str],
    variants: list[str],
    enabled: bool = True,
) -> int:
    """Upsert discovered/catalog models without touching existing rows.

    The conflict clause only refreshes ``last_synced_at``: a re-sync must never
    overwrite the user's ``variants``, nor silently re-enable a model they
    disabled. The INSERT..SELECT carries the ownership predicate.

    ``enabled`` applies to newly inserted rows only. Discovery passes False for
    a builtin provider: ``opencode`` alone advertises 65 models, most unusable
    on a free account, and enabling them all turns a curated six-model picker
    into a seventy-model one on a single Sync click. A custom endpoint has no
    catalog to fall back on, so its discoveries arrive enabled.
    """
    if not model_ids:
        return 0
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO provider_models
                    (user_provider_id, model_id, display_name, variants, is_custom,
                     enabled, last_synced_at)
                SELECT up.id, %s, %s, %s, false, %s, now()
                FROM user_providers up
                WHERE up.id = %s AND up.user_id = %s
                ON CONFLICT (user_provider_id, model_id) DO UPDATE
                    SET last_synced_at = now()
                """,
                [
                    (model_id, model_id, variants, enabled, connection_id, user_id)
                    for model_id in model_ids
                ],
            )
            changed = cur.rowcount
        conn.commit()
    return int(changed or 0)


def _add_model(
    user_id: str,
    connection_id: str,
    model_id: str,
    display_name: str,
    variants: list[str],
) -> bool:
    """Manual add; re-adding a soft-deleted model re-enables it."""
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO provider_models
                    (user_provider_id, model_id, display_name, variants, is_custom,
                     enabled)
                SELECT up.id, %s, %s, %s, true, true
                FROM user_providers up
                WHERE up.id = %s AND up.user_id = %s
                ON CONFLICT (user_provider_id, model_id) DO UPDATE
                    SET enabled = true,
                        is_custom = true,
                        display_name = EXCLUDED.display_name,
                        variants = EXCLUDED.variants
                """,
                (model_id, display_name, variants, connection_id, user_id),
            )
            changed = cur.rowcount
        conn.commit()
    return bool(changed)


def _update_model(
    user_id: str,
    connection_id: str,
    model_id: str,
    *,
    enabled: bool | None,
    variants: list[str] | None,
) -> bool:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE provider_models pm
                SET enabled = COALESCE(%s, pm.enabled),
                    variants = COALESCE(%s, pm.variants)
                FROM user_providers up
                WHERE up.id = pm.user_provider_id
                  AND up.id = %s AND up.user_id = %s AND pm.model_id = %s
                """,
                (enabled, variants, connection_id, user_id, model_id),
            )
            changed = cur.rowcount
        conn.commit()
    return bool(changed)


def _soft_delete_model(user_id: str, connection_id: str, model_id: str) -> bool:
    """Soft delete only: session history must keep resolving the model."""
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE provider_models pm
                SET enabled = false
                FROM user_providers up
                WHERE up.id = pm.user_provider_id
                  AND up.id = %s AND up.user_id = %s AND pm.model_id = %s
                """,
                (connection_id, user_id, model_id),
            )
            changed = cur.rowcount
        conn.commit()
    return bool(changed)


def _mark_validated(user_id: str, connection_id: str) -> bool:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE user_providers SET last_validated_at = now() "
                "WHERE id = %s AND user_id = %s",
                (connection_id, user_id),
            )
            changed = cur.rowcount
        conn.commit()
    return bool(changed)


# ── Serialization ────────────────────────────────────────────────────────────


def _iso(value: Any) -> str | None:
    return value.isoformat() if value else None


def _model_out(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "modelId": model["model_id"],
        "displayName": model["display_name"],
        "variants": model["variants"],
        "isCustom": model["is_custom"],
        "enabled": model["enabled"],
        "contextWindow": model["context_window"],
        "supportsTools": model["supports_tools"],
        "supportsVision": model["supports_vision"],
        "lastSyncedAt": _iso(model["last_synced_at"]),
    }


def _catalog_out(provider: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": provider["id"],
        "slug": provider["slug"],
        "name": provider["name"],
        "apiStyle": provider["api_style"],
        "defaultBaseUrl": provider["default_base_url"],
        "defaultVariants": provider["default_variants"],
        "requiresKey": provider["requires_key"],
        "logoUrl": provider["logo_url"],
        "docsUrl": provider["docs_url"],
        "models": provider["models"],
    }


def _out(
    connection: dict[str, Any], models_by_connection: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """One connection as the UI sees it: `hasKey`, never the key itself."""
    return {
        "id": connection["id"],
        "providerId": connection["provider_id"],
        "slug": connection["slug"],
        "name": connection["custom_name"] or connection["provider_name"] or "",
        "isCustom": connection["is_custom"],
        "baseUrl": connection["base_url"],
        "apiStyle": connection["api_style"],
        "hasKey": connection["has_key"],
        "requiresKey": bool(connection["requires_key"]),
        "extraHeaders": connection["extra_headers"],
        "lastValidatedAt": _iso(connection["last_validated_at"]),
        "createdAt": _iso(connection["created_at"]),
        "updatedAt": _iso(connection["updated_at"]),
        "models": [
            _model_out(model)
            for model in models_by_connection.get(connection["id"], [])
        ],
    }


def _require_connection(user_id: str, connection_id: str) -> dict[str, Any]:
    connection = _get_connection(user_id, connection_id)
    if connection is None:
        # 404, not 403: another user's id must not be confirmed to exist.
        raise HTTPException(status_code=404, detail=MSG_CONNECTION_NOT_FOUND)
    return connection


def _find_model(models: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    for model in models:
        if model["model_id"] == model_id:
            return _model_out(model)
    raise HTTPException(status_code=404, detail=MSG_MODEL_NOT_FOUND)


# ── Discovery ────────────────────────────────────────────────────────────────


def _encrypt(api_key: str | None) -> str | None:
    if not api_key:
        return None
    from api.keys import encrypt  # lazy: api.keys builds the DB pool at import

    return encrypt(api_key)


def _decrypt(stored: str | None) -> str | None:
    if not stored:
        return None
    from api.keys import decrypt

    return decrypt(stored)


def _clean_headers(raw: Any) -> dict[str, str]:
    try:
        return sanitize_extra_headers(raw)
    except UnsafeProviderUrl as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _validate_url(url: str, default_base_url: str | None) -> None:
    """Reject user-supplied URLs that are not safe to fetch.

    ``allow_insecure`` is true only for a byte-identical catalog default, so an
    override -- including one that merely resembles the default -- is strict.
    """
    try:
        assert_safe_provider_url(
            url, allow_insecure=allow_insecure_for(url, default_base_url)
        )
    except UnsafeProviderUrl as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _default_variants(connection: dict[str, Any]) -> list[str]:
    return connection["default_variants"] or ["low"]


def _discovery_headers(
    api_style: str, api_key: str | None, extra_headers: dict[str, Any]
) -> dict[str, str]:
    headers = _clean_headers(extra_headers)
    if api_style == "anthropic_compatible":
        headers.setdefault("anthropic-version", ANTHROPIC_VERSION)
        if api_key:
            headers["x-api-key"] = api_key
    elif api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _fetch_models(url: str, headers: dict[str, str]) -> tuple[int, object | None]:
    """The only network call in this module (monkeypatched in tests)."""
    return safe_get_json(url, headers=headers)


def _discover(connection: dict[str, Any], api_key: str | None) -> tuple[list[str], bool]:
    """`(model ids, discovery_supported)` from `{base_url}/models`.

    Raises HTTPException(400) for an unsafe URL, an unauthorised key, an
    unreachable endpoint or a payload that is not `{data: [...]}`. A 404 means
    the endpoint has no discovery and is reported as such, not as an error.
    """
    base_url = connection["base_url"]
    url = base_url.rstrip("/") + "/models"
    try:
        assert_safe_provider_url(
            url,
            allow_insecure=allow_insecure_for(
                base_url, connection["default_base_url"]
            ),
        )
    except UnsafeProviderUrl as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    headers = _discovery_headers(
        connection["api_style"], api_key, connection["extra_headers"]
    )
    try:
        status, payload = _fetch_models(url, headers)
    except (UnsafeProviderUrl, UpstreamBodyTooLarge, httpx.HTTPError) as exc:
        logger.warning(
            "provider_discovery_unreachable",
            extra={"connection_id": connection["id"], "error": str(exc)},
        )
        raise HTTPException(status_code=400, detail=MSG_UNREACHABLE) from exc

    if status in (401, 403):
        logger.info(
            "provider_discovery_rejected",
            extra={"connection_id": connection["id"], "status": status},
        )
        raise HTTPException(status_code=400, detail=MSG_INVALID_KEY)
    if status == 404:
        return [], False
    if status != 200:
        logger.warning(
            "provider_discovery_status",
            extra={"connection_id": connection["id"], "status": status},
        )
        raise HTTPException(status_code=400, detail=MSG_DISCOVERY_FAILED)

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        logger.warning(
            "provider_discovery_payload", extra={"connection_id": connection["id"]}
        )
        raise HTTPException(status_code=400, detail=MSG_DISCOVERY_FAILED)

    model_ids = [
        str(item["id"]) for item in data if isinstance(item, dict) and item.get("id")
    ]
    return model_ids, True


def _sync_discovered_models(
    user_id: str, connection: dict[str, Any], api_key: str | None
) -> tuple[list[str], bool]:
    model_ids, supported = _discover(connection, api_key)
    if model_ids:
        _sync_models(
            user_id,
            connection["id"],
            model_ids,
            _default_variants(connection),
            enabled=bool(connection["is_custom"]),
        )
    if supported:
        _mark_validated(user_id, connection["id"])
    return model_ids, supported


# ── Request bodies ───────────────────────────────────────────────────────────


class ConnectIn(BaseModel):
    provider_id: int
    api_key: str | None = None
    base_url: str | None = None
    extra_headers: dict[str, str] | None = None


class CustomProviderIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    base_url: str
    api_style: str
    api_key: str | None = None
    extra_headers: dict[str, str] | None = None


class ConnectionPatch(BaseModel):
    base_url: str | None = None
    api_key: str | None = None
    extra_headers: dict[str, str] | None = None
    custom_name: str | None = Field(default=None, min_length=1, max_length=80)


class ModelIn(BaseModel):
    model_id: str = Field(min_length=1, max_length=200)
    display_name: str | None = None
    variants: list[str] | None = None


class ModelPatch(BaseModel):
    enabled: bool | None = None
    variants: list[str] | None = None


# ── Routes ───────────────────────────────────────────────────────────────────


@router.get("")
def list_my_providers(user_id: str = Depends(require_principal)) -> dict[str, Any]:
    """The catalog plus this user's connections (each with its models)."""
    models = _provider_models(user_id)
    return {
        "catalog": [_catalog_out(row) for row in _catalog_rows()],
        "connections": [_out(c, models) for c in _list_connections(user_id)],
    }


@router.post("")
def connect_provider(
    body: ConnectIn, user_id: str = Depends(require_principal)
) -> dict[str, Any]:
    """Connect a builtin provider, seeding the catalog's models and then
    discovering the endpoint's own list.

    The connection is saved before discovery runs, but a discovery failure
    deletes it again before the error is raised, so a wrong key never leaves a
    phantom provider behind. Models discovery finds that are not in the
    catalog are stored disabled (see ``_sync_models``).
    """
    provider = _catalog_by_id().get(body.provider_id)
    if provider is None:
        raise HTTPException(status_code=404, detail=MSG_UNKNOWN_PROVIDER)

    api_key = body.api_key or None
    if provider["requires_key"] and not api_key:
        raise HTTPException(status_code=400, detail=MSG_KEY_REQUIRED)

    base_url = body.base_url or provider["default_base_url"]
    _validate_url(base_url, provider["default_base_url"])
    connection_id = _upsert_connection(
        user_id,
        provider["id"],
        is_custom=False,
        custom_name=None,
        base_url=base_url,
        api_style=provider["api_style"],
        encrypted_key=_encrypt(api_key),
        extra_headers=_clean_headers(body.extra_headers),
    )

    catalog_models = list(provider["models"] or {})
    if catalog_models:
        _sync_models(
            user_id,
            connection_id,
            catalog_models,
            provider["default_variants"] or ["low"],
        )

    connection = _require_connection(user_id, connection_id)
    try:
        _sync_discovered_models(user_id, connection, api_key)
    except HTTPException:
        _discard_connection(user_id, connection_id)
        raise
    return _out(connection, _provider_models(user_id, connection_id))


@router.post("/custom")
def add_custom_provider(
    body: CustomProviderIn, user_id: str = Depends(require_principal)
) -> dict[str, Any]:
    """Add a custom OpenAI/Anthropic-compatible endpoint (strict URL checks:
    no catalog row exists, so there is nothing to be byte-identical to)."""
    if body.api_style not in API_STYLES:
        raise HTTPException(status_code=400, detail=MSG_BAD_API_STYLE)

    _validate_url(body.base_url, None)
    api_key = body.api_key or None
    connection_id = _upsert_connection(
        user_id,
        None,
        is_custom=True,
        custom_name=body.name,
        base_url=body.base_url,
        api_style=body.api_style,
        encrypted_key=_encrypt(api_key),
        extra_headers=_clean_headers(body.extra_headers),
    )

    connection = _require_connection(user_id, connection_id)
    try:
        _sync_discovered_models(user_id, connection, api_key)
    except HTTPException:
        _discard_connection(user_id, connection_id)
        raise
    return _out(connection, _provider_models(user_id, connection_id))


@router.patch("/{connection_id}")
def update_connection(
    connection_id: UUID,
    body: ConnectionPatch,
    user_id: str = Depends(require_principal),
) -> dict[str, Any]:
    """Change the URL, key, headers or custom name. A changed URL is
    re-validated; `api_key: null` clears the stored key."""
    connection = _require_connection(user_id, str(connection_id))
    provided = body.model_dump(exclude_unset=True)
    if not provided:
        raise HTTPException(status_code=400, detail=MSG_NOTHING_TO_UPDATE)

    fields: dict[str, Any] = {}
    if "base_url" in provided:
        base_url = provided["base_url"] or ""
        _validate_url(base_url, connection["default_base_url"])
        fields["base_url"] = base_url
    if "api_key" in provided:
        fields["encrypted_key"] = _encrypt(provided["api_key"])
    if "extra_headers" in provided:
        fields["extra_headers"] = _clean_headers(provided["extra_headers"])
    if provided.get("custom_name"):
        fields["custom_name"] = provided["custom_name"]
    if not fields:
        raise HTTPException(status_code=400, detail=MSG_NOTHING_TO_UPDATE)

    if not _update_connection(user_id, str(connection_id), fields):
        raise HTTPException(status_code=404, detail=MSG_CONNECTION_NOT_FOUND)

    return _out(
        _require_connection(user_id, str(connection_id)),
        _provider_models(user_id, str(connection_id)),
    )


@router.delete("/{connection_id}")
def delete_connection(
    connection_id: UUID, user_id: str = Depends(require_principal)
) -> dict[str, Any]:
    """Delete the user's own connection; its provider_models cascade."""
    if not _delete_connection(user_id, str(connection_id)):
        raise HTTPException(status_code=404, detail=MSG_CONNECTION_NOT_FOUND)
    return {"id": str(connection_id), "deleted": True}


@router.post("/{connection_id}/sync")
def sync_connection(
    connection_id: UUID, user_id: str = Depends(require_principal)
) -> dict[str, Any]:
    """Discover the endpoint's models and store them.

    Never deletes stale rows, and never overwrites an existing row's
    `variants` or `enabled` -- only `last_synced_at`. A provider that answers
    HTTP 404 gets `{"models": [], "discovery_supported": false}` so the user can add
    models by hand.
    """
    connection = _require_connection(user_id, str(connection_id))
    api_key = _decrypt(_encrypted_key(user_id, str(connection_id)))
    _, supported = _sync_discovered_models(user_id, connection, api_key)
    if not supported:
        return {"models": [], "discovery_supported": False}
    models = _provider_models(user_id, str(connection_id)).get(str(connection_id), [])
    return {
        "models": [_model_out(model) for model in models],
        "discovery_supported": True,
    }


@router.post("/{connection_id}/models")
def add_model(
    connection_id: UUID,
    body: ModelIn,
    user_id: str = Depends(require_principal),
) -> dict[str, Any]:
    """Manual model add (`is_custom = true`, variants default `["low"]`)."""
    _require_connection(user_id, str(connection_id))
    _add_model(
        user_id,
        str(connection_id),
        body.model_id,
        body.display_name or body.model_id,
        list(body.variants or ["low"]),
    )
    models = _provider_models(user_id, str(connection_id)).get(str(connection_id), [])
    return _find_model(models, body.model_id)


@router.patch("/{connection_id}/models/{model_id}")
def update_model(
    connection_id: UUID,
    model_id: str,
    body: ModelPatch,
    user_id: str = Depends(require_principal),
) -> dict[str, Any]:
    """Enable/disable a model or change its variants."""
    provided = body.model_dump(exclude_unset=True)
    if not provided:
        raise HTTPException(status_code=400, detail=MSG_NOTHING_TO_UPDATE)

    _require_connection(user_id, str(connection_id))
    variants = provided.get("variants")
    changed = _update_model(
        user_id,
        str(connection_id),
        model_id,
        enabled=provided.get("enabled"),
        variants=list(variants) if variants is not None else None,
    )
    if not changed:
        raise HTTPException(status_code=404, detail=MSG_MODEL_NOT_FOUND)
    models = _provider_models(user_id, str(connection_id)).get(str(connection_id), [])
    return _find_model(models, model_id)


@router.delete("/{connection_id}/models/{model_id}")
def delete_model(
    connection_id: UUID,
    model_id: str,
    user_id: str = Depends(require_principal),
) -> dict[str, Any]:
    """Soft delete: `enabled = false` only, so session history keeps resolving."""
    _require_connection(user_id, str(connection_id))
    if not _soft_delete_model(user_id, str(connection_id), model_id):
        raise HTTPException(status_code=404, detail=MSG_MODEL_NOT_FOUND)
    return {"modelId": model_id, "enabled": False}
