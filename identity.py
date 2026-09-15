"""Server-side identity resolution for every web-facing route.

Deliberately a pure module: no psycopg2, no models, no Qdrant, so it is
unit-testable without the stack (see AGENTS.md "import-time side effects").
The DB lookup the bot path needs is injected, not imported.

Two callers are trusted:

- the Next.js server, which resolves your Better Auth session and signs it::

      X-User-Id:        <users.id>
      X-User-Timestamp: <unix seconds, string>
      X-User-Signature: hex(hmac_sha256(USER_SHARED_SECRET, f"{id}:{ts}"))

- the Telegram bot, which sends ``X-Bot-Secret`` + ``X-Telegram-Id``.

Nothing else gets in: no route reads a user id from a request body for
authorization any more.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from dataclasses import dataclass
from typing import Callable, Literal, Mapping

from fastapi import HTTPException, Request

logger = logging.getLogger("identity")

SIGNATURE_MAX_AGE_SECONDS = 300

USER_ID_HEADER = "X-User-Id"
USER_TS_HEADER = "X-User-Timestamp"
USER_SIG_HEADER = "X-User-Signature"
BOT_SECRET_HEADER = "X-Bot-Secret"
TELEGRAM_ID_HEADER = "X-Telegram-Id"

# ponytail: "permissive" is a temporary rollout aid, not a supported mode: it
# lets pre-migration clients keep using their body/path user_id (with a
# warning) until the frontend proxy is live, then set AUTH_MODE=enforce.
AUTH_MODE = os.environ.get("AUTH_MODE", "permissive")

# Liveness/docs only. Everything else a client can reach must carry an
# identity; tests/test_route_coverage.py enforces that for api/*.py.
PUBLIC_PATHS = frozenset(
    {
        "/chat",
        "/chat/health",
        "/providers",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/favicon.ico",
    }
)

PROTECTED_PREFIXES = (
    "/chat/web",
    "/chat/dense_search",
    "/chat/sparse_search",
    "/chat/hybrid_search",
    "/keys",
    "/sessions",
    "/users",
    "/messages",
    "/me",
    "/search",
)


@dataclass(frozen=True)
class Identity:
    user_id: str
    source: Literal["user", "bot"]


class Unauthorized(Exception):
    """No usable identity was presented."""


class Forbidden(Exception):
    """Identity presented, but not allowed to act on the requested target."""


def sign(user_id: str, timestamp: str, secret: str) -> str:
    """The exact hex digest the Next.js proxy must produce."""
    msg = f"{user_id}:{timestamp}".encode()
    return hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()


def _lower(headers: Mapping[str, str]) -> dict[str, str]:
    return {k.lower(): v for k, v in headers.items()}


def resolve_identity(
    headers: Mapping[str, str],
    *,
    now: float,
    user_secret: str,
    bot_secret: str,
    lookup_telegram_user: Callable[[str], str],
) -> Identity:
    """Headers -> Identity, or raise Unauthorized.

    A user signature wins when present (precedence is pinned by a test).
    """
    h = _lower(headers)

    signature = h.get(USER_SIG_HEADER.lower())
    if signature:
        user_id = h.get(USER_ID_HEADER.lower())
        timestamp = h.get(USER_TS_HEADER.lower())
        if not user_id or not timestamp:
            raise Unauthorized("signature without user id/timestamp")
        try:
            ts = int(timestamp)
        except (TypeError, ValueError):
            raise Unauthorized("non-numeric timestamp") from None
        if abs(now - ts) > SIGNATURE_MAX_AGE_SECONDS:
            raise Unauthorized("stale signature")
        if not user_secret or not hmac.compare_digest(
            sign(user_id, timestamp, user_secret), signature
        ):
            raise Unauthorized("bad signature")
        return Identity(user_id=user_id, source="user")

    bot_secret_header = h.get(BOT_SECRET_HEADER.lower())
    if bot_secret_header:
        telegram_id = h.get(TELEGRAM_ID_HEADER.lower())
        if not bot_secret or not hmac.compare_digest(bot_secret_header, bot_secret):
            raise Unauthorized("bad bot secret")
        if not telegram_id:
            raise Unauthorized("bot secret without telegram id")
        return Identity(user_id=lookup_telegram_user(telegram_id), source="bot")

    raise Unauthorized("no identity headers")


def is_protected_path(path: str) -> bool:
    """True when this path must carry an identity (and is not public)."""
    if path in PUBLIC_PATHS:
        return False
    return any(
        path == prefix or path.startswith(prefix + "/")
        for prefix in PROTECTED_PREFIXES
    )


def identity_of(request: Request) -> Identity | None:
    return getattr(request.state, "identity", None)


def _legacy_allowed() -> bool:
    """Permissive mode keeps pre-migration behaviour for header-less callers."""
    return AUTH_MODE != "enforce"


def require_bot(request: Request) -> Identity | None:
    """Bot-only route. Returns None for a legacy permissive caller."""
    identity = identity_of(request)
    if identity is None:
        if AUTH_MODE == "enforce":
            raise HTTPException(status_code=401, detail="Unauthorized")
        return None
    if identity.source != "bot":
        raise HTTPException(status_code=403, detail="Forbidden")
    return identity


def can_access(request: Request, target_user_id: str) -> bool:
    """True when this request may act on target_user_id.

    Applies to bot requests too: the bot authenticates as the caller it
    forwards, so it may only touch that caller's rows.
    """
    resolved = getattr(request.state, "user_id", None)
    if not resolved:
        return _legacy_allowed()
    return str(target_user_id) == str(resolved)


def require_owner(request: Request, target_user_id: str) -> str:
    """403 (401 with no identity) unless the caller may act on the target."""
    if not can_access(request, target_user_id):
        if getattr(request.state, "user_id", None):
            raise HTTPException(status_code=403, detail="Forbidden")
        raise HTTPException(status_code=401, detail="Unauthorized")
    return str(target_user_id)


def current_user_id(request: Request, legacy_user_id: str | None = None) -> str:
    """The resolved identity, or the legacy body/path id under permissive.

    Callers pass whatever user id their request used to carry, so a client
    that has not migrated yet keeps working during rollout. Under enforce the
    middleware has already rejected a header-less request on a protected path.
    """
    user_id = getattr(request.state, "user_id", None)
    if user_id:
        return str(user_id)
    if _legacy_allowed() and legacy_user_id:
        logger.warning(
            "auth_permissive_legacy_identity",
            extra={"path": request.url.path, "user_id": legacy_user_id},
        )
        return str(legacy_user_id)
    raise HTTPException(status_code=401, detail="Unauthorized")


def optional_user_id(request: Request) -> str | None:
    """For reads that had no identity at all before this plan.

    Returns None under permissive so the old unfiltered query still runs;
    callers must turn None into "match any owner", never into a 500.
    """
    user_id = getattr(request.state, "user_id", None)
    if user_id:
        return str(user_id)
    if AUTH_MODE == "enforce":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return None
