"""SSRF-safe handling for user-supplied provider URLs.

Deliberately standalone: no DB, no FastAPI, so it is unit-testable against
canned DNS answers (see tests/test_provider_url_safety.py). Everything here
treats the URL *and* the upstream body as hostile input:

- the URL must be ``https`` (``http`` only for a byte-identical catalog
  default, e.g. the local ``http://omniroute:20128/v1`` gateway), carry no
  userinfo, and use a port of 443 (or 80/8000/20128 under that same
  exception);
- the hostname is resolved and *every* returned address must be public, so a
  name that answers with both ``1.1.1.1`` and ``10.0.0.5`` is rejected
  (DNS-rebinding defence); link-local/metadata/multicast/reserved stay
  rejected even when ``allow_insecure`` relaxes the private ranges;
- the response is streamed and capped at 1 MiB, and callers only ever get
  ``(status, parsed_or_None)`` -- never the raw upstream body.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
from urllib.parse import urlsplit

import httpx

MAX_BODY_BYTES = 1024 * 1024
MAX_HEADERS = 10
MAX_HEADERS_BYTES = 2 * 1024
HEADER_NAME_RE = re.compile(r"^[A-Za-z0-9-]{1,64}$")

# Headers that would let a user rewrite the request target or impersonate the
# backend's own trusted callers.
DENIED_HEADERS = frozenset(
    {
        "host",
        "content-length",
        "transfer-encoding",
        "connection",
        "cookie",
        "x-forwarded-for",
        "x-bot-secret",
        "x-telegram-id",
        "x-user-id",
    }
)

# ponytail: a fixed port allowlist, not a range. 8000/20128 exist because the
# docker-compose omniroute gateway listens on them; widen only with a reason.
INSECURE_PORTS = frozenset({80, 8000, 20128})

MSG_UNSAFE_URL = "رابط المزوّد غير آمن. استخدم رابط https عامًا."
MSG_BAD_HEADERS = "رؤوس الطلب الإضافية غير صالحة."
MSG_BODY_TOO_LARGE = "استجابة المزوّد أكبر من الحد المسموح."


class UnsafeProviderUrl(ValueError):
    """The URL (or its resolution) is not safe to fetch.

    The message is deliberately generic and never echoes the URL back: the
    caller turns it into a 400 and the URL goes nowhere near the response.
    """


class UpstreamBodyTooLarge(ValueError):
    """The provider streamed more than MAX_BODY_BYTES."""


def insecure_allowed_by_env() -> bool:
    """The deployment-level opt-in for the catalog's own insecure default."""
    return os.environ.get("ALLOW_INSECURE_PROVIDER_URLS") == "1"


def allow_insecure_for(url: str, default_base_url: str | None) -> bool:
    """True only for a byte-identical catalog default with the env opt-in.

    That equality is the whole SSRF exception: a user cannot reach
    ``http://omniroute:20128/v1`` by typing something that merely resembles
    it, and no user-supplied URL ever gets the relaxed checks.
    """
    if not default_base_url or not insecure_allowed_by_env():
        return False
    return url == default_base_url


def assert_safe_provider_url(url: str, *, allow_insecure: bool) -> None:
    """Raise UnsafeProviderUrl unless this URL may be fetched."""
    if not isinstance(url, str) or not url:
        raise UnsafeProviderUrl(MSG_UNSAFE_URL)

    parts = urlsplit(url)
    scheme_ok = parts.scheme == "https" or (allow_insecure and parts.scheme == "http")
    if not scheme_ok or "@" in parts.netloc:
        raise UnsafeProviderUrl(MSG_UNSAFE_URL)

    host = parts.hostname
    if not host:
        raise UnsafeProviderUrl(MSG_UNSAFE_URL)

    try:
        port = parts.port
    except ValueError:
        raise UnsafeProviderUrl(MSG_UNSAFE_URL) from None
    port = port or (443 if parts.scheme == "https" else 80)
    if port != 443 and not (allow_insecure and port in INSECURE_PORTS):
        raise UnsafeProviderUrl(MSG_UNSAFE_URL)

    for address in _resolve(host, port):
        if _is_unsafe_address(address, allow_insecure=allow_insecure):
            raise UnsafeProviderUrl(MSG_UNSAFE_URL)


def _resolve(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError):
        raise UnsafeProviderUrl(MSG_UNSAFE_URL) from None
    return [info[4][0] for info in infos]


def _is_unsafe_address(address: str, *, allow_insecure: bool) -> bool:
    """Reject anything that is not safe to fetch.

    Link-local (which includes the cloud metadata endpoint 169.254.169.254),
    multicast, reserved and unspecified are always rejected. Loopback,
    RFC1918 and IPv6 ULA (fc00::/7) are rejected unless ``allow_insecure``,
    which is reachable *only* for a byte-identical catalog default: the local
    gateway (``http://omniroute:20128/v1``) resolves to a docker-internal
    private address, so the exception has to cover that too or it is inert.
    A literal IP reaches this check through ``getaddrinfo`` like any host.
    """
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return True
    if ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        return True
    if ip.is_loopback or ip.is_private or getattr(ip, "is_site_local", False):
        return not allow_insecure
    return False


def sanitize_extra_headers(raw: dict | None) -> dict[str, str]:
    """Validate user-supplied extra headers, or raise UnsafeProviderUrl."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise UnsafeProviderUrl(MSG_BAD_HEADERS)
    if len(raw) > MAX_HEADERS:
        raise UnsafeProviderUrl(MSG_BAD_HEADERS)

    cleaned: dict[str, str] = {}
    total = 0
    for name, value in raw.items():
        if (
            not isinstance(name, str)
            or not HEADER_NAME_RE.match(name)
            or name.lower() in DENIED_HEADERS
        ):
            raise UnsafeProviderUrl(MSG_BAD_HEADERS)
        if not isinstance(value, str) or "\r" in value or "\n" in value:
            raise UnsafeProviderUrl(MSG_BAD_HEADERS)
        total += len(name.encode("utf-8")) + len(value.encode("utf-8"))
        if total > MAX_HEADERS_BYTES:
            raise UnsafeProviderUrl(MSG_BAD_HEADERS)
        cleaned[name] = value
    return cleaned


def safe_get_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | float | None = None,
) -> tuple[int, object | None]:
    """GET a provider URL, returning ``(status, parsed_json_or_None)``.

    Redirects are not followed (a 3xx is just a status the caller rejects),
    the body is streamed and capped, and the parsed body never leaves this
    process in a response -- callers log details server-side only.
    """
    if timeout is None:
        timeout = httpx.Timeout(15.0, connect=5.0)

    with httpx.Client(follow_redirects=False, timeout=timeout) as client:
        with client.stream("GET", url, headers=headers or {}) as response:
            status = response.status_code
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > MAX_BODY_BYTES:
                    raise UpstreamBodyTooLarge(MSG_BODY_TOO_LARGE)

    parsed: object | None = None
    if status < 300:
        try:
            parsed = json.loads(bytes(body))
        except ValueError:
            parsed = None
    return status, parsed
