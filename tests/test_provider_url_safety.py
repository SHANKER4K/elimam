"""The only automated guard on the SSRF surface: no DB, no network.

DNS is canned so every case is deterministic; ``provider_url`` resolves
through ``socket.getaddrinfo``, so patching that one attribute is enough.
"""

import socket

import pytest

import provider_url
from provider_url import (
    DENIED_HEADERS,
    MAX_BODY_BYTES,
    UnsafeProviderUrl,
    UpstreamBodyTooLarge,
    allow_insecure_for,
    assert_safe_provider_url,
    safe_get_json,
    sanitize_extra_headers,
)


def _canned_dns(monkeypatch, *addresses):
    """Answer every lookup with these addresses (any mix of v4/v6)."""

    def _getaddrinfo(host, port, *args, **kwargs):
        infos = []
        for address in addresses:
            family = socket.AF_INET6 if ":" in address else socket.AF_INET
            infos.append(
                (family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, port, 0, 0))
            )
        return infos

    monkeypatch.setattr(socket, "getaddrinfo", _getaddrinfo)


def _assert(url, *, allow_insecure=False):
    assert_safe_provider_url(url, allow_insecure=allow_insecure)


PUBLIC_URL = "https://api.example.com/v1"


def test_http_is_rejected_without_the_catalog_exception(monkeypatch):
    _canned_dns(monkeypatch, "1.1.1.1")
    with pytest.raises(UnsafeProviderUrl):
        _assert("http://api.example.com/v1")


def test_http_is_allowed_only_with_allow_insecure(monkeypatch):
    _canned_dns(monkeypatch, "1.1.1.1")
    _assert("http://api.example.com/v1", allow_insecure=True)
    _assert("http://api.example.com:80/v1", allow_insecure=True)
    _assert("http://omniroute:20128/v1", allow_insecure=True)


def test_other_schemes_are_rejected_even_with_allow_insecure(monkeypatch):
    _canned_dns(monkeypatch, "1.1.1.1")
    for url in ("ftp://api.example.com/v1", "file:///etc/passwd", "gopher://api.example.com"):
        with pytest.raises(UnsafeProviderUrl):
            _assert(url, allow_insecure=True)


def test_userinfo_is_rejected(monkeypatch):
    _canned_dns(monkeypatch, "1.1.1.1")
    for url in ("https://user:pw@api.example.com/v1", "https://@api.example.com/v1"):
        with pytest.raises(UnsafeProviderUrl):
            _assert(url)


def test_public_port_443_is_accepted(monkeypatch):
    _canned_dns(monkeypatch, "1.1.1.1")
    _assert("https://1.1.1.1/v1")
    _assert("https://1.1.1.1:443/v1")


def test_non_443_port_is_rejected(monkeypatch):
    _canned_dns(monkeypatch, "1.1.1.1")
    with pytest.raises(UnsafeProviderUrl):
        _assert("https://api.example.com:8443/v1")


def test_insecure_ports_only_apply_to_the_catalog_exception(monkeypatch):
    _canned_dns(monkeypatch, "1.1.1.1")
    for url in ("http://api.example.com:8000/v1", "http://api.example.com:20128/v1"):
        with pytest.raises(UnsafeProviderUrl):
            _assert(url)
        _assert(url, allow_insecure=True)


@pytest.mark.parametrize(
    "address",
    ["127.0.0.1", "10.0.0.5", "192.168.1.10", "169.254.169.254", "::1", "fd00::1"],
)
def test_non_routable_addresses_are_rejected(monkeypatch, address):
    _canned_dns(monkeypatch, address)
    with pytest.raises(UnsafeProviderUrl):
        _assert(PUBLIC_URL)


def test_metadata_endpoint_is_rejected_by_literal_ip(monkeypatch):
    _canned_dns(monkeypatch, "169.254.169.254")
    with pytest.raises(UnsafeProviderUrl):
        _assert("https://169.254.169.254/latest/meta-data")


def test_a_name_resolving_to_any_private_address_is_rejected(monkeypatch):
    """DNS-rebinding defence: one bad answer poisons the whole host."""
    _canned_dns(monkeypatch, "1.1.1.1", "10.0.0.5")
    with pytest.raises(UnsafeProviderUrl):
        _assert(PUBLIC_URL)


def test_metadata_endpoint_stays_rejected_under_the_catalog_exception(monkeypatch):
    """allow_insecure relaxes private ranges, never link-local metadata."""
    _canned_dns(monkeypatch, "169.254.169.254")
    with pytest.raises(UnsafeProviderUrl):
        _assert(PUBLIC_URL, allow_insecure=True)
    _canned_dns(monkeypatch, "fe80::1")
    with pytest.raises(UnsafeProviderUrl):
        _assert(PUBLIC_URL, allow_insecure=True)


def test_the_catalog_exception_covers_a_docker_internal_host(monkeypatch):
    """The local gateway resolves inside the compose network, not publicly."""
    _canned_dns(monkeypatch, "172.18.0.4")
    with pytest.raises(UnsafeProviderUrl):
        _assert("http://omniroute:20128/v1")
    _assert("http://omniroute:20128/v1", allow_insecure=True)


def test_unresolvable_host_is_rejected(monkeypatch):
    def _raise(*args, **kwargs):
        raise socket.gaierror("nope")

    monkeypatch.setattr(socket, "getaddrinfo", _raise)
    with pytest.raises(UnsafeProviderUrl):
        _assert(PUBLIC_URL)


def test_hostname_and_port_are_mandatory(monkeypatch):
    _canned_dns(monkeypatch, "1.1.1.1")
    for url in ("https:///v1", "not a url", ""):
        with pytest.raises(UnsafeProviderUrl):
            _assert(url)


def test_allow_insecure_needs_env_opt_in_and_identical_url(monkeypatch):
    default = "http://omniroute:20128/v1"
    monkeypatch.delenv("ALLOW_INSECURE_PROVIDER_URLS", raising=False)
    assert allow_insecure_for(default, default) is False

    monkeypatch.setenv("ALLOW_INSECURE_PROVIDER_URLS", "1")
    assert allow_insecure_for(default, default) is True
    assert allow_insecure_for(default + "/", default) is False
    assert allow_insecure_for("http://evil.test:20128/v1", default) is False
    assert allow_insecure_for(default, None) is False


def test_denied_header_names_are_rejected():
    for name in DENIED_HEADERS:
        for spelling in (name, name.upper()):
            with pytest.raises(UnsafeProviderUrl):
                sanitize_extra_headers({spelling: "x"})


def test_header_values_may_not_contain_crlf():
    with pytest.raises(UnsafeProviderUrl):
        sanitize_extra_headers({"X-API-Key": "a\nb"})
    with pytest.raises(UnsafeProviderUrl):
        sanitize_extra_headers({"X-API-Key": "a\r\nX-User-Id: someone"})


def test_header_name_and_value_types_are_validated():
    with pytest.raises(UnsafeProviderUrl):
        sanitize_extra_headers({"X API Key": "v"})
    with pytest.raises(UnsafeProviderUrl):
        sanitize_extra_headers({"a" * 65: "v"})
    with pytest.raises(UnsafeProviderUrl):
        sanitize_extra_headers({"X-API-Key": 42})
    with pytest.raises(UnsafeProviderUrl):
        sanitize_extra_headers("X-API-Key: v")


def test_header_count_and_total_size_are_capped():
    ten = {f"x-h{i}": "v" for i in range(10)}
    assert len(sanitize_extra_headers(ten)) == 10
    with pytest.raises(UnsafeProviderUrl):
        sanitize_extra_headers({**ten, "x-h10": "v"})
    with pytest.raises(UnsafeProviderUrl):
        sanitize_extra_headers({"x-big": "v" * 2048})


def test_normal_provider_headers_are_accepted():
    assert sanitize_extra_headers(None) == {}
    assert sanitize_extra_headers({}) == {}
    assert sanitize_extra_headers({"anthropic-version": "2023-06-01"}) == {
        "anthropic-version": "2023-06-01"
    }
    assert sanitize_extra_headers({"X-Custom": "ok"}) == {"X-Custom": "ok"}


def _fake_client(monkeypatch, status, body, record=None):
    """Replace httpx.Client so safe_get_json can be driven without a socket."""

    class FakeResponse:
        status_code = status

        def iter_bytes(self, chunk_size=None):
            for start in range(0, len(body), 64 * 1024):
                yield body[start : start + 64 * 1024]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeClient:
        def __init__(self, **kwargs):
            if record is not None:
                record.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def stream(self, method, url, headers=None):
            if record is not None:
                record["request"] = (method, url, headers)
            return FakeResponse()

    monkeypatch.setattr(provider_url.httpx, "Client", FakeClient)


def test_safe_get_json_streams_without_following_redirects(monkeypatch):
    record = {}
    _fake_client(monkeypatch, 200, b'{"data":[{"id":"m-1"}]}', record=record)
    assert safe_get_json(
        "https://api.example.test/v1/models", headers={"Authorization": "Bearer sk"}
    ) == (200, {"data": [{"id": "m-1"}]})
    assert record["follow_redirects"] is False
    assert record["timeout"] is not None
    assert record["request"] == (
        "GET",
        "https://api.example.test/v1/models",
        {"Authorization": "Bearer sk"},
    )


def test_safe_get_json_caps_the_body(monkeypatch):
    _fake_client(monkeypatch, 200, b"x" * (MAX_BODY_BYTES + 1))
    with pytest.raises(UpstreamBodyTooLarge):
        safe_get_json("https://api.example.test/v1/models")


def test_safe_get_json_never_parses_an_error_body(monkeypatch):
    """An upstream body must not travel any further than (status, None)."""
    _fake_client(monkeypatch, 401, b'{"detail":"key sk-secret is invalid"}')
    assert safe_get_json("https://api.example.test/v1/models") == (401, None)


def test_safe_get_json_tolerates_a_non_json_body(monkeypatch):
    _fake_client(monkeypatch, 200, b"<html>nope</html>")
    assert safe_get_json("https://api.example.test/v1/models") == (200, None)
