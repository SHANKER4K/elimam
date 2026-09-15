"""Pure tests for identity.py: no DB, no app import, no network."""

import pytest

from identity import (
    Identity,
    SIGNATURE_MAX_AGE_SECONDS,
    Unauthorized,
    resolve_identity,
    sign,
)

USER_SECRET = "user-secret"
BOT_SECRET = "bot-secret"
NOW = 1_700_000_000.0


def _headers(**extra) -> dict[str, str]:
    return dict(extra)


def _resolve(headers, *, now=NOW, user_secret=USER_SECRET, bot_secret=BOT_SECRET, lookup=None):
    calls: list[str] = []

    def _lookup(telegram_id: str) -> str:
        calls.append(telegram_id)
        return "user-from-telegram"

    identity = resolve_identity(
        _headers(**headers),
        now=now,
        user_secret=user_secret,
        bot_secret=bot_secret,
        lookup_telegram_user=lookup or _lookup,
    )
    return identity, calls


def _signed(user_id="u1", timestamp="1700000000", secret=USER_SECRET):
    return {
        "X-User-Id": user_id,
        "X-User-Timestamp": timestamp,
        "X-User-Signature": sign(user_id, timestamp, secret),
    }


def test_valid_signature():
    identity, _ = _resolve(_signed())
    assert identity == Identity("u1", "user")


def test_wrong_secret_is_rejected():
    with pytest.raises(Unauthorized):
        _resolve(_signed(secret="not-the-secret"))


def test_signature_window_boundaries():
    for offset, allowed in ((299, True), (301, False), (-299, True), (-301, False)):
        headers = _signed(timestamp=str(int(NOW) + offset))
        if allowed:
            assert _resolve(headers)[0] == Identity("u1", "user")
        else:
            with pytest.raises(Unauthorized):
                _resolve(headers)


def test_missing_user_id_with_signature_is_rejected():
    headers = _signed()
    del headers["X-User-Id"]
    with pytest.raises(Unauthorized):
        _resolve(headers)


def test_non_numeric_timestamp_is_rejected():
    headers = _signed(timestamp="not-a-number")
    with pytest.raises(Unauthorized):
        _resolve(headers)


def test_bot_secret_resolves_via_lookup():
    identity, calls = _resolve(
        {"X-Bot-Secret": BOT_SECRET, "X-Telegram-Id": "4242"}
    )
    assert identity == Identity("user-from-telegram", "bot")
    assert calls == ["4242"]


def test_wrong_bot_secret_is_rejected():
    with pytest.raises(Unauthorized):
        _resolve({"X-Bot-Secret": "nope", "X-Telegram-Id": "4242"})


def test_empty_bot_secret_never_authorizes():
    with pytest.raises(Unauthorized):
        _resolve(
            {"X-Bot-Secret": "", "X-Telegram-Id": "4242"},
            bot_secret="",
            lookup=lambda _: "user-from-telegram",
        )


def test_bot_secret_without_telegram_id_is_rejected():
    with pytest.raises(Unauthorized):
        _resolve({"X-Bot-Secret": BOT_SECRET})


def test_no_headers_is_rejected():
    with pytest.raises(Unauthorized):
        _resolve({})


def test_user_signature_wins_over_bot_headers():
    """Precedence is pinned: a signature is never downgraded to bot auth."""
    headers = _signed()
    headers["X-Bot-Secret"] = BOT_SECRET
    headers["X-Telegram-Id"] = "4242"
    identity, calls = _resolve(headers)
    assert identity == Identity("u1", "user")
    assert calls == []


def test_header_names_are_case_insensitive():
    headers = {k.lower(): v for k, v in _signed().items()}
    assert _resolve(headers)[0] == Identity("u1", "user")


def test_signature_window_constant_unchanged():
    assert SIGNATURE_MAX_AGE_SECONDS == 300
