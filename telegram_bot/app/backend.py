from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from aiogram.types.message import Message
import httpx
from telegramify_markdown import markdownify


class BackendError(RuntimeError):
    pass


class BackendNotFound(BackendError):
    pass


class BackendClient:
    """Thin HTTP client. The bot never touches the database directly - every
    operation goes through the backend, per the architecture rule that the
    bot is a transport/interface layer only."""

    def __init__(self, base_url: str, bot_shared_secret: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.bot_shared_secret = bot_shared_secret
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _bot_headers(self) -> dict[str, str]:
        if not self.bot_shared_secret:
            raise BackendError("BOT_SHARED_SECRET is not configured")
        return {"X-Bot-Secret": self.bot_shared_secret}

    def _identity_headers(self, telegram_id: str) -> dict[str, str]:
        if not self.bot_shared_secret:
            raise BackendError("BOT_SHARED_SECRET is not configured")

        return {
            "X-Bot-Secret": self.bot_shared_secret,
            "X-Telegram-Id": str(telegram_id),
        }

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        try:
            response = await self._client.request(
                method, f"{self.base_url}{path}", json=json_body, headers=headers
            )
            response.raise_for_status()
            if response.status_code == 204 or not response.content:
                return None
            return response.json()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            msg = f"Backend returned {exc.response.status_code}: {body}"
            if exc.response.status_code == 404:
                raise BackendNotFound(msg) from exc
            raise BackendError(msg) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise BackendError("Could not communicate with the backend") from exc

    async def fetch_providers(self, path: str) -> dict[str, Any]:
        """Return provider names mapped to their models & variants.
        Example return shape: {'opencode': {'models': {...}}, 'dahl': {'models': {...}}}."""
        result = await self._request_json("GET", path, headers=self._bot_headers())
        providers = result.get("providers") if isinstance(result, dict) else None
        if not isinstance(providers, dict):
            raise BackendError("Invalid provider configuration from backend")
        return providers

    async def find_user(self, telegram_id: str, path_template: str) -> dict[str, Any] | None:
        path = path_template.format(telegram_id=telegram_id)
        try:
            return await self._request_json(
                "GET", path, headers=self._identity_headers(telegram_id)
            )
        except BackendNotFound:
            return None

    async def create_user(
        self, *, path: str, username: str, display_name: str, telegram_id: str
    ) -> dict[str, Any]:
        return await self._request_json(
            "POST",
            path,
            json_body={
                "username": username,
                "display_name": display_name,
                "telegram_id": telegram_id,
            },
            headers=self._identity_headers(telegram_id),
        )

    async def get_active_session(
        self, *, path_template: str, user_id: str, telegram_id: str
    ) -> dict[str, Any] | None:
        path = path_template.format(user_id=user_id)
        try:
            return await self._request_json(
                "GET", path, headers=self._identity_headers(telegram_id)
            )
        except BackendNotFound:
            return None

    async def create_session(
        self,
        *,
        path: str,
        user_id: str,
        telegram_id: str,
        model_provider: str,
        model_name: str,
        model_variant: str,
    ) -> dict[str, Any]:
        return await self._request_json(
            "POST",
            path,
            json_body={
                "user_id": user_id,
                "source": "telegram",
                "model_provider": model_provider,
                "model_name": model_name,
                "model_variant": model_variant,
            },
            headers=self._identity_headers(telegram_id),
        )

    async def update_session_model(
        self,
        *,
        path_template: str,
        session_id: str,
        telegram_id: str,
        model_provider: str,
        model_name: str,
        model_variant: str,
    ) -> dict[str, Any]:
        path = path_template.format(session_id=session_id)
        return await self._request_json(
            "PUT",
            path,
            json_body={
                "model_provider": model_provider,
                "model_name": model_name,
                "model_variant": model_variant,
            },
            headers=self._identity_headers(telegram_id),
        )

    async def reset_session(
        self, *, path_template: str, user_id: str, telegram_id: str
    ) -> dict[str, Any]:
        path = path_template.format(user_id=user_id)
        return await self._request_json(
            "POST", path, headers=self._identity_headers(telegram_id)
        )

    async def key_exists(
        self, *, path_template: str, user_id: str, provider: str, telegram_id: str
    ) -> bool:
        path = path_template.format(user_id=user_id, provider=provider)
        result = await self._request_json(
            "GET", path, headers=self._identity_headers(telegram_id)
        )
        return bool(result and result.get("has_key"))

    async def list_my_providers(self, telegram_id: str) -> dict[str, Any]:
        return await self._request_json("GET", "/me/providers", headers=self._identity_headers(telegram_id))

    async def connect_provider(
        self,
        telegram_id: str,
        provider_id: int,
        api_key: str | None = None,
        base_url: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        json_body = {"provider_id": provider_id}
        if api_key is not None: json_body["api_key"] = api_key
        if base_url is not None: json_body["base_url"] = base_url
        if extra_headers is not None: json_body["extra_headers"] = extra_headers
        return await self._request_json(
            "POST",
            "/me/providers",
            json_body=json_body,
            headers=self._identity_headers(telegram_id),
        )

    async def add_custom_provider(
        self,
        telegram_id: str,
        name: str,
        base_url: str,
        api_style: str,
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        json_body = {"name": name, "base_url": base_url, "api_style": api_style}
        if api_key is not None: json_body["api_key"] = api_key
        if extra_headers is not None: json_body["extra_headers"] = extra_headers
        return await self._request_json(
            "POST",
            "/me/providers/custom",
            json_body=json_body,
            headers=self._identity_headers(telegram_id),
        )

    async def update_provider(
        self,
        telegram_id: str,
        connection_id: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._request_json(
            "PATCH",
            f"/me/providers/{connection_id}",
            json_body=body,
            headers=self._identity_headers(telegram_id),
        )

    async def delete_provider(self, telegram_id: str, connection_id: str) -> dict[str, Any]:
        return await self._request_json(
            "DELETE",
            f"/me/providers/{connection_id}",
            headers=self._identity_headers(telegram_id),
        )

    async def sync_provider(self, telegram_id: str, connection_id: str) -> dict[str, Any]:
        return await self._request_json(
            "POST",
            f"/me/providers/{connection_id}/sync",
            headers=self._identity_headers(telegram_id),
        )

    # DEPRECATED: to be removed in Phase D after `store_key` callers are replaced.
    async def store_key(
        self,
        *,
        path: str,
        user_id: str,
        provider: str,
        api_key: str,
        telegram_id: str,
    ) -> None:
        await self._request_json(
            "POST",
            path,
            json_body={
                "user_id": user_id, "provider": provider, "api_key": api_key
            },
            headers=self._identity_headers(telegram_id),
        )

    async def stream_chat(
        self,
        *,
        path: str,
        sender: Message,
        message: str,
        telegram_id: str,
        username: str | None,
        display_name: str | None,
    ) -> AsyncIterator[str]:
        prompt = (
            "You are in Telegram, so ignore the printing formats of Quran and hadith. "
            "Use block quotes for Quran and hadith instead.\n\n"
            f"{message}"
        )
        headers = self._identity_headers(telegram_id)

        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}{path}",
                json={"message": prompt},
                headers=headers,
            ) as response:
                if response.status_code >= 400:
                    # ponytail: must read inside the stream context;
                    # .text on an unread streaming response raises ResponseNotRead
                    body = (await response.aread()).decode(errors="replace")[:500]
                    raise BackendError(
                        f"Backend returned {response.status_code}: {body}"
                    )

                event: str | None = None
                async for line in response.aiter_lines():
                    if line.startswith("event: "):
                        event = line[7:]
                        continue
                    if not line.startswith("data: "):
                        continue

                    raw = line[6:]
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    if event in {"tool"}:
                        text = f"🔎  **{payload.get('text')}**: **{payload.get('args', {}).get('id', '')}**".strip()
                        mdv2 = markdownify(text)

                        await sender.answer(
                            mdv2,
                            parse_mode="MarkdownV2",
                        )

                    if event == "error":
                        raise BackendError(payload.get("message", "backend error"))
                    if event == "message_start":
                        text = payload.get("text")
                        if text:
                            yield text
                    if event == "text_delta":
                        text = payload.get("text")
                        if text:
                            yield text
                    elif event in {"done", "message_end"}:
                        if event == "done":
                            break
                    elif event is None:
                        text = payload.get("text")
                        if text:
                            yield text
        except httpx.HTTPError as exc:
            raise BackendError("Could not connect to the backend") from exc
