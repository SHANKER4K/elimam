from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# Fill this with the providers/models/variants you want to expose to
# Telegram. Structure: provider -> models -> variants.
# Example:
# PROVIDERS = {
#     "openai": {
#         "models": {
#             "gpt-5": {"variants": ["low", "high", "max"]},
#             "gpt-5-mini": {"variants": ["low", "high"]},
#         }
#     },
# }
PROVIDERS: dict[str, dict] = {
    "opencode": {
        "models": {
            "deepseek-v4-flash-free": {"variants": ["low", "high", "max"]},
            "big-pickle": {"variants": ["low", "high", "max"]},
            "mimo-v2.5-free": {"variants": ["low", "high", "max"]},
        }
    },
}

DEFAULT_VARIANTS = ("low", "high", "max")


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    bot_shared_secret: str
    backend_url: str = "http://localhost:8000"
    chat_path: str = "/chat"
    user_lookup_path: str = "/users/telegram/{telegram_id}"
    user_create_path: str = "/users/add"
    active_session_path: str = "/sessions/active/user/{user_id}"
    session_create_path: str = "/sessions/add"
    session_model_update_path: str = "/sessions/{session_id}/model"
    session_reset_path: str = "/sessions/reset/{user_id}"
    key_exists_path: str = "/keys/{user_id}/{provider}/exists"
    key_add_path: str = "/keys/add"
    request_timeout: float = 60.0

    @classmethod
    def from_env(cls) -> "Settings":
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

        secret = os.getenv("BOT_SHARED_SECRET")
        if not secret:
            raise RuntimeError("BOT_SHARED_SECRET is not set")

        return cls(
            telegram_bot_token=token,
            bot_shared_secret=secret,
            backend_url=os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/"),
            chat_path=os.getenv("BACKEND_CHAT_PATH", "/chat"),
            user_lookup_path=os.getenv("BACKEND_USER_LOOKUP_PATH", "/users/telegram/{telegram_id}"),
            user_create_path=os.getenv("BACKEND_USER_CREATE_PATH", "/users/add"),
            active_session_path=os.getenv(
                "BACKEND_ACTIVE_SESSION_PATH", "/sessions/active/user/{user_id}"
            ),
            session_create_path=os.getenv("BACKEND_SESSION_CREATE_PATH", "/sessions/add"),
            session_model_update_path=os.getenv(
                "BACKEND_SESSION_MODEL_UPDATE_PATH", "/sessions/{session_id}/model"
            ),
            session_reset_path=os.getenv("BACKEND_SESSION_RESET_PATH", "/sessions/reset/{user_id}"),
            key_exists_path=os.getenv(
                "BACKEND_KEY_EXISTS_PATH", "/keys/{user_id}/{provider}/exists"
            ),
            key_add_path=os.getenv("BACKEND_KEY_ADD_PATH", "/keys/add"),
            request_timeout=float(os.getenv("BACKEND_TIMEOUT", "60")),
        )
