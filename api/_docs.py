"""Shared OpenAPI metadata for the FastAPI routers.

Plain constants only. No base class, no factory.
"""

OPENAPI_TAGS: list[dict] = [
    {
        "name": "Chat",
        "description": (
            "Streaming and structured chat against the Islamic scholar agent. "
            "`POST /chat` and `POST /chat/structured` require `X-Bot-Secret` "
            "and `X-Telegram-Id` headers."
        ),
    },
    {
        "name": "Sessions",
        "description": "Conversation sessions (one active per user).",
    },
    {"name": "Users", "description": "User accounts and Telegram identity."},
    {
        "name": "Messages",
        "description": "Persisted message history (read-only via the API; "
        "the chat agent writes its own history to `sessions.pydantic_message`).",
    },
    {
        "name": "API Keys",
        "description": "Per-user provider API keys. Never returned in responses.",
    },
    {
        "name": "Providers",
        "description": "Provider/model configuration read from `config/providers.yaml`.",
    },
]


# Shared error bodies for routes that raise HTTPException(status, detail=...).
# HTTPException always serializes to {"detail": str}, so the schema is a plain
# object — NOT HTTPValidationError (that is the Pydantic 422 shape
# `{"detail": [{"loc":..., "msg":..., "type":...}]}` and FastAPI auto-adds it
# for routes with body validation).
COMMON_ERROR_RESPONSES: dict[int, dict] = {
    status: {
        "description": description,
        "content": {
            "application/json": {
                "schema": {"type": "object", "properties": {"detail": {"type": "string"}}},
                "example": {"detail": example},
            }
        },
    }
    for status, description, example in [
        (400, "Bad request (invalid input or domain error)", "Missing Telegram identity"),
        (401, "Unauthorized (missing or wrong `X-Bot-Secret`)", "Unauthorized"),
        (404, "Resource not found", "Session not found"),
        (409, "Conflict (e.g. user has no active session)", "User has no active session"),
        (422, "Domain validation failure (not Pydantic body validation)", "Quran record not found for [2:153]"),
    ]
}
