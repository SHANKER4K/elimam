# Elimam Telegram Bot

Thin Telegram interface for the Elimam FastAPI backend.

## Flow

- `/start` checks whether the Telegram user already exists.
- New users choose a provider, enter an API key, choose a model, then choose `low`, `high`, or `max`.
- The bot creates the user and the first session through the backend.
- Normal messages are sent to `POST /chat/chat` as `{ "message": "..." }`.
- The backend is responsible for resolving the authenticated Telegram user, active session, model configuration, and API key.
- `/reset` deactivates the current session and creates a new active session using the same model configuration.

## Provider configuration

Edit `app/config.py`:

```python
PROVIDERS = {
    "openai": {
        "models": ["gpt-5", "gpt-5-mini"],
    },
}
```

## Important backend contract

The bot expects these backend operations:

- `GET /users/telegram/{telegram_id}` -> existing user or 404
- `POST /users/add` -> creates user and accepts `api_key`
- `GET /sessions/user_session/{user_id}` -> returns the current/active session
- `POST /sessions/add` -> creates a session and returns its id
- `PUT /sessions/{session_id}` -> can set `is_active=false`
- `POST /chat/chat` -> accepts only `{ "message": "..." }` and returns SSE

