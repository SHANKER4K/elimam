# Backend changes required for persistent Telegram sessions

The Telegram bot sends a stable `session_id` for each Telegram conversation.
The current FastAPI `/chat` code needs to accept and reuse it.

## 1. Add session_id to ChatRequest

```python
class ChatRequest(BaseModel):
    session_id: str
    username: str
    display_name: str
    telegram_id: str
    email: str
    source: str
    external_id: str
    model_name: str
    model_provider: str
    model_variant: str
    message: str
```

## 2. Do not create a new session for every message

The current handler calls `add_user()` and `add_session()` for every request.
Instead:

- find the user using `telegram_id` for Telegram requests;
- create the user only when it does not exist;
- find the session by `req.session_id` and user;
- create it only when it does not exist.

Pseudo-code:

```python
user = get_user_by_telegram_id(req.telegram_id)
if user is None:
    user = create_user(...)

session = get_session(req.session_id)
if session is None:
    session = create_session(
        id=req.session_id,
        user_id=user[0],
        source=req.source,
        external_id=req.external_id,
        model_name=req.model_name,
        model_provider=req.model_provider,
        model_variant=req.model_variant,
    )
```

## 3. Keep the stream call

Once `session_id` exists on `ChatRequest`, this line becomes valid:

```python
return StreamingResponse(
    stream(prompt, session_id=req.session_id),
    media_type="text/event-stream",
)
```

## 4. Fix INSERT ... RETURNING

The current `users.py` and `sessions.py` use `fetchone()` immediately after `INSERT` without a `RETURNING` clause. PostgreSQL will not return a row from that INSERT automatically.

Use something like:

```sql
INSERT INTO users (...)
VALUES (...)
RETURNING id, username, display_name, telegram_id, email
```

and:

```sql
INSERT INTO sessions (...)
VALUES (...)
RETURNING id, user_id, source, external_id, model_name,
          model_provider, model_variant, pydantic_message, is_active
```

## 5. Optional cleaner route

The uploaded code has:

```python
router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("/chat")
```

which produces `/chat/chat`.

For a cleaner API, use:

```python
@router.post("")
```

and set the bot environment variable to:

```env
BACKEND_CHAT_PATH=/chat
```
