# Sequence Diagram

Exact order of interactions for one web chat turn (bot turn differs only in the first two arrows).

```mermaid
sequenceDiagram
  autonumber
  actor U as Browser user
  participant N as Next.js /api/chat
  participant F as FastAPI /chat/web
  participant ID as identity middleware
  participant P as providers.resolve_model_config
  participant S as sessions (Postgres)
  participant A as pydantic-ai Agent
  participant T as search.py tools
  participant Q as Qdrant
  participant L as LLM provider

  U->>N: POST /api/chat {message, model_*, session_id}
  N->>N: auth.api.getSession(cookies)
  alt no session
    N-->>U: 401 Unauthorized
  end
  N->>N: strip api_key/user_id, sign X-User-Id/Timestamp/Signature
  N->>F: POST /chat/web + identity headers
  F->>ID: identity_middleware (protected path)
  ID->>ID: hmac compare, timestamp <= 300s
  ID-->>F: request.state.user_id
  F->>F: uuid.UUID(req.session_id) validate
  F->>S: get_session(session_id) (to_thread)
  S-->>F: session row or 404
  F->>S: ownership check str(row.user_id) != user_id -> 404
  F->>P: resolve_model_config(user_provider_id, provider, model, variant, user_id)
  P->>S: SELECT user_providers LEFT JOIN providers LEFT JOIN provider_models WHERE up.user_id = caller
  S-->>P: url, api_style, encrypted_key, variants, requires_key
  P->>P: decrypt(encrypted_key), variant in variants
  P-->>F: {url, model, variant, api_key}
  opt new web session
    F->>S: INSERT sessions (id, source=web, is_active=false)
  end
  F-->>N: StreamingResponse text/event-stream
  N-->>U: relayed SSE body

  F->>A: _build_agent(provider_slug, model_name, url, key, variant)
  F->>S: load_session -> SELECT pydantic_message
  S-->>F: ModelMessagesTypeAdapter.validate_python(row)
  F->>A: agent.iter(prompt, message_history=history)

  loop until End (max retries=3 on tool errors)
    A->>L: POST /chat/completions (or /messages) with system prompt + tools
    L-->>A: TextPart deltas and/or ToolCallPart
    F-->>N: SSE message_start / text_delta / thinking / tool
    alt model called a tool
      A->>T: tool(args)
      T->>Q: query_points(dense+sparse RRF) or scroll(ids)
      Q-->>T: points with payload
      T->>T: cross-encoder rerank -> top_k
      T-->>A: ToolReturnPart(list of hits)
      F-->>N: SSE tool_result
    end
  end
  A-->>F: End(result.output)
  F->>S: UPDATE sessions SET pydantic_message, updated_at
  F->>S: INSERT INTO messages (session_id, role, content, sequence) per message
  F-->>N: SSE done {output}
  N-->>U: stream ends

  Note over F,S: On ModelHTTPError the provider message is emitted as a<br/>text_delta then re-raised. CancelledError is logged and re-raised.<br/>save_session_messages and append_session_messages run in finally.
```

Telegram variant of the first steps:

```mermaid
sequenceDiagram
  autonumber
  actor TG as Telegram user
  participant B as aiogram handler (chat.py)
  participant C as BackendClient
  participant F as FastAPI /chat
  TG->>B: text message
  B->>C: stream_chat(prompt preamble + text, telegram_id)
  C->>F: POST /chat + X-Bot-Secret + X-Telegram-Id
  F->>F: resolve_telegram_caller -> get_or_create_user_by_telegram_id
  F->>F: get_active_session_row(user_id)
  Note over F: no active session -> 409, bot answers "لا توجد جلسة نشطة. استخدم /start أو /model"
  F-->>C: SSE stream
  loop each event
    C-->>B: yield text from message_start / text_delta
    Note over C,B: `tool` events are posted immediately as 🔎 messages
  end
  B->>B: markdownify + split_text(4096)
  B-->>TG: message.answer(parse_mode=MarkdownV2) per chunk
```

Model-selection flow (`/model`) writes through the same auth path:

```mermaid
sequenceDiagram
  autonumber
  participant B as Bot handler model.py
  participant F as FastAPI
  TG2[Telegram user] ->> B: /model
  B->>F: GET /providers (X-Bot-Secret)
  F-->>B: {providers: {slug: {url, models: {id: {variants}}}}}
  B->>F: GET /me/providers (X-Telegram-Id)
  F-->>B: catalog + caller connections
  B->>F: POST /me/providers (connect) or /me/providers/custom + api_key
  B->>F: POST /me/providers/{id}/sync -> GET {base_url}/models
  B->>F: PUT /sessions/{session_id}/model or POST /sessions/add
  B->>F: POST /sessions/reset/{user_id} (/reset)
```
