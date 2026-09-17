# Data Flow Diagram (DFD)

Where data enters, moves, transforms, and gets stored.

```mermaid
flowchart LR
  E1["Telegram user"]
  E2["Browser user"]
  E3["LLM provider"]
  E4["Operator"]

  P1["1.0 Authenticate<br/>identity.py"]
  P2["2.0 Resolve model config<br/>api/providers.py"]
  P3["3.0 Run agent turn<br/>api/chat.py"]
  P4["4.0 Retrieve<br/>search.py"]
  P5["5.0 Provider discovery<br/>api/me_providers.py"]
  P6["6.0 Persist<br/>api/chat.py, api/sessions.py"]

  D1[("sessions<br/>model triple + pydantic_message + is_active")]
  D2[("messages<br/>role, content, sequence")]
  D3[("users / session / account<br/>identity + auth")]
  D4[("providers<br/>catalog: slug, base_url, models jsonb")]
  D5[("user_providers<br/>base_url, encrypted_key, extra_headers")]
  D6[("provider_models<br/>model_id, variants, enabled")]
  D7[("api_keys (frozen)")]
  D8[("Qdrant<br/>5 collections, dense + sparse vectors + payload")]
  D9[("HF / model_cache volume<br/>GATE-AraBert, reranker, BM25")]

  E1 -->|"text + telegram_id"| P1
  E2 -->|"message + model trio + session_id (signed headers)"| P1
  P1 -->|"users.id"| P2
  P1 -.->|"get_or_create_user_by_telegram_id"| D3
  P1 -->|"request.state.user_id"| P3

  P2 -->|"user_provider_id / slug"| D5
  P2 -->|"catalog fallback"| D4
  P2 -.->|"pm.variants"| D6
  P2 -->|"url, api_style, decrypted key, variant"| P3
  D5 -->|"encrypted_key"| P2
  P2 -.->|"Fernet decrypt (ENCRYPTION_MASTER_KEY)"| P5
  D7 -.->|"never read or written"| P2

  P3 -->|"prompt + history"| D1
  D1 -->|"pydantic_message"| P3
  P3 -->|"tool call: query + filters"| P4
  P4 -->|"payload text"| D8
  D8 -->|"points + payload"| P4
  P4 -.->|"embed / rerank"| D9
  P4 -->|"reranked hits"| P3
  P3 -->|"system prompt + tools + history"| E3
  E3 -->|"text deltas + tool calls"| P3
  P3 -->|"SSE: message_start, text_delta, tool, tool_result, thinking, done"| E1
  P3 -->|same SSE relayed| E2

  P3 -->|"all_messages (JSON)"| P6
  P6 -->|"UPDATE"| D1
  P6 -->|"INSERT with sequence"| D2
  D2 -->|"GET /messages/{session_id}"| E2

  E4 -->|"provider_id / custom base_url / api_key / extra_headers"| P5
  P5 -.->|"SSRF check: https, public IPs, header allowlist"| P5
  P5 -->|"GET {base_url}/models"| E3
  E3 -->|"{data: [model ids]}"| P5
  P5 -->|"encrypt (Fernet)"| D5
  P5 -->|"upsert variant rows"| D6
  D4 -->|"catalog seeds connection + provider_models"| P5

  E4 -->|".env: DATABASE_*, QDRANT_*, ENCRYPTION_MASTER_KEY, BOT_SHARED_SECRET, USER_SHARED_SECRET"| P1
  P3 -->|"JSON logs with request_id"| E4
  P3 -->|"OTLP spans (Opik via logfire)"| E4
```

Trust boundaries (where data is hostile and must be validated):

| Boundary | Incoming data | Control |
| --- | --- | --- |
| Internet → `/chat`, `/chat/web` | message text, model trio, session id | identity middleware, UUID check, `resolve_model_config` raises 400 |
| Internet → `/me/providers` | `base_url`, `extra_headers`, `api_key` | `provider_url.py`: https only (http only for a byte-identical catalog default), every resolved IP public, denied headers, 1 MiB body cap |
| Internet → `/search/*` | `top_k`, `rerank_pool`, filters | Pydantic `extra="forbid"` + fixed caps 100 / 500 |
| LLM → tool loop | tool name + args | Pydantic validation per search request; errors returned to the model, not raised |
| DB → response | key material | `encrypted_key` is decrypted only inside `resolve_model_config`; never logged, never returned |

Data never crosses these lines: the bot does not touch Postgres or Qdrant; the browser never sends a user id (the Next.js server signs it); the LLM never receives raw vectors or other users' rows.
