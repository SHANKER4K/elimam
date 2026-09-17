# Use Case Diagram

What each type of actor can do.

```mermaid
flowchart LR
  TG(("Telegram user"))
  WEB(("Web user<br/>(Better Auth session)"))
  OP(("Operator / dev"))
  PROV(("LLM provider"))
  ADMIN(("Frontend admin"))

  subgraph sys["Elimam backend"]
    subgraph uc_chat["Ask"]
      U1["Ask a question via bot<br/>POST /chat (SSE)"]
      U2["Ask a question in the web app<br/>POST /chat/web (SSE)"]
      U3["Resume a conversation<br/>sessions.pydantic_message"]
      U4["Read own history<br/>GET /messages/{session_id}"]
    end
    subgraph uc_model["Configure model"]
      U5["Browse catalog<br/>GET /providers (bot only)"]
      U6["Connect a provider / store API key<br/>POST /me/providers, /keys/add (deprecated)"]
      U7["Add a custom endpoint<br/>POST /me/providers/custom"]
      U8["Sync model list from provider<br/>POST /me/providers/{id}/sync"]
      U9["Add / patch / disable a model<br/>.../models, .../models/{model_id}"]
      U10["Delete a connection<br/>DELETE /me/providers/{id}"]
      U11["Switch model in place<br/>PUT /sessions/{id}/model"]
      U12["Start a new session, keep model<br/>POST /sessions/reset/{user_id}"]
    end
    subgraph uc_direct["Search API (service-to-service)"]
      U13["dense / sparse / hybrid search<br/>POST /search/*"]
      U14["Fetch exact text by id<br/>get_quran, get_hadith, get_tafsir, get_sunnah"]
    end
    subgraph uc_ops["Operate"]
      U15["Health probe<br/>GET /chat/health"]
      U16["Read JSON logs + Opik traces"]
      U17["Apply Drizzle migrations<br/>(from frontend repo)"]
    end
  end

  TG --> U1
  TG --> U5
  TG --> U6
  TG --> U7
  TG --> U8
  TG --> U11
  TG --> U12
  TG --> U4

  WEB --> U2
  WEB --> U3
  WEB --> U4
  WEB --> U6
  WEB --> U7
  WEB --> U8
  WEB --> U9
  WEB --> U10
  WEB --> U11

  PROV --> U8
  ADMIN --> U9
  OP --> U15
  OP --> U16
  OP --> U17

  U1 -.->|"<<include>>"| UCID["Resolve identity<br/>X-Bot-Secret + X-Telegram-Id"]
  U2 -.->|"<<include>>"| UCID2["Resolve identity<br/>signed X-User-*"]
  U1 -.->|"<<include>>"| UCRET["Retrieve from Qdrant"]
  U2 -.->|"<<include>>"| UCRET
  U6 -.->|"<<include>>"| UCVAL["validate + encrypt key<br/>Fernet"]
  U8 -.->|"<<include>>"| UCVAL
  U8 -.->|"<<include>>"| UCSSRF["SSRF guard on base_url<br/>provider_url.py"]
  U7 -.->|"<<include>>"| UCSSRF
  U1 -.->|"<<extend>>"| UCFAIL["409 no active session<br/>400 no key / unknown model"]
  U2 -.->|"<<extend>>"| UCFAIL
  U13 -.->|"<<include>>"| UCRET
```

Actor notes:

| Actor | How it authenticates | What it can reach |
| --- | --- | --- |
| Telegram user | bot holds `BOT_SHARED_SECRET`, forwards `X-Telegram-Id` | everything the bot exposes: chat, `/keys` shim, sessions, `/me/providers` |
| Web user | Next.js server signs `X-User-Id:X-User-Timestamp` with `USER_SHARED_SECRET` (5 min window) | `/chat/web`, `/sessions`, `/messages`, `/me/providers`, and `/keys` under the allowlist in the Next proxy |
| LLM provider | n/a (outbound) | answers `{base_url}/models` for discovery and chat completions |
| Operator / dev | host access + `.env` | health, logs, traces, DB migrations |
| Frontend admin | `users.role` + ban columns (Better Auth admin plugin) | user management lives in the frontend, not here |

Not reachable from the browser by design: `/providers` (bot-secret only), `/users/add` and `/users/telegram/link` (`require_bot`), and any `/me/*` path other than `/me/providers` (Next.js proxy allowlist).
