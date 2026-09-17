# System Architecture

Major components and how they talk.

```mermaid
flowchart TB
  subgraph clients["Clients"]
    TGU["Telegram user"]
    BR["Browser user"]
    MCPC["MCP client"]
  end

  subgraph edge["Edge"]
    NGX["nginx api.elimam.com<br/>:80 / :443"]
    NEXT["Next.js frontend<br/>/api/chat, /api/backend/[...path], /api/search"]
  end

  subgraph app["Backend service (FastAPI, :8000, image elimam:latest)"]
    SRV["server.py<br/>request-id + identity middleware, routers"]
    RW["api routers<br/>chat, sessions, users, messages, keys, providers, me_providers, search"]
    IDN["identity.py<br/>HMAC user sig / bot secret"]
    SRCH["search.py<br/>dense + sparse + RRF + rerank"]
  end

  subgraph bot["Telegram service (aiogram, image elimam_bot:latest)"]
    AGR["Dispatcher + handlers<br/>start, model, reset, chat"]
    BE["BackendClient (httpx)"]
  end

  subgraph infra["External infrastructure"]
    PG[("PostgreSQL<br/>users, sessions, messages,<br/>providers, user_providers, provider_models,<br/>quran/hadith content")]
    QD[("Qdrant :6333<br/>quran, hadith, tafsir, books, sunnah")]
    LLM["LLM providers<br/>openai / anthropic compatible"]
    OMN["omniroute :20128<br/>local OpenAI-compatible gateway"]
    HF[("HF / fastembed model cache<br/>named volume model_cache")]
    OPIK["Opik via logfire OTLP"]
  end

  subgraph legacy["Legacy, not in compose"]
    MCP["server_mcp.py (FastMCP)"]
    CLI["cli/, web.py, get_books.py"]
  end

  TGU --> AGR --> BE
  BE -->|"HTTP + X-Bot-Secret + X-Telegram-Id"| SRV
  BR --> NEXT
  NEXT -->|"signed X-User-Id/Timestamp/Signature"| SRV
  NGX --> SRV
  MCPC --> MCP --> SRCH

  SRV --> RW
  SRV --> IDN
  RW --> IDN
  RW --> SRCH
  RW -->|"psycopg2 pool"| PG
  SRCH -->|"query_points / scroll"| QD
  SRCH --> HF
  RW -->|"pydantic-ai"| LLM
  RW --> OMN
  SRV --> OPIK
  CLI -.-> QD
```

Key points:

- There is exactly one search layer (`search.py`) and one identity layer (`identity.py`).
- The bot is transport only: it never opens the DB or Qdrant, only `BackendClient` HTTP.
- Backend publishes no host port; nginx and the Next.js proxy are the only entries.
- Postgres and Qdrant are external to compose (`.env`: `DATABASE_*`, `QDRANT_URL`).
- `providers` (DB table) is the model catalog source of truth — no YAML at runtime.
