# Function / Module Dependency

Which modules import which. Arrows point at the dependency.

```mermaid
flowchart TD
  server["server.py"]
  chat["api/chat.py"]
  sessions["api/sessions.py"]
  users["api/users.py"]
  messages["api/messages.py"]
  keys["api/keys.py"]
  providers["api/providers.py"]
  meprov["api/me_providers.py"]
  apisearch["api/search.py"]
  search["search.py"]
  identity["identity.py"]
  providerurl["provider_url.py"]
  dbconn["db/connection.py"]
  skillmd["skills/turath-index-skill.md"]
  mcp["server_mcp.py"]
  botmain["telegram_bot/app/main.py"]
  botbackend["telegram_bot/app/backend.py"]
  bothandlers["telegram_bot/app/handlers/*"]
  legacy["cli/, web.py, get_books.py"]

  server --> chat
  server --> sessions
  server --> users
  server --> messages
  server --> keys
  server --> providers
  server --> meprov
  server --> apisearch
  server --> dbconn
  server --> identity
  server -->|"logfire + OpikSpanProcessor"| opik["opik / logfire"]

  chat --> search
  chat --> sessions
  chat --> users
  chat --> providers
  chat --> dbconn
  chat --> identity
  chat --> skillmd
  chat --> pai["pydantic_ai, pydantic_ai_harness, pydantic_graph"]

  apisearch --> search
  providers --> keys
  providers --> dbconn
  meprov --> providerurl
  meprov --> dbconn
  meprov -->|"lazy import"| keys
  meprov --> identity
  sessions --> dbconn
  sessions --> identity
  users --> dbconn
  users --> identity
  messages --> dbconn
  messages --> identity
  keys --> dbconn
  keys --> identity
  keys -->|"Fernet"| crypto["cryptography"]
  identity --> stdlib["stdlib only: hmac, hashlib, os"]

  search --> qdrant["qdrant_client"]
  search --> st["sentence_transformers (GATE-AraBert-v1, mmarco reranker)"]
  search --> fe["fastembed (Qdrant/bm25)"]
  search --> camel["camel_tools (dediac, normalize_alef)"]
  search --> dbconn
  dbconn -->|"ThreadedConnectionPool"| pg[("PostgreSQL")]
  providerurl --> httpx["httpx + socket/ipaddress"]
  search --> qd[("Qdrant")]
  mcp --> search

  botmain --> botbackend
  botmain --> bothandlers
  botmain --> botconfig["app/config.py"]
  bothandlers --> botbackend
  bothandlers --> botstates["app/states.py"]
  botbackend -->|"httpx"| server

  legacy -.-> search
```

Import-time coupling (the reason tests need the stack):

| Module | Side effect at import |
| --- | --- |
| `db/connection.py` | builds the psycopg2 pool from `DATABASE_*` in `.env` |
| `search.py` | loads GATE-AraBert-v1, the cross-encoder and BM25, connects to Qdrant |
| `api/chat.py` | calls `setup_indexes()` and reads `./skills/turath-index-skill.md` |
| `api/providers.py` | lazy — reads the `providers` table on first use, then caches in-process |
| `identity.py`, `provider_url.py` | pure, no DB/model imports (unit-testable standalone) |
