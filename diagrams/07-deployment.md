# Deployment

`docker compose up` from `backend/` runs backend + telegram bot + nginx + omniroute. Postgres and Qdrant are external.

```mermaid
flowchart TB
  NET((Internet))

  subgraph host["VPS / Docker host"]
    subgraph compose["docker compose network"]
      NGX["nginx:1.27-alpine<br/>elimam_nginx<br/>ports 80:80, 443:443<br/>server_name api.elimam.com<br/>proxy_pass http://backend:8000"]
      BE["backend<br/>image elimam:latest<br/>expose 8000, NO published port<br/>uvicorn server:app --host 0.0.0.0 --port 8000"]
      BOT["telegram<br/>image elimam_bot:latest<br/>python -m app.main (aiogram long polling)<br/>BACKEND_URL=http://backend:8000"]
      OMN["omniroute<br/>diegosouzapw/omniroute:latest<br/>127.0.0.1:20128"]
    end

    subgraph vol["volumes / binds"]
      SKILL["./skills/turath-index-skill.md -> /app/skills/turath-index-skill"]
      ENVF["./.env -> /app/.env:ro"]
      HFC["/home/ismail/.cache/huggingface -> /models/hf (HF_HOME)"]
      MC["named volume model_cache -> /models (XDG_CACHE_HOME)<br/>fastembed BM25"]
      OVD["named volume omniroute-data -> /app/data"]
      CERT["./nginx/conf.d, ./certbot/{www,conf}"]
    end
  end

  subgraph external["External services (from .env)"]
    PG[("PostgreSQL<br/>DATABASE_HOST/PORT/USER/NAME/PASSWORD<br/>sslmode=require by default")]
    QD[("Qdrant :6333<br/>QDRANT_URL + QDRANT_API_KEY")]
    LLMP["LLM providers<br/>https://.../v1"]
    OPIKS["Opik collector<br/>OTEL_EXPORTER_OTLP_ENDPOINT + HEADERS"]
  end

  NEXT["Next.js frontend (separate deploy)<br/>API_URL / NEXT_PUBLIC_API_URL"]

  NET --> NGX
  NET --> NEXT
  NET --> BOT
  NGX --> BE
  NEXT -->|"signed X-User-* headers"| BE
  BOT -->|"X-Bot-Secret + X-Telegram-Id"| BE
  BE --> PG
  BE --> QD
  BE --> LLMP
  BE -.-> OMN
  BE --> OPIKS
  BE --- vol
  OMN --- OVD
  NGX --- CERT
```

Startup ordering and gating:

```mermaid
sequenceDiagram
  autonumber
  participant D as docker compose
  participant B as backend
  participant T as telegram
  participant N as nginx
  D->>B: build (python:3.12-slim, 2-stage, torch 2.13.0+cpu)
  Note over B: import-time: psycopg2 pool, Qdrant client,<br/>SentenceTransformer + cross-encoder + BM25, setup_indexes()
  D->>B: healthcheck every 120s, start_period 120s<br/>GET /chat/health via urllib
  B-->>D: healthy
  D->>T: start (depends_on: service_healthy)
  D->>N: start (depends_on: service_healthy)
  D->>O: start omniroute (depends_on: backend healthy)
```

Deployment facts and risks:

- **Image build**: multi-stage `python:3.12-slim`; `requirements.txt` (not `pyproject.toml`) is what the Dockerfile installs; `torch==2.13.0+cpu` comes from the PyTorch CPU extra index.
- **Health endpoint** is `/chat/health` (not `/health`), and `/chat` is in `PUBLIC_PATHS` so it works before identity rollout.
- **No port on backend** — reachable only inside the compose network as `http://backend:8000` (nginx, bot, and the Next.js deploy must be on that network or a tunnel).
- **Model cache is the slow bit**: first boot downloads GATE-AraBert-v1 into the HF bind mount; `start_period: 120s` exists for exactly that.
- **`HF_HOME=/models/hf`** and **`XDG_CACHE_HOME=/models/cache`** keep the HF and fastembed caches inside mounted volumes so a restart is not a re-download.
- Required secrets in `.env`: `TELEGRAM_BOT_TOKEN`, `BOT_SHARED_SECRET` (must be identical in both services), `ENCRYPTION_MASTER_KEY`, `USER_SHARED_SECRET` (shared with the Next.js proxy), plus `DATABASE_*`, `QDRANT_*`, `OTEL_*`.
- `docker-compose.yaml.tmp` and `telegram_bot/Dockerfile.tmp` are leftovers, not used by `up`.
- `heroku.yml` exists but the compose file is the live deployment.
- `AUTH_MODE` (default `permissive`) is the rollout switch; set `enforce` when no pre-migration clients remain.
