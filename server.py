"""
FastAPI server for the Islamic scholar agent.
Run: uvicorn backend.server:app --reload
"""

import contextvars
import json
import logging
import os
import time
import uuid
import logfire

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

from api.chat import router as chat_router
from api.sessions import router as sessions_router
from api.users import router as users_router
from api.messages import router as messages_router
from api.keys import router as keys_router
from api.providers import router as providers_router
from api.search import router as search_router
from opik.integrations.otel import OpikSpanProcessor


# ── Logging ────────────────────────────────────────────────────────────────
# stdlib logging only; JSON lines to stdout so `docker compose logs backend`
# and the local uvicorn terminal both give greppable, timestamped, level-tagged
# records with per-request correlation from REQUEST_ID_CONTEXT.
# ponytail: no structlog/loguru/Sentry/OTLP; add Sentry when you want
# push-notified failures, OTel spans when you need cross-service latency.
REQUEST_ID_CONTEXT: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


class _RequestIdFilter(logging.Filter):
    """Injects the current request_id into every record so one request's logs
    can be grepped / correlated. Shares the contextvar with the middleware."""

    def filter(self, record):
        record.request_id = REQUEST_ID_CONTEXT.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "time": self.formatTime(record, "%Y-%m-%d T%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        if record.exc_info and record.exc_info[0] is not None:
            payload["exc"] = self.formatException(record.exc_info)
        for k in ("method", "path", "status", "duration_ms", "tool", "n_messages"):
            v = getattr(record, k, None)
            if v is not None:
                payload[k] = v
        return json.dumps(payload, ensure_ascii=False, default=str)


def _configure_logging():
    fmt = JsonFormatter()
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    handler.addFilter(_RequestIdFilter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

    logging.getLogger("server").info("logging configured")


_configure_logging()


# ── FastAPI app ──────────────────────────────────────────────────────────────
logger = logging.getLogger("server")

app = FastAPI(title="Islamic Scholar API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ponytail: lock to your Next.js origin in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logfire.configure(
    send_to_logfire=False,
    additional_span_processors=[OpikSpanProcessor()],
)
logfire.instrument_pydantic_ai()


@app.middleware("http")
async def request_logging(request: Request, call_next):
    """One JSON line per HTTP request, correlated by request_id. Accepts an
    inbound X-Request-ID (set by a reverse proxy) or mints one."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    token = REQUEST_ID_CONTEXT.set(request_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        # Re-raise so the global handler below also sees it (and logs it once).
        raise
    finally:
        # Only log on the happy path here; exceptions log themselves with context.
        if "response" in locals():
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": getattr(response, "status_code", 0),
                    "duration_ms": duration_ms,
                },
            )
        REQUEST_ID_CONTEXT.reset(token)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Last resort: log the traceback WITH request context, return a clean 500.
    Guards against leaking internals to clients."""
    logger.error(
        "unhandled_exception",
        exc_info=exc,
        extra={"method": request.method, "path": request.url.path},
    )
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


app.include_router(chat_router)
app.include_router(sessions_router)
app.include_router(users_router)
app.include_router(messages_router)
app.include_router(keys_router)
app.include_router(providers_router)
app.include_router(search_router)
