"""Runnable check for issue #19: env-driven logfire configuration in server.py.

Boots the real server module with env vars set, then asserts the pipeline did
what the issue demands. Needs the DB/Qdrant env from .env to import server.
Run from the repo root: .venv/bin/python temp/test_logfire_config.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

# Dummy values on purpose: config must wire exporters without contacting them.
# `--off` runs the negative gate instead: envs unset -> nothing may attach.
if "--off" not in sys.argv:
    os.environ["LOGFIRE_TOKEN"] = "dummy-token"
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://127.0.0.1:9999"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = "x-opik-api-key=fake"
    os.environ["ENVIRONMENT"] = "test"
else:
    # `.env` (via load_dotenv) carries real values, so empty strings - not
    # popping - are the honest "nothing configured" state: they defeat both
    # gates (logfire's own OTEL check uses os.getenv falsy too).
    os.environ["LOGFIRE_TOKEN"] = ""
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = ""
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = ""
    os.environ["ENVIRONMENT"] = ""

import server  # noqa: E402  (import-time side effects are the point)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # noqa: E402
import pydantic_ai  # noqa: E402


def _walk(obj):
    """Yield every leaf span processor AND exporter in the logfire pipeline."""
    if isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _walk(item)
        return
    yield obj
    for attr in ("_span_processors", "span_processor", "_processor", "_real_span_processor", "processor", "_exporter", "_span_exporter", "_batch_processor"):
        inner = getattr(obj, attr, None)
        if inner is not None:
            yield from _walk(inner)
            return


provider = server.logfire._internal.config.GLOBAL_CONFIG._tracer_provider.provider
pipeline = list(_walk(provider._active_span_processor))
print("pipeline:", [type(p).__name__ for p in pipeline], file=sys.stderr)

if "--off" in sys.argv:
    assert not any(isinstance(p, server.OpikSpanProcessor) for p in pipeline), (
        "OpikSpanProcessor attached although OTEL_EXPORTER_OTLP_ENDPOINT is unset"
    )
    assert not any(isinstance(p, OTLPSpanExporter) for p in pipeline), (
        "OTLP exporter attached although OTEL_EXPORTER_OTLP_ENDPOINT is unset"
    )
else:
    assert any(isinstance(p, server.OpikSpanProcessor) for p in pipeline), (
        "OpikSpanProcessor missing while OTEL_EXPORTER_OTLP_ENDPOINT is set"
    )
    assert any(
        isinstance(p, OTLPSpanExporter) for p in pipeline
    ), "no OTLP exporter while OTEL_EXPORTER_OTLP_ENDPOINT is set (Opik export dead)"

instrumentation = pydantic_ai.Agent._instrument_default
assert instrumentation is not True and instrumentation.include_content is False, (
    "pydantic-ai instrumentation must mask prompt/answer content"
)

mode = "off (envs unset)" if "--off" in sys.argv else "on (envs set)"
print(f"test_logfire_config OK [{mode}]: pipeline gates correct, content masked, server boots")
