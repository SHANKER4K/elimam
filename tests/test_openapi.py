"""Regression: every FastAPI route must render usable docs in /openapi.json.

Catches a new route added without metadata, and protects the specific
invariants the docs rely on (SSE content type on /chat).
"""

from fastapi.testclient import TestClient

from server import app

client = TestClient(app)

METHODS = {"get", "post", "put", "delete", "patch"}


def iter_operations(spec):
    for path, methods in spec["paths"].items():
        for method, op in methods.items():
            if method in METHODS:
                yield path, method, op


def test_openapi_has_documentation_for_every_route():
    spec = client.get("/openapi.json").json()

    problems: list[str] = []
    for path, method, op in iter_operations(spec):
        label = f"{method.upper()} {path}"
        if not op.get("summary"):
            problems.append(f"{label}: missing summary")
        if not op.get("tags"):
            problems.append(f"{label}: missing tags")
        if "200" not in op.get("responses", {}) or not op["responses"]["200"].get("description"):
            problems.append(f"{label}: 200 response has no description")

    # SSE endpoint must advertise text/event-stream (not application/json).
    sse = spec["paths"]["/chat"]["post"]["responses"]["200"]["content"]
    if "text/event-stream" not in sse:
        problems.append("POST /chat: 200 content does not advertise text/event-stream")

    assert not problems, "OpenAPI problems:\n" + "\n".join(problems)
