"""Static gate: every route declared in api/*.py must be public or protected.

Reads the source instead of importing the app on purpose - importing api.chat
pulls in transformers, fastembed and Qdrant (see AGENTS.md).
"""

import re
from pathlib import Path

from identity import PUBLIC_PATHS, is_protected_path

API_DIR = Path(__file__).resolve().parent.parent / "api"

PREFIX_RE = re.compile(r"APIRouter\(\s*prefix\s*=\s*[\"']([^\"']*)[\"']")
ROUTE_RE = re.compile(r"@router\.(get|post|put|patch|delete)\(\s*[\"']([^\"']*)[\"']")


def _join(prefix: str, path: str) -> str:
    if not path:
        return prefix
    return f"{prefix.rstrip('/')}/{path.lstrip('/')}"


def extract_routes(api_dir: Path = API_DIR) -> set[tuple[str, str]]:
    """{(full_path, METHOD)} for every @router decorator under api_dir."""
    routes: set[tuple[str, str]] = set()
    for source_file in sorted(api_dir.glob("*.py")):
        source = source_file.read_text(encoding="utf-8")
        prefix_match = PREFIX_RE.search(source)
        prefix = prefix_match.group(1) if prefix_match else ""
        for method, path in ROUTE_RE.findall(source):
            routes.add((_join(prefix, path), method.upper()))
    return routes


def test_route_extraction_is_not_silently_empty():
    routes = extract_routes()
    assert routes, "route regex matched nothing - fix the extractor, not the assertion"
    assert ("/keys/add", "POST") in routes


def test_every_route_is_public_or_protected():
    uncovered = sorted(
        path
        for path, _method in extract_routes()
        if path not in PUBLIC_PATHS and not is_protected_path(path)
    )
    assert not uncovered, (
        "these routes have no identity requirement - add their prefix to "
        f"identity.PROTECTED_PREFIXES (or the path to PUBLIC_PATHS with a "
        f"reason): {uncovered}"
    )
