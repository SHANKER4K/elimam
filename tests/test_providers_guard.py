"""Static guards on api/providers.py, which cannot be imported here.

Importing it pulls in db.connection (a live pool) and api.keys, so - like
tests/test_route_coverage.py - this reads the source instead. Two invariants
the custom-providers refactor depends on:

1. the per-connection lookup filters on the owner (`up.user_id = %s`);
   without it one user can spend another user's provider connection.
2. the catalog comes from the DB, not the deleted config/providers.yaml.
"""

import re
from pathlib import Path

SOURCE = (Path(__file__).resolve().parent.parent / "api" / "providers.py").read_text(
    encoding="utf-8"
)


def test_custom_connections_survive_the_provider_join():
    """A custom connection has `provider_id IS NULL`.

    An inner `JOIN providers` drops its row, so `resolve_model_config` raised
    "Unknown model provider" for every custom endpoint: the user could add one
    but no request could ever use it.
    """
    assert not re.search(r"(?<!LEFT )JOIN providers", SOURCE), (
        "providers must be LEFT JOINed: an inner join hides custom "
        "connections (provider_id IS NULL)"
    )
    assert SOURCE.count("LEFT JOIN providers p ON p.id = up.provider_id") == 2


def test_connection_lookup_filters_on_owner():
    assert "FROM user_providers" in SOURCE
    lookups = re.findall(r"FROM user_providers up.*?\([^)]*\)", SOURCE, re.DOTALL)
    assert lookups, "no user_providers lookup found - fix the extractor"
    for query in lookups:
        assert "up.user_id = %s" in query, (
            "every user_providers lookup must filter on the owner; "
            f"missing in: {query}"
        )


def test_catalog_is_read_from_the_database():
    assert "FROM providers" in SOURCE
    assert "load_providers(" not in SOURCE
    assert "providers.yaml" not in SOURCE


def test_public_names_are_intact():
    for name in ("def load_catalog(", "def resolve_model_config(", "def invalidate_catalog_cache("):
        assert name in SOURCE
