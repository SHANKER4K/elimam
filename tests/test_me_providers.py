"""Route tests for api/me_providers.py: no DB, no network.

Every data-access function the routes call is monkeypatched, so no
connection object is ever constructed -- monkeypatching `get_conn` would hand
route code a real connection and that is exactly how a row once reached
production. The only network edge (`_fetch_models`) is canned too.
"""

import copy
import socket
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.me_providers as mp

PROVIDER = {
    "id": 7,
    "slug": "acme",
    "name": "Acme",
    "api_style": "openai_compatible",
    "default_base_url": "https://api.acme.test/v1",
    "logo_url": None,
    "default_variants": ["low", "high"],
    "requires_key": True,
    "docs_url": None,
    "models": {"acme-1": {"variants": ["low", "high"]}},
}

FREE = {
    "id": 8,
    "slug": "free",
    "name": "Free",
    "api_style": "openai_compatible",
    "default_base_url": "http://omniroute:20128/v1",
    "logo_url": None,
    "default_variants": ["low"],
    "requires_key": False,
    "docs_url": None,
    "models": {},
}

CATALOG = {row["id"]: row for row in (PROVIDER, FREE)}

# Real uuid-shaped ids: the routes type their path params as UUID.
CID = "11111111-1111-1111-1111-111111111111"
OTHER_CID = "22222222-2222-2222-2222-222222222222"


def _client(user_id: str | None = "user-1") -> TestClient:
    app = FastAPI()
    if user_id is not None:

        @app.middleware("http")
        async def _identity(request, call_next):
            request.state.user_id = user_id
            return await call_next(request)

    app.include_router(mp.router)
    return TestClient(app)


def _patch_dns(monkeypatch, mapping=None):
    """Canned DNS: literals resolve to themselves, names default to 1.1.1.1."""
    mapping = mapping or {}

    def _getaddrinfo(host, port, *args, **kwargs):
        if ":" in host or host.replace(".", "").isdigit():
            addresses = (host,)
        else:
            addresses = tuple(mapping.get(host, ("1.1.1.1",)))
        return [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, port, 0, 0))
            for address in addresses
        ]

    monkeypatch.setattr(socket, "getaddrinfo", _getaddrinfo)


def _model_row(model_id, *, variants=("low",), is_custom=False, enabled=True):
    return {
        "model_id": model_id,
        "display_name": model_id,
        "variants": list(variants),
        "is_custom": is_custom,
        "enabled": enabled,
        "context_window": None,
        "supports_tools": None,
        "supports_vision": None,
        "last_synced_at": None,
    }


@pytest.fixture
def env(monkeypatch):
    _patch_dns(monkeypatch)
    store = {
        "owners": {},
        "connections": {},
        "models": {},
        "calls": [],
        "discovery": (200, {"data": [{"id": "acme-9"}]}),
    }

    def _records(name, *args):
        store["calls"].append((name, *args))

    def _catalog_rows():
        return [copy.deepcopy(PROVIDER), copy.deepcopy(FREE)]

    def _list_connections(user_id):
        _records("list_connections", user_id)
        return [
            connection
            for cid, connection in store["connections"].items()
            if store["owners"][cid] == user_id
        ]

    def _get_connection(user_id, connection_id):
        _records("get_connection", user_id, connection_id)
        if store["owners"].get(connection_id) != user_id:
            return None
        return store["connections"][connection_id]

    def _provider_models(user_id, connection_id=None):
        _records("provider_models", user_id, connection_id)
        if connection_id is not None:
            return {connection_id: copy.deepcopy(store["models"].get(connection_id, []))}
        return {
            cid: copy.deepcopy(models)
            for cid, models in store["models"].items()
            if store["owners"].get(cid) == user_id
        }

    def _upsert_connection(user_id, provider_id, **fields):
        _records("upsert_connection", user_id, provider_id, fields)
        catalog_row = CATALOG.get(provider_id)
        cid = fields["custom_name"] or str(uuid.uuid5(uuid.NAMESPACE_URL, f"c-{provider_id}"))
        store["connections"][cid] = {
            "id": cid,
            "provider_id": provider_id,
            "is_custom": fields["is_custom"],
            "custom_name": fields["custom_name"],
            "base_url": fields["base_url"],
            "api_style": fields["api_style"],
            "has_key": bool(fields["encrypted_key"]),
            "extra_headers": dict(fields["extra_headers"]),
            "last_validated_at": None,
            "created_at": None,
            "updated_at": None,
            "slug": catalog_row and catalog_row["slug"],
            "provider_name": catalog_row and catalog_row["name"],
            "default_base_url": catalog_row and catalog_row["default_base_url"],
            "requires_key": catalog_row and catalog_row["requires_key"],
            "default_variants": catalog_row and catalog_row["default_variants"] or [],
        }
        store["owners"][cid] = user_id
        return cid

    def _update_connection(user_id, connection_id, fields):
        _records("update_connection", user_id, connection_id, fields)
        if store["owners"].get(connection_id) != user_id:
            return False
        connection = store["connections"][connection_id]
        for key, value in fields.items():
            connection["has_key" if key == "encrypted_key" else key] = bool(value) if key == "encrypted_key" else value
        return True

    def _delete_connection(user_id, connection_id):
        _records("delete_connection", user_id, connection_id)
        if store["owners"].get(connection_id) != user_id:
            return False
        store["connections"].pop(connection_id, None)
        store["owners"].pop(connection_id, None)
        store["models"].pop(connection_id, None)
        return True

    def _sync_models(user_id, connection_id, model_ids, variants, enabled=True):
        _records(
            "sync_models", user_id, connection_id, list(model_ids), list(variants), enabled
        )
        if store["owners"].get(connection_id) != user_id:
            return 0
        rows = store["models"].setdefault(connection_id, [])
        for model_id in model_ids:
            if not any(row["model_id"] == model_id for row in rows):
                row = _model_row(model_id, variants=variants)
                row["enabled"] = enabled
                rows.append(row)
        return len(model_ids)

    def _add_model(user_id, connection_id, model_id, display_name, variants):
        _records("add_model", user_id, connection_id, model_id)
        rows = store["models"].setdefault(connection_id, [])
        row = next((row for row in rows if row["model_id"] == model_id), None)
        if row is None:
            row = _model_row(model_id, variants=variants, is_custom=True)
            rows.append(row)
        row.update(
            display_name=display_name,
            variants=list(variants),
            is_custom=True,
            enabled=True,
        )
        return True

    def _update_model(user_id, connection_id, model_id, *, enabled, variants):
        _records("update_model", user_id, connection_id, model_id, enabled, variants)
        if store["owners"].get(connection_id) != user_id:
            return False
        row = next(
            (row for row in store["models"].get(connection_id, []) if row["model_id"] == model_id),
            None,
        )
        if row is None:
            return False
        if enabled is not None:
            row["enabled"] = enabled
        if variants is not None:
            row["variants"] = list(variants)
        return True

    def _soft_delete_model(user_id, connection_id, model_id):
        _records("soft_delete_model", user_id, connection_id, model_id)
        if store["owners"].get(connection_id) != user_id:
            return False
        row = next(
            (row for row in store["models"].get(connection_id, []) if row["model_id"] == model_id),
            None,
        )
        if row is None:
            return False
        row["enabled"] = False
        return True

    def _mark_validated(user_id, connection_id):
        _records("mark_validated", user_id, connection_id)
        return True

    def _encrypted_key(user_id, connection_id):
        _records("encrypted_key", user_id, connection_id)
        if store["owners"].get(connection_id) != user_id:
            return None
        return "enc:sk-secret" if store["connections"][connection_id]["has_key"] else None

    def _fetch_models(url, headers):
        store["fetch"] = (url, headers)
        return store["discovery"]

    monkeypatch.setattr(mp, "_catalog_rows", _catalog_rows)
    monkeypatch.setattr(mp, "_list_connections", _list_connections)
    monkeypatch.setattr(mp, "_get_connection", _get_connection)
    monkeypatch.setattr(mp, "_provider_models", _provider_models)
    monkeypatch.setattr(mp, "_upsert_connection", _upsert_connection)
    monkeypatch.setattr(mp, "_update_connection", _update_connection)
    monkeypatch.setattr(mp, "_delete_connection", _delete_connection)
    monkeypatch.setattr(mp, "_sync_models", _sync_models)
    monkeypatch.setattr(mp, "_add_model", _add_model)
    monkeypatch.setattr(mp, "_update_model", _update_model)
    monkeypatch.setattr(mp, "_soft_delete_model", _soft_delete_model)
    monkeypatch.setattr(mp, "_mark_validated", _mark_validated)
    monkeypatch.setattr(mp, "_encrypted_key", _encrypted_key)
    monkeypatch.setattr(mp, "_fetch_models", _fetch_models)
    monkeypatch.setattr(mp, "_encrypt", lambda key: f"enc:{key}" if key else None)
    monkeypatch.setattr(mp, "_decrypt", lambda stored: stored[len("enc:") :] if stored else None)

    def _principal_user_ids():
        return {call[1] for call in store["calls"] if len(call) > 1 and isinstance(call[1], str)}

    store["principal_user_ids"] = _principal_user_ids
    return store


def _seed(store, cid, owner, **connection_overrides):
    connection = {
        "id": cid,
        "provider_id": PROVIDER["id"],
        "is_custom": False,
        "custom_name": None,
        "base_url": PROVIDER["default_base_url"],
        "api_style": PROVIDER["api_style"],
        "has_key": True,
        "extra_headers": {},
        "last_validated_at": None,
        "created_at": None,
        "updated_at": None,
        "slug": PROVIDER["slug"],
        "provider_name": PROVIDER["name"],
        "default_base_url": PROVIDER["default_base_url"],
        "requires_key": True,
        "default_variants": PROVIDER["default_variants"],
    }
    connection.update(connection_overrides)
    store["connections"][cid] = connection
    store["owners"][cid] = owner
    store["models"][cid] = [_model_row("acme-1", variants=("low", "high"))]
    return connection


# ── identity ─────────────────────────────────────────────────────────────────


def test_every_route_requires_identity(env):
    client = _client(user_id=None)
    assert client.get("/me/providers").status_code == 401
    assert client.post("/me/providers", json={"provider_id": 7}).status_code == 401
    assert client.post("/me/providers/custom", json={}).status_code == 401
    assert client.patch(f"/me/providers/{CID}", json={"api_key": "x"}).status_code == 401
    assert client.delete(f"/me/providers/{CID}").status_code == 401
    assert client.post(f"/me/providers/{CID}/sync").status_code == 401
    assert client.post(f"/me/providers/{CID}/models", json={"model_id": "m"}).status_code == 401


def test_body_user_id_is_never_used(env):
    response = _client("user-1").post(
        "/me/providers",
        json={"provider_id": 7, "api_key": "sk-secret", "user_id": "someone-else"},
    )
    assert response.status_code == 200
    assert env["principal_user_ids"]() == {"user-1"}, env["calls"]


# ── listing / key hygiene ────────────────────────────────────────────────────


def test_list_returns_catalog_and_connections_without_the_key(env):
    _seed(env, CID, "user-1")
    _seed(env, OTHER_CID, "user-2", has_key=False)

    payload = _client("user-1").get("/me/providers").json()
    assert [row["id"] for row in payload["catalog"]] == [7, 8]
    assert payload["catalog"][0]["defaultVariants"] == ["low", "high"]
    assert [row["id"] for row in payload["connections"]] == [CID]
    assert payload["connections"][0]["hasKey"] is True
    assert payload["connections"][0]["models"][0]["modelId"] == "acme-1"


def test_connect_stores_the_key_but_never_returns_it(env):
    response = _client("user-1").post(
        "/me/providers", json={"provider_id": 7, "api_key": "sk-secret"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["hasKey"] is True
    assert "sk-secret" not in response.text and "enc:sk-secret" not in response.text
    assert [call[1] for call in env["calls"] if call[0] == "upsert_connection"] == ["user-1"]


def test_connect_seeds_catalog_models_then_discovers(env):
    response = _client("user-1").post(
        "/me/providers", json={"provider_id": 7, "api_key": "sk-secret"}
    )
    assert response.status_code == 200
    syncs = [call for call in env["calls"] if call[0] == "sync_models"]
    assert syncs[0][3] == ["acme-1"] and syncs[0][4] == ["low", "high"]
    assert syncs[1][3] == ["acme-9"]
    assert sorted(model["modelId"] for model in response.json()["models"]) == ["acme-1", "acme-9"]
    assert env["fetch"][0] == "https://api.acme.test/v1/models"
    assert env["fetch"][1]["Authorization"] == "Bearer sk-secret"
    assert [call for call in env["calls"] if call[0] == "mark_validated"]


def test_connect_requires_a_key_when_the_provider_does(env):
    response = _client("user-1").post("/me/providers", json={"provider_id": 7})
    assert response.status_code == 400
    assert not [call for call in env["calls"] if call[0] == "upsert_connection"]


def test_connect_unknown_provider_is_404(env):
    assert _client("user-1").post("/me/providers", json={"provider_id": 99}).status_code == 404


# ── URL safety ───────────────────────────────────────────────────────────────


def test_connect_rejects_an_http_override(env):
    response = _client("user-1").post(
        "/me/providers",
        json={"provider_id": 7, "api_key": "sk", "base_url": "http://api.acme.test/v1"},
    )
    assert response.status_code == 400
    assert not [call for call in env["calls"] if call[0] == "upsert_connection"]


def test_catalog_default_is_exempt_only_when_byte_identical(env, monkeypatch):
    monkeypatch.setenv("ALLOW_INSECURE_PROVIDER_URLS", "1")
    _patch_dns(monkeypatch, {"omniroute": ("172.18.0.4",)})

    similar = _client("user-1").post(
        "/me/providers",
        json={"provider_id": 8, "base_url": "http://omniroute:20128/v1/"},
    )
    assert similar.status_code == 400

    identical = _client("user-1").post("/me/providers", json={"provider_id": 8})
    assert identical.status_code == 200


def test_a_private_address_is_still_rejected_for_a_user_url(env, monkeypatch):
    _patch_dns(monkeypatch, {"api.acme.test": ("172.18.0.4",)})
    response = _client("user-1").post(
        "/me/providers", json={"provider_id": 7, "api_key": "sk"}
    )
    assert response.status_code == 400


def test_custom_provider_url_is_strict(env):
    client = _client("user-1")
    body = {"name": "Mine", "base_url": "http://127.0.0.1:8000/v1", "api_style": "openai_compatible"}
    assert client.post("/me/providers/custom", json=body).status_code == 400

    body["base_url"] = "https://10.0.0.5/v1"
    assert client.post("/me/providers/custom", json=body).status_code == 400


def test_custom_provider_is_added_and_discovered(env):
    response = _client("user-1").post(
        "/me/providers/custom",
        json={
            "name": "Mine",
            "base_url": "https://llm.example.test/v1",
            "api_style": "anthropic_compatible",
            "api_key": "sk-secret",
        },
    )
    assert response.status_code == 200
    assert response.json()["isCustom"] is True
    assert response.json()["providerId"] is None
    assert env["fetch"][0] == "https://llm.example.test/v1/models"
    assert env["fetch"][1]["x-api-key"] == "sk-secret"
    assert env["fetch"][1]["anthropic-version"] == "2023-06-01"


def test_custom_provider_rejects_an_unknown_api_style(env):
    response = _client("user-1").post(
        "/me/providers/custom",
        json={"name": "Mine", "base_url": "https://llm.example.test/v1", "api_style": "grpc"},
    )
    assert response.status_code == 400


# ── ownership ────────────────────────────────────────────────────────────────


def test_another_users_connection_is_404_not_403(env):
    _seed(env, OTHER_CID, "user-2")
    client = _client("user-1")

    assert client.patch(f"/me/providers/{OTHER_CID}", json={"api_key": "sk"}).status_code == 404
    assert client.delete(f"/me/providers/{OTHER_CID}").status_code == 404
    assert client.post(f"/me/providers/{OTHER_CID}/sync").status_code == 404
    assert client.post(f"/me/providers/{OTHER_CID}/models", json={"model_id": "m"}).status_code == 404
    assert (
        client.patch(f"/me/providers/{OTHER_CID}/models/acme-1", json={"enabled": False}).status_code
        == 404
    )
    assert client.delete(f"/me/providers/{OTHER_CID}/models/acme-1").status_code == 404
    assert [call[1] for call in env["calls"] if call[0] == "update_connection"] == []


def test_non_uuid_connection_id_is_rejected_by_the_path_type(env):
    assert _client("user-1").delete("/me/providers/nope").status_code == 422


# ── patch / delete ───────────────────────────────────────────────────────────


def test_patch_url_is_revalidated(env):
    _seed(env, CID, "user-1")
    client = _client("user-1")

    assert (
        client.patch(f"/me/providers/{CID}", json={"base_url": "http://api.acme.test/v1"}).status_code
        == 400
    )
    ok = client.patch(f"/me/providers/{CID}", json={"api_key": "sk-new", "custom_name": "Alias"})
    assert ok.status_code == 200
    assert ok.json()["name"] == "Alias"
    assert ok.json()["hasKey"] is True


def test_patch_with_no_fields_is_400(env):
    _seed(env, CID, "user-1")
    assert _client("user-1").patch(f"/me/providers/{CID}", json={}).status_code == 400


def test_delete_connection_removes_it(env):
    _seed(env, CID, "user-1")
    client = _client("user-1")
    assert client.delete(f"/me/providers/{CID}").json() == {"id": CID, "deleted": True}
    assert client.get("/me/providers").json()["connections"] == []


# ── discovery statuses ───────────────────────────────────────────────────────


@pytest.mark.parametrize("status", [401, 403])
def test_discovery_rejects_an_invalid_key_generically(env, status):
    _seed(env, CID, "user-1")
    env["discovery"] = (status, None)
    response = _client("user-1").post(f"/me/providers/{CID}/sync")
    assert response.status_code == 400
    assert response.json()["detail"] == mp.MSG_INVALID_KEY
    assert "sk-secret" not in response.text


def test_discovery_404_is_reported_not_raised(env):
    _seed(env, CID, "user-1")
    env["discovery"] = (404, None)
    response = _client("user-1").post(f"/me/providers/{CID}/sync")
    assert response.status_code == 200
    assert response.json() == {"models": [], "discovery_supported": False}


def test_discovery_other_status_is_generic_400(env):
    _seed(env, CID, "user-1")
    env["discovery"] = (502, None)
    response = _client("user-1").post(f"/me/providers/{CID}/sync")
    assert response.status_code == 400
    assert response.json()["detail"] == mp.MSG_DISCOVERY_FAILED


def test_discovery_garbage_payload_is_generic_400(env):
    _seed(env, CID, "user-1")
    env["discovery"] = (200, {"models": ["nope"]})
    assert _client("user-1").post(f"/me/providers/{CID}/sync").status_code == 400


def test_sync_stores_discovered_models_with_provider_variants(env):
    _seed(env, CID, "user-1")
    env["discovery"] = (200, {"data": [{"id": "m-1"}, {"id": "m-2"}]})
    response = _client("user-1").post(f"/me/providers/{CID}/sync")
    assert response.status_code == 200
    body = response.json()
    assert body["discovery_supported"] is True
    assert sorted(model["modelId"] for model in body["models"]) == ["acme-1", "m-1", "m-2"]
    sync = [call for call in env["calls"] if call[0] == "sync_models"][-1]
    assert sync[3] == ["m-1", "m-2"] and sync[4] == ["low", "high"]


def test_sync_rejects_a_stored_denied_header(env):
    _seed(env, CID, "user-1", extra_headers={"Host": "evil.test"})
    response = _client("user-1").post(f"/me/providers/{CID}/sync")
    assert response.status_code == 400
    assert "fetch" not in env


def test_sync_uses_the_stored_key_only_for_the_owner(env):
    _seed(env, CID, "user-1")
    _client("user-1").post(f"/me/providers/{CID}/sync")
    assert env["fetch"][1]["Authorization"] == "Bearer sk-secret"
    assert [call[1] for call in env["calls"] if call[0] == "encrypted_key"] == ["user-1"]


# ── manual models ────────────────────────────────────────────────────────────


def test_manual_model_add_toggle_and_soft_delete(env):
    _seed(env, CID, "user-1")
    client = _client("user-1")

    added = client.post(f"/me/providers/{CID}/models", json={"model_id": "custom-1"})
    assert added.status_code == 200
    assert added.json()["isCustom"] is True
    assert added.json()["variants"] == ["low"]
    assert added.json()["enabled"] is True

    toggled = client.patch(
        f"/me/providers/{CID}/models/custom-1", json={"enabled": False, "variants": ["max"]}
    )
    assert toggled.status_code == 200
    assert toggled.json()["enabled"] is False and toggled.json()["variants"] == ["max"]

    deleted = client.delete(f"/me/providers/{CID}/models/custom-1")
    assert deleted.json() == {"modelId": "custom-1", "enabled": False}
    # soft delete: the row is still there, and still disabled
    assert env["models"][CID][-1]["model_id"] == "custom-1"
    assert env["models"][CID][-1]["enabled"] is False


def test_model_patch_requires_a_field(env):
    _seed(env, CID, "user-1")
    assert (
        _client("user-1").patch(f"/me/providers/{CID}/models/acme-1", json={}).status_code == 400
    )


def test_missing_model_is_404(env):
    _seed(env, CID, "user-1")
    assert (
        _client("user-1").patch(
            f"/me/providers/{CID}/models/nope", json={"enabled": False}
        ).status_code
        == 404
    )
    assert _client("user-1").delete(f"/me/providers/{CID}/models/nope").status_code == 404


def test_discovered_models_on_a_builtin_provider_arrive_disabled(env):
    """`opencode` advertises 65 models. A Sync click must not flood the picker
    with models the user cannot use; the catalog's six stay enabled."""
    _seed(env, CID, "user-1")
    env["discovery"] = (200, {"data": [{"id": "extra-1"}]})
    assert _client("user-1").post(f"/me/providers/{CID}/sync").status_code == 200
    sync = [call for call in env["calls"] if call[0] == "sync_models"][-1]
    assert sync[5] is False
    assert env["models"][CID][-1]["enabled"] is False


def test_discovered_models_on_a_custom_provider_arrive_enabled(env):
    """A custom endpoint has no catalog to fall back on, so what discovery
    finds has to be usable immediately."""
    _seed(env, CID, "user-1", is_custom=True, provider_id=None, custom_name="Mine")
    env["discovery"] = (200, {"data": [{"id": "extra-1"}]})
    assert _client("user-1").post(f"/me/providers/{CID}/sync").status_code == 200
    sync = [call for call in env["calls"] if call[0] == "sync_models"][-1]
    assert sync[5] is True


def test_a_create_that_fails_discovery_leaves_nothing_behind(env):
    """The 400 never carries the connection id, so a saved row would be
    unreachable and would show up as a broken provider in settings."""
    env["discovery"] = (401, None)
    response = _client("user-1").post(
        "/me/providers/custom",
        json={
            "name": "Mine",
            "base_url": "https://llm.example.test/v1",
            "api_style": "openai_compatible",
            "api_key": "sk-bad",
        },
    )
    assert response.status_code == 400
    assert [c for c in env["calls"] if c[0] == "delete_connection"], "row was not discarded"
    assert env["connections"] == {}
