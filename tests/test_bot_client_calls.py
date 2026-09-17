"""Static guard: every `backend.<method>()` call in the bot matches the client.

Two real failures motivate this. `manage_provider_sync` called
`backend.sync_provider(cid)` where the signature is `sync_provider(telegram_id,
connection_id)`, so the handler raised TypeError. And `store_key` is
keyword-only, so a positional call would have failed the same way.

Parsed with `ast`, not imported: importing the bot needs aiogram and a token.
"""

import ast
import pathlib

BOT = pathlib.Path(__file__).resolve().parents[1] / "telegram_bot" / "app"


def _client_signatures() -> dict[str, dict]:
    """name -> {required: names with no default, positional: names, vararg: bool}."""
    tree = ast.parse((BOT / "backend.py").read_text())
    signatures: dict[str, dict] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        positional = [a.arg for a in args.args if a.arg != "self"]
        optional = positional[: len(positional) - len(args.defaults)] if args.defaults else positional
        required = list(optional)
        for name, default in zip(args.kwonlyargs, args.kw_defaults):
            if default is None:
                required.append(name.arg)
        signatures[node.name] = {
            "required": required,
            "positional": positional,
            "vararg": args.vararg is not None,
        }
    return signatures


def _calls():
    for path in sorted(BOT.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "backend"
            ):
                yield path, node, func.attr


def test_every_backend_call_targets_a_real_method():
    signatures = _client_signatures()
    missing = [
        f"{path.name}:{node.lineno} -> backend.{name}"
        for path, node, name in _calls()
        if name not in signatures
    ]
    assert not missing, "unknown BackendClient method(s): " + ", ".join(missing)


def test_every_backend_call_supplies_every_required_argument():
    signatures = _client_signatures()
    problems = []
    for path, node, name in _calls():
        sig = signatures[name]
        keywords = {k.arg for k in node.keywords if k.arg}
        # Positional arguments fill the leading parameters by position.
        provided = set(sig["positional"][: len(node.args)]) | keywords
        missing = [p for p in sig["required"] if p not in provided]

        if missing:
            problems.append(
                f"{path.name}:{node.lineno} backend.{name}() missing {missing}"
            )
        if not sig["vararg"] and len(node.args) > len(sig["positional"]):
            problems.append(
                f"{path.name}:{node.lineno} backend.{name}() got {len(node.args)} "
                f"positional argument(s), takes {len(sig['positional'])}"
            )
    assert not problems, "\n".join(problems)


def test_the_signatures_are_actually_being_read():
    """A broken extractor would make the checks above vacuous."""
    signatures = _client_signatures()
    assert signatures["sync_provider"]["required"] == [
        "telegram_id",
        "connection_id",
    ]
    # Defaulted parameters are not required...
    assert signatures["connect_provider"]["required"] == ["telegram_id", "provider_id"]
    assert "api_key" not in signatures["connect_provider"]["required"]
    # ...but a keyword-only parameter with no default is, even though it cannot
    # be passed positionally.
    assert signatures["store_key"]["required"] == [
        "path",
        "user_id",
        "provider",
        "api_key",
        "telegram_id",
    ]
    assert len(signatures) > 10
