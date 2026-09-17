"""Static guard for the Telegram model wizard.

The `/model` flow once entered `ModelChange.api_key` and never handled it: the
refactor deleted the message handler, so a user who picked a provider without a
key sent the key into a void. Nothing crashed and nothing was logged.

Parsed with `ast` (not imported) so this needs no aiogram, no env, no tokens.
"""

import ast
import pathlib

BOT = pathlib.Path(__file__).resolve().parents[1] / "telegram_bot" / "app"
MODEL_HANDLERS = BOT / "handlers" / "model.py"
STATES = BOT / "states.py"


def _states_group(name: str) -> set[str]:
    """The declared attributes of one StatesGroup."""
    tree = ast.parse(STATES.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return {
                stmt.targets[0].id
                for stmt in node.body
                if isinstance(stmt, ast.Assign)
                and isinstance(stmt.targets[0], ast.Name)
            }
    raise AssertionError(f"StatesGroup {name} not found")


def _decorated_states(tree: ast.AST, attribute: str) -> set[str]:
    """States named in `@router.<attribute>(ModelChange.<state>, ...)`."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            if not (isinstance(func, ast.Attribute) and func.attr == attribute):
                continue
            for arg in decorator.args:
                if (
                    isinstance(arg, ast.Attribute)
                    and isinstance(arg.value, ast.Name)
                    and arg.value.id == "ModelChange"
                ):
                    found.add(arg.attr)
    return found


def test_model_change_states_are_unique_and_fully_handled():
    handled = _decorated_states(ast.parse(MODEL_HANDLERS.read_text()), "message")
    handled |= _decorated_states(ast.parse(MODEL_HANDLERS.read_text()), "callback_query")

    declared = _states_group("ModelChange")
    assert declared == {
        "provider",
        "api_key",
        "model",
        "variant",
        "custom_name",
        "custom_url",
        "custom_style",
        "custom_api_key",
        "update_api_key",
    }, declared

    # Every state the wizard can land in must have a handler, or input vanishes.
    assert declared <= handled, f"states with no handler: {sorted(declared - handled)}"


def test_text_collecting_states_are_handled_as_messages():
    """`callback_query` cannot carry text: these must be @router.message."""
    message = _decorated_states(ast.parse(MODEL_HANDLERS.read_text()), "message")
    needs_text = {"api_key", "custom_name", "custom_url", "custom_api_key", "update_api_key"}
    assert needs_text <= message, sorted(needs_text - message)


def test_no_state_is_declared_twice():
    """Duplicate `State()` assignments silently shadow each other."""
    tree = ast.parse(STATES.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ModelChange":
            names = [
                stmt.targets[0].id
                for stmt in node.body
                if isinstance(stmt, ast.Assign)
            ]
            assert len(names) == len(set(names)), f"duplicate states: {names}"
            return
    raise AssertionError("StatesGroup ModelChange not found")
