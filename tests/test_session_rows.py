"""Regression guard for the sessions row reader.

`_row_to_session` indexes a fixed column list. Two things broke `/reset`,
`/sessions/add` and `/sessions/{id}/model` in production:

1. a RETURNING clause that selected only 7 of those columns, and
2. an off-by-one guard (`len(row) > 6`) that let `row[7]` be read anyway.

Both are checked here without a database.
"""

import ast
import pathlib

import api.sessions as sessions


def test_session_columns_width_matches_the_indexes_the_reader_uses():
    columns = [c.strip() for c in sessions.SESSION_COLUMNS.split(",")]
    assert len(columns) == 8, columns
    assert columns[7] == "user_provider_id"


def test_reader_accepts_a_narrow_legacy_row_and_a_full_one():
    narrow = ("s1", "u1", "telegram", "openai", "gpt-4o", "high", True)
    wide = narrow + ("11111111-1111-1111-1111-111111111111",)

    assert sessions._row_to_session(narrow)["user_provider_id"] is None
    assert (
        sessions._row_to_session(wide)["user_provider_id"]
        == "11111111-1111-1111-1111-111111111111"
    )
    assert sessions._row_to_session(None) is None


def test_every_returning_clause_uses_the_shared_column_list():
    source = pathlib.Path(sessions.__file__).read_text()
    lines = source.splitlines()
    returning = [line.strip() for line in lines if "RETURNING" in line]
    assert returning, "no RETURNING clauses found"
    for line in returning:
        assert "SESSION_COLUMNS" in line, line

    # ...and it must sit in an f-string, or the placeholder is sent to Postgres
    # verbatim (that was a live 500 on PUT /sessions/{id}/model).
    literals = [
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "{SESSION_COLUMNS}" in node.value
    ]
    assert not literals, f"SESSION_COLUMNS in a non-f string: {literals}"
