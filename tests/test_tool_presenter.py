"""The tool presentation contract, checked without importing `api.chat`.

`api.chat` pulls in every provider SDK (groq/mistral/xai extras), so the tool
list is read out of the source instead of imported.
"""

import ast
import pathlib

from api.tool_presenter import PREVIEW_ROWS, PRESENTERS, describe_call, describe_result

CHAT = pathlib.Path(__file__).parent.parent / "api" / "chat.py"


def _registered_tools() -> set[str]:
    tree = ast.parse(CHAT.read_text())

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "FunctionToolset":
            for keyword in node.keywords:
                if keyword.arg == "tools":
                    return {element.id for element in keyword.value.elts}

    raise AssertionError("no FunctionToolset(tools=[...]) found in api/chat.py")


def test_every_registered_tool_has_a_presenter():
    # A tool without one still streams, but shows up as its raw English name.
    assert _registered_tools() <= set(PRESENTERS)


def test_unknown_tool_degrades_to_its_own_name():
    assert describe_call("some_future_tool", {})["title"] == "some_future_tool"


def test_search_call_reads_like_a_sentence():
    described = describe_call(
        "search",
        {
            "collection": "quran",
            "query_text": "الرحمن",
            "filters": {"surah_number": 1},
        },
    )

    assert described["title"] == "بحث في المصادر"
    assert described["subtitle"] == "القرآن · الرحمن · surah_number=1"


def test_hadith_call_drops_the_float_noise():
    described = describe_call(
        "get_hadith", {"req": {"book": "bukhari", "hadith_number": 1.0}}
    )

    assert described["subtitle"] == "bukhari · 1"


def test_result_reports_count_duration_and_a_bounded_preview():
    rows = [
        {"ids": f"2:{i}", "preview": "نص" * 500, "collection": "quran"}
        for i in range(5)
    ]

    described = describe_result("search", rows, 812)

    assert described["status"] == "ok"
    assert described["count"] == 5
    assert described["duration_ms"] == 812
    assert described["summary"] == "5 نتيجة"
    assert len(described["preview"]) == PREVIEW_ROWS
    assert all(len(row["text"]) <= 160 for row in described["preview"])


def test_error_result_becomes_an_error_card():
    described = describe_result("search", [{"error": "ConnectError: tei"}], 10)

    assert described["status"] == "error"
    assert "ConnectError" in described["message"]
    assert described["preview"] == []


def test_empty_result_is_not_an_error():
    described = describe_result("find_book", [], 3)

    assert described["status"] == "empty"
    assert described["count"] == 0
