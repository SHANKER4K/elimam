"""Human-readable presentation for the tool events the UI renders.

The SSE layer streams raw tool names and raw results. Without this module a
client has to know every tool name, every argument shape and every payload
shape to show a step — which is why the Telegram bot printed
`🔎 search:` and nothing else. Each tool gets a label, an Arabic subtitle built
from its arguments, and a one-line summary of its result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

COLLECTION_LABELS = {
    "quran": "القرآن",
    "hadith": "الحديث",
    "tafsir": "التفسير",
    "books": "الكتب",
    "sunnah": "الآثار",
}

PREVIEW_ROWS = 3
PREVIEW_CHARS = 160


def _no_args(args: dict) -> str:
    return ""


def _ids_of(args: dict) -> str:
    """Tools with one pydantic model arg arrive as {"req": {"ids": ...}}."""
    req = args.get("req")
    if isinstance(req, dict):
        return str(req.get("ids") or req.get("hadith_number") or "")
    return ""


def _search_subtitle(args: dict) -> str:
    collection = COLLECTION_LABELS.get(args.get("collection", ""), "")
    query = str(args.get("query_text") or "")
    filters = args.get("filters") or {}

    detail = ""
    if isinstance(filters, dict) and filters:
        detail = " · ".join(f"{k}={v}" for k, v in filters.items())

    parts = [p for p in (collection, query, detail) if p]
    return " · ".join(parts)


def _list_subtitle(args: dict) -> str:
    return ""


def _find_subtitle(args: dict) -> str:
    return str(args.get("name") or "")


def _hadith_subtitle(args: dict) -> str:
    req = args.get("req") or {}
    number = req.get("hadith_number")
    if isinstance(number, float) and number.is_integer():
        number = int(number)
    return " · ".join(
        str(part) for part in (req.get("book"), number) if part is not None
    )


def _chunk_subtitle(args: dict) -> str:
    return ":".join(
        str(args[key])
        for key in ("category", "book_id", "chunk_index")
        if args.get(key) is not None
    )


def _count_summary(result: Any) -> str:
    if isinstance(result, list):
        return f"{len(result)} نتيجة"
    if isinstance(result, dict):
        return "مقتطف واحد"
    return "تم"


def _is_error(result: Any) -> bool:
    return (
        isinstance(result, list)
        and len(result) == 1
        and isinstance(result[0], dict)
        and "error" in result[0]
    )


def _preview(result: Any) -> list[dict]:
    """At most PREVIEW_ROWS short rows — the UI's card, not the model's context."""
    if isinstance(result, list):
        rows = [row for row in result if isinstance(row, dict)][:PREVIEW_ROWS]
        return [
            {
                "ids": row.get("ids") or _ids_of({"req": row}),
                "text": str(row.get("preview") or row.get("text") or "")[
                    :PREVIEW_CHARS
                ],
            }
            for row in rows
        ]
    if isinstance(result, dict):
        return [
            {
                "ids": result.get("ids"),
                "text": str(result.get("text") or "")[:PREVIEW_CHARS],
            }
        ]
    return []


@dataclass(frozen=True)
class ToolPresenter:
    label: str
    icon: str
    subtitle: Callable[[dict], str] = _no_args
    summary: Callable[[Any], str] = _count_summary


PRESENTERS: dict[str, ToolPresenter] = {
    "search": ToolPresenter("بحث في المصادر", "search", _search_subtitle),
    "get_quran": ToolPresenter("آية قرآنية", "quran", _ids_of),
    "get_tafsir": ToolPresenter("تفسير آية", "tafsir", _ids_of),
    "get_hadith": ToolPresenter("حديث", "hadith", _hadith_subtitle),
    "get_book": ToolPresenter("مقتطف من كتاب", "book", _chunk_subtitle),
    "get_sunnah": ToolPresenter("أثر", "athar", _chunk_subtitle),
    "get_chunk": ToolPresenter("قراءة مقتطف", "book", lambda a: str(a.get("ids") or "")),
    "get_books_books": ToolPresenter("قائمة الكتب", "list"),
    "get_books_sunnah": ToolPresenter("قائمة كتب الآثار", "list"),
    "get_books_categories": ToolPresenter("قائمة التصنيفات", "list"),
    "find_book": ToolPresenter("مطابقة اسم كتاب", "list", _find_subtitle),
    "find_category": ToolPresenter("مطابقة تصنيف", "list", _find_subtitle),
}


def presenter(tool_name: str) -> ToolPresenter:
    # Unknown tool: still render something instead of breaking the stream.
    return PRESENTERS.get(tool_name) or ToolPresenter(tool_name, "tool")


def describe_call(tool_name: str, args: dict) -> dict:
    p = presenter(tool_name)
    return {"title": p.label, "icon": p.icon, "subtitle": p.subtitle(args)}


def describe_result(tool_name: str, result: Any, duration_ms: int | None) -> dict:
    p = presenter(tool_name)

    if _is_error(result):
        return {
            "title": p.label,
            "icon": p.icon,
            "status": "error",
            "message": result[0]["error"],
            "duration_ms": duration_ms,
            "preview": [],
        }

    count = len(result) if isinstance(result, list) else None

    return {
        "title": p.label,
        "icon": p.icon,
        "status": "empty" if count == 0 else "ok",
        "count": count,
        "summary": p.summary(result),
        "duration_ms": duration_ms,
        "preview": _preview(result),
    }
