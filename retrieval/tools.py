"""Agent-facing tools: the flat `search` tool plus the by-id getters.

These are the only functions the LLM sees, so their signatures stay small and
they translate failures into tool-friendly `[{"error": ...}]` results. The
retrieval machinery behind them lives in `retrieve.Retrieval` and raises.
"""

from typing import Any

from rapidfuzz import fuzz, process

from .books import BOOKS_LIST, CATEGORIES_NAMES, SUNNAH_BOOKS
from .retrieve import CollectionName, retrieval
from .schemas import (
    BooksSearchRequest,
    HadithId,
    HadithSearchRequest,
    QuranId,
    QuranSearchRequest,
    SunnahSearchRequest,
    TafsirId,
    TafsirSearchRequest,
)

_COLLECTION_REQUEST = {
    "quran": QuranSearchRequest,
    "hadith": HadithSearchRequest,
    "tafsir": TafsirSearchRequest,
    "books": BooksSearchRequest,
    "sunnah": SunnahSearchRequest,
}

# ponytail: the HTTP route caps top_k at 100, but the model picks this number and
# nothing stopped it asking for 5000 candidates — 157 serial TEI rerank round
# trips inside one tool call. The reranker is batched at 32, so 50 is two batches.
MAX_TOP_K = 50

# A `books` chunk is ~600 words; 50 of them in full is tens of thousands of
# tokens the model rarely reads, and the same JSON goes over SSE to the UI.
PREVIEW_CHARS = 200

# Per-collection payload fields a UI needs to label a hit. `text` is truncated
# to `preview`; everything else in the payload (headers trees, embeddings) stays
# server-side and is reachable through get_chunk().
_DISPLAY_FIELDS = {
    "quran": ("surah", "surah_number", "ayah_number"),
    "hadith": ("book", "grade"),
    "tafsir": ("tafsir_book", "surah_number", "ayah_number"),
    "books": ("book_name", "category_name", "all_authors", "author_death"),
    "sunnah": (
        "book_name",
        "category_name",
        "all_authors",
        "author_death",
        "athar_number",
    ),
}


def _compact(collection: CollectionName, point: dict) -> dict:
    payload = point.get("payload") or {}
    text = payload.get("text") or ""

    row = {
        "collection": collection,
        "ids": payload.get("ids"),
        "score": round(point.get("reranker_score", point.get("score")) or 0.0, 4),
        "preview": text[:PREVIEW_CHARS] + ("…" if len(text) > PREVIEW_CHARS else ""),
    }
    for field in _DISPLAY_FIELDS[collection]:
        if field in payload:
            row[field] = payload[field]
    return row


async def search(
    collection: CollectionName,
    query_text: str,
    top_k: int = 10,
    filters: dict[str, Any] | None = None,
) -> list[dict]:
    """Search one collection (hybrid dense+BM25 RRF, cross-encoder reranked).

    `query_text` must be Arabic; search only the collection relevant to the
    question. `top_k` is capped at 50. Results are compact: `ids`, `score`, a
    `preview` of the first 200 characters, and the collection's own labels
    (surah/ayah, book/grade, book_name/category_name). Call `get_chunk` with an
    `ids` value when you need the full text. `filters` keys by collection —
    quran: surah_number, surah; hadith: book, grade; tafsir: surah_number,
    surah, ayah_number; books: book_id, book_name, category_name, all_authors,
    author_death, book_date; sunnah: books' keys plus athar_number. Text fields
    take a str or list[str]; numeric fields take an int, list[int], or a range
    {eq,lt,gt,lte,gte}.
    """
    # ponytail: one flat tool replaces dense/sparse/hybrid (2.3k token schema
    # each — the model never chose among them). The generic `filters` dict is
    # coerced by the collection's typed request model, so validation still holds.
    request = _COLLECTION_REQUEST[collection](
        collection=collection,
        query_text=query_text,
        top_k=min(top_k, MAX_TOP_K),
        filters=filters,
    )

    try:
        points = await retrieval.retrieve(
            collection=request.collection,
            query_text=request.query_text,
            top_k=request.top_k,
            rerank_pool=request.rerank_pool,
            filters=request.filters,
        )
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]

    return [_compact(collection, point) for point in points]


def get_chunk(collection: CollectionName, ids: str) -> dict:
    """Full payload of one chunk, addressed by the `ids` value from a search hit.

    Use when the 200-character `preview` is not enough — the whole passage is
    needed for reading or quoting.
    """
    return retrieval.by_id(collection, ids)


def get_quran(req: QuranId):
    "get a specific ayah ex(1:1)"
    return retrieval.by_id("quran", req.ids)


def get_hadith(req: HadithId):
    "Get a specific hadith"
    return retrieval.by_hadith(req.book, req.hadith_number)


def get_tafsir(req: TafsirId):
    "Get a specific tafsir"
    return retrieval.by_id("tafsir", f"{req.book}:{req.ids}")


def get_book(category: int, book_id: int, chunk_index: int):
    "Get a specific chunk"
    return retrieval.by_id("books", f"{category}:{book_id}:{chunk_index}")


def get_sunnah(category: int, book_id: int, chunk_index: int):
    "Get a specific athar"
    return retrieval.by_id("sunnah", f"{category}:{book_id}:{chunk_index}")


def get_books_books() -> list:
    """List all Arabic book titles (static).

    Hardcoded titles — no database I/O, instant.

    Use when you need to filter data by book name.

    Returns:
        Fresh copy of BOOKS_LIST (e.g. 'منهاج السنة النبوية', ...).
    """
    return list(BOOKS_LIST)


def get_books_sunnah() -> list:
    """List all Arabic book titles for the sunnah collection (static).

    Hardcoded titles — no database I/O, instant.

    Use when you need to filter data by sunnah books names.

    Returns:
        Fresh copy of SUNNAH_BOOKS (e.g. 'مسند أحمد - ط الرسالة', ...).
    """
    return list(SUNNAH_BOOKS)


def get_books_categories() -> list:
    """List all books and sunnah books categories (static).

    Hardcoded titles — no database I/O, instant.

    Use when you need to filter data by categories.

    Returns:
        Fresh copy of CATEGORIES_NAMES (e.g. 'العقيدة', ...).
    """
    return list(CATEGORIES_NAMES)


# -----------------------------
# Fuzzy title lookup
# -----------------------------

# The listings carry the edition suffix ("- ط عطاءات العلم"), so an exact
# book_name filter is unusable from a loose request like "شرح الطحاوية". These
# helpers map loose names onto the exact titles the filter needs.
#
# ponytail: token_set_ratio for titles — measured on the real list, WRatio ranks
# an unrelated book first for "شرح الطحاوية" (الدرة البهية at 86) while
# token_set_ratio puts all three الطحاوية editions at 100 and is insensitive to
# the extra edition words. Categories are a short list of prefix-heavy names, so
# partial_ratio fits them better ("سنه" -> "كتب السنة"). Known ceiling: a typo
# sharing no token scores weakly; upgrade path is a per-token Levenshtein pass.
#
# Cutoffs calibrated over all 260 titles (query = first two words): every real
# query scores 100 (token_set treats it as a subset), while absent books sit at
# 53-74 ("صحيح البخاري" -> 74, "رياض الصالحين" -> 70). 80 keeps every real hit
# and drops all of them. Categories get 70: eight known options, and the caller
# can always fall back to get_books_categories.
FUZZY_CUTOFF = 80
CATEGORY_CUTOFF = 70


def _fold(text: str) -> str:
    """Normalize for fuzzy matching only.

    `preprocess()` must stay index-compatible (the stored vectors were built
    without teh-marbuta folding), so the fold happens here instead: it is what
    makes "عقيده" reach "العقيدة".
    """
    return retrieval.preprocess(text).replace("ة", "ه")


_BOOKS_NORMALIZED = [_fold(title) for title in BOOKS_LIST]
_SUNNAH_NORMALIZED = [_fold(title) for title in SUNNAH_BOOKS]
_CATEGORIES_NORMALIZED = [_fold(name) for name in CATEGORIES_NAMES]


def _fuzzy(
    query: str,
    titles: list[str],
    normalized: list[str],
    limit: int,
    scorer,
    score_cutoff: float,
) -> list[dict]:
    matches = process.extract(
        _fold(query),
        normalized,
        scorer=scorer,
        limit=limit,
        score_cutoff=score_cutoff,
    )
    # `matches` yields (normalized_choice, score, index into `titles`).
    return [
        {"title": titles[index], "score": round(float(score), 1)}
        for _, score, index in matches
    ]


def find_book(
    name: str,
    collection: CollectionName = "books",
    limit: int = 5,
) -> list[dict]:
    """Find the exact book title to use in a `book_name` filter.

    Pass the user's loose book name (`شرح الطحاوية`, `منهاج السنة`); returns the
    matching titles from the corpus, best first, each with a 0-100 score. Two
    editions of the same book are separate titles — pick deliberately. An empty
    list means the book is not in the corpus: do not invent a title.

    Returns:
        [{"title": "شرح العقيدة الطحاوية - ط الرسالة", "score": 100.0}, ...]
    """
    if collection == "sunnah":
        return _fuzzy(
            name,
            SUNNAH_BOOKS,
            _SUNNAH_NORMALIZED,
            limit,
            fuzz.token_set_ratio,
            FUZZY_CUTOFF,
        )
    return _fuzzy(
        name,
        BOOKS_LIST,
        _BOOKS_NORMALIZED,
        limit,
        fuzz.token_set_ratio,
        FUZZY_CUTOFF,
    )


def find_category(name: str, limit: int = 8) -> list[dict]:
    """Find the exact category name to use in a `category_name` filter.

    Returns:
        [{"title": "العقيدة", "score": 100.0}, ...]
    """
    return _fuzzy(
        name,
        CATEGORIES_NAMES,
        _CATEGORIES_NORMALIZED,
        limit,
        fuzz.partial_ratio,
        CATEGORY_CUTOFF,
    )
