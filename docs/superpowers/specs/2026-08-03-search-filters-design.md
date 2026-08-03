# Search Filters Design

**Date:** 2026-08-03
**Status:** Approved (design)
**Branch:** feat/qdrant-search (extends the search-functions work)

## Goal

Add metadata filters to the four search functions (`dense_search`, `sparse_search`, `hybrid_search`, `hybrid_search_weighted`) so callers can narrow results by any allowed payload key — with equal and comparison operators for numeric keys.

## Scope

- Modify only `backend/search.py` and `backend/test_search.py`.
- Getters (`get_quran`/`get_hadith`/`get_tafsir`/`get_book`), `backend/server.py`, `retrieve_function.py` are **out of scope** (server wiring remains a separate follow-up).
- Return type stays `list[dict]` (keys `id`, `version`, `score`, `payload`).

## Filter Registry

Per-collection allowlist of filterable keys with types. Drives validation and payload indexes.

```python
FILTER_SCHEMA = {
    "quran":  {"surah_number": "int", "surah": "str"},
    "hadith": {"book": "str", "grade": "str"},
    "tafsir": {"surah_number": "int", "surah": "str", "ayah_number": "int"},
    "books":  {"book_id": "int", "book_name": "str", "category_name": "str",
               "all_authors": "str", "author_death": "int", "bood_data": "int"},
}
```

Notes:
- `bood_data` is the literal payload key in the books collection (typo baked into the data at index time; holds the author death-year int).
- Not all payload keys are filterable — only this allowlist (user decision). `ids`/`text`/`source`/etc. are excluded.
- All bukhari hadith points have empty `grade` (verified: 7,589/7,589) — tests must not combine `book=bukhari` with a grade filter.

## Filter Value Semantics

| Value form | Meaning | Keys allowed |
|---|---|---|
| scalar `"الفاتحة"` / `1` | equal (`MatchValue`) | any |
| list `["الفاتحة", "البقرة"]` | equal-any, OR within the key (`MatchAny`) | any; non-empty; element type must match key type |
| dict `{"gte": 2, "lt": 5}` | comparisons (`Range`) | int keys only; ops `eq`, `lt`, `gt`, `lte`, `gte`; `eq` → `MatchValue` |

- Multiple keys always AND (`must`).
- Unknown key, wrong value type, unknown op, empty list, bool value, or dict on a str key → `ValueError` with a message naming the key.
- `filters=None` or `{}` → no filter (behavior identical to today).

## Helper

`_build_filter(collection: str, filters: dict | None) -> models.Filter | None`

Pure function: translates the dict into a qdrant-client `models.Filter`. No Qdrant I/O — unit-testable.

## Function Wiring

All four search functions gain a trailing parameter `filters: dict | None = None` (backward compatible; existing calls unchanged).

- `dense_search(collection, query_text, top_k=10, filters=None)` → pass `query_filter=_build_filter(...)` to `client.query_points`.
- `sparse_search(collection, query_text, top_k=10, filters=None)` → same.
- `hybrid_search(collection, query_text, top_k=10, pool=50, filters=None)` → same (top-level `query_filter` applies to both prefetches; verified against server 1.18.3).
- `hybrid_search_weighted(collection, query_text, top_k=10, pool=50, weights=(0.7, 0.3), filters=None)` → add `"filter": _build_filter(...).model_dump()` to the raw HTTP body (verified: server accepts `filter` in the `/points/query` body).

Client API note: qdrant-client 1.18.0's `query_points` takes the filter as `query_filter=`, not `filter=` (unknown kwargs raise `AssertionError`).

## Payload Indexes

`setup_indexes()` extends to index every filter key so filtered searches stay fast (books has ~148k points). Schema mapping: `"int"` → `PayloadSchemaType.INTEGER`, `"str"` → `PayloadSchemaType.KEYWORD`. Derived from `FILTER_SCHEMA` (single source of truth) plus the pre-existing getter fields; set-based so duplicates collapse (idempotent).

## Tests (TDD, live Qdrant except pure helper tests)

| Test | Function | Filter |
|---|---|---|
| `_build_filter` scalar+AND | pure | `{"surah": "الفاتحة", "surah_number": 1}` → 2 must conditions |
| `_build_filter` range | pure | `{"surah_number": {"gte": 2, "lt": 5}}` → Range |
| `_build_filter` list OR | pure | `{"surah": ["الفاتحة", "البقرة"]}` → MatchAny |
| `_build_filter` validation | pure | unknown key / bad type / bad op / empty list / bool → `ValueError` |
| dense string filter | `dense_search("quran", "الرحمن", filters={"surah": "الفاتحة"})` | all hits `surah == "الفاتحة"` |
| sparse filter | `sparse_search("hadith", "قال", filters={"grade": "Sahih"})` | all hits `grade == "Sahih"`, len > 0 |
| hybrid int filter | `hybrid_search("tafsir", "الرحمن", filters={"surah_number": 1})` | all hits `surah_number == 1` |
| weighted range | `hybrid_search_weighted("books", "العقيدة", filters={"author_death": {"lt": 500}})` | all hits `author_death < 500` |
| AND two keys | `dense_search("quran", "الرحمن", filters={"surah_number": 1, "surah": "الفاتحة"})` | both hold |
| list OR integration | `dense_search("quran", "الرحمن", filters={"surah": ["الفاتحة", "البقرة"]})` | `surah` in set |
| regression | existing 9 tests unchanged | default `filters=None` |

All term/filter assumptions verified against the live collections (quran 6,236; hadith 28,949; tafsir 31,180; books 148,411).

## Verification

- Full suite from project root: `./.venv/bin/python -m pytest backend/test_search.py -q` → 18 passed.
- Smoke: filtered calls on all four collections + `setup_indexes()` re-run (idempotent).
- Indexes present on server: 18 payload indexes across 4 collections (quran 3, hadith 3, tafsir 5, books 7; list via client).
