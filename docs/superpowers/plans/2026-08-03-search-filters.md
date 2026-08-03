# Search Filters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add metadata filters (equal + comparisons on numeric keys) to the four search functions so callers can narrow results by any allowed payload key.

**Architecture:** A per-collection registry `FILTER_SCHEMA` (key → "int"/"str") drives a pure translator `_build_filter(collection, filters) -> models.Filter | None`. All four search functions gain a trailing `filters: dict | None = None` kwarg (backward compatible). The client-based functions pass it as `query_filter=` to `query_points`; the weighted function injects `"filter"` into its raw HTTP body. `setup_indexes()` extends to index every filter key.

**Tech Stack:** Python 3.14, qdrant-client 1.18.0 (models.Filter/FieldCondition/MatchValue/MatchAny/Range, PayloadSchemaType), httpx, fastembed, sentence-transformers 5.6.0, Qdrant server 1.18.3 at localhost:6333.

## Global Constraints

- Only `backend/search.py` and `backend/test_search.py` are modified. Docs, `backend/server.py`, `retrieve_function.py`, getters, and the trailing module-level demo calls are untouched.
- All functions keep the uniform return type `list[dict]` (keys `id`, `version`, `score`, `payload`).
- qdrant-client 1.18.0: `query_points` takes the filter as `query_filter=`, NOT `filter=` (unknown kwargs raise `AssertionError`). `FusionQuery` forbids `weights` — the weighted path stays raw HTTP.
- Test file convention: mid-file `from search import ...` imports, one per feature block (do NOT restructure).
- Test command (project root only — the ONNX model path is CWD-relative): `./.venv/bin/python -m pytest backend/test_search.py -q`
- All bukhari points have empty `grade` (7,589/7,589) — never combine `book=bukhari` with a grade filter in tests.
- `FILTER_SCHEMA` is the single source of truth for filter keys; `setup_indexes()` derives filter-key indexes from it (Task 4).

---
### Task 1: Filter registry + `_build_filter` helper

**Files:**
- Modify: `backend/search.py` — insert after `_sparse_query`, before `def dense_search`
- Test: `backend/test_search.py` — append at end

**Interfaces:**
- Consumes: `models` (already imported)
- Produces: `FILTER_SCHEMA: dict[str, dict[str, str]]` and `_build_filter(collection: str, filters: dict | None) -> models.Filter | None` — used by Tasks 2, 3, 4

- [x] **Step 1: Write the failing tests**

Append to `backend/test_search.py`:

```python
from search import FILTER_SCHEMA, _build_filter


def test_build_filter_scalar_and_conditions():
    f = _build_filter("quran", {"surah": "الفاتحة", "surah_number": 1})
    must = f.model_dump()["must"]
    assert len(must) == 2
    assert {"key": "surah", "match": {"value": "الفاتحة"}} in must
    assert {"key": "surah_number", "match": {"value": 1}} in must


def test_build_filter_range_ops():
    f = _build_filter("quran", {"surah_number": {"gte": 2, "lt": 5}})
    cond = f.model_dump()["must"][0]
    assert cond["key"] == "surah_number"
    assert cond["range"] == {"gt": None, "gte": 2, "lt": 5, "lte": None}


def test_build_filter_dict_eq():
    f = _build_filter("books", {"book_id": {"eq": 121}})
    assert f.model_dump()["must"][0]["match"] == {"value": 121}


def test_build_filter_list_or():
    f = _build_filter("quran", {"surah": ["الفاتحة", "البقرة"]})
    cond = f.model_dump()["must"][0]
    assert cond["match"] == {"any": ["الفاتحة", "البقرة"]}


def test_build_filter_validation_errors():
    for bad in (
        {"gradee": "Sahih"},                          # unknown key
        {"surah_number": "abc"},                      # str on int key
        {"surah": 5},                                 # int on str key
        {"surah": {"gte": 1}},                        # ops on str key
        {"surah_number": {"gte": 2, "magic": 1}},     # unknown op
        {"surah_number": []},                         # empty list
        {"surah_number": True},                       # bool value
        {"surah_number": None},                       # None value
    ):
        with pytest.raises(ValueError):
            _build_filter("quran", bad)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/python -m pytest backend/test_search.py -q`
Expected: FAIL — `ImportError: cannot import name 'FILTER_SCHEMA'`

- [x] **Step 3: Write the implementation**

Insert into `backend/search.py` after `_sparse_query` (before `def dense_search`):

```python
# Allowed filter keys per collection with their value type.
# "int" keys accept scalars, lists, or {eq|lt|gt|lte|gte: number} dicts.
# "str" keys accept scalars or lists. Anything else raises ValueError.
FILTER_SCHEMA: dict[str, dict[str, str]] = {
    "quran": {"surah_number": "int", "surah": "str"},
    "hadith": {"book": "str", "grade": "str"},
    "tafsir": {"surah_number": "int", "surah": "str", "ayah_number": "int"},
    "books": {
        "book_id": "int",
        "book_name": "str",
        "category_name": "str",
        "all_authors": "str",
        "author_death": "int",
        "bood_data": "int",
    },
}

_OPS = {"eq", "lt", "gt", "lte", "gte"}


def _build_filter(collection: str, filters: dict | None) -> models.Filter | None:
    """Translate {key: value} filters into a Qdrant Filter (or None for no filter).

    Value forms (multiple keys are AND-combined):
      scalar       -> equal match (MatchValue)
      list         -> equal to any of the values (MatchAny, OR within the key)
      dict (int)   -> {eq|lt|gt|lte|gte: number}; eq becomes MatchValue, the
                      remaining ops combine into one Range condition

    Raises ValueError on unknown keys, wrong value types, unknown ops, empty
    lists, bool values, and None values.
    """
    if not filters:
        return None
    schema = FILTER_SCHEMA[collection]
    conditions = []
    for key, value in filters.items():
        if key not in schema:
            raise ValueError(
                f"Unknown filter key {key!r} for {collection!r}; allowed: {sorted(schema)}"
            )
        kind = schema[key]
        if isinstance(value, list):
            if not value:
                raise ValueError(f"Filter {key!r}: list must not be empty")
            if kind == "str":
                if not all(isinstance(v, str) for v in value):
                    raise ValueError(f"Filter {key!r}: list elements must be str")
            elif any(
                isinstance(v, bool) or not isinstance(v, (int, float)) for v in value
            ):
                raise ValueError(f"Filter {key!r}: list elements must be numbers")
            conditions.append(
                models.FieldCondition(key=key, match=models.MatchAny(any=value))
            )
        elif kind == "int":
            if isinstance(value, bool):
                raise ValueError(f"Filter {key!r}: bool is not allowed")
            if isinstance(value, dict):
                bad_ops = set(value) - _OPS
                if bad_ops:
                    raise ValueError(
                        f"Filter {key!r}: unknown op(s) {sorted(bad_ops)}; "
                        f"allowed: {sorted(_OPS)}"
                    )
                if any(
                    isinstance(v, bool) or not isinstance(v, (int, float))
                    for v in value.values()
                ):
                    raise ValueError(f"Filter {key!r}: op values must be numbers")
                if "eq" in value:
                    conditions.append(
                        models.FieldCondition(
                            key=key, match=models.MatchValue(value=value["eq"])
                        )
                    )
                ranges = {
                    op: value[op] for op in ("lt", "gt", "lte", "gte") if op in value
                }
                if ranges:
                    conditions.append(
                        models.FieldCondition(key=key, range=models.Range(**ranges))
                    )
            elif isinstance(value, (int, float)):
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                )
            else:
                raise ValueError(
                    f"Filter {key!r}: expected a number or {{op: number}}, "
                    f"got {type(value).__name__}"
                )
        else:  # "str" keys
            if isinstance(value, dict):
                raise ValueError(
                    f"Filter {key!r}: comparison ops only work on integer keys"
                )
            if not isinstance(value, str):
                raise ValueError(
                    f"Filter {key!r}: expected str, got {type(value).__name__}"
                )
            conditions.append(
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
            )
    return models.Filter(must=conditions)
```

- [x] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest backend/test_search.py -q`
Expected: PASS — `14 passed` (9 existing + 5 new; new tests are pure, no Qdrant I/O)

- [x] **Step 5: Commit**

```bash
git add backend/search.py backend/test_search.py
git commit -m "feat: filter schema and filter builder"
```

---
### Task 2: Filters on `dense_search` + `sparse_search`

**Files:**
- Modify: `backend/search.py` — `dense_search` and `sparse_search` (signatures + one kwarg each + docstring Filters section)
- Test: `backend/test_search.py` — append at end

**Interfaces:**
- Consumes: `_build_filter(collection, filters)` from Task 1
- Produces: `dense_search(collection, query_text, top_k=10, filters=None)` and `sparse_search(collection, query_text, top_k=10, filters=None)` — both still return `list[dict]`

- [x] **Step 1: Write the failing tests**

Append to `backend/test_search.py`:

```python
def test_dense_search_filters_by_string():
    hits = dense_search("quran", "الرحمن", top_k=5, filters={"surah": "الفاتحة"})
    assert len(hits) > 0
    assert all(h["payload"]["surah"] == "الفاتحة" for h in hits)


def test_sparse_search_filters_by_grade():
    hits = sparse_search("hadith", "قال", top_k=5, filters={"grade": "Sahih"})
    assert len(hits) > 0
    assert all(h["payload"]["grade"] == "Sahih" for h in hits)
```

(`dense_search`/`sparse_search` are already imported mid-file from the earlier block.)

- [x] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/python -m pytest backend/test_search.py -q`
Expected: FAIL — `TypeError: dense_search() got an unexpected keyword argument 'filters'`

- [x] **Step 3: Write the implementation**

Change `def dense_search(collection: str, query_text: str, top_k: int = 10) -> list:` to:

```python
def dense_search(
    collection: str, query_text: str, top_k: int = 10, filters: dict | None = None
) -> list:
    """Search one collection by dense vector similarity.

    Embeds `query_text` with the ONNX model (preprocessed like training data)
    and returns the `top_k` nearest points by cosine over the `dense` vectors.

    Args:
        collection: One of "quran", "hadith", "tafsir", "books".
        query_text: Free-text query (Arabic).
        top_k: Number of results to return.
        filters: Optional {key: value} metadata filters — see _build_filter.
            Keys must be in FILTER_SCHEMA[collection]; values are scalars
            (equal), lists (equal-any), or {eq|lt|gt|lte|gte} dicts on int keys.

    Returns:
        List of dicts: {"id", "version", "score", "payload"} — same shape as
        every other search function.
    """
    return [
        p.model_dump()
        for p in client.query_points(
            collection_name=collection,
            query=_dense_query(query_text),
            using="dense",
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            query_filter=_build_filter(collection, filters),
        ).points
    ]
```

Change `def sparse_search(collection: str, query_text: str, top_k: int = 10) -> list:` to:

```python
def sparse_search(
    collection: str, query_text: str, top_k: int = 10, filters: dict | None = None
) -> list:
    """Search one collection by exact BM25 keyword match.

    Embeds `query_text` with fastembed Qdrant/bm25 and searches the `sparse`
    vectors. Catches exact term matches that dense search misses.

    Args:
        collection: One of "quran", "hadith", "tafsir", "books".
        query_text: Free-text query (Arabic). Use exact terms — no stemming.
        top_k: Number of results to return.
        filters: Optional {key: value} metadata filters — same semantics as
            dense_search (see _build_filter).

    Returns:
        List of dicts: {"id", "version", "score", "payload"}.
    """
    return [
        p.model_dump()
        for p in client.query_points(
            collection_name=collection,
            query=_sparse_query(query_text),
            using="sparse",
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            query_filter=_build_filter(collection, filters),
        ).points
    ]
```

- [x] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest backend/test_search.py -q`
Expected: PASS — `16 passed`

- [x] **Step 5: Commit**

```bash
git add backend/search.py backend/test_search.py
git commit -m "feat: filters on dense and sparse search"
```

---
### Task 3: Filters on `hybrid_search` + `hybrid_search_weighted`

**Files:**
- Modify: `backend/search.py` — `hybrid_search` and `hybrid_search_weighted` (signatures + docstrings + `query_filter=` / body `"filter"`)
- Test: `backend/test_search.py` — append at end

**Interfaces:**
- Consumes: `_build_filter(collection, filters)` from Task 1
- Produces: `hybrid_search(collection, query_text, top_k=10, pool=50, filters=None)` and `hybrid_search_weighted(collection, query_text, top_k=10, pool=50, weights=(0.7, 0.3), filters=None)` — both still return `list[dict]`

- [x] **Step 1: Write the failing tests**

Append to `backend/test_search.py`:

```python
def test_hybrid_search_filters_by_int():
    hits = hybrid_search("tafsir", "الرحمن", top_k=5, filters={"surah_number": 1})
    assert len(hits) > 0
    assert all(h["payload"]["surah_number"] == 1 for h in hits)


def test_hybrid_weighted_filters_by_range():
    hits = hybrid_search_weighted(
        "books", "العقيدة", top_k=5, filters={"author_death": {"lt": 500}}
    )
    assert len(hits) > 0
    assert all(h["payload"]["author_death"] < 500 for h in hits)


def test_dense_search_filters_and_and_list_or():
    and_hits = dense_search(
        "quran", "الرحمن", top_k=5, filters={"surah_number": 1, "surah": "الفاتحة"}
    )
    assert len(and_hits) > 0
    assert all(
        h["payload"]["surah_number"] == 1 and h["payload"]["surah"] == "الفاتحة"
        for h in and_hits
    )
    or_hits = dense_search(
        "quran", "الرحمن", top_k=5, filters={"surah": ["الفاتحة", "البقرة"]}
    )
    assert len(or_hits) > 0
    assert all(h["payload"]["surah"] in ("الفاتحة", "البقرة") for h in or_hits)
```

(`hybrid_search`/`hybrid_search_weighted` are already imported mid-file.)

- [x] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/python -m pytest backend/test_search.py -q`
Expected: FAIL — `TypeError: hybrid_search() got an unexpected keyword argument 'filters'`

- [x] **Step 3: Write the implementation**

Change `def hybrid_search(collection: str, query_text: str, top_k: int = 10, pool: int = 50) -> list:` to:

```python
def hybrid_search(
    collection: str,
    query_text: str,
    top_k: int = 10,
    pool: int = 50,
    filters: dict | None = None,
) -> list:
    """Hybrid search: dense + sparse in parallel, fused with RRF.

    Runs dense (cosine) and sparse (BM25) searches in parallel via prefetch,
    then fuses their candidate lists with Reciprocal Rank Fusion. RRF is
    rank-based, so the incomparable cosine/BM25 score scales don't matter.

    Args:
        collection: One of "quran", "hadith", "tafsir", "books".
        query_text: Free-text query (Arabic).
        top_k: Number of results to return.
        pool: Candidates retrieved per retriever before fusion. Larger = slower;
            50 is enough before reranking.
        filters: Optional {key: value} metadata filters — same semantics as
            dense_search (see _build_filter). Applied to both retrievers.

    Returns:
        List of dicts: {"id", "version", "score", "payload"} — same type as
        every other search function.
    """
    return [
        p.model_dump()
        for p in client.query_points(
            collection_name=collection,
            prefetch=[
                models.Prefetch(
                    query=_dense_query(query_text), using="dense", limit=pool
                ),
                models.Prefetch(
                    query=_sparse_query(query_text), using="sparse", limit=pool
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            query_filter=_build_filter(collection, filters),
        ).points
    ]
```

Change `hybrid_search_weighted`'s signature and body: signature becomes

```python
def hybrid_search_weighted(
    collection: str,
    query_text: str,
    top_k: int = 10,
    pool: int = 50,
    weights: tuple[float, float] = (0.7, 0.3),
    filters: dict | None = None,
) -> list:
```

Add a `filters` line to the docstring Args block:

```python
        filters: Optional {key: value} metadata filters — same semantics as
            dense_search (see _build_filter). Sent in the raw HTTP body.
```

And before the `response = httpx.post(...)` call, insert:

```python
    query_filter = _build_filter(collection, filters)
    if query_filter is not None:
        body["filter"] = query_filter.model_dump()
```

- [x] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest backend/test_search.py -q`
Expected: PASS — `19 passed`

- [x] **Step 5: Commit**

```bash
git add backend/search.py backend/test_search.py
git commit -m "feat: filters on hybrid and weighted hybrid search"
```

---
### Task 4: Payload indexes for every filter key

**Files:**
- Modify: `backend/search.py` — `setup_indexes()` (top of file)
- Test: `backend/test_search.py` — append at end

**Interfaces:**
- Consumes: `FILTER_SCHEMA` from Task 1 (referenced at call time — defined later in the module, which is fine)
- Produces: `setup_indexes()` covering getter fields + every filter key; index schema mapping `"int"` → `PayloadSchemaType.INTEGER`, `"str"` → `PayloadSchemaType.KEYWORD`

- [x] **Step 1: Write the failing test**

Append to `backend/test_search.py`:

```python
def test_setup_indexes_covers_filter_keys():
    from qdrant_client import QdrantClient

    setup_indexes()
    client = QdrantClient(url=QDRANT_URL)
    expected = {
        "quran": {"surah_number", "surah"},
        "hadith": {"grade"},
        "tafsir": {"surah_number", "surah", "ayah_number"},
        "books": {
            "book_name", "category_name", "all_authors", "author_death", "bood_data",
        },
    }
    for collection, fields in expected.items():
        schema = client.get_collection(collection).payload_schema
        assert fields <= set(schema)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/python -m pytest backend/test_search.py -q`
Expected: FAIL — assertion error on `"surah_number"` (or the first missing index) in `test_setup_indexes_covers_filter_keys`; the other 19 tests still pass.

- [x] **Step 3: Write the implementation**

Replace the whole current `setup_indexes()` (the version with the hardcoded `index_fields` dict) with:

```python
def setup_indexes():
    """Create payload indexes for every field used in getters and filters.

    Idempotent — safe to call repeatedly. Makes getter lookups and filtered
    searches use indexed access instead of full payload scans (books has
    ~148k points).
    """
    kind_schema = {
        "int": models.PayloadSchemaType.INTEGER,
        "str": models.PayloadSchemaType.KEYWORD,
    }
    # Getter-only fields (ids, book, tafsir_book, book_id) are not in
    # FILTER_SCHEMA; filter keys are derived from it (single source of truth).
    fields: dict[str, set[tuple[str, models.PayloadSchemaType]]] = {
        "quran": {("ids", models.PayloadSchemaType.KEYWORD)},
        "hadith": {
            ("ids", models.PayloadSchemaType.KEYWORD),
            ("book", models.PayloadSchemaType.KEYWORD),
        },
        "tafsir": {
            ("ids", models.PayloadSchemaType.KEYWORD),
            ("tafsir_book", models.PayloadSchemaType.KEYWORD),
        },
        "books": {
            ("ids", models.PayloadSchemaType.KEYWORD),
            ("book_id", models.PayloadSchemaType.INTEGER),
        },
    }
    for collection, keys in FILTER_SCHEMA.items():
        fields[collection].update(
            (key, kind_schema[kind]) for key, kind in keys.items()
        )
    for collection, index_set in fields.items():
        for field, schema in index_set:
            client.create_payload_index(collection, field_name=field, field_schema=schema)
```

- [x] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest backend/test_search.py -q`
Expected: PASS — `20 passed`

- [x] **Step 5: Commit**

```bash
git add backend/search.py backend/test_search.py
git commit -m "feat: payload indexes for all filter keys"
```

---
### Task 5: Full verification

**Files:** none (run-only, like the previous plan's Task 5)

- [x] **Step 1: Run the full suite**

Run: `./.venv/bin/python -m pytest backend/test_search.py -q`
Expected: PASS — `20 passed`

- [x] **Step 2: Smoke filtered search on every collection**

Run from the project root (CWD-relative model path):

```bash
./.venv/bin/python -c "
from backend.search import *

setup_indexes()  # second run — must not error (idempotent)
smokes = (
    ('quran', {'surah_number': 1}),
    ('hadith', {'grade': 'Sahih'}),
    ('tafsir', {'surah_number': {'gte': 1, 'lte': 3}}),
    ('books', {'author_death': {'lt': 500}}),
)
for name, flt in smokes:
    r = hybrid_search(name, 'التوحيد', top_k=3, pool=10, filters=flt)
    print(f'{name:8s} hybrid+filter {flt}: {len(r)} hits')
    if r:
        print('   first payload keys ok:', all(k in r[0] for k in ('id', 'score', 'payload')))
"
```

Expected: 4 lines, each `> 0 hits`, each first-hit dict has `id`/`score`/`payload`; `setup_indexes()` re-run does not raise.

- [x] **Step 3: Confirm index count on the server**

Run:

```bash
./.venv/bin/python -c "
from backend.search import client
for c in ('quran', 'hadith', 'tafsir', 'books'):
    schema = client.get_collection(c).payload_schema
    print(c, len(schema), sorted(schema))
"
```

Expected: quran 3 (`ids, surah, surah_number`), hadith 3 (`book, grade, ids`), tafsir 5 (`ayah_number, ids, surah, surah_number, tafsir_book`), books 7 (`all_authors, author_death, bood_data, book_id, book_name, category_name, ids`) — 18 total.

- [x] **Step 4: Commit if anything changed**

```bash
git status --short
git commit -am "chore: verified filters against live collections" || true
```

Only commit if Step 2/3 required a fix (unlikely). If the working tree shows unrelated pre-existing changes, do NOT sweep them in — skip the commit.

## Verification Summary

- Full suite: `20 passed` from the project root.
- Filtered smoke on all 4 collections (hybrid + range/equal/grade filters).
- 18 payload indexes across the 4 collections.
- Regression: existing 9 tests unchanged and green (filters default `None`).
