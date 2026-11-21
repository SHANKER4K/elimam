"""Checks for the retrieval package that need no Qdrant, no TEI, no model.

The point of the class refactor was that importing it must be free: with the old
`search.py`, `import search` connected to Qdrant and loaded fastembed BM25, so
every one of these assertions was unreachable without the full stack running.
"""

import asyncio
from types import SimpleNamespace

from retrieval import Retrieval, get_books_books, get_books_sunnah, tools
from retrieval.schemas import HadithFilters, QuranFilters, QuranSearchRequest


def test_constructing_and_filtering_creates_no_clients():
    r = Retrieval(url="http://127.0.0.1:1")

    # cached_property only fills these on first real use.
    assert "client" not in r.__dict__
    assert "sparse_model" not in r.__dict__
    assert "embedding_client" not in r.__dict__
    assert "reranker_client" not in r.__dict__

    # ...and the pure part of the pipeline still works.
    assert r.preprocess("بِسْمِ اللَّهِ") == "بسم الله"
    # Quranic marks + superscript alef + alef wasla, as pyarabic handles them.
    assert r.preprocess("الرَّحْمَٰنِ") == "الرحمن"
    assert r.preprocess("ٱلْحَمْدُ") == "الحمد"


def test_build_filter_maps_every_value_shape():
    assert Retrieval.build_filter(None) is None
    assert Retrieval.build_filter(QuranFilters()) is None

    any_filter = Retrieval.build_filter(QuranFilters(surah_number=[1, 2]))
    assert any_filter.must[0].key == "surah_number"
    assert any_filter.must[0].match.any == [1, 2]

    value_filter = Retrieval.build_filter(HadithFilters(book="bukhari"))
    assert value_filter.must[0].match.value == "bukhari"

    range_filter = Retrieval.build_filter(QuranFilters(surah_number={"gte": 5}))
    assert range_filter.must[0].range.gte == 5

    eq_filter = Retrieval.build_filter(QuranFilters(surah_number={"eq": 7}))
    assert eq_filter.must[0].match.value == 7


def test_book_lists_are_fresh_copies():
    first, second = get_books_books(), get_books_books()
    assert first and first == second and first is not second

    first.pop()
    assert len(get_books_books()) == len(second)

    assert get_books_sunnah()


def test_typo_filters_are_rejected_by_the_request_model():
    # The single flat `search` tool relies on this to validate a raw dict.
    request = QuranSearchRequest(
        collection="quran", query_text="الرحمن", filters={"surah_number": [1, 2]}
    )
    assert request.filters.surah_number == [1, 2]

    try:
        QuranSearchRequest(
            collection="quran", query_text="الرحمن", filters={"surah": [1]}
        )
    except ValueError:
        pass
    else:
        raise AssertionError("a non-text surah filter must not validate")


class _FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class _FakeReranker:
    """Scores a text by its length and records every batch it saw."""

    def __init__(self):
        self.batches = []

    async def post(self, url, json, timeout=None):
        self.batches.append(json["texts"])
        return _FakeResponse(
            [{"index": i, "score": float(len(t))} for i, t in enumerate(json["texts"])]
        )


def test_rerank_batches_and_maps_indices_back():
    # 40 candidates at 16 per batch = 3 TEI calls; the index offset is the part
    # that silently breaks (every point must keep its own score).
    points = [{"id": i, "payload": {"text": "ن" * (i + 1)}} for i in range(40)]
    r = Retrieval(
        url="http://127.0.0.1:1", rerank_batch_size=16, jev_api_key=""
    )  # jev_api_key="": a filled-in .env must not route this test through Jev
    fake = _FakeReranker()
    r.__dict__["reranker_client"] = fake

    top = asyncio.run(r.rerank("الرحمن", points, top_k=5))

    assert [len(b) for b in fake.batches] == [16, 16, 8]
    assert len(top) == 5
    assert [p["reranker_score"] for p in top] == [40.0, 39.0, 38.0, 37.0, 36.0]
    assert all("reranker_score" in p for p in points), "a batch went unscored"


def test_retrieve_dispatches_on_mode():
    r = Retrieval(url="http://127.0.0.1:1")
    calls = []

    async def fake_points(collection, query_text, limit, query_filter):
        calls.append(collection)
        return []

    for mode in ("dense", "sparse", "hybrid"):
        r.__dict__[f"_{mode}_points"] = fake_points
        assert asyncio.run(
            r.retrieve(collection="quran", query_text="الرحمن", mode=mode)
        ) == []

    assert calls == ["quran", "quran", "quran"]


def test_search_tool_clamps_top_k(monkeypatch):
    seen = {}

    async def fake_retrieve(**kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(tools, "retrieval", SimpleNamespace(retrieve=fake_retrieve))
    asyncio.run(tools.search("quran", "الرحمن", top_k=5000))

    assert seen["top_k"] == tools.MAX_TOP_K
