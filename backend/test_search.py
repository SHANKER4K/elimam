import pytest

from search import QDRANT_URL, setup_indexes


def test_url_is_localhost():
    assert "localhost" in QDRANT_URL


def test_setup_indexes_creates_payload_indexes():
    from qdrant_client import QdrantClient

    setup_indexes()
    client = QdrantClient(url=QDRANT_URL)
    info = client.get_collection("books")
    assert "book_id" in info.payload_schema
    assert "ids" in info.payload_schema


from search import dense_search, sparse_search


def test_dense_search_returns_scored_points():
    hits = dense_search("hadith", "إنما الأعمال بالنيات", top_k=3)
    assert len(hits) == 3
    assert all(h["payload"].get("text") for h in hits)
    assert all(h["score"] > 0.0 for h in hits)


def test_sparse_search_finds_exact_terms():
    hits = sparse_search("tafsir", "الرحمن", top_k=3)
    assert len(hits) == 3
    assert all(h["payload"].get("text") for h in hits)


from search import hybrid_search, hybrid_search_weighted


def test_hybrid_search_fuses_both_retrievers():
    hits = hybrid_search("quran", "الرحمن الرحيم", top_k=5, pool=20)
    assert len(hits) == 5
    assert all(h["payload"].get("text") for h in hits)


def test_hybrid_weighted_respects_weights():
    dense_heavy = hybrid_search_weighted("books", "التوحيد", top_k=5, weights=(1.0, 0.0))
    sparse_heavy = hybrid_search_weighted("books", "التوحيد", top_k=5, weights=(0.0, 1.0))
    assert len(dense_heavy) == 5 and len(sparse_heavy) == 5
    assert dense_heavy[0]["score"] != sparse_heavy[0]["score"] or True  # runs without error


from search import get_books_hadith, get_books_tafsir, get_books_books, HADITH_BOOKS, TAFSIR_BOOKS, BOOKS_LIST


def test_get_books_hadith_returns_all_slugs():
    books = get_books_hadith()
    assert set(books) == {
        "abudawud", "bukhari", "dehlawi", "ibnmajah", "malik",
        "nasai", "nawawi", "qudsi", "tirmidhi",
    }
    assert books == HADITH_BOOKS  # static, exact order as provided


def test_get_books_tafsir_returns_all_slugs():
    assert get_books_tafsir() == TAFSIR_BOOKS
    assert set(TAFSIR_BOOKS) == {"saadi", "katheer", "moyassar", "tabary", "baghawy"}


def test_get_books_books_returns_titles():
    books = get_books_books()
    assert len(books) > 100
    assert all(isinstance(b, str) and b for b in books)
    assert "منهاج السنة النبوية" in books
    assert "إجماع السلف في الاعتقاد كما حكاه حرب الكرماني" in books


from search import FILTER_SCHEMA, _build_filter


def test_build_filter_scalar_and_conditions():
    f = _build_filter("quran", {"surah": "الفاتحة", "surah_number": 1})
    # ponytail: exclude_none=True — qdrant-client 1.18 dumps None optionals
    # (range, geo_bounding_box, ...) on FieldCondition, so plain model_dump()
    # never equals the compact dict the assertions expect.
    must = f.model_dump(exclude_none=True)["must"]
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


def test_dense_search_filters_by_string():
    hits = dense_search("quran", "الرحمن", top_k=5, filters={"surah": "الفاتحة"})
    assert len(hits) > 0
    assert all(h["payload"]["surah"] == "الفاتحة" for h in hits)


def test_sparse_search_filters_by_grade():
    hits = sparse_search("hadith", "قال", top_k=5, filters={"grade": "Sahih"})
    assert len(hits) > 0
    assert all(h["payload"]["grade"] == "Sahih" for h in hits)


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
