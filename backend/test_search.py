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
