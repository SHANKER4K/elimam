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
