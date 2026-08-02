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
