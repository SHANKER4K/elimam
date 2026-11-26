"""#25: `book_id` / author filters on books+sunnah, against a real Qdrant.

The live payloads carry the book id only inside `ids` ("category:book:chunk")
and the author under `authors`, so `book_id` / `all_authors` filters matched
nothing. These points mirror that shape exactly.

Needs a Qdrant at QDRANT_TEST_URL (default http://localhost:6333, e.g.
`docker run -p 6333:6333 qdrant/qdrant`); skipped otherwise. Uses throwaway
collections, never the real ones.
"""

import os
import uuid

import pytest
from qdrant_client import QdrantClient, models

from retrieval import Retrieval
from retrieval.schemas import BooksFilters, SunnahFilters
from vectordb.backfill_book_ids import backfill

URL = os.environ.get("QDRANT_TEST_URL", "http://localhost:6333")


def _client() -> QdrantClient | None:
    try:
        client = QdrantClient(url=URL, timeout=2)
        client.get_collections()
        return client
    except Exception:
        return None


pytestmark = pytest.mark.skipif(_client() is None, reason=f"no Qdrant at {URL}")

LIVE_SHAPED_POINTS = [
    ("1:121:0", "حرب الكرماني", "مسائل حرب", "العقيدة"),
    ("1:121:1", "حرب الكرماني", "مسائل حرب", "العقيدة"),
    ("3:323:0", "ابن تيمية", "منهاج السنة النبوية", "الفقه"),
]


@pytest.fixture
def books():
    client = _client()
    name = f"test_books_{uuid.uuid4().hex[:8]}"
    client.create_collection(
        name, vectors_config={"dense": models.VectorParams(size=2, distance=models.Distance.COSINE)}
    )
    client.upsert(
        name,
        points=[
            models.PointStruct(
                id=i,
                vector={"dense": [1.0, float(i)]},
                payload={"ids": ids, "authors": author, "book_name": book, "category_name": cat,
                         "page": "23, 24", "text": f"نص {ids}"},
            )
            for i, (ids, author, book, cat) in enumerate(LIVE_SHAPED_POINTS)
        ],
        wait=True,
    )
    retrieval = Retrieval(url=URL)
    retrieval.setup_indexes(collections={"books": name})
    yield client, retrieval, name
    client.delete_collection(name)


def _count(client, name, filters):
    return client.count(name, count_filter=Retrieval.build_filter(filters), exact=True).count


def test_live_shaped_data_reproduces_the_bug_and_the_check_flags_it(books):
    client, retrieval, name = books
    assert _count(client, name, BooksFilters(book_id=121)) == 0
    empty = retrieval.empty_filter_indexes(collections={"books": name})
    assert ("books", "book_id") in empty
    assert ("books", "authors") not in empty  # authors is populated already


def test_backfill_makes_book_id_filters_work_and_is_idempotent(books):
    client, retrieval, name = books
    assert backfill(client, name) == 3
    assert backfill(client, name) == 0  # second run: nothing left to do

    assert _count(client, name, BooksFilters(book_id=121)) == 2
    assert _count(client, name, BooksFilters(book_id=[121, 323])) == 3
    payload = client.retrieve(name, [1])[0].payload
    assert (payload["category_id"], payload["book_id"], payload["chunk_index"]) == (1, 121, 1)
    empty = retrieval.empty_filter_indexes(collections={"books": name})
    assert not {"book_id", "category_id", "chunk_index", "authors"} & {f for _, f in empty}
    # These synthetic points have no author_death/book_date, and the check says so.
    assert ("books", "author_death") in empty


def test_author_filter_uses_the_stored_field_and_accepts_the_old_name(books):
    client, _, name = books
    assert BooksFilters(authors="ابن تيمية").model_fields_set == {"authors"}
    assert _count(client, name, BooksFilters(authors="حرب الكرماني")) == 2
    # The old name still validates (a model may have learned it) and maps to `authors`.
    assert _count(client, name, BooksFilters.model_validate({"all_authors": ["ابن تيمية"]})) == 1
    assert SunnahFilters.model_validate({"all_authors": "x"}).authors == "x"


def test_get_book_needs_no_category(books):
    client, retrieval, name = books
    backfill(client, name)
    hit = retrieval.by_book(name, book_id=121, chunk_index=1)
    assert hit["ids"] == "1:121:1"
    miss = retrieval.by_book(name, book_id=121, chunk_index=99)
    assert miss["error"] == "not_found"
