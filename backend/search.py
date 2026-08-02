import httpx
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import normalize_alef_ar

QDRANT_URL = "http://localhost:6333"

model = SentenceTransformer(
    "models/gate_arabert_onnx",
    backend="onnx",
    model_kwargs={"file_name": "model_int8.onnx"},
)

# BM25 TF vectors; IDF is applied by Qdrant via Modifier.IDF (collection config)
sparse_model = SparseTextEmbedding("Qdrant/bm25")

client = QdrantClient(url=QDRANT_URL)


def setup_indexes():
    """Create payload indexes for every field used in filters.

    Must run once per server start (idempotent — safe to call repeatedly).
    Makes get_quran/get_hadith/get_tafsir/get_book use indexed lookups instead
    of a full payload scan (books collection has ~148k points).
    """
    index_fields = {
        "quran": [("ids", models.PayloadSchemaType.KEYWORD)],
        "hadith": [
            ("ids", models.PayloadSchemaType.KEYWORD),
            ("book", models.PayloadSchemaType.KEYWORD),
        ],
        "tafsir": [
            ("ids", models.PayloadSchemaType.KEYWORD),
            ("tafsir_book", models.PayloadSchemaType.KEYWORD),
        ],
        "books": [
            ("ids", models.PayloadSchemaType.KEYWORD),
            ("book_id", models.PayloadSchemaType.INTEGER),
        ],
    }
    for collection, fields in index_fields.items():
        for field, schema in fields:
            client.create_payload_index(
                collection, field_name=field, field_schema=schema
            )


def _preprocess(text: str) -> str:
    """Normalize Arabic text exactly like embed_docs did when generating the embeddings."""
    text = dediac_ar(text)
    text = normalize_alef_ar(text)
    text = text.replace("\u200f", "")
    return text.replace("  ", "")


def _dense_query(text: str) -> list[float]:
    """Embed a query with the ONNX model, preprocessed to match training."""
    return model.encode([_preprocess(text)])[0].tolist()


def _sparse_query(text: str) -> models.SparseVector:
    """Embed a query with fastembed BM25. Raw text (no dediac) — matches how sparse
    vectors were indexed in update_qdrant.py."""
    sv = list(sparse_model.embed([text]))[0]
    return models.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())


def dense_search(collection: str, query_text: str, top_k: int = 10) -> list:
    """Search one collection by dense vector similarity.

    Embeds `query_text` with the ONNX model (preprocessed like training data)
    and returns the `top_k` nearest points by cosine over the `dense` vectors.

    Args:
        collection: One of "quran", "hadith", "tafsir", "books".
        query_text: Free-text query (Arabic).
        top_k: Number of results to return.

    Returns:
        List of dicts: {"id", "version", "score", "payload"} — same shape as
        every other search function.
    """
    return [p.model_dump() for p in client.query_points(
        collection_name=collection,
        query=_dense_query(query_text),
        using="dense",
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    ).points]


def sparse_search(collection: str, query_text: str, top_k: int = 10) -> list:
    """Search one collection by exact BM25 keyword match.

    Embeds `query_text` with fastembed Qdrant/bm25 and searches the `sparse`
    vectors. Catches exact term matches that dense search misses.

    Args:
        collection: One of "quran", "hadith", "tafsir", "books".
        query_text: Free-text query (Arabic). Use exact terms — no stemming.
        top_k: Number of results to return.

    Returns:
        List of dicts: {"id", "version", "score", "payload"}.
    """
    return [p.model_dump() for p in client.query_points(
        collection_name=collection,
        query=_sparse_query(query_text),
        using="sparse",
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    ).points]


# %%


def get_quran(id: str):
    val, _ = client.scroll(
        collection_name="quran",
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="ids",
                    match=models.MatchValue(value=id),
                )
            ]
        ),
        with_vectors=False,
    )
    if val:
        return val[0].payload
    return []


def get_hadith(book: str, hadith_number: float):
    val, _ = client.scroll(
        collection_name="hadith",
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="ids",
                    match=models.MatchValue(value=f"{book}:{hadith_number}"),
                )
            ]
        ),
        with_vectors=False,
    )
    if val:
        return val[0].payload
    return []


def get_tafsir(book: str, id: str):
    val, _ = client.scroll(
        collection_name="tafsir",
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="ids",
                    match=models.MatchValue(value=f"{book}:{id}"),
                )
            ]
        ),
        with_vectors=False,
    )
    if val:
        return val[0].payload
    return []


def get_book(category: int, book_id: int, chunk_index: int):
    val, _ = client.scroll(
        collection_name="books",
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="ids",
                    match=models.MatchValue(
                        value=f"{category}:{book_id}:{chunk_index}"
                    ),
                )
            ]
        ),
        with_vectors=False,
    )
    if val:
        return val[0].payload
    return []


get_tafsir("tabary", "1:7")
get_hadith("bukhari", 2.0)
get_book(1, 121, 1)
