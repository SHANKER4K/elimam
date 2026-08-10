from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "models/gate_arabert_onnx",
    backend="onnx",
    model_kwargs={"file_name": "model_int8.onnx"},
)


client = QdrantClient(url="localhost:6333")


def get_quran(id: str):

    return client.scroll(
        collection_name="quran",
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="ids",
                    match=models.MatchValue(value=id),
                ),
            ]
        ),
    )
sparse_model = SparseTextEmbedding("Qdrant/bm25")


def _sparse_query(text: str) -> models.SparseVector:
    """Embed a query with fastembed BM25. Raw text (no dediac) — matches how sparse
    vectors were indexed in update_qdrant.py."""
    sv = list(sparse_model.embed([text]))[0]
    return models.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())


def sparse_search(collection: str, query_text: str, top_k: int = 10) -> list:
    return [
        p.model_dump()
        for p in client.query_points(
            collection_name=collection,
            query=_sparse_query(query_text),
            using="sparse",
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        ).points
    ]
client.query_points?

sparse_search('quran','الرحمن',2)



















