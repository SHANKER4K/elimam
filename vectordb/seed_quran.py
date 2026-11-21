from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    Distance,
    VectorParams,
    SparseVector,
    models,
)
from fastembed import SparseTextEmbedding
from tqdm import tqdm

from _text import clean

SOURCE_KEYS = ("ids", "documents", "metadatas", "embeddings")


def seed_quran(
    data: dict,
    qdrant_url: str = "http://localhost:6333",
    sparse_embedding_model: str = "qdrant/bm25",
) -> None:
    """Upsert the `quran` collection.

    `data` carries the four keys the payload needs:
    `ids`, `documents` (verse text), `metadatas` (surah_number, surah, ...) and
    `embeddings` (768-d dense vectors).

    ponytail: the source-loading step (chromadb export / datasets) is not in this
    repo, so it is not guessed here — prepare `data` and pass it in. The
    text-cleaning and batching below are all that matter for the index.
    """
    missing = [key for key in SOURCE_KEYS if key not in data]
    if missing:
        raise ValueError(f"data is missing {missing}")

    client = QdrantClient(url=qdrant_url)

    if not client.collection_exists("quran"):
        client.create_collection(
            collection_name="quran",
            vectors_config={
                "dense": VectorParams(size=768, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

    texts = [clean(text) for text in data["documents"]]
    sparse_model = SparseTextEmbedding(sparse_embedding_model)
    sparse_vecs = list(sparse_model.embed(texts))

    points = [
        PointStruct(
            id=i + 1,
            vector={
                "dense": data["embeddings"][i],
                "sparse": SparseVector(
                    indices=sv.indices.tolist(), values=sv.values.tolist()
                ),
            },
            payload={
                "ids": data["ids"][i],
                "text": texts[i],
                **data["metadatas"][i],
            },
        )
        for i, sv in enumerate(sparse_vecs)
    ]

    for i in tqdm(range(0, len(points), 100), desc="Adding: quran"):
        client.upsert(
            collection_name="quran",
            wait=True,
            points=points[i : i + 100],
        )
