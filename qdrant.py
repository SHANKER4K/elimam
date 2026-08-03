from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    Distance,
    VectorParams,
    SparseVector,
    models,
)
from fastembed import SparseTextEmbedding
import chromadb
from tqdm import tqdm

from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import (
    normalize_alef_ar,
    normalize_teh_marbuta_ar,
    normalize_alef_maksura_ar,
)


client = QdrantClient(url="http://localhost:6333")
chroma = chromadb.PersistentClient(path="Islam")

sparse_model = SparseTextEmbedding(
    "Qdrant/bm25"
)  # BM25 TF vectors; IDF applied by Qdrant via Modifier.IDF


def clean(text):
    text = dediac_ar(text)
    text = normalize_alef_ar(text)
    text = text.replace("\u200f", "")
    text = text.replace("  ", "")
    return text


for name in ["quran"]:
    collection = chroma.get_or_create_collection(name=name)
    data = collection.get(include=["documents", "embeddings", "metadatas"])

    sparse_vecs = list(
        sparse_model.embed([clean(d) for d in data["documents"]])
    )  # batch — one call, not per-point

    points = []
    for i in range(len(data["ids"])):
        id = i + 1
        # if name == "quran":
        #     surah, verse = data["ids"][0].split(":")
        #     id = surah * 1000 + verse
        sv = sparse_vecs[i]
        points.append(
            PointStruct(
                id=id,
                vector={
                    "dense": data["embeddings"][i],
                    "sparse": SparseVector(
                        indices=sv.indices.tolist(), values=sv.values.tolist()
                    ),
                },
                payload={
                    "ids": data["ids"][i],
                    "text": data["documents"][i],
                    **data["metadatas"][i],
                },
            )
        )

    # ----

    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config={
                "dense": models.VectorParams(
                    size=768,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

    for i in tqdm(range(0, len(points), 100), desc=f"Adding: {name}"):
        client.upsert(
            collection_name=name,
            wait=True,
            points=points[i : i + 100],
        )
