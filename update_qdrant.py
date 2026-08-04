import torch
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    Distance,
    VectorParams,
    SparseVector,
    models,
)
from fastembed import SparseTextEmbedding
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import (
    normalize_alef_ar,
    normalize_teh_marbuta_ar,
    normalize_alef_maksura_ar,
)
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

model = SentenceTransformer(
    "models/gate_arabert_onnx",
    backend="onnx",
    model_kwargs={"file_name": "model_int8.onnx"},
)

client = QdrantClient(url="http://localhost:6333")

if client.collection_exists("books"):
    client.delete_collection("books")

client.create_collection(
    "books",
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

# %%
from datasets import load_dataset

ds = load_dataset("SHK4K/islam_books")
df = ds["train"].to_pandas()
df.head()
del ds

for book in df["book_id"].unique():
    chunks_num = np.arange(1, len(df[df["book_id"] == book]) + 1)
    df.loc[df["book_id"] == book, "chunk_num"] = chunks_num


def book_meta(row):
    return {
        "book_id": int(row.book_id),
        "book_name": row.book_name,
        "bood_data": row.book_date,
        "category_name": row.category_name,
        "all_authors": row.all_authors,
        "headers": row.headers_tree,
        "chunk_page": row.chunk_page,
        "author_death": row.main_author_death,
        "source": f"https://shamela.ws/book/{row.book_id}/{row.chunk_page.split(', ')[0]}",
    }


# %%
def clean(text):
    text = dediac_ar(text)
    text = normalize_alef_ar(text)
    text = text.replace("\u200f", "")
    text = text.replace("  ", "")
    return text


def embed_docs(text):
    text = clean(text)
    return model([text])


# %%
from tqdm import tqdm

book_ids, docs, metadatas = [], [], []

for row in tqdm(df.itertuples(), total=len(df)):
    ids = (
        str(row.book_category) + ":" + str(row.book_id) + ":" + str(int(row.chunk_num))
    )

    docs.append(row.chunks)

    metadata = book_meta(row)

    book_ids.append(ids)
    metadatas.append(metadata)


# embs = [embed_docs(d) for d in tqdm(docs)]

# %%
embs = torch.load("tensors/books_emb.pt").tolist()

# %%
sparse_model = SparseTextEmbedding(
    "Qdrant/bm25"
)  # BM25 TF vectors; IDF applied by Qdrant via Modifier.IDF

sparse_vecs = list(
    sparse_model.embed([clean(d) for d in docs])
)  # batch — one call, not per-point

points = []
for i in tqdm(range(len(book_ids))):
    sv = sparse_vecs[i]
    points.append(
        PointStruct(
            id=i + 1,
            vector={
                # 'dense': embs[i],
                "sparse": SparseVector(
                    indices=sv.indices.tolist(), values=sv.values.tolist()
                ),
            },
            payload={
                "ids": book_ids[i],
                "text": docs[i],
                **metadatas[i],
            },
        )
    )

# %%
del embs, df

# %%


for i in tqdm(range(0, len(points), 100), desc=f"Adding: {'books'}"):
    client.upsert(
        collection_name="books",
        wait=True,
        points=points[i : i + 100],
    )
