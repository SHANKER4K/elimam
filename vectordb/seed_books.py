"""
Ingest SHK4K/islam_books dataset into Qdrant (hybrid dense + sparse).
- Loads dataset from HuggingFace
- Cleans Arabic text (dediac, normalize_alef, remove RTL mark)
- Generates dense embeddings (GATE-AraBert-v1) on CUDA with FP16
- Generates sparse BM25 vectors (FastEmbed)
- Creates Qdrant collection 'books' if missing
- Batch-upserts points with dual vectors
"""

import torch
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, PointStruct, Modifier
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import normalize_alef_ar

# =========================
# Configuration
# =========================
DATASET_NAME = "SHK4K/islam_books"
COLLECTION_NAME = "books"
QDRANT_URL = "http://localhost:6333"  # Change if needed

DENSE_MODEL_NAME = "Omartificial-Intelligence-Space/GATE-AraBert-v1"
SPARSE_MODEL_NAME = "Qdrant/bm25"

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

DENSE_BATCH_SIZE = 64       # GPU embedding batch
SPARSE_BATCH_SIZE = 128     # FastEmbed sparse batch
UPSERT_BATCH_SIZE = 256     # Qdrant upsert batch

# =========================
# Helpers: Text Cleaning
# =========================

def clean_text(text: str) -> str:
    text = dediac_ar(text)
    text = normalize_alef_ar(text)
    text = text.replace("\u200f", "")  # Remove RTL mark
    text = text.replace("  ", " ")     # Collapse double spaces
    return text.strip()

# =========================
# Metadata Functions
# =========================
def book_meta(row) -> dict:
    """Build payload metadata from a dataset row (namedtuple from itertuples)."""
    # Safely handle headers_tree if it's a list/dict
    headers_tree = row.headers_tree
    if hasattr(headers_tree, "tolist"):
        headers_tree = headers_tree.tolist()
    
    return {
        "book_id": int(row.book_id),
        "book_name": row.book_name,
        "book_date": row.book_date,
        "category_name": row.category_name,
        "all_authors": row.all_authors,
        "headers": headers_tree,
        "chunk_page": row.chunk_page,
        "author_death": row.main_author_death,
        "source": f"https://shamela.ws/book/{row.book_id}/{str(row.chunk_page).split(', ')[0]}",
    }

# =========================
# Main Pipeline
# =========================
def main():
    # 1. Load Dataset
    print(f"Loading dataset '{DATASET_NAME}'...")
    ds = load_dataset(DATASET_NAME)
    df = ds["train"].to_pandas()
    del ds

    # 2. Assign chunk numbers per book
    print("Assigning chunk numbers...")
    df["chunk_num"] = 0
    for book_id in tqdm(df["book_id"].unique(), desc="Chunking"):
        mask = df["book_id"] == book_id
        df.loc[mask, "chunk_num"] = np.arange(1, len(df[mask]) + 1)

    # 3. Initialize Models
    print("Loading dense model (CUDA + FP16)...")
    dense_model = SentenceTransformer(
        DENSE_MODEL_NAME,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    if torch.cuda.is_available():
        dense_model.half()  # FP16 for speed

    print("Loading sparse BM25 model...")
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

    # 4. Initialize Qdrant Client
    client = QdrantClient(url=QDRANT_URL)

    # 5. Create Collection if missing
    if not client.collection_exists(COLLECTION_NAME):
        print(f"Creating collection '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=768,  # GATE-AraBert-v1 dimension
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(
                    modifier=Modifier.IDF,
                ),
            },
        )
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

    # 6. Prepare Data for Embedding
    print("Preparing documents...")
    docs = df["chunks"].apply(clean_text).tolist()
    total_docs = len(docs)

    # 7. Generate Dense Embeddings (Batched on GPU)
    print("Generating dense embeddings...")
    dense_embeddings = []
    for i in tqdm(range(0, total_docs, DENSE_BATCH_SIZE), desc="Dense Embedding"):
        batch_docs = docs[i:i + DENSE_BATCH_SIZE]
        batch_embs = dense_model.encode(
            batch_docs,
            batch_size=len(batch_docs),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        dense_embeddings.extend(batch_embs.tolist())

    # 8. Generate Sparse Embeddings (Batched)
    print("Generating sparse embeddings...")
    sparse_embeddings = list(
        sparse_model.embed(docs, batch_size=SPARSE_BATCH_SIZE)
    )

    # 9. Build and Upsert Points in Batches
    print("Building points and upserting to Qdrant...")
    points_buffer = []

    for i in tqdm(range(total_docs), desc="Upserting"):
        row = df.iloc[i]
        doc_id = f"{row.book_category}:{row.book_id}:{int(row.chunk_num)}"
        
        payload = book_meta(row)
        payload["ids"] = doc_id
        payload["text"] = docs[i]

        point = PointStruct(
            id=i + 1,  # Sequential integer IDs
            vector={
                DENSE_VECTOR_NAME: dense_embeddings[i],
                SPARSE_VECTOR_NAME: models.SparseVector(
                    indices=sparse_embeddings[i].indices.tolist(),
                    values=sparse_embeddings[i].values.tolist(),
                ),
            },
            payload=payload,
        )
        points_buffer.append(point)

        # Flush buffer when full
        if len(points_buffer) >= UPSERT_BATCH_SIZE:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points_buffer,
                wait=True
            )
            points_buffer = []

    # Flush remaining points
    if points_buffer:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points_buffer,
            wait=True
        )

    print("✅ Ingestion completed successfully!")

if __name__ == "__main__":
    main()