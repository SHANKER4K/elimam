from fastapi import FastAPI
import chromadb
from typing import Dict, Any
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import normalize_alef_ar
from functools import lru_cache
import json
from fastapi.middleware.cors import CORSMiddleware
from whoosh_search import (
    hybrid_search_verses,
    hybrid_search_hadith,
    hybrid_search_tafsir,
    hybrid_search_aqeedah,
)
from sentence_transformers import CrossEncoder


# Custom Embedding Function
@register_embedding_function
class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformerEmbeddingFunction(
            model_name=model_name,
            normalize_embeddings=False,
            backend="onnx",
            model_kwargs={"file_name": "model_int8.onnx"},
        )

    def __call__(self, docs: Documents) -> Embeddings:
        return self.model(docs)

    @staticmethod
    def name() -> str:
        return "GATE-AraBert-v1"

    def get_config(self) -> Dict[str, Any]:
        return dict(model_name=self.model_name)

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "EmbeddingFunction":
        return MyEmbeddingFunction(config["model_name"])


@lru_cache(maxsize=1)
def ranker_loader(model):
    return CrossEncoder(
        model, backend="onnx", model_kwargs={"file_name": "model_int8.onnx"}
    )


@lru_cache(maxsize=1)
def load_assets(name, reranker_name):
    print("Loading Model:")
    model = MyEmbeddingFunction(name)
    print("Model Loaded")

    print("Loading Reranker")
    reranker = ranker_loader(reranker_name)
    print("Loaded")
    # Create Client
    client = chromadb.PersistentClient(
        path="../Islam",
    )

    quran_collection = client.get_or_create_collection(
        name="quran", embedding_function=model
    )

    tafsir_collection = client.get_or_create_collection(
        name="tafsir", embedding_function=model
    )

    hadith_collection = client.get_or_create_collection(
        name="hadith", embedding_function=model
    )
    aqeedah_collection = client.get_or_create_collection(
        name="aqeedah", embedding_function=model
    )
    return (
        model,
        reranker,
        quran_collection,
        tafsir_collection,
        hadith_collection,
        aqeedah_collection,
    )


(
    model,
    reranker,
    quran_collection,
    tafsir_collection,
    hadith_collection,
    aqeedah_collection,
) = load_assets("../models/gate_arabert_onnx", "../models/gate_reranker_onnx")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Check server health and collection sizes."""
    return {
        "status": "ok",
        "quran_count": quran_collection.count(),
        "tafsir_count": tafsir_collection.count(),
        "hadith_count": hadith_collection.count(),
        "aqeedah_count": aqeedah_collection.count(),
    }


@app.get("/get-ayah")
def get_ayahs(surah, ayah):
    """Get verse(s) from the Quran.

    Args:
        surah: surah number (e.g. 1)
        ayah: ayah number or range (e.g. "1" or "1-5")

    Returns:
        list[dict] — each dict has keys: ids, documents, metadatas.
        Metadata format: {"surah": str, "surah_number": int, "ayah_number": int}
    """
    ayat = [ayah]
    if "-" in ayah:
        ayat = ayah.split("-")

    ayat = [
        quran_collection.get(f"{surah}:{aya}")
        for aya in range(int(ayat[0]), int(ayat[-1]) + 1)
    ]
    return ayat


@app.get("/get-hadith")
def get_hadith(book, hadith_number):
    """Get a specific hadith by book and hadith number.

    Args:
        book: one of abudaud, bukhari, ibnmaja, muslim, nesai, tirmizi
        hadith: hadith number

    Returns:
        dict — keys: ids, documents, metadatas.
        Metadata format: {"book": str, "chapter_number": int, "chapter": str,
                         "hadith_number": int, "grade": str, "sanad": str}
    """
    return hadith_collection.get(f"{book}:{hadith_number}.0")


@app.get("/get-tafsir")
def get_tafsir(book: str, surah: str, ayah: str):
    """Get tafsir for a verse from one or all books.

    Args:
        book: one of tabary, katheer, saadi, baghawy, or '*' for all
        surah: surah number
        ayah: ayah number or range (e.g. "1" or "1-5")

    Returns:
        list[dict] — each dict has keys: ids, documents, metadatas.
        Metadata format: {"aya": str, "surah_name": str, "surah_numer": int,
                         "aya_number": int, "tafsir_book": str,
                         "tafsir_book_short": str, "era": str}
    """
    books = ["tabary", "katheer", "saadi", "baghawy"] if book == "*" else [book]
    if "-" in ayah:
        start, end = map(int, ayah.split("-"))
        ayat = list(range(start, end + 1))
    else:
        ayat = [int(ayah)]
    return [tafsir_collection.get(f"{b}:{surah}:{a}") for b in books for a in ayat]


@app.get("/get-aqeedah")
def get_aqeedah(book_category: int, book_id: str, chunk_num: str | None = None):
    """Get a specific aqeedah chunk by book ID and optionally page.

    Args:
        book_id: book ID (string to preserve leading zeros if any)
        chunk_page: optional page number

    Returns:
        dict — keys: ids, documents, metadatas.
        Metadata format: {"book_id": int, "book_name": str, "category_name": str,
                         "all_authors": str, "chunk_page": str, "book_pages": int,
                         "source": str}
    """
    key = f"{book_category}:{book_id}:{chunk_num}"
    return aqeedah_collection.get(key)


@app.get("/quran-search")
def search_quran(
    query: str, k: int = 10, where: str | None = None, where_document: str | None = None
):
    """Semantic search over the Quran corpus. Both `where` and `where_document` are JSON strings matching ChromaDB's filter types."""
    query = normalize_alef_ar(dediac_ar(query))
    docs = quran_collection.query(
        query_texts=[query],
        n_results=k,
        where=json.loads(where) if where else None,
        where_document=json.loads(where_document) if where_document else None,
    )

    texts = [normalize_alef_ar(dediac_ar(d["text"])) for d in docs]
    ranks = reranker.rank(query, texts)
    res = []
    for r in ranks:
        if r["score"] > 0.3:
            doc = docs[r["corpus_id"]]
            res.append({**doc, "score": r["score"]})
    return res


@app.get("/hadith-search")
def search_hadith(
    query: str, k: int = 10, where: str | None = None, where_document: str | None = None
):
    """Semantic search over the hadith corpus."""
    query = normalize_alef_ar(dediac_ar(query))
    docs = hadith_collection.query(
        query_texts=[query],
        n_results=k,
        where=json.loads(where) if where else None,
        where_document=json.loads(where_document) if where_document else None,
    )

    texts = [normalize_alef_ar(dediac_ar(d["text"])) for d in docs]
    ranks = reranker.rank(query, texts)
    res = []
    for r in ranks:
        if r["score"] > 0.3:
            doc = docs[r["corpus_id"]]
            res.append({**doc, "score": int(r["score"])})
    return res


@app.get("/tafsir-search")
def search_tafsir(
    query: str, k: int = 10, where: str | None = None, where_document: str | None = None
):
    """Semantic search over the tafsir corpus."""
    query = normalize_alef_ar(dediac_ar(query))

    docs = tafsir_collection.query(
        query_texts=[query],
        n_results=k,
        where=json.loads(where) if where else None,
        where_document=json.loads(where_document) if where_document else None,
    )

    texts = [normalize_alef_ar(dediac_ar(d["text"])) for d in docs]
    ranks = reranker.rank(query, texts)
    res = []
    for r in ranks:
        if r["score"] > 0.3:
            doc = docs[r["corpus_id"]]
            res.append({**doc, "score": int(r["score"])})
    return res


@app.get("/aqeedah-search")
def search_aqeedah(
    query: str, k: int = 10, where: str | None = None, where_document: str | None = None
):
    """Semantic search over the aqeedah corpus."""
    query = normalize_alef_ar(dediac_ar(query))
    docs = aqeedah_collection.query(
        query_texts=[query],
        n_results=k,
        where=json.loads(where) if where else None,
        where_document=json.loads(where_document) if where_document else None,
    )

    texts = [normalize_alef_ar(dediac_ar(d["text"])) for d in docs]
    ranks = reranker.rank(query, texts)
    res = []
    for r in ranks:
        if r["score"] > 0.3:
            doc = docs[r["corpus_id"]]
            res.append({**doc, "score": int(r["score"])})
    return res


# =====================
# Hybrid Search (Whoosh + ChromaDB)
# =====================


@app.get("/hybrid-quran-search")
def hybrid_quran_search(
    query: str,
    k: int = 10,
    alpha: float = 0.6,
    threshold: float = 0.3,
    where: str | None = None,
    where_document: str | None = None,
):
    """Hybrid search على القرآن: Semantic + Keyword (BM25).

    Args:
        query: نص البحث
        k: عدد النتائج (default 10)
        alpha: وزن semantic (0.0 = keyword فقط, 1.0 = semantic فقط)
        where: JSON string for metadata filter
        where_document: JSON string for document content filter

    Returns:
        قائمة نتائج مرتبة:
        [{id, text, hybrid_score, keyword_score, semantic_score, surah, ayah_number, surah_number}, ...]
    """
    query = normalize_alef_ar(dediac_ar(query))
    _where = json.loads(where) if where else None
    _where_document = json.loads(where_document) if where_document else None

    def semantic_fn(q, n):
        return quran_collection.query(
            query_texts=[q], n_results=n, where=_where, where_document=_where_document
        )

    results = hybrid_search_verses(
        query, k=k, semantic_func=semantic_fn if alpha > 0 else None, alpha=alpha
    )

    # إضافة أسماء السور وأرقام الآيات
    for r in results:
        # استخرج meta من ChromaDB إذا كانت موجودة
        if "meta" in r:
            r["surah"] = r["meta"].get("surah", "")
            r["surah_number"] = r["meta"].get("surah_number", 0)
            r["ayah_number"] = r["meta"].get("ayah_number", 0)
            del r["meta"]
        else:
            # من Whoosh
            pass  # Whoosh تخزنها في الحقول

    if alpha > 0:
        texts = [normalize_alef_ar(dediac_ar(d["text"])) for d in results]
        ranks = reranker.rank(query, texts)

        res = []
        for r in ranks:
            if r["score"] > threshold:
                doc = results[int(r["corpus_id"])]
                res.append({**doc, "score": float(r["score"])})
    else:
        res = results
    return res


@app.get("/hybrid-hadith-search")
def hybrid_hadith_search(
    query: str,
    k: int = 10,
    alpha: float = 0.6,
    threshold: float = 0.3,
    where: str | None = None,
    where_document: str | None = None,
):
    """Hybrid search على الحديث: Semantic + Keyword (BM25).

    Args:
        query: نص البحث
        k: عدد النتائج
        alpha: وزن semantic (0.0 = keyword فقط, 1.0 = semantic فقط)
        where: JSON string for metadata filter
        where_document: JSON string for document content filter
    """
    query = normalize_alef_ar(dediac_ar(query))
    _where = json.loads(where) if where else None
    _where_document = json.loads(where_document) if where_document else None

    def semantic_fn(q, n):
        return hadith_collection.query(
            query_texts=[q], n_results=n, where=_where, where_document=_where_document
        )

    results = hybrid_search_hadith(
        query, k=k, semantic_func=semantic_fn if alpha > 0 else None, alpha=alpha
    )

    # استخرج meta من ChromaDB إذا كانت موجودة
    for r in results:
        if "meta" in r:
            r["book"] = r["meta"].get("book", "")
            r["book_full"] = r["meta"].get("book_full", "")
            r["grade"] = r["meta"].get("grade", "")
            r["hadith_number"] = r["meta"].get("hadith_number", "")
            r["section"] = r["meta"].get("section", "")
            r["section_number"] = r["meta"].get("section_number", 0)
            r["source"] = r["meta"].get("source", "")
            del r["meta"]

    if alpha > 0:
        texts = [normalize_alef_ar(dediac_ar(d["text"])) for d in results]
        ranks = reranker.rank(query, texts)
        res = []
        for r in ranks:
            if r["score"] > threshold:
                doc = results[int(r["corpus_id"])]
                res.append({**doc, "score": float(r["score"])})
    else:
        res = results
    return res


@app.get("/hybrid-aqeedah-search")
def hybrid_aqeedah_search(
    query: str,
    k: int = 10,
    alpha: float = 0.6,
    threshold: float = 0.3,
    where: str | None = None,
    where_document: str | None = None,
):
    """Hybrid search على العقيدة: Semantic + Keyword (BM25).

    Args:
        query: نص البحث
        k: عدد النتائج
        alpha: وزن semantic (0.0 = keyword فقط, 1.0 = semantic فقط)
        where: JSON string for metadata filter
        where_document: JSON string for document content filter
    """
    query = normalize_alef_ar(dediac_ar(query))
    _where = json.loads(where) if where else None
    _where_document = json.loads(where_document) if where_document else None

    def semantic_fn(q, n):
        return aqeedah_collection.query(
            query_texts=[q], n_results=n, where=_where, where_document=_where_document
        )

    results = hybrid_search_aqeedah(
        query, k=k, semantic_func=semantic_fn if alpha > 0 else None, alpha=alpha
    )

    for r in results:
        if "meta" in r:
            r["book_id"] = r["meta"].get("book_id", 0)
            r["book_name"] = r["meta"].get("book_name", "")
            r["category_name"] = r["meta"].get("category_name", "")
            r["all_authors"] = r["meta"].get("all_authors", "")
            r["chunk_page"] = r["meta"].get("chunk_page", "")
            r["book_pages"] = r["meta"].get("book_pages", 0)
            r["source"] = r["meta"].get("source", "")
            del r["meta"]

    if alpha > 0:
        texts = [normalize_alef_ar(dediac_ar(d["text"])) for d in results]
        ranks = reranker.rank(query, texts)
        res = []
        for r in ranks:
            if r["score"] > threshold:
                doc = results[int(r["corpus_id"])]
                res.append({**doc, "score": float(r["score"])})
    else:
        res = results
    return res


@app.get("/hybrid-tafsir-search")
def hybrid_tafsir_search(
    query: str,
    k: int = 10,
    alpha: float = 0.6,
    threshold: float = 0.3,
    where: str | None = None,
    where_document: str | None = None,
):
    """Hybrid search على التفسير: Semantic + Keyword (BM25).

    Args:
        query: نص البحث
        k: عدد النتائج
        alpha: وزن semantic (0.0 = keyword فقط, 1.0 = semantic فقط)
        where: JSON string for metadata filter
        where_document: JSON string for document content filter
    """
    query = normalize_alef_ar(dediac_ar(query))
    _where = json.loads(where) if where else None
    _where_document = json.loads(where_document) if where_document else None

    def semantic_fn(q, n):
        return tafsir_collection.query(
            query_texts=[q], n_results=n, where=_where, where_document=_where_document
        )

    results = hybrid_search_tafsir(
        query, k=k, semantic_func=semantic_fn if alpha > 0 else None, alpha=alpha
    )

    # استخرج meta من ChromaDB إذا كانت موجودة
    for r in results:
        if "meta" in r:
            r["tafsir_book"] = r["meta"].get("tafsir_book", "")
            r["surah"] = r["meta"].get("surah", "")
            r["surah_number"] = r["meta"].get("surah_number", 0)
            r["ayah_number"] = r["meta"].get("ayah_number", 0)
            del r["meta"]

    if alpha > 0:
        texts = [normalize_alef_ar(dediac_ar(d["text"])) for d in results]
        ranks = reranker.rank(query, texts)
        res = []
        for r in ranks:
            if r["score"] > threshold:
                doc = results[int(r["corpus_id"])]
                res.append({**doc, "score": float(r["score"])})
    else:
        res = results
    return res
