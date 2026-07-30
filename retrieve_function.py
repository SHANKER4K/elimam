import chromadb
from typing import Dict, Any
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import normalize_alef_ar
from functools import lru_cache
import json
from cli.whoosh_search import (
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
        path="Islam",
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
) = load_assets("models/gate_arabert_onnx", "models/gate_reranker_onnx")


def _flatten(doc):
    """Turn a ChromaDB get result into a flat dict."""
    if doc is None or not doc.get("ids"):
        return None
    result = {"id": doc["ids"][0], "text": doc.get("documents", [""])[0]}
    meta = doc.get("metadatas", [{}])[0]
    if meta:
        result.update(meta)
    return result


def get_ayahs(surah: int, ayah: str):
    """Get verse(s) from the Quran.

    Args:
        surah: surah number (e.g. 1)
        ayah: ayah number or range (e.g. "1" or "1-5")

    Returns:
        list[dict] — each flat dict has keys: id, text, surah, surah_number, ayah_number
    """
    ayat = [ayah]
    if "-" in ayah:
        ayat = ayah.split("-")

    ayat = [
        _flatten(quran_collection.get(f"{surah}:{aya}"))
        for aya in range(int(ayat[0]), int(ayat[-1]) + 1)
    ]

    return [a for a in ayat if a is not None]


def get_hadith(book, hadith_number):
    """Get a specific hadith by book and hadith number.

    Args:
        book: one of abudaud, bukhari, ibnmaja, muslim, nesai, tirmizi
        hadith: hadith number

    Returns:
        dict — flat keys: id, text, book, chapter_number, chapter,
               hadith_number, grade, sanad
    """
    return _flatten(hadith_collection.get(f"{book}:{hadith_number}.0"))


def get_tafsir(book: str, surah: str, ayah: str):
    """Get tafsir for a verse from one or all books.

    Args:
        book: one of tabary, katheer, saadi, baghawy, or '*' for all
        surah: surah number
        ayah: ayah number or range (e.g. "1" or "1-5")

    Returns:
        list[dict] — each flat dict has keys: id, text, aya, surah_name,
               surah_number, aya_number, tafsir_book, tafsir_book_short, era
    """
    books = ["tabary", "katheer", "saadi", "baghawy"] if book == "*" else [book]
    if "-" in ayah:
        start, end = map(int, ayah.split("-"))
        ayat = list(range(start, end + 1))
    else:
        ayat = [int(ayah)]
    results = [tafsir_collection.get(f"{b}:{surah}:{a}") for b in books for a in ayat]
    return [r for r in (_flatten(d) for d in results) if r is not None]


def get_aqeedah(book_category: int, book_id: str, chunk_num: str | None = None):
    """Get a specific aqeedah chunk by book ID and optionally page.

    Args:
        book_id: book ID (string to preserve leading zeros if any)
        chunk_page: optional page number

    Returns:
        dict — flat keys: id, text, book_id, book_name, category_name,
               all_authors, chunk_page, book_pages, source
    """
    key = f"{book_category}:{book_id}:{chunk_num}"
    return _flatten(aqeedah_collection.get(key))


# ponytail: Whoosh internal scores stripped before returning — agent only needs content + reranker score
_HYBRID_NOISE = frozenset(
    {"hybrid_score", "keyword_score", "semantic_score", "score_norm"}
)


def _clean_results(results, query, threshold):
    """Flatten meta, strip Whoosh noise, rerank, filter by threshold."""
    for r in results:
        if "meta" in r:
            r.update(r.pop("meta"))
        for k in _HYBRID_NOISE:
            r.pop(k, None)

    texts = [normalize_alef_ar(dediac_ar(d["text"])) for d in results]
    ranks = reranker.rank(query, texts)
    return [
        {**results[int(r["corpus_id"])], "score": float(r["score"])}
        for r in ranks
        if r["score"] > threshold
    ]


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
        k: عدد النتائج
        alpha: وزن semantic (0.0 = keyword فقط, 1.0 = semantic فقط)
        where: JSON string for metadata filter
        where_document: JSON string for document content filter

    Returns:
        [{id, text, score, surah, surah_number, ayah_number}, ...]
    """
    query = normalize_alef_ar(dediac_ar(query))
    _where = json.loads(where) if where else None
    _where_document = json.loads(where_document) if where_document else None

    def semantic_fn(q, n):
        return quran_collection.query(
            query_texts=[q], n_results=n, where=_where, where_document=_where_document
        )

    results = hybrid_search_verses(query, k=k, semantic_func=semantic_fn, alpha=alpha)
    return _clean_results(results, query, threshold)


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

    Returns:
        [{id, text, score, book, grade, hadith_number, ...}, ...]
    """
    query = normalize_alef_ar(dediac_ar(query))
    _where = json.loads(where) if where else None
    _where_document = json.loads(where_document) if where_document else None

    def semantic_fn(q, n):
        return hadith_collection.query(
            query_texts=[q], n_results=n, where=_where, where_document=_where_document
        )

    results = hybrid_search_hadith(query, k=k, semantic_func=semantic_fn, alpha=alpha)
    return _clean_results(results, query, threshold)


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

    Returns:
        [{id, text, score, book_name, category_name, ...}, ...]
    """
    query = normalize_alef_ar(dediac_ar(query))
    _where = json.loads(where) if where else None
    _where_document = json.loads(where_document) if where_document else None

    def semantic_fn(q, n):
        return aqeedah_collection.query(
            query_texts=[q], n_results=n, where=_where, where_document=_where_document
        )

    results = hybrid_search_aqeedah(query, k=k, semantic_func=semantic_fn, alpha=alpha)
    return _clean_results(results, query, threshold)


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

    Returns:
        [{id, text, score, tafsir_book, surah, ayah_number}, ...]
    """
    query = normalize_alef_ar(dediac_ar(query))
    _where = json.loads(where) if where else None
    _where_document = json.loads(where_document) if where_document else None

    def semantic_fn(q, n):
        return tafsir_collection.query(
            query_texts=[q], n_results=n, where=_where, where_document=_where_document
        )

    results = hybrid_search_tafsir(query, k=k, semantic_func=semantic_fn, alpha=alpha)
    return _clean_results(results, query, threshold)
