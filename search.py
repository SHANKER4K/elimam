import httpx
import json
import os
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import normalize_alef_ar

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

print("Loading Model")
model = SentenceTransformer(
    "Omartificial-Intelligence-Space/GATE-AraBert-v1", model_kwargs={"dtype": "float16"}
)
print("Done")

# BM25 TF vectors; IDF is applied by Qdrant via Modifier.IDF (collection config)
sparse_model = SparseTextEmbedding("Qdrant/bm25")

client = QdrantClient(url=QDRANT_URL)


# Allowed filter keys per collection with their value type.
# "int" keys accept scalars, lists, or {eq|lt|gt|lte|gte: number} dicts.
# "str" keys accept scalars or lists. Anything else raises ValueError.
FILTER_SCHEMA: dict[str, dict[str, str]] = {
    "quran": {"surah_number": "int", "surah": "str"},
    "tafsir": {"surah_number": "int", "surah": "str", "ayah_number": "int"},
}


def setup_indexes():
    """Create payload indexes for every field used in getters and filters.

    Idempotent — safe to call repeatedly. Makes getter lookups and filtered
    searches use indexed access instead of full payload scans.
    """
    kind_schema = {
        "int": models.PayloadSchemaType.INTEGER,
        "str": models.PayloadSchemaType.KEYWORD,
    }
    fields: dict[str, set[tuple[str, models.PayloadSchemaType]]] = {
        "quran": {("ids", models.PayloadSchemaType.KEYWORD)},
        "tafsir": {
            ("ids", models.PayloadSchemaType.KEYWORD),
            ("tafsir_book", models.PayloadSchemaType.KEYWORD),
        },
    }
    for collection, keys in FILTER_SCHEMA.items():
        fields[collection].update(
            (key, kind_schema[kind]) for key, kind in keys.items()
        )
    for collection, index_set in fields.items():
        for field, schema in index_set:
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


_OPS = {"eq", "lt", "gt", "lte", "gte"}


def _build_filter(collection: str, filters: dict | None) -> models.Filter | None:
    """Translate {key: value} filters into a Qdrant Filter (or None for no filter)."""
    if not filters or filters == {}:
        return None

    if collection not in FILTER_SCHEMA:
        raise ValueError(
            f"Unknown collection {collection!r}; allowed: {sorted(FILTER_SCHEMA)}"
        )

    schema = FILTER_SCHEMA[collection]
    conditions = []
    for key, value in filters.items():
        if key not in schema:
            raise ValueError(
                f"Unknown filter key {key!r} for {collection!r}; allowed: {sorted(schema)}"
            )
        kind = schema[key]
        if isinstance(value, list):
            if not value:
                raise ValueError(f"Filter {key!r}: list must not be empty")
            if kind == "str":
                if not all(isinstance(v, str) for v in value):
                    raise ValueError(f"Filter {key!r}: list elements must be str")
            elif any(
                isinstance(v, bool) or not isinstance(v, (int, float)) for v in value
            ):
                raise ValueError(f"Filter {key!r}: list elements must be numbers")
            conditions.append(
                models.FieldCondition(key=key, match=models.MatchAny(any=value))
            )
        elif kind == "int":
            if isinstance(value, bool):
                raise ValueError(f"Filter {key!r}: bool is not allowed")
            if isinstance(value, dict):
                bad_ops = set(value) - _OPS
                if bad_ops:
                    raise ValueError(
                        f"Filter {key!r}: unknown op(s) {sorted(bad_ops)}; "
                        f"allowed: {sorted(_OPS)}"
                    )
                if any(
                    isinstance(v, bool) or not isinstance(v, (int, float))
                    for v in value.values()
                ):
                    raise ValueError(f"Filter {key!r}: op values must be numbers")
                if "eq" in value:
                    conditions.append(
                        models.FieldCondition(
                            key=key, match=models.MatchValue(value=value["eq"])
                        )
                    )
                ranges = {
                    op: value[op] for op in ("lt", "gt", "lte", "gte") if op in value
                }
                if ranges:
                    conditions.append(
                        models.FieldCondition(key=key, range=models.Range(**ranges))
                    )
            elif isinstance(value, (int, float)):
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                )
            else:
                raise ValueError(
                    f"Filter {key!r}: expected a number or {{op: number}}, "
                    f"got {type(value).__name__}"
                )
        else:  # "str" keys
            if isinstance(value, dict):
                raise ValueError(
                    f"Filter {key!r}: comparison ops only work on integer keys"
                )
            if not isinstance(value, str):
                raise ValueError(
                    f"Filter {key!r}: expected str, got {type(value).__name__}"
                )
            conditions.append(
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
            )
    return models.Filter(must=conditions)


def dense_search(
    collection: str, query_text: str, top_k: int = 10, filters: dict | None = None
) -> list:
    """Search one collection by dense vector similarity.

    Args:
        collection: One of "quran", "tafsir".
        query_text: Free-text query (Arabic).
        top_k: Number of results to return.
        filters: Optional metadata filters.
            Allowed keys per collection:
              quran  -> surah_number (int), surah (str)
              tafsir -> surah_number (int), surah (str), ayah_number (int)
    """
    if isinstance(filters, str):
        filters = json.loads(filters)
    try:
        res = [
            p.model_dump()
            for p in client.query_points(
                collection_name=collection.strip(),
                query=_dense_query(query_text),
                using="dense",
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                query_filter=_build_filter(collection, filters),
            ).points
        ]
    except ValueError:
        raise
    except Exception as e:
        return [{"error": f"{type(e).__name__}: {e}"}]
    return res


def sparse_search(
    collection: str, query_text: str, top_k: int = 10, filters: dict | None = None
) -> list:
    """Search one collection by exact BM25 keyword match.

    Args:
        collection: One of "quran", "tafsir".
        query_text: Free-text query (Arabic).
        top_k: Number of results to return.
        filters: Optional metadata filters.
            Allowed keys per collection:
              quran  -> surah_number (int), surah (str)
              tafsir -> surah_number (int), surah (str), ayah_number (int)
    """
    if isinstance(filters, str):
        filters = json.loads(filters)
    try:
        res = [
            p.model_dump()
            for p in client.query_points(
                collection_name=collection.strip(),
                query=_sparse_query(query_text),
                using="sparse",
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                query_filter=_build_filter(collection, filters),
            ).points
        ]
    except ValueError:
        raise
    except Exception as e:
        return [{"error": f"{type(e).__name__}: {e}"}]
    return res


def hybrid_search(
    collection: str,
    query_text: str,
    top_k: int = 10,
    pool: int = 50,
    filters: dict | None = None,
) -> list:
    """Hybrid search: dense + sparse in parallel, fused with RRF.

    Args:
        collection: One of "quran", "tafsir".
        query_text: Free-text query (Arabic).
        top_k: Number of results to return.
        pool: Candidates retrieved per retriever before fusion.
        filters: Optional metadata filters.
            Allowed keys per collection:
              quran  -> surah_number (int), surah (str)
              tafsir -> surah_number (int), surah (str), ayah_number (int)
    """
    if isinstance(filters, str):
        filters = json.loads(filters)
    query_text = _preprocess(query_text)
    try:
        res = [
            p.model_dump()
            for p in client.query_points(
                collection_name=collection.strip(),
                prefetch=[
                    models.Prefetch(
                        query=_dense_query(query_text), using="dense", limit=pool
                    ),
                    models.Prefetch(
                        query=_sparse_query(query_text), using="sparse", limit=pool
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                query_filter=_build_filter(collection, filters),
            ).points
        ]
    except ValueError:
        raise
    except Exception as e:
        return [{"error": f"{type(e).__name__}: {e}"}]
    return res


def hybrid_search_weighted(
    collection: str,
    query_text: str,
    top_k: int = 10,
    pool: int = 50,
    weights: tuple[float, float] = (0.7, 0.3),
    filters: dict | None = None,
) -> list:
    """Hybrid search with per-retriever weights (weighted RRF).

    Args:
        collection: One of "quran", "tafsir".
        query_text: Free-text query (Arabic).
        top_k: Number of results to return.
        pool: Candidates retrieved per retriever before fusion.
        weights: (dense_weight, sparse_weight).
        filters: Optional metadata filters.
            Allowed keys per collection:
              quran  -> surah_number (int), surah (str)
              tafsir -> surah_number (int), surah (str), ayah_number (int)
    """
    if isinstance(filters, str):
        filters = json.loads(filters)

    query_text = _preprocess(query_text)

    body = {
        "prefetch": [
            {"query": _dense_query(query_text), "using": "dense", "limit": pool},
            {
                "query": _sparse_query(query_text).model_dump(),
                "using": "sparse",
                "limit": pool,
            },
        ],
        "query": {"fusion": "rrf", "weights": list(weights)},
        "limit": top_k,
        "with_payload": True,
        "with_vectors": False,
    }
    try:
        query_filter = _build_filter(collection.strip(), filters)
        if query_filter is not None:
            body["filter"] = query_filter.model_dump()
        response = httpx.post(
            f"{QDRANT_URL}/collections/{collection}/points/query",
            json=body,
            timeout=30,
        )
        response.raise_for_status()
    except ValueError:
        raise
    except Exception as e:
        return [{"error": f"{type(e).__name__}: {e}"}]
    return response.json()["result"]["points"]


def get_quran(id: str):
    """Get one Quran ayah by its ids string (e.g. "1:1"). Returns the payload dict or []"""
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


def get_tafsir(book: str, id: str):
    """Get one tafsir entry by book slug + surah:ayah id (e.g. "tabary", "1:7"). Returns the payload dict or []"""
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


TAFSIR_BOOKS = ["saadi", "katheer", "moyassar", "tabary", "baghawy"]


def get_books_tafsir() -> list:
    """List all tafsir books (static).

    Hardcoded slugs — no database I/O, instant.

    Use when you need to filter data by tafsir book.

    Returns:
        ['saadi', 'katheer', 'moyassar', 'tabary', 'baghawy'] (a fresh copy).
    """
    return list(TAFSIR_BOOKS)
