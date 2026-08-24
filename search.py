import httpx
import json
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import normalize_alef_ar
from dotenv import load_dotenv
import os

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


def setup_indexes():
    """Create payload indexes for every field used in getters and filters.

    Idempotent — safe to call repeatedly. Makes getter lookups and filtered
    searches use indexed access instead of full payload scans (books has
    ~148k points).
    """
    kind_schema = {
        "int": models.PayloadSchemaType.INTEGER,
        "str": models.PayloadSchemaType.KEYWORD,
    }
    # Getter-only fields (ids, book, tafsir_book, book_id) are not in
    # FILTER_SCHEMA; filter keys are derived from it (single source of truth).
    fields: dict[str, set[tuple[str, models.PayloadSchemaType]]] = {
        "quran": {("ids", models.PayloadSchemaType.KEYWORD)},
        "hadith": {
            ("ids", models.PayloadSchemaType.KEYWORD),
            ("book", models.PayloadSchemaType.KEYWORD),
        },
        "tafsir": {
            ("ids", models.PayloadSchemaType.KEYWORD),
            ("tafsir_book", models.PayloadSchemaType.KEYWORD),
        },
        "books": {
            ("ids", models.PayloadSchemaType.KEYWORD),
            ("book_id", models.PayloadSchemaType.INTEGER),
        },
        "sunnah": {
            ("ids", models.PayloadSchemaType.KEYWORD),
            ("book_id", models.PayloadSchemaType.INTEGER),
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


# Allowed filter keys per collection with their value type.
# "int" keys accept scalars, lists, or {eq|lt|gt|lte|gte: number} dicts.
# "str" keys accept scalars or lists. Anything else raises ValueError.
FILTER_SCHEMA: dict[str, dict[str, str]] = {
    "quran": {"surah_number": "int", "surah": "str"},
    "hadith": {"book": "str", "grade": "str"},
    "tafsir": {"surah_number": "int", "surah": "str", "ayah_number": "int"},
    "books": {
        "book_id": "int",
        "book_name": "str",
        "category_name": "str",
        "all_authors": "str",
        "author_death": "int",
        "book_date": "int",
    },
    "sunnah": {
        "book_id": "int",
        "book_name": "str",
        "category_name": "str",
        "all_authors": "str",
        "author_death": "int",
        "book_date": "int",
        "athar_number": "int",
    },
}

_OPS = {"eq", "lt", "gt", "lte", "gte"}


def _build_filter(collection: str, filters: dict | None) -> models.Filter | None:
    """Translate {key: value} filters into a Qdrant Filter (or None for no filter).

    Value forms (multiple keys are AND-combined):
      scalar       -> equal match (MatchValue)
      list         -> equal to any of the values (MatchAny, OR within the key)
      dict (int)   -> {eq|lt|gt|lte|gte: number}; eq becomes MatchValue, the
                      remaining ops combine into one Range condition

    Raises ValueError on unknown keys, wrong value types, unknown ops, empty
    lists, bool values, and None values.
    """
    if not filters or filters == {}:
        return None

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

    Embeds `query_text` with the ONNX model (preprocessed like training data)
    and returns the `top_k` nearest points by cosine over the `dense` vectors.

    Args:
        collection: One of "quran", "hadith", "tafsir", "books", "sunnah".
        query_text: Free-text query (Arabic).
        top_k: Number of results to return.
        filters: Optional metadata filters to narrow results before scoring.
            Default None (or {}) = no filtering. Multiple keys are ANDed —
            every condition must match.

            Value forms:
              scalar       equal match            {"surah": "الفاتحة"}
              list         equal-any (OR in key)  {"surah": ["الفاتحة", "البقرة"]}
              dict (int)   comparisons eq/lt/gt/lte/gte
                                                   {"surah_number": {"gte": 2, "lt": 10}}
                                                   {"author_death": {"lt": 500}}

            Allowed keys per collection (see FILTER_SCHEMA):
              quran  -> surah_number (int), surah (str)
              hadith -> book (str), grade (str)
              tafsir -> surah_number (int), surah (str), ayah_number (int)
              books  -> book_id (int), book_name (str), category_name (str),
                        all_authors (str), author_death (int), book_date (int)

              sunnah  -> book_id (int), book_name (str), category_name (str),
                        all_authors (str), author_death (int), book_date (int), athar_number (int)

            Raises ValueError for unknown keys, wrong value types, unknown
            operators, empty lists, and bool values.

    Returns:
        List of dicts: {"id", "version", "score", "payload"} — same shape as
        every other search function.
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

    except Exception as e:
        return [e]
    return res


def sparse_search(
    collection: str, query_text: str, top_k: int = 10, filters: dict | None = None
) -> list:
    """Search one collection by exact BM25 keyword match.

    Embeds `query_text` with fastembed Qdrant/bm25 and searches the `sparse`
    vectors. Catches exact term matches that dense search misses.

    Args:
        collection: One of "quran", "hadith", "tafsir", "books", "sunnah".
        query_text: Free-text query (Arabic). Use exact terms — no stemming.
        top_k: Number of results to return.
        filters: Optional metadata filters to narrow results before scoring.
            Default None (or {}) = no filtering. Multiple keys are ANDed —
            every condition must match.

            Value forms:
              scalar       equal match            {"surah": "الفاتحة"}
              list         equal-any (OR in key)  {"surah": ["الفاتحة", "البقرة"]}
              dict (int)   comparisons eq/lt/gt/lte/gte
                                                   {"surah_number": {"gte": 2, "lt": 10}}
                                                   {"author_death": {"lt": 500}}

            Allowed keys per collection (see FILTER_SCHEMA):
              quran  -> surah_number (int), surah (str)
              hadith -> book (str), grade (str)
              tafsir -> surah_number (int), surah (str), ayah_number (int)
              books  -> book_id (int), book_name (str), category_name (str),
                        all_authors (str), author_death (int), book_date (int)
            sunnah  -> book_id (int), book_name (str), category_name (str),
                        all_authors (str), author_death (int), book_date (int), athar_number (int)

            Raises ValueError for unknown keys, wrong value types, unknown
            operators, empty lists, and bool values.

    Returns:
        List of dicts: {"id", "version", "score", "payload"}.
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
    except Exception as e:
        return [e]
    return res


def hybrid_search(
    collection: str,
    query_text: str,
    top_k: int = 10,
    pool: int = 50,
    filters: dict | None = None,
) -> list:
    """Hybrid search: dense + sparse in parallel, fused with RRF.

    Runs dense (cosine) and sparse (BM25) searches in parallel via prefetch,
    then fuses their candidate lists with Reciprocal Rank Fusion. RRF is
    rank-based, so the incomparable cosine/BM25 score scales don't matter.

    Args:
        collection: One of "quran", "hadith", "tafsir", "books", "sunnah".
        query_text: Free-text query (Arabic).
        top_k: Number of results to return.
        pool: Candidates retrieved per retriever before fusion. Larger = slower;
            50 is enough before reranking.
        filters: Optional metadata filters to narrow results before scoring.
            Default None (or {}) = no filtering. Multiple keys are ANDed —
            every condition must match. Applied to both the dense and the
            sparse retrievers before fusion.

            Value forms:
              scalar       equal match            {"surah": "الفاتحة"}
              list         equal-any (OR in key)  {"surah": ["الفاتحة", "البقرة"]}
              dict (int)   comparisons eq/lt/gt/lte/gte
                                                   {"surah_number": {"gte": 2, "lt": 10}}
                                                   {"author_death": {"lt": 500}}

            Allowed keys per collection (see FILTER_SCHEMA):
              quran  -> surah_number (int), surah (str)
              hadith -> book (str), grade (str)
              tafsir -> surah_number (int), surah (str), ayah_number (int)
              books  -> book_id (int), book_name (str), category_name (str),
                        all_authors (str), author_death (int), book_date (int)
            sunnah  -> book_id (int), book_name (str), category_name (str),
                        all_authors (str), author_death (int), book_date (int), athar_number (int)

            Raises ValueError for unknown keys, wrong value types, unknown
            operators, empty lists, and bool values.

    Returns:
        List of dicts: {"id", "version", "score", "payload"} — same type as
        every other search function.
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
    except Exception as e:
        return [e]
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

    Same as `hybrid_search` but each retriever's rank contribution is scaled
    by `weights` (dense, sparse). Sent as raw HTTP because the installed
    qdrant-client's FusionQuery model forbids the `weights` field, while the
    Qdrant server supports it.

    Args:
        collection: One of "quran", "hadith", "tafsir", "books", "sunnah".
        query_text: Free-text query (Arabic).
        top_k: Number of results to return.
        pool: Candidates retrieved per retriever before fusion.
        weights: (dense_weight, sparse_weight). Both must be >= 0; the bigger
            one dominates the ranking. Tune per collection with real queries.
        filters: Optional metadata filters to narrow results before scoring.
            Default None (or {}) = no filtering. Multiple keys are ANDed —
            every condition must match. Sent in the raw HTTP request body.

            Value forms:
              scalar       equal match            {"surah": "الفاتحة"}
              list         equal-any (OR in key)  {"surah": ["الفاتحة", "البقرة"]}
              dict (int)   comparisons eq/lt/gt/lte/gte
                                                   {"surah_number": {"gte": 2, "lt": 10}}
                                                   {"author_death": {"lt": 500}}

            Allowed keys per collection (see FILTER_SCHEMA):
              quran  -> surah_number (int), surah (str)
              hadith -> book (str), grade (str)
              tafsir -> surah_number (int), surah (str), ayah_number (int)
              books  -> book_id (int), book_name (str), category_name (str),
                        all_authors (str), author_death (int), book_date (int)
            sunnah  -> book_id (int), book_name (str), category_name (str),
                        all_authors (str), author_death (int), book_date (int), athar_number (int)

            Raises ValueError for unknown keys, wrong value types, unknown
            operators, empty lists, and bool values.

    Returns:
        List of dicts: {"id", "version", "score", "payload"} — already the
        raw JSON shape from the HTTP response, same type as the other three.
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
    except Exception as e:
        return [e]
    return response.json()["result"]["points"]


# %%


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


def get_hadith(book: str, hadith_number: float):
    """Get one hadith by book slug + number (e.g. "bukhari", 2.0). Returns the payload dict or []"""
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


def get_book(category: int, book_id: int, chunk_index: int):
    """Get one book chunk by category, book_id, chunk_index. Returns the payload dict or []"""
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


def get_suunah(category: int, book_id: int, chunk_index: int):
    """Get one book chunk by category, book_id, chunk_index. Returns the payload dict or []"""
    val, _ = client.scroll(
        collection_name="sunnah",
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


HADITH_BOOKS = [
    "abudawud",
    "bukhari",
    "dehlawi",
    "ibnmajah",
    "malik",
    "nasai",
    "nawawi",
    "qudsi",
    "tirmidhi",
]

TAFSIR_BOOKS = ["saadi", "katheer", "moyassar", "tabary", "baghawy"]

BOOKS_LIST = [
    "إجماع السلف في الاعتقاد كما حكاه حرب الكرماني",
    "الجواب الصحيح لمن بدل دين المسيح لابن تيمية",
    "اتباع السنن واجتناب البدع",
    "الصواعق المرسلة على الجهمية والمعطلة - ط عطاءات العلم",
    "شفاء العليل في مسائل القضاء والقدر والحكمة والتعليل - ط عطاءات العلم",
    "ثلاثة الأصول وشروط الصلاة والقواعد الأربع",
    "تحقيق الإيمان لابن تيمية",
    "تحقيق الاحتجاج بالقدر لابن تيمية",
    "العقيدة الصحيحة وما يضادها ونواقض الإسلام",
    "السنة لعبد الله بن أحمد",
    "أصول الدين الإسلامي مع قواعده الأربع",
    "الروح - ابن القيم - ط عطاءات العلم",
    "إقامة البراهين على حكم من استغاث بغير الله أو صدق الكهنة والعرافين",
    "كشف الشبهات - ت القاسم",
    "كتاب العلل الواقع بآخر جامع الترمذي - ت بشار",
    "القصيدة التائية في القدر",
    "الحث على التجارة - من «الجامع» للخلال - ت العوضي",
    "رأس الحسين",
    "منهاج السنة النبوية",
    "العقيدة التي حكاها أبو الفضل التميمي عن الإمام أحمد - المطبوع بآخر طبقات الحنابلة",
    "السنة لأبي بكر بن الخلال",
    "بر الوالدين - البخاري - ت مكي",
    "أصول الإيمان لمحمد بن عبد الوهاب - ت الجوابرة",
    "المعجم الكبير للطبراني",
    "العلل ومعرفة الرجال لأحمد رواية ابنه عبد الله",
    "بيان تلبيس الجهمية في تأسيس بدعهم الكلامية",
    "الرد على من قال بفناء الجنة والنار",
    "العقل وفضله لابن أبي الدنيا",
    "هداية الحيارى في أجوبة اليهود والنصارى - ط عطاءات العلم",
    "الأسامي والكنى - الإمام أحمد",
    "سؤالات أبي داود للإمام أحمد",
    "اصطناع المعروف لابن أبي الدنيا",
    "العلل ومعرفة الرجال لأحمد رواية المروذي وغيره ت صبحي السامرائي",
    "مسائل الإمام أحمد رواية ابنه عبد الله",
    "مسائل الإمام أحمد رواية ابنه أبي الفضل صالح",
    "الإخلاص والنية لابن أبي الدنيا",
    "فتاوى مهمة لعموم الأمة",
    "اختصاص القرآن بعوده إلى الرحيم الرحمن",
    "أصول السنة لأحمد بن حنبل",
    "الروح - ابن القيم - ط العلمية",
    "العلو للعلي الغفار",
    "تحريم النظر في كتب الكلام",
    "تميز الصدق من المين في محاورة الرجلين",
    "كلمة الإخلاص وتحقيق معناها - ط المكتب الإسلامي",
    "كشف الأوهام والإلتباس عن تشبيه بعض الأغبياء من الناس",
    "فتييان تتعلقان بتكفير الجهمية",
    "المنتقى من منهاج الاعتدال",
    "سيرة الإمام أحمد بن حنبل - لابنه صالح",
    "الاستقامة",
    "قصر الأمل لابن أبي الدنيا",
    "زيارة القبور والاستنجاد بالمقبور",
    "مختصر العلو للعلي العظيم",
    "الصارم المسلول على شاتم الرسول",
    "الصواعق المرسلة على الجهمية والمعطلة - ط العاصمة",
    "العرش للذهبي",
    "أحاديث في الفتن والحوادث (مطبوع ضمن مؤلفات الشيخ محمد بن عبد الوهاب، الجزء الحادي عشر)",
    "أحاديث في الفتن والحوادث ط القاسم",
    "أصول الإيمان لمحمد بن عبد الوهاب - ضمن مجموع مؤلفاته",
    "إقامة الحجة والدليل وإيضاح المحجة والسبيل",
    "الانتصار لحزب الله الموحدين والرد على المجادل عن المشركين",
    "الإيمان لابن تيمية",
    "البيان المبدي لشناعة القول المجدي",
    "الجواهر المضية لمجدد الدعوة النجدية",
    "الحسنة والسيئة",
    "الدرة البهية شرح القصيدة التائية في حل المشكلة القدرية",
    "الرد على البردة",
    "الرد على الجهمية والزنادقة للإمام أحمد ت صبري",
    "الرد على المنطقيين",
    "الرسائل الشخصية (مطبوع ضمن مؤلفات الشيخ محمد بن عبد الوهاب، الجزء السادس)",
    "الرسالة الأكملية في ما يجب لله من صفات الكمال",
    "الرسالة المدنية في تحقيق المجاز والحقيقة في صفات الله (مطبوع ضمن الفتوى الحموية الكبرى)",
    "الرسالة المفيدة",
    "الضياء الشارق في رد شبهات الماذق المارق",
    "سنن الترمذي - ت بشار",
    "مكائد الشيطان",
    "ذم الكذب - من الصمت وآداب اللسان",
    "مسائل الجاهلية",
    "الأربعون حديثا للآجري",
    "الإبانة الكبرى - ابن بطة",
    "الأمر بالمعروف والنهي عن المنكر- ابن أبي الدنيا",
    "الحث على التجارة - من «الجامع» للخلال - ت الحداد",
    "الزهد لابن أبي الدنيا",
    "اليقين لابن أبي الدنيا",
    "تحريم النرد والشطرنج والملاهي للآجري",
    "ذم البغى لابن أبي الدنيا",
    "ذم الملاهي لابن أبي الدنيا",
    "صفة الجنة لابن أبي الدنيا ت سليم",
    "ذم الغيبة والنميمة لابن أبي الدنيا",
    "كلام الليالي والأيام لابن أبي الدنيا",
    "أدب النفوس للآجري",
    "الأمر بالمعروف والنهي عن المنكر - من «الجامع» للخلال",
    "التوبة لابن أبي الدنيا",
    "التوكل على الله لابن أبي الدنيا",
    "الرقة والبكاء لابن أبي الدنيا",
    "الصبر والثواب عليه لابن أبي الدنيا",
    "العقوبات لابن أبي الدنيا",
    "القراءة عند القبور - من «الجامع» للخلال",
    "المطر والرعد والبرق لابن أبي الدنيا",
    "شرح العقيدة الطحاوية - ط الرسالة",
    "الزهد لأحمد بن حنبل",
    "القواعد الأربع (مطبوع ضمن مؤلفات الشيخ محمد بن عبد الوهاب، الجزء الأول)",
    "القول السديد في الرد على من أنكر تقسيم التوحيد",
    "القول السديد شرح كتاب التوحيد ط النفائس",
    "الكلمات النافعة في المكفرات الواقعة",
    "التوضيح والبيان لشجرة الإيمان",
    "رسالة في حكم السحر والكهانة مع بعض الفتاوى المهمة",
    "شرح السنة للبربهاري",
    "العقيدة الصحيحة وما يضادها",
    "الواسطة بين الحق والخلق",
    "العقائد الإسلامية لابن باديس",
    "الإخنائية أو الرد على الإخنائي ت العنزي",
    "الجوع لابن أبي الدنيا",
    "الفرج بعد الشدة لابن أبي الدنيا",
    "فضائل عثمان بن عفان لعبد الله بن أحمد",
    "مسند الشافعي",
    "التنبيهات اللطيفة على ما احتوت عليه العقيدة الواسطية من المباحث المنيفة",
    "ذم اللواط للآجري",
    "مسند الشافعي - ترتيب سنجر",
    "منهج أهل السنة والجماعة في السمع والطاعة",
    "خلق أفعال العباد للبخاري",
    "أصول السنة لابن أبي زمنين",
    "تفسير أسماء الله الحسنى للسعدي",
    "حكم الإسلام فيمن زعم أن القرآن متناقض",
    "العلل ومعرفة الرجال لأحمد رواية المروذي وغيره ت وصي الله عباس",
    "التمسك بالسنن والتحذير من البدع",
    "القول السديد شرح كتاب التوحيد ط الوزارة",
    "شرح العقيدة الطحاوية - ط المكتب الإسلامي التاسعة",
    "القبور لابن أبي الدنيا",
    "شرح العقيدة الطحاوية - ط الأوقاف السعودية - بتعليقات أحمد شاكر",
    "الرسالة العرشية",
    "فضائل رمضان لابن أبي الدنيا",
    "قرى الضيف لابن أبي الدنيا",
    "بغية المرتاد في الرد على المتفلسفة والقرامطة والباطنية",
    "تأسيس التقديس في كشف تلبيس داود بن جرجيس",
    "تنبيه ذوي الألباب السليمة عن والوقوع في الألفاظ المبتدعة الوخيمة",
    "ثلاثة الأصول (مطبوع ضمن مؤلفات الشيخ محمد بن عبد الوهاب، الجزء الأول)",
    "جواب أهل السنة النبوية في نقض كلام الشيعة والزيدية (مطبوع ضمن الرسائل والمسائل النجدية، الجزء الرابع، القسم الأول)",
    "حقوق آل البيت",
    "أصول الإيمان لابن باز",
    "دحض شبهات على التوحيد من سوء الفهم لثلاثة أحاديث",
    "رسالة في أصول الدين",
    "رسالة في القرآن وكلام الله",
    "فضل الإسلام (مطبوع ضمن مؤلفات الشيخ محمد بن عبد الوهاب، الجزءالأول)",
    "شرح العقيدة الأصفهانية",
    "شرح ثلاثة الأصول لابن باز",
    "شرح حديث النزول",
    "قاعدة جامعة في توحيد الله وإخلاص الوجه والعمل له عبادة واستعانة",
    "التوحيد لابن عبد الوهاب",
    "فائدة جليلة في قواعد الأسماء الحسنى",
    "كشف الشبهتين",
    "كشف غياهب الظلام عن أوهام جلاء الأوهام",
    "نونية ابن القيم الكافية الشافية - ط مكتبة ابن تيمية",
    "مجموعة رسائل في التوحيد والإيمان (مطبوع ضمن مؤلفات الشيخ محمد بن عبد الوهاب، الجزء الأول)",
    "اقتضاء الصراط المستقيم لمخالفة أصحاب الجحيم",
    "النبوات لابن تيمية",
    "شفاء العليل في مسائل القضاء والقدر والحكمة والتعليل - ط المعرفة",
    "مفيد المستفيد في كفر تارك التوحيد (مطبوع ضمن مؤلفات الشيخ محمد بن عبد الوهاب، الجزء الأول)",
    "منهاج أهل الحق والاتباع في مخالفة أهل الجهل والابتداع",
    "الأهوال لابن أبي الدنيا",
    "صفة الجنة لابن أبي الدنيا ت العساسلة",
    "محاسبة النفس لابن أبي الدنيا",
    "فضل قيام الليل والتهجد للآجري",
    "مقتل علي لابن أبي الدنيا",
    "قاعدة عظيمة في الفرق بين عبادات أهل الإسلام والإيمان وعبادات أهل الشرك والنفاق",
    "مجابو الدعوة لابن أبي الدنيا",
    "الإخوان لابن أبي الدنيا",
    "الأدب المفرد - ت عبد الباقي",
    "الإشراف في منازل الأشراف لابن أبي الدنيا",
    "الأشربة لأحمد بن حنبل",
    "الاعتبار وأعقاب السرور لابن أبي الدنيا",
    "الأولياء لابن أبي الدنيا",
    "التواضع والخمول لابن أبي الدنيا",
    "التوحيد لابن خزيمة",
    "الحلم لابن أبي الدنيا",
    "الرد على الجهمية لابن منده - ط المكتبة الأثرية",
    "الرضا عن الله بقضائه لابن أبي الدنيا",
    "السنن المأثورة للشافعي",
    "الشريعة للآجري",
    "الشكر لابن أبي الدنيا",
    "الصمت وآداب اللسان",
    "العمر والشيب لابن أبي الدنيا",
    "الغرباء للآجري",
    "المتمنين لابن أبي الدنيا",
    "المحتضرين لابن أبي الدنيا",
    "المرض والكفارات لابن أبي الدنيا",
    "المنامات لابن أبي الدنيا",
    "النفقة على العيال لابن أبي الدنيا",
    "الهم والحزن لابن أبي الدنيا",
    "الوجل والتوثق بالعمل لابن أبي الدنيا",
    "الورع لابن أبي الدنيا",
    "حسن الظن بالله لابن أبي الدنيا",
    "ذم المسكر لابن أبي الدنيا",
    "صفة النار لابن أبي الدنيا",
    "فضائل الصحابة لأحمد بن حنبل",
    "قضاء الحوائج لابن أبي الدنيا",
    "مداراة الناس لابن أبي الدنيا",
    "مكارم الأخلاق لابن أبي الدنيا",
    "من عاش بعد الموت لابن أبي الدنيا",
    "إصلاح المال",
    "نونية ابن القيم الكافية الشافية - ط عطاءات العلم",
    "الجامع لعلوم الإمام أحمد - أصول الفقه",
    "التسعينية",
    "السيف المسلول على من سب الرسول",
    "الرد على الجهمية للدارمي - ت الشوامي",
    "نقض الدارمي على المريسي - ت الشوامي",
    "العقيدة الواسطية - ت ابن مانع",
    "الفتوى الحموية الكبرى",
    "اجتماع الجيوش الإسلامية - ط عطاءات العلم",
    "الإيمان الأوسط - ط ابن الجوزي",
    "الجامع لعلوم الإمام أحمد - علوم الحديث",
    "الجامع لعلوم الإمام أحمد - شرح الأحاديث والآثار",
    "الجامع لعلوم الإمام أحمد - علل الحديث",
    "الجامع لعلوم الإمام أحمد - التفسير وعلوم القرآن",
    "الجامع لعلوم الإمام أحمد - الفقه",
    "الجامع لعلوم الإمام أحمد - العقيدة",
    "الجامع لعلوم الإمام أحمد - الرجال",
    "الجامع لعلوم الإمام أحمد - الأدب والزهد",
    "مسائل الإمام أحمد رواية أبي داود السجستاني",
    "مسند الشافعي - ترتيب السندي",
    "الفرقان بين أولياء الرحمن وأولياء الشيطان",
    "درء تعارض العقل والنقل",
    "مختصر منهاج السنة",
    "ذم الدنيا",
    "جواب في الحلف بغير الله والصلاة إلى القبور، ويليه: فصل في الاستغاثة",
    "اجتماع الجيوش الإسلامية - ت المعتق",
    "القناعة والتعفف",
    "العبودية",
    "قاعدة جليلة في التوسل والوسيلة",
    "مسألة في الكنائس",
    "العقيدة الواسطية - ت أشرف عبد المقصود",
    "التدمرية",
    "تحقيق القول في مسألة: عيسى كلمة الله والقرآن كلام الله",
    "مسألة فيما إذا كان في العبد محبة لما هو خير وحق ومحمود في نفسه",
    "هداية الحيارى في أجوبة اليهود والنصارى - ط دار القلم",
    "الصفدية",
    "مسند أحمد - ط الرسالة",
    "سؤالات الاثرم لأحمد بن حنبل",
    "حديث سفيان بن عيينة رواية المروزي",
    "العزلة والانفراد",
    "من حديث سفيان الثوري - ت عامر صبري",
    "حلم معاوية لابن أبي الدنيا",
    "الهواتف = هواتف الجنان لابن أبي الدنيا",
    "بيان التوحيد الذي بعث الله به الرسل جميعا وبعث به خاتمهم محمدا عليه السلام",
    "كشف الشبهات - ط الأوقاف السعودية",
    "لمعة الاعتقاد",
    "نواقض الإسلام",
    "وجوب تحكيم شرع الله ونبذ ما خالفه",
    "معنى لا إله إلا الله - محمد بن عبد الوهاب",
    "التحذير من البدع",
    "الناهية عن طعن أمير المؤمنين معاوية",
    "مسند الدارمي - ت الزهراني",
    "المقدمة الزهرا في إيضاح الإمامة الكبرى",
    "رسالة الشرك ومظاهره",
    "الإخنائية أو الرد على الإخنائي ت زهوي",
    "عقيدة السلف - مقدمة أبي زيد القيرواني لكتابه الرسالة",
    "مسند أحمد - ت شاكر - ط دار الحديث",
    "سنن أبي داود - ت الأرنؤوط",
    "مسألة في توحيد الفلاسفة",
    "مقدمة تشتمل على أن جميع الرسل كان دينهم الإسلام",
    "كلمة الإخلاص وتحقيق معناها - ضمن رسائل ابن رجب",
]

SUNNAH_BOOKS = [
    "السنة لعبد الله بن أحمد",
    "الحث على التجارة - من «الجامع» للخلال - ت العوضي",
    "السنة لأبي بكر بن الخلال",
    "بر الوالدين - البخاري - ت مكي",
    "المعجم الكبير للطبراني",
    "العلل ومعرفة الرجال لأحمد رواية ابنه عبد الله",
    "العقل وفضله لابن أبي الدنيا",
    "الأسامي والكنى - الإمام أحمد",
    "سؤالات أبي داود للإمام أحمد",
    "اصطناع المعروف لابن أبي الدنيا",
    "العلل ومعرفة الرجال لأحمد رواية المروذي وغيره ت صبحي السامرائي",
    "مسائل الإمام أحمد رواية ابنه عبد الله",
    "الإخلاص والنية لابن أبي الدنيا",
    "اختصاص القرآن بعوده إلى الرحيم الرحمن",
    "أصول السنة لأحمد بن حنبل",
    "قصر الأمل لابن أبي الدنيا",
    "سنن الترمذي - ت بشار",
    "مكائد الشيطان",
    "ذم الكذب - من الصمت وآداب اللسان",
    "الأربعون حديثا للآجري",
    "الإبانة الكبرى - ابن بطة",
    "الأمر بالمعروف والنهي عن المنكر- ابن أبي الدنيا",
    "الحث على التجارة - من «الجامع» للخلال - ت الحداد",
    "الزهد لابن أبي الدنيا",
    "اليقين لابن أبي الدنيا",
    "تحريم النرد والشطرنج والملاهي للآجري",
    "ذم البغى لابن أبي الدنيا",
    "ذم الملاهي لابن أبي الدنيا",
    "صفة الجنة لابن أبي الدنيا ت سليم",
    "ذم الغيبة والنميمة لابن أبي الدنيا",
    "كلام الليالي والأيام لابن أبي الدنيا",
    "أدب النفوس للآجري",
    "التوبة لابن أبي الدنيا",
    "التوكل على الله لابن أبي الدنيا",
    "الرقة والبكاء لابن أبي الدنيا",
    "الصبر والثواب عليه لابن أبي الدنيا",
    "العقوبات لابن أبي الدنيا",
    "المطر والرعد والبرق لابن أبي الدنيا",
    "الزهد لأحمد بن حنبل",
    "شرح السنة للبربهاري",
    "العقائد الإسلامية لابن باديس",
    "الجوع لابن أبي الدنيا",
    "الفرج بعد الشدة لابن أبي الدنيا",
    "فضائل عثمان بن عفان لعبد الله بن أحمد",
    "ذم اللواط للآجري",
    "مسند الشافعي - ترتيب سنجر",
    "أصول السنة لابن أبي زمنين",
    "العلل ومعرفة الرجال لأحمد رواية المروذي وغيره ت وصي الله عباس",
    "القبور لابن أبي الدنيا",
    "فضائل رمضان لابن أبي الدنيا",
    "قرى الضيف لابن أبي الدنيا",
    "الأهوال لابن أبي الدنيا",
    "صفة الجنة لابن أبي الدنيا ت العساسلة",
    "محاسبة النفس لابن أبي الدنيا",
    "فضل قيام الليل والتهجد للآجري",
    "مقتل علي لابن أبي الدنيا",
    "مجابو الدعوة لابن أبي الدنيا",
    "الإخوان لابن أبي الدنيا",
    "الأدب المفرد - ت عبد الباقي",
    "الإشراف في منازل الأشراف لابن أبي الدنيا",
    "الاعتبار وأعقاب السرور لابن أبي الدنيا",
    "الأولياء لابن أبي الدنيا",
    "التواضع والخمول لابن أبي الدنيا",
    "التوحيد لابن خزيمة",
    "الحلم لابن أبي الدنيا",
    "الرد على الجهمية لابن منده - ط المكتبة الأثرية",
    "الرضا عن الله بقضائه لابن أبي الدنيا",
    "السنن المأثورة للشافعي",
    "الشريعة للآجري",
    "الشكر لابن أبي الدنيا",
    "الصمت وآداب اللسان",
    "العمر والشيب لابن أبي الدنيا",
    "الغرباء للآجري",
    "المتمنين لابن أبي الدنيا",
    "المحتضرين لابن أبي الدنيا",
    "المرض والكفارات لابن أبي الدنيا",
    "المنامات لابن أبي الدنيا",
    "النفقة على العيال لابن أبي الدنيا",
    "الهم والحزن لابن أبي الدنيا",
    "الوجل والتوثق بالعمل لابن أبي الدنيا",
    "الورع لابن أبي الدنيا",
    "حسن الظن بالله لابن أبي الدنيا",
    "ذم المسكر لابن أبي الدنيا",
    "صفة النار لابن أبي الدنيا",
    "فضائل الصحابة لأحمد بن حنبل",
    "قضاء الحوائج لابن أبي الدنيا",
    "مداراة الناس لابن أبي الدنيا",
    "مكارم الأخلاق لابن أبي الدنيا",
    "من عاش بعد الموت لابن أبي الدنيا",
    "إصلاح المال",
    "الجامع لعلوم الإمام أحمد - علوم الحديث",
    "الجامع لعلوم الإمام أحمد - شرح الأحاديث والآثار",
    "الجامع لعلوم الإمام أحمد - علل الحديث",
    "الجامع لعلوم الإمام أحمد - العقيدة",
    "الجامع لعلوم الإمام أحمد - الرجال",
    "الجامع لعلوم الإمام أحمد - الأدب والزهد",
    "مسند الشافعي - ترتيب السندي",
    "ذم الدنيا",
    "القناعة والتعفف",
    "مسند أحمد - ط الرسالة",
    "سؤالات الاثرم لأحمد بن حنبل",
    "حديث سفيان بن عيينة رواية المروزي",
    "العزلة والانفراد",
    "من حديث سفيان الثوري - ت عامر صبري",
    "حلم معاوية لابن أبي الدنيا",
    "الهواتف = هواتف الجنان لابن أبي الدنيا",
    "مسند الدارمي - ت الزهراني",
    "مسند أحمد - ت شاكر - ط دار الحديث",
    "كتاب العلل الواقع بآخر جامع الترمذي - ت بشار",
    "القراءة عند القبور - من «الجامع» للخلال",
]

CATEGORIES_NAMES = [
    "العقيدة",
    "كتب السنة",
    "العلل والسؤلات الحديثية",
    "التراجم والطبقات",
    "الفقه الحنبلي",
    "الرقائق والآداب والأذكار",
    "علوم الحديث",
    "شروح الحديث",
]


def get_books_hadith() -> list:
    """List all hadith books (static).

    Hardcoded slugs — no database I/O, instant.

    Use when you need to filter data by hadith book.
    Returns:
        ['abudawud', 'bukhari', 'dehlawi', 'ibnmajah', 'malik',
         'nasai', 'nawawi', 'qudsi', 'tirmidhi'] (a fresh copy).
    """
    return list(HADITH_BOOKS)


def get_books_tafsir() -> list:
    """List all tafsir books (static).

    Hardcoded slugs — no database I/O, instant.

    Use when you need to filter data by tafsir book.

    Returns:
        ['saadi', 'katheer', 'moyassar', 'tabary', 'baghawy'] (a fresh copy).
    """
    return list(TAFSIR_BOOKS)


def get_books_books() -> list:
    """List all Arabic book titles (static).

    Hardcoded titles — no database I/O, instant. ~260 titles as provided.

    Use when you need to filter data by book name.

    Returns:
        Fresh copy of BOOKS_LIST (e.g. 'منهاج السنة النبوية', ...).
    """
    return list(BOOKS_LIST)


def get_books_sunnah() -> list:
    """List all Arabic book titles (static).

    Hardcoded titles — no database I/O, instant. ~260 titles as provided.

    Use when you need to filter data by sunnah books names.

    Returns:
        Fresh copy of BOOKS_LIST (e.g. 'منهاج السنة النبوية', ...).
    """
    return list(SUNNAH_BOOKS)


def get_books_categories() -> list:
    """List all books and sunnah books categories (static).

    Hardcoded titles — no database I/O, instant. ~8 categories as provided.

    Use when you need to filter data by categories.

    Returns:
        Fresh copy of CATEGORIES_NAMES (e.g. 'العقيدة', ...).
    """
    return list(CATEGORIES_NAMES)
