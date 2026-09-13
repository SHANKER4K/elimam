from __future__ import annotations

import os
from functools import lru_cache
from typing import Annotated, Literal, TypeAlias

from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import normalize_alef_ar
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    conlist,
    field_validator,
    model_validator,
)
from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

CollectionName = Literal["quran", "hadith", "tafsir", "books", "sunnah"]


# -----------------------------
# Models
# -----------------------------


@lru_cache(maxsize=1)
def load_model(model_name: str):
    print("Loading Model")
    value = SentenceTransformer(
        model_name,
        model_kwargs={"dtype": "float16"},
        # backend="onnx",
    )
    print("Done")
    return value


@lru_cache(maxsize=1)
def ranker_loader(model_name: str):
    return CrossEncoder(
        model_name,
        # backend="onnx",
        trust_remote_code=True,
    )


model = load_model("Omartificial-Intelligence-Space/GATE-AraBert-v1")
reranker = ranker_loader("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
sparse_model = SparseTextEmbedding("Qdrant/bm25")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    cloud_inference=True,
)


class IntRangeFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eq: StrictInt | None = None
    lt: StrictInt | None = None
    gt: StrictInt | None = None
    lte: StrictInt | None = None
    gte: StrictInt | None = None

    @model_validator(mode="after")
    def require_operator(self):
        if not self.model_fields_set:
            raise ValueError("at least one comparison operator is required")
        return self


IntFilter: TypeAlias = StrictInt | conlist(StrictInt, min_length=1) | IntRangeFilter
StrFilter: TypeAlias = StrictStr | conlist(StrictStr, min_length=1)


class QuranFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    surah_number: IntFilter | None = None
    surah: StrFilter | None = None


class HadithFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    book: StrFilter | None = None
    grade: StrFilter | None = None


class TafsirFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    surah_number: IntFilter | None = None
    surah: StrFilter | None = None
    ayah_number: IntFilter | None = None


class BooksFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    book_id: IntFilter | None = None
    book_name: StrFilter | None = None
    category_name: StrFilter | None = None
    all_authors: StrFilter | None = None
    author_death: IntFilter | None = None
    book_date: IntFilter | None = None


class SunnahFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    book_id: IntFilter | None = None
    book_name: StrFilter | None = None
    category_name: StrFilter | None = None
    all_authors: StrFilter | None = None
    author_death: IntFilter | None = None
    book_date: IntFilter | None = None
    athar_number: IntFilter | None = None


class QuranSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection: Literal["quran"]
    query_text: StrictStr = Field(min_length=1)
    top_k: StrictInt = Field(default=10, ge=1)
    filters: QuranFilters | None = None
    rerank_pool: StrictInt = Field(default=50, ge=1)


class HadithSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection: Literal["hadith"]
    query_text: StrictStr = Field(min_length=1)
    top_k: StrictInt = Field(default=10, ge=1)
    filters: HadithFilters | None = None
    rerank_pool: StrictInt = Field(default=50, ge=1)


class TafsirSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection: Literal["tafsir"]
    query_text: StrictStr = Field(min_length=1)
    top_k: StrictInt = Field(default=10, ge=1)
    filters: TafsirFilters | None = None
    rerank_pool: StrictInt = Field(default=50, ge=1)


class BooksSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection: Literal["books"]
    query_text: StrictStr = Field(min_length=1)
    top_k: StrictInt = Field(default=10, ge=1)
    filters: BooksFilters | None = None
    rerank_pool: StrictInt = Field(default=50, ge=1)


class SunnahSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection: Literal["sunnah"]
    query_text: StrictStr = Field(min_length=1)
    top_k: StrictInt = Field(default=10, ge=1)
    filters: SunnahFilters | None = None
    rerank_pool: StrictInt = Field(default=50, ge=1)


SearchRequest: TypeAlias = Annotated[
    QuranSearchRequest
    | HadithSearchRequest
    | TafsirSearchRequest
    | BooksSearchRequest
    | SunnahSearchRequest,
    Field(discriminator="collection"),
]


# -----------------------------
# Text and filter conversion
# -----------------------------


def _preprocess(text: str) -> str:
    text = dediac_ar(text)
    text = normalize_alef_ar(text)
    text = text.replace("\u200f", "")
    return " ".join(text.split())


def _dense_query(text: str) -> list[float]:
    return model.encode([_preprocess(text)])[0].tolist()


def _sparse_query(text: str) -> models.SparseVector:
    vector = list(sparse_model.embed([text]))[0]
    return models.SparseVector(
        indices=vector.indices.tolist(),
        values=vector.values.tolist(),
    )


def _build_filter(filters: BaseModel | None) -> models.Filter | None:
    if filters is None or not filters.model_fields_set:
        return None

    conditions: list[models.FieldCondition] = []

    for key in filters.model_fields_set:
        value = getattr(filters, key)

        if isinstance(value, list):
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchAny(any=value),
                )
            )
            continue

        if isinstance(value, IntRangeFilter):
            if value.eq is not None:
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value.eq),
                    )
                )

            ranges = {
                operator: getattr(value, operator)
                for operator in ("lt", "gt", "lte", "gte")
                if getattr(value, operator) is not None
            }
            if ranges:
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        range=models.Range(**ranges),
                    )
                )
            continue

        conditions.append(
            models.FieldCondition(
                key=key,
                match=models.MatchValue(value=value),
            )
        )

    return models.Filter(must=conditions)


# -----------------------------
# Search helpers
# -----------------------------


def _rerank(query_text: str, points: list[dict], top_k: int) -> list[dict]:
    if not points:
        return points

    pairs = [
        (query_text, _preprocess(point.get("payload", {}).get("text", "")))
        for point in points
    ]
    scores = reranker.predict(pairs, batch_size=32)

    for point, score in zip(points, scores):
        point["reranker_score"] = float(score)

    points.sort(key=lambda point: point["reranker_score"], reverse=True)
    return points[:top_k]


def _query_points(
    *,
    collection: str,
    query,
    using: str,
    limit: int,
    query_filter: models.Filter | None,
) -> list[dict]:
    return [
        point.model_dump()
        for point in client.query_points(
            collection_name=collection,
            query=query,
            using=using,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            query_filter=query_filter,
        ).points
    ]


def dense_search(
    request: QuranSearchRequest
    | HadithSearchRequest
    | TafsirSearchRequest
    | BooksSearchRequest
    | SunnahSearchRequest,
) -> list[dict]:
    """Dense vector search followed by cross-encoder reranking."""
    try:
        query_text = _preprocess(request.query_text)
        points = _query_points(
            collection=request.collection,
            query=_dense_query(query_text),
            using="dense",
            limit=request.rerank_pool,
            query_filter=_build_filter(request.filters),
        )
        return _rerank(query_text, points, request.top_k)
    except ValueError:
        raise
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]


def sparse_search(
    request: QuranSearchRequest
    | HadithSearchRequest
    | TafsirSearchRequest
    | BooksSearchRequest
    | SunnahSearchRequest,
) -> list[dict]:
    """Sparse BM25 search followed by cross-encoder reranking."""
    try:
        query_text = _preprocess(request.query_text)

        points = _query_points(
            collection=request.collection,
            query=_sparse_query(query_text),
            using="sparse",
            limit=request.rerank_pool,
            query_filter=_build_filter(request.filters),
        )
        return _rerank(query_text, points, request.top_k)
    except ValueError:
        raise
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]


def hybrid_search(
    request: QuranSearchRequest
    | HadithSearchRequest
    | TafsirSearchRequest
    | BooksSearchRequest
    | SunnahSearchRequest,
) -> list[dict]:
    """Dense+sparse RRF search followed by cross-encoder reranking."""
    try:
        query_text = _preprocess(request.query_text)
        points = [
            point.model_dump()
            for point in client.query_points(
                collection_name=request.collection,
                prefetch=[
                    models.Prefetch(
                        query=_dense_query(query_text),
                        using="dense",
                        limit=request.rerank_pool,
                    ),
                    models.Prefetch(
                        query=_sparse_query(query_text),
                        using="sparse",
                        limit=request.rerank_pool,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=request.rerank_pool,
                with_payload=True,
                with_vectors=False,
                query_filter=_build_filter(request.filters),
            ).points
        ]
        return _rerank(query_text, points, request.top_k)
    except ValueError:
        raise
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]


# -----------------------------
# Index setup and getters
# -----------------------------


def setup_indexes() -> None:
    """Create payload indexes used by getters and filter fields."""
    filter_types = {
        "int": models.PayloadSchemaType.INTEGER,
        "str": models.PayloadSchemaType.KEYWORD,
    }
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

    filter_schema = {
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

    for collection, schema in filter_schema.items():
        fields[collection].update(
            (key, filter_types[kind]) for key, kind in schema.items()
        )

    for collection, index_set in fields.items():
        for field, schema in index_set:
            client.create_payload_index(
                collection,
                field_name=field,
                field_schema=schema,
            )


def _get_by_id(collection: str, value: str):
    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="ids",
                    match=models.MatchValue(value=value),
                )
            ]
        ),
        with_vectors=False,
    )
    return points[0].payload if points else []


def get_quran(id: str):
    return _get_by_id("quran", id)


def get_hadith(book: str, hadith_number: float):
    return _get_by_id("hadith", f"{book}:{hadith_number}")


def get_tafsir(book: str, id: str):
    return _get_by_id("tafsir", f"{book}:{id}")


def get_book(category: int, book_id: int, chunk_index: int):
    return _get_by_id("books", f"{category}:{book_id}:{chunk_index}")


def get_sunnah(category: int, book_id: int, chunk_index: int):
    return _get_by_id("sunnah", f"{category}:{book_id}:{chunk_index}")


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
