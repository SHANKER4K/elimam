from typing import Annotated, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    conlist,
    model_validator,
)

class QuranId(BaseModel):
    ids: str = Field(description="In that format 1:1")


class TafsirId(BaseModel):
    book: Literal["saadi", "katheer", "moyassar", "tabary", "baghawy"]
    ids: str = Field(description="In that format 1:1")


class HadithId(BaseModel):
    book: Literal[
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
    hadith_number: float


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
