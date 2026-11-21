"""Retrieval layer for the turath collections.

    from retrieval import retrieval, get_quran, search

    await search("quran", "الرحمن الرحيم", top_k=5)
    await retrieval.retrieve(collection="tafsir", query_text="...", mode="dense")
"""

from .books import BOOKS_LIST, CATEGORIES_NAMES, SUNNAH_BOOKS
from .citations import (
    CitationRenderer,
    CitationStream,
    plain_text,
    render_citations,
    render_plain,
)
from .retrieve import CollectionName, Retrieval, RetrievalMode, retrieval
from .schemas import (
    BooksFilters,
    BooksSearchRequest,
    HadithFilters,
    HadithId,
    HadithSearchRequest,
    IntFilter,
    IntRangeFilter,
    QuranFilters,
    QuranId,
    QuranSearchRequest,
    SearchRequest,
    StrFilter,
    SunnahFilters,
    SunnahSearchRequest,
    TafsirFilters,
    TafsirId,
    TafsirSearchRequest,
)
from .tools import (
    find_book,
    find_category,
    get_book,
    get_books_books,
    get_books_categories,
    get_books_sunnah,
    get_chunk,
    get_hadith,
    get_quran,
    get_sunnah,
    get_tafsir,
    search,
)
