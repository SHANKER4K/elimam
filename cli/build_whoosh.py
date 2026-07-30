#!/usr/bin/env python3
"""
بناء Whoosh indexes من بيانات ChromaDB.
يشمل: القرآن، الحديث، التفسير، العقيدة.
"""

import os
import sys
import time
from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, TEXT, ID, NUMERIC, STORED
from whoosh.analysis import StandardAnalyzer
import chromadb
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import normalize_alef_ar
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WHOOSH_DIR = os.path.join(BASE_DIR, "whoosh_index")
CHROMA_DIR = os.path.join(BASE_DIR, "..", "Islam")

# ===== Standard Analyzer (بدون stemming — نحافظ على النص القرآني كما هو) =====
standard_analyzer = StandardAnalyzer(stoplist=None)

# ===== Schema لكل مجموعة =====
QURAN_SCHEMA = Schema(
    id=ID(stored=True, unique=True),
    surah=STORED,
    surah_number=NUMERIC(stored=True),
    ayah_number=NUMERIC(stored=True),
    text=TEXT(stored=True, analyzer=standard_analyzer),
)

HADITH_SCHEMA = Schema(
    id=ID(stored=True, unique=True),
    book=STORED,
    book_full=STORED,
    hadith_number=STORED,
    section=STORED,
    section_number=STORED,
    grade=STORED,
    source=STORED,
    text=TEXT(stored=True, analyzer=standard_analyzer),
)

TAFSIR_SCHEMA = Schema(
    id=ID(stored=True, unique=True),
    surah=STORED,
    surah_number=NUMERIC(stored=True),
    ayah_number=NUMERIC(stored=True),
    tafsir_book=STORED,
    text=TEXT(stored=True, analyzer=standard_analyzer),
)

AQEEDAH_SCHEMA = Schema(
    id=ID(stored=True, unique=True),
    book_id=NUMERIC(stored=True),
    book_name=STORED,
    category_name=STORED,
    all_authors=STORED,
    chunk_page=STORED,
    book_pages=NUMERIC(stored=True),
    source=STORED,
    text=TEXT(stored=True, analyzer=standard_analyzer),
)


def get_chromadb_connection():
    """الاتصال بـ ChromaDB وجلب الكوليكشنز (بدون embedding function — فقط للقراءة)."""
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )

    quran = client.get_collection("quran")
    hadith = client.get_collection("hadith")
    tafsir = client.get_collection("tafsir")
    aqeedah = client.get_collection("aqeedah")

    return quran, hadith, tafsir, aqeedah


def normalize_text(text: str) -> str:
    """تطبيع النص: إزالة التشكيل وتوحيد ألف."""
    return normalize_alef_ar(dediac_ar(text))


def build_quran_index(quran_collection):
    """بناء Whoosh index للقرآن."""
    ix_path = os.path.join(WHOOSH_DIR, "quran")
    os.makedirs(ix_path, exist_ok=True)

    if exists_in(ix_path):
        print("  ← Quran index موجود، يتخطى...")
        return open_dir(ix_path)

    ix = create_in(ix_path, QURAN_SCHEMA)
    writer = ix.writer(procs=2, limitmb=256)

    print("  جلب آيات القرآن من ChromaDB...")
    all_data = quran_collection.get()
    total = len(all_data["ids"])
    print(f"  {total} آية. بدء الفهرسة...")

    for i, (id_, doc, meta) in tqdm(enumerate(
        zip(all_data["ids"], all_data["documents"], all_data["metadatas"])
    ),desc='building quran index'):
        writer.add_document(
            id=id_,
            text=normalize_text(doc),
            surah=str(meta.get("surah", "")),
            surah_number=int(meta.get("surah_number", 0)),
            ayah_number=int(meta.get("ayah_number", 0)),
        )
        if (i + 1) % 1000 == 0:
            print(f"    ... {i + 1}/{total}")

    writer.commit()
    print(f"  ✅ Quran index: {total} وثيقة")
    return ix


def build_hadith_index(hadith_collection):
    """بناء Whoosh index للحديث."""
    ix_path = os.path.join(WHOOSH_DIR, "hadith")
    os.makedirs(ix_path, exist_ok=True)

    if exists_in(ix_path):
        print("  ← Hadith index موجود، يتخطى...")
        return open_dir(ix_path)

    ix = create_in(ix_path, HADITH_SCHEMA)
    writer = ix.writer(procs=2, limitmb=256)

    print("  جلب أحاديث من ChromaDB...")
    all_data = hadith_collection.get()
    total = len(all_data["ids"])
    print(f"  {total} حديث. بدء الفهرسة...")

    for i, (id_, doc, meta) in tqdm(enumerate(
        zip(all_data["ids"], all_data["documents"], all_data["metadatas"])
    ),desc='Building hadith index'):
        writer.add_document(
            id=id_,
            text=normalize_text(doc),
            book=str(meta.get("book", "")),
            book_full=str(meta.get("book_full", "")),
            hadith_number=str(meta.get("hadith_number", "")),
            section=str(meta.get("section", "")),
            section_number=str(meta.get("section_number", "")),
            source=str(meta.get("source", "")),
            grade=str(meta.get("grade", "")),
        )

    writer.commit()
    print(f"  ✅ Hadith index: {total} حديث")
    return ix


def build_aqeedah_index(aqeedah_collection):
    """بناء Whoosh index للعقيدة."""
    ix_path = os.path.join(WHOOSH_DIR, "aqeedah")
    os.makedirs(ix_path, exist_ok=True)

    if exists_in(ix_path):
        print("  ← Aqeedah index موجود، يتخطى...")
        return open_dir(ix_path)

    ix = create_in(ix_path, AQEEDAH_SCHEMA)
    writer = ix.writer(procs=2, limitmb=256)

    print("  جلب نصوص العقيدة من ChromaDB...")
    all_data = aqeedah_collection.get()
    total = len(all_data["ids"])
    print(f"  {total} نص. بدء الفهرسة...")

    for i, (id_, doc, meta) in tqdm(enumerate(
        zip(all_data["ids"], all_data["documents"], all_data["metadatas"])
    ), desc='building aqeedah index'):
        writer.add_document(
            id=id_,
            text=normalize_text(doc),
            book_id=int(meta.get("book_id", 0)),
            book_name=str(meta.get("book_name", "")),
            category_name=str(meta.get("category_name", "")),
            all_authors=str(meta.get("all_authors", "")),
            chunk_page=str(meta.get("chunk_page", "")),
            book_pages=int(meta.get("book_pages", 0)),
            source=str(meta.get("source", "")),
        )

    writer.commit()
    print(f"  ✅ Aqeedah index: {total} نص")
    return ix


def build_tafsir_index(tafsir_collection):
    """بناء Whoosh index للتفسير."""
    ix_path = os.path.join(WHOOSH_DIR, "tafsir")
    os.makedirs(ix_path, exist_ok=True)

    if exists_in(ix_path):
        print("  ← Tafsir index موجود، يتخطى...")
        return open_dir(ix_path)

    ix = create_in(ix_path, TAFSIR_SCHEMA)
    writer = ix.writer(procs=2, limitmb=256)

    print("  جلب تفاسير من ChromaDB...")
    all_data = tafsir_collection.get()
    total = len(all_data["ids"])
    print(f"  {total} تفسير. بدء الفهرسة...")

    for i, (id_, doc, meta) in tqdm(enumerate(
        zip(all_data["ids"], all_data["documents"], all_data["metadatas"])
    ),desc='Building hadith index'):
        writer.add_document(
            id=id_,
            text=normalize_text(doc),
            surah=str(meta.get("surah", "")),
            surah_number=int(meta.get("surah_number", 0)),
            ayah_number=int(meta.get("aya_number", 0)),
            tafsir_book=str(meta.get("tafsir_book", "")),
        )

    writer.commit()
    print(f"  ✅ Tafsir index: {total} تفسير")
    return ix


def main():
    start = time.time()
    print("=" * 50)
    print("  بناء Whoosh Indexes")
    print("=" * 50)

    # 1. الاتصال بـ ChromaDB
    print("\n[1] الاتصال بـ ChromaDB...")
    quran, hadith, tafsir, aqeedah = get_chromadb_connection()
    print(f"    القرآن: {quran.count()} | الحديث: {hadith.count()} | التفسير: {tafsir.count()} | العقيدة: {aqeedah.count()}")

    # 2. بناء indexes
    print("\n[2] بناء Quran index...")
    build_quran_index(quran)

    print("\n[3] بناء Hadith index...")
    build_hadith_index(hadith)

    print("\n[4] بناء Tafsir index...")
    build_tafsir_index(tafsir)

    print("\n[5] بناء Aqeedah index...")
    build_aqeedah_index(aqeedah)

    elapsed = time.time() - start
    print(f"\n✅ تم! {elapsed:.1f} ثانية")
    print(f"   المسار: {WHOOSH_DIR}")


if __name__ == "__main__":
    main()
