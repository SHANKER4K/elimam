#!/usr/bin/env python3
"""
واجهة Whoosh للـ hybrid search.
تستعملها server.py بإضافة endpoint /hybrid-search
"""

import os
import json
from functools import lru_cache

from whoosh.index import open_dir, exists_in
from whoosh.qparser import QueryParser, FuzzyTermPlugin, OrGroup
from whoosh.query import Or, And, Term, Variations


WHOOSH_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whoosh_index")


def _chromadb_to_results(raw, default_semantic_score=1.0):
    """تحويل response ChromaDB (dict) إلى list من dictionaries.

    ChromeDB query ترجع dict بهذا الشكل:
        {"ids": [["id1", "id2"]], "documents": [["text1", "text2"]],
         "metadatas": [[{...}, {...}]], "distances": [[0.5, 0.7]]}

    هذه الدالة تحوّله إلى list لإستخدامه في الـ hybrid search endpoints.
    """
    results = []
    docs = raw.get("documents", [[]])[0]
    ids_ = raw.get("ids", [[]])[0]
    dists = raw.get("distances", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]

    for i in range(len(docs)):
        sim = 1.0 - (dists[i] / 2.0) if i < len(dists) else default_semantic_score
        item = {
            "id": ids_[i] if i < len(ids_) else "",
            "text": docs[i],
            "keyword_score": 0.0,
            "semantic_score": max(0.0, sim),
            "hybrid_score": max(0.0, sim),
        }
        if i < len(metas) and metas[i]:
            item["meta"] = metas[i]
        results.append(item)
    return results


# ===== فتح الـ indexes =====
@lru_cache(maxsize=1)
def get_quran_index():
    path = os.path.join(WHOOSH_BASE, "quran")
    if not exists_in(path):
        raise FileNotFoundError("Quran Whoosh index غير موجود. شغّل build_whoosh.py أولاً")
    return open_dir(path)


@lru_cache(maxsize=1)
def get_hadith_index():
    path = os.path.join(WHOOSH_BASE, "hadith")
    if not exists_in(path):
        raise FileNotFoundError("Hadith Whoosh index غير موجود. شغّل build_whoosh.py أولاً")
    return open_dir(path)


@lru_cache(maxsize=1)
def get_aqeedah_index():
    path = os.path.join(WHOOSH_BASE, "aqeedah")
    if not exists_in(path):
        raise FileNotFoundError("Aqeedah Whoosh index غير موجود. شغّل build_whoosh.py أولاً")
    return open_dir(path)


@lru_cache(maxsize=1)
def get_tafsir_index():
    path = os.path.join(WHOOSH_BASE, "tafsir")
    if not exists_in(path):
        raise FileNotFoundError("Tafsir Whoosh index غير موجود. شغّل build_whoosh.py أولاً")
    return open_dir(path)


# ===== دوال البحث الأساسية =====
def keyword_search(ix, query_text, limit=30):
    """
    بحث باستخدام Whoosh (BM25 + ArabicAnalyzer).
    يرجع قائمة dictionaries: {id, text, score, score_norm, ...all schema fields}
    """
    with ix.searcher() as searcher:
        # استخدم OrGroup: أي كلمة تطابق = نتيجة (BM25-style)
        parser = QueryParser("text", schema=ix.schema, group=OrGroup)
        parser.add_plugin(FuzzyTermPlugin())

        # جرّب الاستعلام أولاً مع الـ analyzer
        query = parser.parse(query_text)
        results = searcher.search(query, limit=limit)

        output = []
        for r in results:
            # اجمع كل الحقول الموجودة في الـ schema (عدا id و text نضيفهم يدوي)
            item = {
                "id": r["id"],
                "text": r["text"],
            }
            # أضف باقي الحقول ديناميكياً من الـ schema
            for field in ix.schema._fields:
                if field not in ("id", "text", "score") and field in r:
                    item[field] = r[field]
            item["score"] = r.score

            # Normalize score إلى 0-1
            if results.has_exact_length() and results.scored_length() > 1:
                top_score = results[0].score if len(results) > 0 else 1.0
                item["score_norm"] = r.score / top_score if top_score > 0 else 0.0
            else:
                item["score_norm"] = r.score / (r.score + 1.0)  # sigmoid-like normalization

            output.append(item)

    return output


# ===== Hybrid Search =====
def hybrid_search_verses(query_text, k=10, semantic_func=None, alpha=0.6):
    """
    Hybrid search على القرآن.
    
    Args:
        query_text: نص الاستعلام
        k: عدد النتائج
        semantic_func: دالة تأخذ query و k وترجع ChromaDB result 
                      (quran_collection.query مع normalize)
        alpha: وزن semantic (1.0 = سيمانتيك فقط، 0.0 = كلمات فقط)
    
    Returns:
        قائمة نتائج مرتبة
    """
    # 1. Keyword search (Whoosh)
    ix = get_quran_index()
    keyword_results = keyword_search(ix, query_text, limit=max(k,30))
    
    if not keyword_results:
        # إذا Whoosh ما جاب شيء، رجع السيمانتيك فقط
        if semantic_func:
            raw = semantic_func(query_text, k)
            return _chromadb_to_results(raw)
        return []
    
    # 2. Semantic search (ChromaDB) — استدعاء خارجي
    semantic_results = []
    if semantic_func:
        raw = semantic_func(query_text, k)
        # semantic_func يرجع structure زي ChromaDB
        docs = raw.get("documents", [[]])[0]
        dists = raw.get("distances", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        ids_ = raw.get("ids", [[]])[0]
        
        for i in range(len(docs)):
            # المسافة → تشابه (cosine distance range 0-2)
            sim = 1.0 - (dists[i] / 2.0)
            semantic_results.append({
                "id": ids_[i],
                "text": docs[i],
                "meta": metas[i],
                "score_norm": max(0.0, sim),  # 0-1
            })
    
    # 3. دمج النتائج
    combined = {}
    
    # معالجة نتائج Whoosh
    whoosh_max = max((r["score_norm"] for r in keyword_results), default=1.0)
    for r in keyword_results:
        combined[r["id"]] = {
            **r,
            "keyword_score": r["score_norm"] / whoosh_max if whoosh_max > 0 else 0,
            "semantic_score": 0.0,
        }
    
    # معالجة نتائج السيمانتيك
    sem_max = max((r["score_norm"] for r in semantic_results), default=1.0)
    for r in semantic_results:
        if r["id"] in combined:
            combined[r["id"]]["semantic_score"] = r["score_norm"] / sem_max if sem_max > 0 else 0
            combined[r["id"]]["meta"] = r["meta"]
        else:
            combined[r["id"]] = {
                "id": r["id"],
                "text": r["text"],
                "keyword_score": 0.0,
                "semantic_score": r["score_norm"] / sem_max if sem_max > 0 else 0,
                "meta": r["meta"],
            }
    
    # 4. حساب الـ hybrid score
    for item in combined.values():
        item["hybrid_score"] = (
            alpha * item["semantic_score"]
            + (1 - alpha) * item["keyword_score"]
        )
    
    # 5. ترتيب وأخذ top-K
    sorted_results = sorted(
        combined.values(),
        key=lambda x: x["hybrid_score"],
        reverse=True,
    )
    
    return sorted_results[:k]


def hybrid_search_hadith(query_text, k=10, semantic_func=None, alpha=0.6):
    """Hybrid search على الحديث — نفس فكرة القرآن."""
    ix = get_hadith_index()
    keyword_results = keyword_search(ix, query_text, limit=max(k,30))

    if not keyword_results:
        if semantic_func:
            raw = semantic_func(query_text, k)
            return _chromadb_to_results(raw)
        return []

    semantic_results = []
    if semantic_func:
        raw = semantic_func(query_text, k)
        docs = raw.get("documents", [[]])[0]
        dists = raw.get("distances", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        ids_ = raw.get("ids", [[]])[0]
        
        for i in range(len(docs)):
            sim = 1.0 - (dists[i] / 2.0)
            semantic_results.append({
                "id": ids_[i],
                "text": docs[i],
                "meta": metas[i],
                "score_norm": max(0.0, sim),
            })
    
    combined = {}
    whoosh_max = max((r["score_norm"] for r in keyword_results), default=1.0)
    for r in keyword_results:
        combined[r["id"]] = {
            **r,
            "keyword_score": r["score_norm"] / whoosh_max if whoosh_max > 0 else 0,
            "semantic_score": 0.0,
        }

    sem_max = max((r["score_norm"] for r in semantic_results), default=1.0)
    for r in semantic_results:
        if r["id"] in combined:
            combined[r["id"]]["semantic_score"] = r["score_norm"] / sem_max if sem_max > 0 else 0
            combined[r["id"]]["meta"] = r["meta"]
        else:
            combined[r["id"]] = {
                "id": r["id"],
                "text": r["text"],
                "keyword_score": 0.0,
                "semantic_score": r["score_norm"] / sem_max if sem_max > 0 else 0,
                "meta": r["meta"],
            }
    
    for item in combined.values():
        item["hybrid_score"] = (
            alpha * item["semantic_score"]
            + (1 - alpha) * item["keyword_score"]
        )
    
    sorted_results = sorted(
        combined.values(),
        key=lambda x: x["hybrid_score"],
        reverse=True,
    )
    
    return sorted_results[:k]


def hybrid_search_aqeedah(query_text, k=10, semantic_func=None, alpha=0.6):
    """Hybrid search على العقيدة — نفس فكرة القرآن."""
    ix = get_aqeedah_index()
    keyword_results = keyword_search(ix, query_text, limit=max(k,30))

    if not keyword_results:
        if semantic_func:
            raw = semantic_func(query_text, k)
            return _chromadb_to_results(raw)
        return []

    semantic_results = []
    if semantic_func:
        raw = semantic_func(query_text, k)
        docs = raw.get("documents", [[]])[0]
        dists = raw.get("distances", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        ids_ = raw.get("ids", [[]])[0]

        for i in range(len(docs)):
            sim = 1.0 - (dists[i] / 2.0)
            semantic_results.append({
                "id": ids_[i],
                "text": docs[i],
                "meta": metas[i],
                "score_norm": max(0.0, sim),
            })

    combined = {}
    whoosh_max = max((r["score_norm"] for r in keyword_results), default=1.0)
    for r in keyword_results:
        combined[r["id"]] = {
            **r,
            "keyword_score": r["score_norm"] / whoosh_max if whoosh_max > 0 else 0,
            "semantic_score": 0.0,
        }

    sem_max = max((r["score_norm"] for r in semantic_results), default=1.0)
    for r in semantic_results:
        if r["id"] in combined:
            combined[r["id"]]["semantic_score"] = r["score_norm"] / sem_max if sem_max > 0 else 0
            combined[r["id"]]["meta"] = r["meta"]
        else:
            combined[r["id"]] = {
                "id": r["id"],
                "text": r["text"],
                "keyword_score": 0.0,
                "semantic_score": r["score_norm"] / sem_max if sem_max > 0 else 0,
                "meta": r["meta"],
            }

    for item in combined.values():
        item["hybrid_score"] = (
            alpha * item["semantic_score"]
            + (1 - alpha) * item["keyword_score"]
        )

    sorted_results = sorted(
        combined.values(),
        key=lambda x: x["hybrid_score"],
        reverse=True,
    )

    return sorted_results[:k]


def hybrid_search_tafsir(query_text, k=10, semantic_func=None, alpha=0.6):
    """Hybrid search على التفسير — نفس فكرة القرآن."""
    ix = get_tafsir_index()
    keyword_results = keyword_search(ix, query_text, limit=max(k,30))

    if not keyword_results:
        if semantic_func:
            raw = semantic_func(query_text, k)
            return _chromadb_to_results(raw)
        return []

    semantic_results = []
    if semantic_func:
        raw = semantic_func(query_text, k)
        docs = raw.get("documents", [[]])[0]
        dists = raw.get("distances", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        ids_ = raw.get("ids", [[]])[0]
        
        for i in range(len(docs)):
            sim = 1.0 - (dists[i] / 2.0)
            semantic_results.append({
                "id": ids_[i],
                "text": docs[i],
                "meta": metas[i],
                "score_norm": max(0.0, sim),
            })
    
    combined = {}
    whoosh_max = max((r["score_norm"] for r in keyword_results), default=1.0)
    for r in keyword_results:
        combined[r["id"]] = {
            **r,
            "keyword_score": r["score_norm"] / whoosh_max if whoosh_max > 0 else 0,
            "semantic_score": 0.0,
        }

    sem_max = max((r["score_norm"] for r in semantic_results), default=1.0)
    for r in semantic_results:
        if r["id"] in combined:
            combined[r["id"]]["semantic_score"] = r["score_norm"] / sem_max if sem_max > 0 else 0
            combined[r["id"]]["meta"] = r["meta"]
            # نسخ tafsir_book من ChromaDB لو Whoosh كان فاضي (defensive)
            meta = r.get("meta") or {}
            if not combined[r["id"]].get("tafsir_book") and meta.get("tafsir_book"):
                combined[r["id"]]["tafsir_book"] = meta["tafsir_book"]
        else:
            combined[r["id"]] = {
                "id": r["id"],
                "text": r["text"],
                "keyword_score": 0.0,
                "semantic_score": r["score_norm"] / sem_max if sem_max > 0 else 0,
                "meta": r["meta"],
            }
    
    for item in combined.values():
        item["hybrid_score"] = (
            alpha * item["semantic_score"]
            + (1 - alpha) * item["keyword_score"]
        )
    
    sorted_results = sorted(
        combined.values(),
        key=lambda x: x["hybrid_score"],
        reverse=True,
    )
    
    return sorted_results[:k]
