"""Verify alpha=0 skips semantic + reranker; alpha>0 still reranks."""
import sys, time
sys.path.insert(0, ".")
import server

from whoosh_search import get_quran_index, keyword_search

q = "الرحمن الرحيم"
k = 10

# --- alpha=0: should be keyword-only ---
t0 = time.perf_counter()
r0 = server.hybrid_quran_search(q, k=k, alpha=0.0)
t_alpha0 = time.perf_counter() - t0
assert r0, "alpha=0 returned nothing"
assert all("keyword_score" in r for r in r0), "missing keyword_score"
assert all(r["hybrid_score"] == r["keyword_score"] for r in r0), "alpha=0 fusion must be pure keyword"
assert all("surah_number" in r and "ayah_number" in r for r in r0), "whoosh meta fields missing"
print(f"alpha=0  : {t_alpha0:.3f}s  {len(r0)} results")

# matches raw whoosh keyword search exactly (ids + order)
ix = get_quran_index()
kw = keyword_search(ix, q, limit=max(k, 30))[:k]
assert [r["id"] for r in r0] == [r["id"] for r in kw], "alpha=0 != pure keyword"
print("          == pure keyword result set (ids + order)")

# --- alpha=1: reranker still applies ---
t0 = time.perf_counter()
r1 = server.hybrid_quran_search(q, k=k, alpha=1.0)
t_alpha1 = time.perf_counter() - t0
assert [r["id"] for r in r1] != [r["id"] for r in r0], "alpha=1 should rerank/reorder"
print(f"alpha=1  : {t_alpha1:.3f}s  {len(r1)} results (reordered by reranker)")

assert t_alpha0 < t_alpha1, "alpha=0 should be faster than alpha=1"
print(f"\nspeedup : {t_alpha1/t_alpha0:.1f}x  (before fix alpha=0 took ~0.79s)")

# smoke: other 3 hybrid endpoints, alpha=0
for fn, qq in [
    (server.hybrid_hadith_search, "من كذب علي"),
    (server.hybrid_tafsir_search, "بسم الله الرحمن الرحيم"),
    (server.hybrid_aqeedah_search, "توحيد الألوهية"),
]:
    t0 = time.perf_counter()
    out = fn(qq, k=k, alpha=0.0)
    dt = time.perf_counter() - t0
    assert all(r["hybrid_score"] == r["keyword_score"] for r in out), f"{fn.__name__} alpha=0 must be pure keyword"
    print(f"{fn.__name__:26s} alpha=0: {dt:.3f}s  {len(out)} results")

print("\nALL CHECKS PASSED")
