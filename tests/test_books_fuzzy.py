"""Fuzzy book-title lookup. Pure string work — no Qdrant, no models."""

from retrieval.tools import find_book, find_category


def test_partial_title_reaches_the_exact_edition_titles():
    results = find_book("شرح الطحاوية")

    assert results[0]["title"] == "شرح العقيدة الطحاوية - ط الرسالة"
    assert results[0]["score"] == 100.0
    assert len(results) == 3  # the three editions
    assert all("الطحاوية" in r["title"] for r in results)


def test_editions_are_returned_together_so_the_caller_picks():
    titles = [r["title"] for r in find_book("الصواعق المرسلة")]

    assert len(titles) == 2
    assert any("عطاءات العلم" in t for t in titles)
    assert any("العاصمة" in t for t in titles)


def test_exact_title_scores_100():
    assert find_book("منهاج السنة النبوية")[0] == {
        "title": "منهاج السنة النبوية",
        "score": 100.0,
    }


def test_absent_books_return_nothing_instead_of_a_wrong_title():
    # Both are famous books that are NOT in the corpus. A confident wrong hit
    # here would make the model filter on the wrong title and get zero chunks.
    assert find_book("فتح الباري") == []
    assert find_book("صحيح البخاري") == []


def test_sunnah_collection_uses_its_own_list():
    titles = [r["title"] for r in find_book("مسند أحمد", collection="sunnah")]

    assert titles[0] == "مسند أحمد - ط الرسالة"
    assert all(t in titles for t in ["مسند أحمد - ط الرسالة"])


def test_category_lookup_folds_teh_marbuta_and_the_article():
    assert find_category("عقيده")[0]["title"] == "العقيدة"
    assert find_category("الفقه الحنبلي")[0]["title"] == "الفقه الحنبلي"


def test_limit_is_respected():
    assert len(find_book("الزهد", limit=2)) <= 2
