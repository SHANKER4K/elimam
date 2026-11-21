"""The seeds and the query path must preprocess text the same way.

They used to be two copies, and they had drifted: only the seeds folded alef
maksura, so a query for "هدى" could never match an indexed "هدى" -> "هدا".
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "vectordb"))

from _text import clean  # noqa: E402

from retrieval.retrieve import retrieval  # noqa: E402


def test_seed_clean_is_the_query_preprocess():
    samples = [
        "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
        "ٱلْحَمْدُ لِلَّهِ",
        "آمَنُوا",
        "الْعَـــرَبِيَّةُ",
        "قَالَ  يَا   آمِنُوا\u200f",
    ]

    for sample in samples:
        assert clean(sample) == retrieval.preprocess(sample)


def test_seed_clean_strips_quranic_marks_and_folds_alef():
    assert clean("الرَّحْمَٰنِ") == "الرحمن"
    assert clean("آمَنُوا") == "امنوا"
    assert clean("إِبْرَاهِيمُ") == "ابراهيم"
