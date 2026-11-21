"""The one text-cleaning definition shared by the seeds and the query side.

The index and the query must preprocess text identically; before this module they
were separate copies, and they had already drifted (only the seeds folded alef
maksura). `clean` is `Retrieval.preprocess` and nothing else.
"""

import pathlib
import sys

# Seeding runs as `python vectordb/seed_*.py`, which puts `vectordb/` on sys.path
# but not the repo root, where the `retrieval` package lives.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from retrieval.retrieve import retrieval  # noqa: E402


def clean(text: str) -> str:
    """Strip diacritics, normalize alef forms, drop U+200F, collapse whitespace."""
    return retrieval.preprocess(text)
