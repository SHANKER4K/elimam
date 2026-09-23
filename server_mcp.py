import os
import sys as _sys

# Pre-load the venv's mcp.server.lowlevel.server so it registers on sys.modules.
# This resolves `from mcp.server import FastMCP` when running
# `python server_mcp.py` outside the venv.
_venv_site = "/home/ismail/Documents/projects/python/ML/Summer/projects/islam/.venv/lib/python3.13/site-packages"
if _venv_site not in _sys.path:
    _sys.path.insert(0, _venv_site)

try:
    import mcp.server.lowlevel.server  # noqa: F401
except ImportError:
    pass

from mcp.server import FastMCP

from search import (
    dense_search,
    get_books_tafsir,
    get_quran,
    get_tafsir,
    hybrid_search,
    hybrid_search_weighted,
    sparse_search,
)

mcp = FastMCP("El Imam")

mcp.add_tool(
    get_quran,
    name="get_quran",
    description="Retrieve specific ayahs, verses, or passages directly from the Quran.",
)
mcp.add_tool(
    get_tafsir,
    name="get_tafsir",
    description="Retrieve exegetical commentary (Tafsir) for specific Quranic verses.",
)
mcp.add_tool(
    get_books_tafsir,
    name="get_books_tafsir",
    description="Fetch available Tafsir books, authors, or commentary collections.",
)
mcp.add_tool(
    dense_search,
    name="dense_search",
    description="Perform semantic search using dense vector embeddings.",
)
mcp.add_tool(
    sparse_search,
    name="sparse_search",
    description="Perform BM25/keyword-based sparse retrieval across indexed texts.",
)
mcp.add_tool(
    hybrid_search,
    name="hybrid_search",
    description="Combine dense semantic search and sparse keyword search for balanced results.",
)
mcp.add_tool(
    hybrid_search_weighted,
    name="hybrid_search_weighted",
    description="Perform hybrid search with customizable weighting parameters between dense and sparse results.",
)

# server_mcp.py
if __name__ == "__main__":
    mcp.run()
