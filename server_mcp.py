"""MCP server exposing the same retrieval tools as the chat agent.

Run:  python server_mcp.py

Tools and descriptions come from `retrieval` (docstrings become the MCP tool
descriptions), so the HTTP agent and MCP clients cannot drift apart.
"""

from mcp.server import FastMCP

from retrieval import (
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

mcp = FastMCP("El Imam")

for tool in (
    search,
    get_quran,
    get_hadith,
    get_tafsir,
    get_book,
    get_sunnah,
    get_chunk,
    find_book,
    find_category,
    get_books_books,
    get_books_sunnah,
    get_books_categories,
):
    mcp.add_tool(tool)

if __name__ == "__main__":
    mcp.run()
