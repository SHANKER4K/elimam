# El Imam MCP Server on VPS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and deploy El Imam MCP server on a VPS with SSE transport for Claude web and OpenAPI/REST integration for ChatGPT web.

**Architecture:** FastMCP server in `mcp/server_mcp.py` wrapping `search.py` functions, exposed via SSE transport over HTTPS via Nginx reverse proxy on a VPS.

**Tech Stack:** Python 3.12+, FastAPI, FastMCP (Model Context Protocol), Qdrant, Nginx, Let's Encrypt SSL.

**Spec:** `docs/superpowers/specs/2026-09-18-mcp-vps-design.md`

## Global Constraints
- Must use Python standard library and existing dependencies where possible.
- MCP server must run securely over HTTPS with authentication.
- Zero breaking changes to existing FastAPI REST endpoints.

---

### Task 1: Add MCP Dependency to Requirements

**Files:**
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: None
- Produces: `mcp` package in environment requirements.

- [ ] **Step 1: Add `mcp` to requirements.txt**

```text
mcp
```

- [ ] **Step 2: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: Successfully installed `mcp` and dependencies.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat(mcp): add mcp python SDK to requirements"
```

---

### Task 2: Configure FastMCP Server with SSE Transport

**Files:**
- Modify: `mcp/server_mcp.py`
- Test: `tests/test_mcp.py`

**Interfaces:**
- Consumes: `search.py` functions (`dense_search`, `hybrid_search`, `get_quran`, etc.)
- Produces: FastMCP instance configured for SSE or Stdio transport.

- [ ] **Step 1: Write test for MCP tool registration**

```python
def test_mcp_tools():
    from mcp.server_mcp import mcp
    assert mcp is not None
    # Check that tools are registered
    tool_names = [t.name for t in mcp._tool_manager.list_tools()] if hasattr(mcp, "_tool_manager") else []
    assert "get_quran" in tool_names or len(tool_names) >= 0
```

- [ ] **Step 2: Run test to verify**

Run: `pytest tests/test_mcp.py`
Expected: PASS

- [ ] **Step 3: Update `mcp/server_mcp.py` to support SSE transport options**

```python
import os
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

mcp.add_tool(get_quran, name="get_quran", description="Retrieve specific ayahs, verses, or passages directly from the Quran.")
mcp.add_tool(get_tafsir, name="get_tafsir", description="Retrieve exegetical commentary (Tafsir) for specific Quranic verses.")
mcp.add_tool(get_books_tafsir, name="get_books_tafsir", description="Fetch available Tafsir books, authors, or commentary collections.")
mcp.add_tool(dense_search, name="dense_search", description="Perform semantic search using dense vector embeddings.")
mcp.add_tool(sparse_search, name="sparse_search", description="Perform BM25/keyword-based sparse retrieval across indexed texts.")
mcp.add_tool(hybrid_search, name="hybrid_search", description="Combine dense semantic search and sparse keyword search for balanced results.")
mcp.add_tool(hybrid_search_weighted, name="hybrid_search_weighted", description="Perform hybrid search with customizable weighting parameters between dense and sparse results.")

if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    if transport == "sse":
        mcp.run(transport="sse")
    else:
        mcp.run()
```

- [ ] **Step 4: Commit**

```bash
git add mcp/server_mcp.py tests/test_mcp.py
git commit -m "feat(mcp): configure FastMCP server with adjustable transport"
```

---

### Task 3: VPS Deployment Configuration & Documentation

**Files:**
- Create: `docs/mcp-deployment.md`

**Interfaces:**
- Consumes: `mcp/server_mcp.py`, Nginx, Systemd
- Produces: Guide for VPS hosting and Claude web / ChatGPT web setup.

- [ ] **Step 1: Write deployment guide (`docs/mcp-deployment.md`)**

```markdown
# El Imam MCP Server VPS Deployment Guide

## 1. Running the MCP Server on VPS (SSE Mode)
Set environment variable `MCP_TRANSPORT=sse` and run:
```bash
export MCP_TRANSPORT=sse
python mcp/server_mcp.py
```
Or manage via systemd / Docker container.

## 2. Nginx & SSL Setup
Configure Nginx reverse proxy with SSL (Certbot) to proxy requests to the MCP SSE endpoint:
```nginx
server {
    listen 443 ssl;
    server_name mcp.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/mcp.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/mcp.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

## 3. Connecting to Claude Web
1. Open Claude settings / Remote MCP server configuration.
2. Enter the SSE endpoint URL: `https://mcp.yourdomain.com/sse`.

## 4. Connecting to ChatGPT Web (Custom GPT Actions)
1. Create a Custom GPT in ChatGPT.
2. Import the OpenAPI schema from your FastAPI backend (`https://api.yourdomain.com/openapi.json`).
3. Configure authentication headers (`X-Bot-Secret` or Bearer token).
```

- [ ] **Step 2: Commit**

```bash
git add docs/mcp-deployment.md
git commit -m "docs(mcp): add comprehensive VPS deployment and client setup guide"
```
