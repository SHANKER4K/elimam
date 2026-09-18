# Architectural Design: El Imam MCP Server on VPS for Claude & ChatGPT Web

## Objective
Enable users to access El Imam's Islamic scholar search and retrieval tools directly from **Claude web** (via Remote MCP over Server-Sent Events / SSE) and **ChatGPT web** (via Custom GPT actions / OpenAPI).

## Architecture & Components

### 1. Transport Layer
- **Claude Web (MCP SSE):** FastMCP will be configured with SSE transport (`transport="sse"` or mounted via Starlette/FastAPI) to expose `/mcp/sse` and `/mcp/messages`.
- **ChatGPT Web (OpenAPI / Actions):** ChatGPT Custom GPTs use standard REST APIs. FastAPI automatically exposes OpenAPI documentation at `/openapi.json`, and our existing `/search/*` endpoints serve as the action backend.

### 2. Security & Authentication
- **SSL / TLS:** Required by both Claude web and ChatGPT web. Nginx reverse proxy with Let's Encrypt SSL certificates.
- **Authentication:** Bearer token / API key header (`Authorization: Bearer <token>`) or `X-Bot-Secret` to secure MCP and API endpoints.

### 3. File Changes
- `requirements.txt`: Add `mcp` library.
- `mcp/server_mcp.py`: Configure FastMCP for production deployment / ASGI / SSE transport.
- `docker-compose.yaml` / Nginx configuration: Expose ports and handle reverse proxying with SSL.

## Verification Plan
1. Test local MCP server startup and SSE endpoint availability.
2. Verify FastAPI OpenAPI schema generation for ChatGPT Actions.
3. Test remote connection from Claude web / MCP client.
