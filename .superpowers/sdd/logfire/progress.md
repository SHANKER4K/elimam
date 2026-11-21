# SDD ledger — plan: inline chat plan for issue #19 (Logfire observability)

Pre-flight: no shared interfaces (single workstream, 3 tasks).

Task A1 (env-driven logfire in server.py): complete
Task A2 (.env.example + docker-compose env): complete
Task A3 (verification): complete — temp/test_logfire_config.py --off and default both OK; pytest 111/111.

Rulings:
1. Ruling: Opik gate is OTEL_EXPORTER_OTLP_ENDPOINT, not the issue's OPIK_URL — logfire itself
   attaches the OTLP exporter from that var (config.py:1427-1435), and compose/.env already wire
   Opik through it; OpikSpanProcessor is only ID-propagation glue. Gating on a new OPIK_URL would
   break the live export path. — cost if wrong: Opik export silently dies (caught by test).
2. Ruling: logfire[fastapi] added to both manifests — instrument_fastapi requires
   opentelemetry-instrumentation-fastapi; boot check caught it before any deploy would.
3. Ruling: skip psycopg2 DB spans — logfire.instrument_psycopg is psycopg3-only, repo is psycopg2;
   upgrade path (opentelemetry-instrumentation-psycopg2) noted in server.py ponytail comment.
4. Ruling: message bodies masked via instrument_pydantic_ai(include_content=False) — privacy-first
   default per plan review point; secrets safe via capture_headers=False + default scrubbing.

Final review: fresh reviewer (subagent ses_e006d5b81ffeMClX7N242entEM) — Ready to merge, Yes.
0 Critical, 0 Important, 5 Minor (deferred):
  1. ENVIRONMENT="" beats default in bare-uvicorn runs; compose unaffected (or "development" fix deferred).
  2. off-mode check doesn't assert logfire cloud sink absence mechanically; _walk reach-in brittle across logfire upgrades.
  3. .env.example lacks a warning against LOGFIRE_HTTPX_CAPTURE_ALL / OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_* (would re-enable header capture).
  4. SSE accept-stream-duration-spans decision not ledgered → ledgered now below.
  5. user-supplied provider URLs could embed keys in query strings; url.full ships to sinks. Low risk.
Task 5: Ruling: SSE chat spans accepted as stream-duration spans (issue's explicit verification point) — the
  request span covering the whole agent run is useful latency correlation, and reviewer confirmed the OTel
  middleware wraps the full custom stack with no conflict — cost if wrong: one long-lived span per chat turn;
  revisit via instrument_fastapi(excluded_urls=...) if span volume ever matters.
Final: fixed none — all findings Minor by effect (cosmetic/docs); no fix pass entered.
