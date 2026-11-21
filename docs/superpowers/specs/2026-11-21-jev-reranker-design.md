# Jev as the reranker

2026-11-21 · status: awaiting approval

## Intent

`Retrieval.rerank()` is the one rerank stage shared by every search in the app:
`POST /search/{dense,sparse,hybrid}_search` (the browse UI) and the agent's `search`
tool both end there. Today it calls a local TEI cross-encoder
(`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) with batches of 32 texts.

Replace that with Jev — `jev-1.13-free` on OpenCode Go's System One endpoint
(`https://opencode.ai/inference/systemone/v1/systemone`) — which answers a typed
yes/no question about one query–passage pair and returns a calibrated probability.
The TEI path stays, demoted to the fallback and the A/B control.

Why: Jev scores the pair on whether the passage answers the question rather than on
surface similarity, the key is free to us, and one Noul per pair is TypeSafe's own
re-ranking recipe (CLERC: 5% → 30% top-1 against a BM25 shortlist).

## Goals

- Rerank by one `noul` per (query, chunk) pair, sorted highest first.
- Keep the payload field name `reranker_score` — the UI already reads it.
- A search never fails because Jev is unavailable: on any Jev error, that request
  falls back to the existing TEI path.
- One env switch back to TEI, so the two rerankers can be compared on the same
  queries without a code change.

## Non-goals

- No passage gating, dropping, or contradiction detection (TypeSafe's "classifying
  RAG passages" recipe). The caller gets the `top_k` it asked for.
- No change to candidate generation: Qdrant RRF, filters, collections stay as they are.
- No per-user Jev keys. This is infrastructure, one server-side key like
  `FREE_PROVIDER_API_KEY`; it is not a provider a user picks.
- No new Python dependency (plain `httpx`, already used).
- Citations, thinking, tool events, and the SSE contract are untouched.

## Constraints

- TypeSafe guidance for non-English state: instructions and criteria in English,
  state in the content's own language (Arabic here).
- One request per pair — their rerank cookbook does not batch different pairs into
  one request, and `state` is shared by all questions in a request.
- Limits: 32k tokens per question, 64k per request. Chunk text is capped well below.
- The endpoint sits behind Cloudflare, which answers a generic SDK/urllib
  User-Agent with `403 error code: 1010`. Send `User-Agent: elimam-backend/0.1`.
- The stored OpenCode Console key currently returns `api_key_expired`. A fresh key
  is a prerequisite for the smoke test and for calibration.
- Jev is a remote single point of failure with a free, time-limited model id.

## Approaches considered

**A. Jev replaces the cross-encoder; TEI kept as the fallback.** *(picked)*
`rerank()` sends the head of the RRF pool to Jev; on any failure it calls the
existing TEI method for that request. Satisfies "Jev is the reranker", keeps search
alive when the free model ends or a key expires, and gives the A/B for free.

**B. Jev only; delete the reranker service.** One dependency fewer and the simplest
end state, but a dead key or a 429 becomes a 502 on search until a human intervenes,
and there is nothing to compare against afterwards.

**C. Cascade: TEI reranks the pool, Jev re-ranks its top 12–15.** Best quality
ceiling and the fewest Jev calls (~0.5 s), but it adds a hop and can only reorder what
the cross-encoder already liked — it cannot rescue a good chunk TEI buried.

## Design

### Config (environment)

| var | default | meaning |
|---|---|---|
| `JEV_API_KEY` | — | required to use Jev; absent or `JEV_RERANK=off` → TEI path |
| `JEV_API_URL` | `https://opencode.ai/inference/systemone/v1/systemone` | full endpoint |
| `JEV_MODEL` | `jev-1.13-free` | pinned id; the alias moves under a tuned threshold |
| `JEV_POOL` | `24` | how many candidates Jev sees |

No `JEV_RERANK=on` needed: a present key enables it. `JEV_RERANK=off` is the kill
switch, and deleting the key does the same thing.

### The call

One `POST` per candidate, `state = {"query": <arabic query>, "passage": <chunk>}`,
question:

```json
{"type": "noul",
 "instructions": "Does the passage contain information that answers the question?",
 "criteria": {"true": "The passage states, or directly supports, what the question asks.",
              "false": "The passage shares the topic but does not answer the question."}}
```

`answers.<id>.noul` (0–1) becomes `reranker_score`. Chunk text goes through the
existing `preprocess()` (pyarabic, no diacritics) and is truncated to 1200
characters — a ceiling worth knowing, since a `books` chunk can be longer than that.

Up to 8 in flight (`asyncio.Semaphore`), 12 s per request, whole stage bounded by the
same `asyncio.gather` shape the TEI batching already uses.

### Pool size

`n = min(rerank_pool, max(top_k, JEV_POOL))` — the tail of the RRF pool is dropped
when Jev is on. With today's defaults (`top_k=10`, `rerank_pool=50`) Jev scores 24 and
the caller still gets its 10. A `top_k=50` request costs 50 calls, which is why the
existing `MAX_TOP_K`/`MAX_RERANK_POOL` caps stay where they are.

### Failure

If any pair in the request fails — HTTP error, timeout, unparseable body, missing
`noul` — the whole request falls back to TEI. No mixed score scales inside one
result list. The cost of that rule: one flaky pair buys a TEI call as well. Jev
failures are expected to be correlated (expired key, 429, egress), not per-pair.

`ponytail:` the 1200-char truncation and the 24-candidate pool are the two knobs that
trade quality for latency; both are measured against TEI before the defaults are
trusted.

### The check

`tests/test_jev_rerank.py`, standalone (no Qdrant, no network, no models): an
`httpx.MockTransport` returning canned System One answers asserts the order is by
`noul`; a transport that raises asserts `_rerank_tei` was used instead. This is the
smallest thing that fails if the wiring or the fallback breaks.

### Calibration (after a fresh key)

Same queries, both rerankers, compare top-1 and top-10 by eye against the corpus.
The threshold question — whether a low `noul` should also drop a result — is
deliberately not in this change.

## Calls I made

- A present key is the enable switch; `JEV_RERANK=off` is the escape hatch.
- Any failed pair falls back for the whole request (deterministic, one scale).
- `preprocess()` is reused rather than sending raw diacritized text.
- 1200-character truncation, 24-candidate default pool, concurrency 8.
- `JEV_MODEL` pinned to `jev-1.13-free` rather than an alias, so a release cannot
  move the numbers under us silently.

## Left to the builder

- The exact question and criteria wording is a starting point; it gets tuned once
  there is a key to measure with.
- Whether a second Noul (e.g. "does it contradict the question's premise") earns its
  latency is an open question for later, not this change.
