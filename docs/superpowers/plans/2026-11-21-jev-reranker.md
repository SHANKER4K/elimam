# Jev Reranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `Retrieval.rerank()` scores the head of the RRF pool with Jev (`jev-1.13-free` over System One) instead of the local cross-encoder, falling back to TEI on any Jev failure.

**Architecture:** One `POST /v1/systemone` per (query, chunk) pair, `type: noul`, run concurrently under a semaphore; the returned probability becomes `reranker_score` and the candidates sort on it. The existing TEI batching is kept verbatim and renamed `_rerank_tei`, called whenever Jev is disabled or the request fails.

**Tech Stack:** Python 3.13, `httpx` (already a dependency), `asyncio`, pytest with `asyncio.run` (the repo has no pytest-asyncio).

**Spec:** `docs/superpowers/specs/2026-11-21-jev-reranker-design.md`

## Global Constraints

- No new Python dependency; `httpx` and `asyncio` only.
- The payload key stays `reranker_score`; nothing outside `retrieval/` changes shape.
- Instructions and criteria in English, `state` content in its original Arabic.
- `User-Agent: elimam-backend/0.1` on every Jev request (Cloudflare answers a generic UA with `403 error code: 1010`).
- Model id default `jev-1.13-free`, pinned; endpoint default `https://opencode.ai/inference/systemone/v1/systemone`.
- `tests/` is gitignored but tracked — new test files need `git add -f`.

## Review Focus

- **A chunk longer than 1200 characters** (`books` has these): only the first 1200 are sent, and the returned point must still carry its full text untouched.
- **A point with empty/whitespace `payload.text`**: skipped exactly like the TEI path skips it, never sent as an empty passage.
- **A 200 response with no `answers.<id>.noul`**: must count as a failure and fall back to TEI — not score `0.0` and silently sink to the bottom.
- **One 429/timeout among 24 pairs**: the whole request falls back to TEI and the caller still gets `top_k` results.
- **No key, or `JEV_RERANK=off`**: behaviour must be byte-identical to today's TEI path, so nobody lands on a mixed or missing score scale.
- **`top_k=100` from the UI**: 100 pairs are sent, the existing `MAX_TOP_K`/`MAX_RERANK_POOL` caps still bound it, and nothing hangs.

---

### Task 1: Jev pair scorer

**Files:**
- Modify: `retrieval/retrieve.py` (constructor, plus new members near the other clients/rerank helpers)
- Test: `tests/test_jev_rerank.py` (create)

**Interfaces:**
- Consumes: `Retrieval.preprocess(text) -> str` (existing).
- Produces:
  - `Retrieval.__init__(..., jev_url: str | None = None, jev_api_key: str | None = None, jev_model: str | None = None, jev_pool: int = 24)` — `None` reads the env var; `""` disables Jev explicitly.
  - `Retrieval.jev_enabled -> bool` (property)
  - `Retrieval.jev_client -> httpx.AsyncClient` (`cached_property`)
  - `Retrieval._jev_score(self, query_text: str, text: str) -> float` (async)
  - module constants `JEV_URL`, `JEV_MODEL`, `JEV_PASSAGE_CHARS = 1200`, `JEV_CONCURRENCY = 8`, `JEV_QUESTION`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_jev_rerank.py
import asyncio, json, httpx
from retrieval.retrieve import Retrieval, JEV_PASSAGE_CHARS

def _jev(score_fn, seen: list[dict], status: int = 200):
    """A client whose handler records every request and answers score_fn(passage)."""
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append({"url": str(request.url), "body": body,
                     "auth": request.headers.get("authorization"),
                     "ua": request.headers.get("user-agent")})
        if status != 200:
            return httpx.Response(status, json={"detail": "nope"})
        noul = score_fn(body["state"]["passage"])
        return httpx.Response(200, json={"model": "jev-1.13-free",
                                         "answers": {"answers_question": {"type": "noul", "noul": noul}},
                                         "usage": {"input_tokens": 10, "output_tokens": 1}})
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))

def test_jev_score_sends_one_pair_and_reads_the_noul():
    seen = []
    r = Retrieval(url="http://127.0.0.1:1", jev_api_key="test-key")
    r.__dict__["jev_client"] = _jev(lambda passage: 0.87, seen)

    score = asyncio.run(r._jev_score("من هو الله", "اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ"))

    assert score == 0.87
    assert len(seen) == 1
    assert seen[0]["url"] == "https://opencode.ai/inference/systemone/v1/systemone"
    assert seen[0]["auth"] == "Bearer test-key"
    assert seen[0]["ua"] == "elimam-backend/0.1"
    assert seen[0]["body"]["model"] == "jev-1.13-free"
    assert seen[0]["body"]["questions"]["answers_question"]["type"] == "noul"
    assert seen[0]["body"]["state"]["query"] == "من هو الله"
    # preprocess() strips diacritics; the passage is capped, not rejected
    assert "َ" not in seen[0]["body"]["state"]["passage"]
    assert len(seen[0]["body"]["state"]["passage"]) <= JEV_PASSAGE_CHARS
```

- [ ] **Step 2: Run it and watch it fail**

Run: `.venv/bin/python -m pytest tests/test_jev_rerank.py -k jev_score -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'jev_api_key'`.

- [ ] **Step 3: Implement config, client and `_jev_score`**

Constructor additions (same style as `reranker_url`; note `is None`, not `or`, so `""` disables):

```python
self.jev_url = jev_url if jev_url is not None else os.environ.get("JEV_API_URL", JEV_URL)
self.jev_api_key = jev_api_key if jev_api_key is not None else os.environ.get("JEV_API_KEY")
self.jev_model = jev_model if jev_model is not None else os.environ.get("JEV_MODEL", JEV_MODEL)
self.jev_pool = jev_pool
```

```python
@property
def jev_enabled(self) -> bool:
    return bool(self.jev_api_key) and os.environ.get("JEV_RERANK", "on").strip().lower() != "off"

@cached_property
def jev_client(self) -> httpx.AsyncClient:
    # No base_url: the endpoint is one absolute URL and an empty relative path
    # would make httpx guess how to join it.
    return httpx.AsyncClient(
        timeout=httpx.Timeout(12.0, connect=5.0),
        headers={
            "Authorization": f"Bearer {self.jev_api_key}",
            "User-Agent": "elimam-backend/0.1",
        },
    )

async def _jev_score(self, query_text: str, text: str) -> float:
    """One System One Noul for one (query, passage) pair; 0..1."""
    response = await self.jev_client.post(
        self.jev_url,
        json={
            "model": self.jev_model,
            "state": {"query": query_text, "passage": text[:JEV_PASSAGE_CHARS]},
            "questions": {"answers_question": JEV_QUESTION},
        },
    )
    response.raise_for_status()
    return float(response.json()["answers"]["answers_question"]["noul"])
```

- [ ] **Step 4: Run the test again**

Run: `.venv/bin/python -m pytest tests/test_jev_rerank.py -k jev_score -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add retrieval/retrieve.py && git add -f tests/test_jev_rerank.py
git commit -m "feat(retrieval): add a Jev System One pair scorer"
```

### Task 2: `rerank()` prefers Jev, falls back to TEI

**Files:**
- Modify: `retrieval/retrieve.py` (rename `rerank` → `_rerank_tei`, add new `rerank`)
- Modify: `tests/test_retrieval.py:100-113` (pin `jev_api_key=""` so a developer's `.env` cannot flip the test onto Jev)
- Test: `tests/test_jev_rerank.py` (append)

**Interfaces:**
- Consumes: `_jev_score`, `jev_enabled`, `jev_pool` from Task 1; `_rerank_batch` and `preprocess` (existing).
- Produces: `Retrieval.rerank(self, query_text: str, points: list[dict], top_k: int) -> list[dict]` (async) — same name, signature and `reranker_score` contract as today. `Retrieval._rerank_tei(self, query_text: str, points: list[dict], top_k: int) -> list[dict]` (async) — the old body, unchanged.

- [ ] **Step 1: Write the failing tests**

```python
def _points(n: int) -> list[dict]:
    return [{"id": i, "payload": {"text": f"مقطع رقم {i} عن الصبر"}} for i in range(n)]

def test_rerank_sorts_by_noul_and_caps_the_fan_out():
    seen = []
    points = _points(10)
    r = Retrieval(url="http://127.0.0.1:1", jev_api_key="k", jev_pool=4)
    r.__dict__["jev_client"] = _jev(lambda text: float(text.split()[2]), seen)

    top = asyncio.run(r.rerank("الصبر", points, top_k=2))

    assert len(seen) == 4                          # jev_pool caps the fan-out
    assert [p["id"] for p in top] == [3, 2]        # highest noul first
    assert all("reranker_score" in p for p in points[:4])
    assert all("reranker_score" not in p for p in points[4:])   # tail untouched

def test_rerank_sends_at_least_top_k_pairs():
    seen = []
    r = Retrieval(url="http://127.0.0.1:1", jev_api_key="k", jev_pool=4)
    r.__dict__["jev_client"] = _jev(lambda text: 0.5, seen)

    asyncio.run(r.rerank("الصبر", _points(10), top_k=8))
    assert len(seen) == 8                          # max(top_k, jev_pool)

def test_rerank_falls_back_to_tei_when_jev_fails():
    r = Retrieval(url="http://127.0.0.1:1", jev_api_key="k")

    async def boom(query_text, text):
        raise httpx.ConnectError("no route")

    async def tei(query_text, points, top_k):
        return [{"id": "tei"}]

    r.__dict__["jev_client"] = _jev(lambda text: 0.5, [])
    r._jev_score = boom
    r._rerank_tei = tei

    assert asyncio.run(r.rerank("الصبر", _points(3), top_k=2)) == [{"id": "tei"}]

def test_rerank_uses_tei_when_jev_is_disabled(monkeypatch):
    r = Retrieval(url="http://127.0.0.1:1", jev_api_key="k")
    async def tei(query_text, points, top_k):
        return [{"id": "tei"}]
    r._rerank_tei = tei

    monkeypatch.setenv("JEV_RERANK", "off")
    assert asyncio.run(r.rerank("الصبر", _points(3), top_k=2)) == [{"id": "tei"}]
    monkeypatch.delenv("JEV_RERANK")

    assert Retrieval(url="http://127.0.0.1:1", jev_api_key="").jev_enabled is False
```


- [ ] **Step 2: Run them and watch them fail**

Run: `.venv/bin/python -m pytest tests/test_jev_rerank.py -v`
Expected: FAIL — `AttributeError: 'Retrieval' object has no attribute '_rerank_tei'` and the sorts come back in TEI order.

- [ ] **Step 3: Rename the old method and add the dispatcher**

Rename `rerank` to `_rerank_tei` verbatim (docstring: "The local cross-encoder path; also the fallback when Jev is unavailable."). New method:

```python
async def rerank(self, query_text: str, points: list[dict], top_k: int) -> list[dict]:
    """Re-rank with Jev, or the cross-encoder when Jev is off or fails."""
    if not points or not self.jev_enabled:
        return await self._rerank_tei(query_text, points, top_k)

    head = points[: min(len(points), max(top_k, self.jev_pool))]
    valid = [p for p in head if (p.get("payload", {}).get("text") or "").strip()]

    # ponytail: one bad pair falls back for the whole request rather than mixing
    # two score scales in one result list. Jev failures are correlated (key, 429,
    # egress), not per-pair, so the wasted TEI call is the cheap side of that bet.
    semaphore = asyncio.Semaphore(JEV_CONCURRENCY)
    async def score(point: dict) -> float:
        async with semaphore:
            return await self._jev_score(query_text, self.preprocess(point["payload"]["text"]))

    try:
        scores = await asyncio.gather(*(score(p) for p in valid))
    except Exception:
        return await self._rerank_tei(query_text, points, top_k)

    for point, value in zip(valid, scores):
        point["reranker_score"] = value
    valid.sort(key=lambda point: point["reranker_score"], reverse=True)
    return (valid + points[len(head):])[:top_k]
```

- [ ] **Step 4: Pin the existing TEI test to TEI, then run everything**

In `tests/test_retrieval.py:104`, construct `Retrieval(url="http://127.0.0.1:1", rerank_batch_size=16, jev_api_key="")` so a filled-in `.env` cannot route that test through Jev.

Run: `.venv/bin/python -m pytest tests/test_jev_rerank.py tests/test_retrieval.py -v`
Expected: PASS, including the unchanged batch-map assertions `[16, 16, 8]` and scores `[40.0, 39.0, ...]`.

- [ ] **Step 5: Commit**

```bash
git add retrieval/retrieve.py tests/test_retrieval.py && git add -f tests/test_jev_rerank.py
git commit -m "feat(retrieval): re-rank with Jev, fall back to the cross-encoder"
```

### Task 3: Environment surface

**Files:**
- Modify: `.env.example`
- Modify: `docker-compose.yaml` (backend `environment:` block, beside the other secrets)

**Interfaces:**
- Consumes: names read in Task 1 (`JEV_API_URL`, `JEV_API_KEY`, `JEV_MODEL`, `JEV_RERANK`) and `jev_pool`'s default.
- Produces: nothing importable; this task exists so the operator can fill the key in.

- [ ] **Step 1: Add the block to `.env.example`, under the reranker line**

```
RERANKER_URL=http://reranker:80
# Jev (System One) reranker. Leave JEV_API_KEY empty to keep using the
# cross-encoder above. Console key: https://opencode.ai/console/account#api-keys
# jev-1.13-free is free for a limited time and can disappear without notice.
JEV_API_URL=https://opencode.ai/inference/systemone/v1/systemone
JEV_API_KEY=
JEV_MODEL=jev-1.13-free
JEV_RERANK=on              # "off" = cross-encoder only, for comparing the two
```

- [ ] **Step 2: Pass them through in `docker-compose.yaml`**

Under `backend.environment`, next to `EMBEDDING`-adjacent entries:

```yaml
      - JEV_API_URL=${JEV_API_URL:-https://opencode.ai/inference/systemone/v1/systemone}
      - JEV_API_KEY=${JEV_API_KEY:-}
      - JEV_MODEL=${JEV_MODEL:-jev-1.13-free}
      - JEV_RERANK=${JEV_RERANK:-on}
```

(`/app/.env` is already mounted, so `load_dotenv()` would pick these up anyway; listing them keeps every secret in the same place. Heroku deployments set them as config vars instead.)

- [ ] **Step 3: Verify nothing regressed**

Run: `docker compose config >/dev/null` — expected: no error, no "variable is not set" warning beyond an empty `JEV_API_KEY`.
Run: `.venv/bin/python -m pytest tests -q` — expected: PASS (unchanged count plus the new file).

- [ ] **Step 4: Commit**

```bash
git add .env.example docker-compose.yaml
git commit -m "chore: expose the Jev reranker env vars"
```
