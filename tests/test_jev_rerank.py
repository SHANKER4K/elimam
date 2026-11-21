"""Checks for the Jev reranker: request shape, ordering, and the TEI fallback.

No Qdrant, no TEI, no network — the System One endpoint is faked with an
httpx.MockTransport, so these run anywhere `import retrieval` runs.
"""

import asyncio
import json

import httpx

from retrieval.retrieve import JEV_PASSAGE_CHARS, Retrieval


def _jev(score_fn, seen: list[dict], status: int = 200):
    """A client whose handler records every request and answers score_fn(passage)."""

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(
            {
                "url": str(request.url),
                "body": body,
                "auth": request.headers.get("authorization"),
                "ua": request.headers.get("user-agent"),
            }
        )
        if status != 200:
            return httpx.Response(status, json={"detail": "nope"})
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13-free",
                "answers": {
                    "answers_question": {
                        "type": "noul",
                        "noul": score_fn(body["state"]["passage"]),
                    }
                },
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_jev_score_sends_one_pair_and_reads_the_noul():
    seen = []
    r = Retrieval(url="http://127.0.0.1:1", jev_api_key="test-key")
    r.__dict__["jev_client"] = _jev(lambda passage: 0.87, seen)

    score = asyncio.run(
        r._jev_score("من هو الله", "اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ")
    )

    assert score == 0.87
    assert len(seen) == 1
    assert seen[0]["url"] == "https://opencode.ai/inference/systemone/v1/systemone"
    assert seen[0]["body"]["model"] == "jev-1.13-free"
    assert seen[0]["body"]["questions"]["answers_question"]["type"] == "noul"
    assert seen[0]["body"]["state"]["query"] == "من هو الله"
    # preprocess() drops diacritics; the passage is capped, never rejected.
    assert "َ" not in seen[0]["body"]["state"]["passage"]
    assert len(seen[0]["body"]["state"]["passage"]) <= JEV_PASSAGE_CHARS


def test_jev_client_authenticates_and_names_itself():
    r = Retrieval(url="http://127.0.0.1:1", jev_api_key="test-key")

    assert r.jev_client.headers["authorization"] == "Bearer test-key"
    # Cloudflare answers a generic SDK user agent with 403/1010.
    assert r.jev_client.headers["user-agent"] == "elimam-backend/0.1"


def _points(n: int) -> list[dict]:
    return [{"id": i, "payload": {"text": f"مقطع رقم {i} عن الصبر"}} for i in range(n)]


def _index(passage: str) -> float:
    return float(passage.split()[2])


def test_rerank_sorts_by_noul_and_caps_the_fan_out():
    seen = []
    points = _points(10)
    r = Retrieval(url="http://127.0.0.1:1", jev_api_key="k", jev_pool=4)
    r.__dict__["jev_client"] = _jev(_index, seen)

    top = asyncio.run(r.rerank("الصبر", points, top_k=2))

    assert len(seen) == 4  # jev_pool caps the fan-out
    assert [p["id"] for p in top] == [3, 2]  # highest noul first
    assert all("reranker_score" in p for p in points[:4])
    assert all("reranker_score" not in p for p in points[4:])  # tail untouched


def test_rerank_sends_at_least_top_k_pairs():
    seen = []
    r = Retrieval(url="http://127.0.0.1:1", jev_api_key="k", jev_pool=4)
    r.__dict__["jev_client"] = _jev(_index, seen)

    asyncio.run(r.rerank("الصبر", _points(10), top_k=8))

    assert len(seen) == 8  # max(top_k, jev_pool), never more than the pool


def test_rerank_keeps_the_tail_when_top_k_exceeds_the_scored_head():
    r = Retrieval(url="http://127.0.0.1:1", jev_api_key="k", jev_pool=4)
    r.__dict__["jev_client"] = _jev(_index, [])

    top = asyncio.run(r.rerank("الصبر", _points(6), top_k=6))

    assert len(top) == 6  # 4 scored + 2 unscored tail, never short-changed


def test_rerank_falls_back_to_tei_when_jev_fails():
    r = Retrieval(url="http://127.0.0.1:1", jev_api_key="k")

    async def boom(query_text, text):
        raise httpx.ConnectError("no route")

    async def tei(query_text, points, top_k):
        return [{"id": "tei"}]

    r.__dict__["jev_client"] = _jev(_index, [])
    r._jev_score = boom
    r._rerank_tei = tei

    assert asyncio.run(r.rerank("الصبر", _points(3), top_k=2)) == [{"id": "tei"}]


def test_rerank_falls_back_to_tei_on_a_200_without_a_noul():
    r = Retrieval(url="http://127.0.0.1:1", jev_api_key="k")
    r.__dict__["jev_client"] = _jev(_index, [], status=200)
    # A 200 whose body has no answers.<id>.noul must not score 0.0 and sink.
    r.__dict__["jev_client"] = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"answers": {}})
        )
    )

    async def tei(query_text, points, top_k):
        return [{"id": "tei"}]

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
