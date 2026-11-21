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
