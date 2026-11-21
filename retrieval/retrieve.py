"""Retrieval over the Qdrant turath collections.

One class owns the whole retrieval mechanism — the Qdrant client, the BM25
sparse model, the embedding/reranker HTTP clients, filter building, and the
dense+sparse RRF + cross-encoder rerank pipeline — so callers only ever touch
`retrieve()`. Everything heavy is created on first use (`cached_property`), so
importing this module needs no Qdrant, no model download, and no network.
"""

from __future__ import annotations

import asyncio
import logging
import os
from functools import cached_property
from typing import Literal

import httpx
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from pydantic import BaseModel
from pyarabic.araby import normalize_alef, strip_diacritics
from qdrant_client import QdrantClient, models

from .schemas import IntRangeFilter

load_dotenv()

CollectionName = Literal["quran", "hadith", "tafsir", "books", "sunnah"]
RetrievalMode = Literal["hybrid", "dense", "sparse"]

# Jev (TypeSafe System One) reranker. English instructions, Arabic state: that is
# TypeSafe's guidance for non-English content, and their own reranking recipe is
# one Noul per query-candidate pair.
JEV_URL = "https://opencode.ai/inference/systemone/v1/systemone"
JEV_MODEL = "jev-1.13-free"
JEV_CONCURRENCY = 8
# One deadline for the whole stage, not per request: at 8 in flight a 24-pair
# search is 3 waves, and 3 x the per-request timeout is not a search timeout.
JEV_STAGE_TIMEOUT = 15.0
# ponytail: one pair per request, no batching (state is shared by every question in
# a request), so a chunk longer than this is judged on its head. Upgrade path: raise
# it and pay the tokens, or split the chunk and take the max noul.
JEV_PASSAGE_CHARS = 1200
JEV_QUESTION = {
    "type": "noul",
    "instructions": "Does the passage contain information that answers the question?",
    "criteria": {
        "true": "The passage states, or directly supports, what the question asks.",
        "false": "The passage shares the topic but does not answer the question.",
    },
}

logger = logging.getLogger(__name__)


class Retrieval:
    def __init__(
        self,
        *,
        url: str | None = None,
        api_key: str | None = None,
        embedding_url: str | None = None,
        reranker_url: str | None = None,
        # ponytail: TEI rejects requests with more texts than max_client_batch_size
        # (422 "batch size N > maximum allowed batch size"), 32 on the deployed
        # reranker. Upgrade path: read /info once at startup instead of hardcoding.
        rerank_batch_size: int = 32,
        # None reads the env var; "" disables Jev explicitly (tests rely on that,
        # since load_dotenv() would otherwise hand them the operator's real key).
        jev_url: str | None = None,
        jev_api_key: str | None = None,
        jev_model: str | None = None,
        jev_pool: int | None = None,
    ) -> None:
        self.url = url or os.environ.get("QDRANT_URL", "http://localhost:6333")
        self.api_key = api_key or os.environ.get("QDRANT_API_KEY")
        self.embedding_url = embedding_url or os.environ.get(
            "EMBEDDING_URL", "http://embedding:80"
        )
        self.reranker_url = reranker_url or os.environ.get(
            "RERANKER_URL", "http://reranker:80"
        )
        self.rerank_batch_size = rerank_batch_size
        self.jev_url = (
            jev_url if jev_url is not None else os.environ.get("JEV_API_URL", JEV_URL)
        )
        self.jev_api_key = (
            jev_api_key if jev_api_key is not None else os.environ.get("JEV_API_KEY")
        )
        self.jev_model = (
            jev_model
            if jev_model is not None
            else os.environ.get("JEV_MODEL", JEV_MODEL)
        )
        self.jev_pool = (
            jev_pool if jev_pool is not None else int(os.environ.get("JEV_POOL", 24))
        )

    # -----------------------------
    # Lazily created dependencies
    # -----------------------------

    @cached_property
    def client(self) -> QdrantClient:
        return QdrantClient(url=self.url, api_key=self.api_key, cloud_inference=True)

    @cached_property
    def sparse_model(self) -> SparseTextEmbedding:
        return SparseTextEmbedding("Qdrant/bm25")

    @cached_property
    def embedding_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.embedding_url, timeout=30.0)

    @cached_property
    def reranker_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.reranker_url, timeout=60.0)

    @property
    def jev_enabled(self) -> bool:
        """A key is the switch; JEV_RERANK=off is the kill switch back to TEI."""
        return (
            bool(self.jev_api_key)
            and os.environ.get("JEV_RERANK", "on").strip().lower() != "off"
        )

    @cached_property
    def jev_client(self) -> httpx.AsyncClient:
        # No base_url: the endpoint arrives as one absolute URL and an empty
        # relative path would leave httpx guessing how to join them.
        return httpx.AsyncClient(
            timeout=httpx.Timeout(12.0, connect=5.0),
            headers={
                "Authorization": f"Bearer {self.jev_api_key}",
                # Cloudflare in front of the endpoint answers a generic SDK or
                # library user agent with 403 "error code: 1010".
                "User-Agent": "elimam-backend/0.1",
                # OpenCode Go asks every client to send a stable session id per
                # conversation; without it the request "cannot be routed
                # efficiently". One id per process = one long conversation.
                "x-opencode-session": f"elimam-{os.urandom(8).hex()}",
            },
        )

    # -----------------------------
    # Vectors and filters
    # -----------------------------

    def preprocess(self, text: str) -> str:
        # pyarabic replacement for camel-tools dediac_ar + normalize_alef_ar.
        # strip_diacritics also drops Quranic marks and the superscript alef;
        # normalize_alef folds أ إ آ ٱ — and ى, which used to be a separate
        # normalize_alef_maksura_ar call on the indexing side only.
        text = strip_diacritics(text)
        text = normalize_alef(text)
        text = text.replace("\u200f", "")
        return " ".join(text.split())

    async def dense_vector(self, text: str) -> list[float]:
        response = await self.embedding_client.post(
            "/embed",
            json={"inputs": [self.preprocess(text)]},
        )
        response.raise_for_status()
        return response.json()[0]

    def sparse_vector(self, text: str) -> models.SparseVector:
        vector = list(self.sparse_model.embed([text]))[0]
        return models.SparseVector(
            indices=vector.indices.tolist(),
            values=vector.values.tolist(),
        )

    @staticmethod
    def build_filter(filters: BaseModel | None) -> models.Filter | None:
        if filters is None or not filters.model_fields_set:
            return None

        conditions: list[models.FieldCondition] = []

        for key in filters.model_fields_set:
            value = getattr(filters, key)

            if isinstance(value, list):
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value),
                    )
                )
                continue

            if isinstance(value, IntRangeFilter):
                if value.eq is not None:
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value.eq),
                        )
                    )

                ranges = {
                    operator: getattr(value, operator)
                    for operator in ("lt", "gt", "lte", "gte")
                    if getattr(value, operator) is not None
                }
                if ranges:
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            range=models.Range(**ranges),
                        )
                    )
                continue

            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
            )

        return models.Filter(must=conditions)

    # -----------------------------
    # Reranking
    # -----------------------------

    async def _jev_score(self, query_text: str, text: str) -> float:
        """One System One Noul for one (query, passage) pair; 0..1."""
        response = await self.jev_client.post(
            self.jev_url,
            json={
                "model": self.jev_model,
                "state": {
                    "query": query_text,
                    "passage": self.preprocess(text)[:JEV_PASSAGE_CHARS],
                },
                "questions": {"answers_question": JEV_QUESTION},
            },
        )
        response.raise_for_status()
        return float(response.json()["answers"]["answers_question"]["noul"])

    async def _rerank_batch(self, query_text: str, batch: list[str]) -> list[dict]:
        """One TEI /rerank call; returns [{"index": i, "score": s}, ...]."""
        response = await self.reranker_client.post(
            "/rerank",
            json={"query": query_text, "texts": batch, "top_n": len(batch)},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()

    async def rerank(
        self,
        query_text: str,
        points: list[dict],
        top_k: int,
    ) -> list[dict]:
        """Re-rank with Jev, or with the cross-encoder when Jev is off or fails."""
        if not points or not self.jev_enabled:
            return await self._rerank_tei(query_text, points, top_k)

        # Jev is one HTTP round trip per pair, so it only ever sees the head of the
        # pool — but never fewer candidates than the caller asked to be returned.
        head = points[: min(len(points), max(top_k, self.jev_pool))]
        pairs: list[tuple[dict, str]] = []
        for point in head:
            text = point.get("payload", {}).get("text")
            # Same guard as the TEI path: a missing or non-string text is skipped,
            # never sent as an empty passage.
            if isinstance(text, str) and text.strip():
                pairs.append((point, text))

        # Nothing scoreable in the head is a TEI answer, not an empty result.
        if not pairs:
            return await self._rerank_tei(query_text, points, top_k)

        semaphore = asyncio.Semaphore(JEV_CONCURRENCY)

        async def score(point: dict, text: str) -> float:
            async with semaphore:
                return await self._jev_score(query_text, text)

        # ponytail: one bad pair falls back for the whole request instead of mixing
        # two score scales in one result list. Jev failures are correlated (key,
        # 429, egress), not per-pair, so paying one TEI call is the cheap side of
        # that bet. The tasks are cancelled rather than left running, and the stage
        # has one deadline so N waves of a slow endpoint cannot stall a search.
        tasks = [asyncio.create_task(score(point, text)) for point, text in pairs]
        try:
            async with asyncio.timeout(JEV_STAGE_TIMEOUT):
                scores = await asyncio.gather(*tasks)
        except Exception:
            for task in tasks:
                task.cancel()
            logger.warning(
                "jev rerank failed; falling back to the cross-encoder", exc_info=True
            )
            return await self._rerank_tei(query_text, points, top_k)

        for (point, _), value in zip(pairs, scores):
            point["reranker_score"] = value

        valid = [point for point, _ in pairs]
        valid.sort(key=lambda point: point["reranker_score"], reverse=True)
        return (valid + points[len(head) :])[:top_k]

    async def _rerank_tei(
        self,
        query_text: str,
        points: list[dict],
        top_k: int,
    ) -> list[dict]:
        """The local cross-encoder path, and the fallback when Jev is unavailable."""
        if not points:
            return points

        # 1. Keep only points that actually carry a text payload, remembering
        # their order so batch-local reranker indices map back correctly.
        valid_points: list[dict] = []
        texts: list[str] = []

        for point in points:
            payload_text = point.get("payload", {}).get("text")
            if payload_text and isinstance(payload_text, str) and payload_text.strip():
                texts.append(self.preprocess(payload_text))
                valid_points.append(point)

        if not valid_points:
            return points[:top_k]

        # 2. TEI caps a /rerank request at max_client_batch_size texts, so send
        # candidates in chunks and offset indices back into `texts`. The chunks
        # are independent, so they run concurrently; at most a handful per call.
        starts = range(0, len(texts), self.rerank_batch_size)
        batches = [texts[start : start + self.rerank_batch_size] for start in starts]
        batch_results = await asyncio.gather(
            *(self._rerank_batch(query_text, batch) for batch in batches)
        )

        for start, results in zip(starts, batch_results):
            for result in results:
                valid_points[start + result["index"]]["reranker_score"] = float(
                    result["score"]
                )

        valid_points.sort(
            key=lambda point: point.get("reranker_score", -float("inf")),
            reverse=True,
        )

        return valid_points[:top_k]

    # -----------------------------
    # Candidate generation
    # -----------------------------

    def _query_points(
        self,
        *,
        collection: str,
        query,
        using: str,
        limit: int,
        query_filter: models.Filter | None,
    ) -> list[dict]:
        return [
            point.model_dump()
            for point in self.client.query_points(
                collection_name=collection,
                query=query,
                using=using,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                query_filter=query_filter,
            ).points
        ]

    async def _dense_points(
        self,
        collection: str,
        query_text: str,
        limit: int,
        query_filter: models.Filter | None,
    ) -> list[dict]:
        return self._query_points(
            collection=collection,
            query=await self.dense_vector(query_text),
            using="dense",
            limit=limit,
            query_filter=query_filter,
        )

    async def _sparse_points(
        self,
        collection: str,
        query_text: str,
        limit: int,
        query_filter: models.Filter | None,
    ) -> list[dict]:
        return self._query_points(
            collection=collection,
            query=self.sparse_vector(query_text),
            using="sparse",
            limit=limit,
            query_filter=query_filter,
        )

    async def _hybrid_points(
        self,
        collection: str,
        query_text: str,
        limit: int,
        query_filter: models.Filter | None,
    ) -> list[dict]:
        return [
            point.model_dump()
            for point in self.client.query_points(
                collection_name=collection,
                prefetch=[
                    models.Prefetch(
                        query=await self.dense_vector(query_text),
                        using="dense",
                        limit=limit,
                    ),
                    models.Prefetch(
                        query=self.sparse_vector(query_text),
                        using="sparse",
                        limit=limit,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
                with_vectors=False,
                query_filter=query_filter,
            ).points
        ]

    # -----------------------------
    # Public retrieval
    # -----------------------------

    async def retrieve(
        self,
        *,
        collection: CollectionName,
        query_text: str,
        top_k: int = 10,
        rerank_pool: int = 50,
        filters: BaseModel | None = None,
        mode: RetrievalMode = "hybrid",
    ) -> list[dict]:
        """Search one collection and rerank the candidates.

        `mode` picks the candidate generator: "hybrid" (dense+BM25 RRF, the
        default and what the agent uses), "dense" (semantic only), or "sparse"
        (BM25 only). All three end in the same cross-encoder rerank.
        """
        query_text = self.preprocess(query_text)
        query_filter = self.build_filter(filters)

        fetchers = {
            "dense": self._dense_points,
            "sparse": self._sparse_points,
            "hybrid": self._hybrid_points,
        }

        points = await fetchers[mode](
            collection, query_text, rerank_pool, query_filter
        )
        return await self.rerank(query_text, points, top_k)

    def by_id(self, collection: str, key: str):
        """Fetch the single payload whose `ids` field equals `key`."""
        points, _ = self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="ids",
                        match=models.MatchValue(value=key),
                    )
                ]
            ),
            with_vectors=False,
        )
        return points[0].payload if points else []

    def by_hadith(self, book: str, hadith_number: float):
        """Fetch a hadith by book + number.

        The hadith collection has no `ids` payload field (the seed never wrote
        one), so `by_id` cannot find it - the identity lives in book +
        hadith_number. Range [n, n] is the numeric match: MatchValue rejects
        floats, and has been the source of silent empty lookups.
        """
        points, _ = self.client.scroll(
            collection_name="hadith",
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="book", match=models.MatchValue(value=book)
                    ),
                    models.FieldCondition(
                        key="hadith_number",
                        range=models.Range(gte=hadith_number, lte=hadith_number),
                    ),
                ]
            ),
            with_vectors=False,
        )
        if not points:
            return []
        payload = dict(points[0].payload)
        # Same id convention the citation registry will use (issue #7):
        # hadith -> "book:number", number kept as the stored float.
        payload.setdefault("ids", f"{book}:{float(hadith_number)}")
        return payload

    # -----------------------------
    # Index setup
    # -----------------------------

    def setup_indexes(self) -> None:
        """Create payload indexes used by the getters and filter fields."""
        filter_types = {
            "int": models.PayloadSchemaType.INTEGER,
            "str": models.PayloadSchemaType.KEYWORD,
            "float": models.PayloadSchemaType.FLOAT,
        }
        fields: dict[str, set[tuple[str, models.PayloadSchemaType]]] = {
            "quran": {("ids", models.PayloadSchemaType.KEYWORD)},
            "hadith": {
                ("ids", models.PayloadSchemaType.KEYWORD),
                ("book", models.PayloadSchemaType.KEYWORD),
            },
            "tafsir": {
                ("ids", models.PayloadSchemaType.KEYWORD),
                ("tafsir_book", models.PayloadSchemaType.KEYWORD),
            },
            "books": {
                ("ids", models.PayloadSchemaType.KEYWORD),
                ("book_id", models.PayloadSchemaType.INTEGER),
            },
            "sunnah": {
                ("ids", models.PayloadSchemaType.KEYWORD),
                ("book_id", models.PayloadSchemaType.INTEGER),
            },
        }

        filter_schema = {
            "quran": {"surah_number": "int", "surah": "str"},
            "hadith": {"book": "str", "grade": "str", "hadith_number": "float"},
            "tafsir": {"surah_number": "int", "surah": "str", "ayah_number": "int"},
            "books": {
                "book_id": "int",
                "book_name": "str",
                "category_name": "str",
                "all_authors": "str",
                "author_death": "int",
                "book_date": "int",
            },
            "sunnah": {
                "book_id": "int",
                "book_name": "str",
                "category_name": "str",
                "all_authors": "str",
                "author_death": "int",
                "book_date": "int",
                "athar_number": "int",
            },
        }

        for collection, schema in filter_schema.items():
            fields[collection].update(
                (key, filter_types[kind]) for key, kind in schema.items()
            )

        for collection, index_set in fields.items():
            for field, schema in index_set:
                self.client.create_payload_index(
                    collection,
                    field_name=field,
                    field_schema=schema,
                )


# ponytail: one process-wide instance. The clients and the BM25 model are global
# resources anyway, and every dependency above is lazy, so this costs nothing at
# import and still lets tests build a throwaway `Retrieval(url=...)` instead.
retrieval = Retrieval()
