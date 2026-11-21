"""Retrieval over the Qdrant turath collections.

One class owns the whole retrieval mechanism — the Qdrant client, the BM25
sparse model, the embedding/reranker HTTP clients, filter building, and the
dense+sparse RRF + cross-encoder rerank pipeline — so callers only ever touch
`retrieve()`. Everything heavy is created on first use (`cached_property`), so
importing this module needs no Qdrant, no model download, and no network.
"""

from __future__ import annotations

import asyncio
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

    # -----------------------------
    # Index setup
    # -----------------------------

    def setup_indexes(self) -> None:
        """Create payload indexes used by the getters and filter fields."""
        filter_types = {
            "int": models.PayloadSchemaType.INTEGER,
            "str": models.PayloadSchemaType.KEYWORD,
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
            "hadith": {"book": "str", "grade": "str"},
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
