from typing import Literal

from fastapi import APIRouter, HTTPException

from retrieval import SearchRequest, retrieval

router = APIRouter(prefix="/search", tags=["Search"])

# ponytail: a fixed cap, not a rate limiter. Each request runs a cross-encoder
# rerank; 100/500 is far above any real client and stops the amplification.
MAX_TOP_K = 100
MAX_RERANK_POOL = 500


def _enforce_caps(request: SearchRequest) -> None:
    if request.top_k > MAX_TOP_K or request.rerank_pool > MAX_RERANK_POOL:
        raise HTTPException(
            status_code=400,
            detail=f"top_k must be <= {MAX_TOP_K} and rerank_pool <= {MAX_RERANK_POOL}",
        )


async def _run(
    request: SearchRequest, mode: Literal["dense", "sparse", "hybrid"]
) -> list:
    _enforce_caps(request)

    try:
        return await retrieval.retrieve(
            collection=request.collection,
            query_text=request.query_text,
            top_k=request.top_k,
            rerank_pool=request.rerank_pool,
            filters=request.filters,
            mode=mode,
        )
    except Exception as exc:
        # ponytail: the class raises; Qdrant/TEI failures are upstream, so answer
        # 502 instead of the old 200-with-an-error-object-inside-the-list.
        raise HTTPException(
            status_code=502, detail=f"{type(exc).__name__}: {exc}"
        ) from exc


@router.post("/dense_search")
async def dense_search_endpoint(request: SearchRequest) -> list:
    return await _run(request, "dense")


@router.post("/sparse_search")
async def sparse_search_endpoint(request: SearchRequest) -> list:
    return await _run(request, "sparse")


@router.post("/hybrid_search")
async def hybrid_search_endpoint(request: SearchRequest) -> list:
    return await _run(request, "hybrid")
