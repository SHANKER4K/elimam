from fastapi import APIRouter, HTTPException

from search import (
    SearchRequest,
    dense_search,
    sparse_search,
    hybrid_search,
)

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


@router.post("/dense_search")
async def dense_search_endpoint(request: SearchRequest) -> list:
    _enforce_caps(request)
    return await dense_search(request)


@router.post("/sparse_search")
async def sparse_search_endpoint(request: SearchRequest) -> list:
    _enforce_caps(request)
    return await sparse_search(request)


@router.post("/hybrid_search")
async def hybrid_search_endpoint(request: SearchRequest) -> list:
    _enforce_caps(request)
    return await hybrid_search(request)
