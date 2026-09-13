from fastapi import APIRouter

from search import (
    SearchRequest,
    dense_search,
    sparse_search,
    hybrid_search,
)

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("/dense_search")
def dense_search_endpoint(request: SearchRequest) -> list:
    return dense_search(request)


@router.post("/sparse_search")
def sparse_search_endpoint(request: SearchRequest) -> list:
    return sparse_search(request)


@router.post("/hybrid_search")
def hybrid_search_endpoint(request: SearchRequest) -> list:
    return hybrid_search(request)
