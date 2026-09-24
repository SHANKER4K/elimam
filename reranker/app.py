import os
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder


MODEL_NAME = "ALJIACHI/Mizan-Rerank-v1"

# Start conservatively on a 4-vCPU VPS.
BATCH_SIZE = 4

# Don't blindly send 8192-token documents to the reranker.
# Increase this after benchmarking.
MAX_LENGTH = 2048

model: CrossEncoder | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)

    print(f"Loading reranker: {MODEL_NAME}")

    model = CrossEncoder(
        MODEL_NAME,
        max_length=MAX_LENGTH,
        trust_remote_code=True,
        device="cpu",
    )

    model.model.eval()

    print("Reranker loaded successfully.")

    yield

    print("Shutting down reranker.")


app = FastAPI(
    title="Al-Imam Reranker",
    version="1.0.0",
    lifespan=lifespan,
)


class RerankRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10000)
    texts: list[str] = Field(min_length=1, max_length=50)


class RerankResult(BaseModel):
    index: int
    score: float
    text: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
    }


@app.post("/rerank")
def rerank(request: RerankRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Reranker is not ready",
        )

    start = time.perf_counter()

    pairs = [[request.query, text] for text in request.texts]

    with torch.inference_mode():
        scores = model.predict(
            pairs,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    results = [
        {
            "index": i,
            "score": float(score),
            "text": text,
        }
        for i, (score, text) in enumerate(zip(scores, request.texts))
    ]

    results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    elapsed = time.perf_counter() - start

    return {
        "results": results,
        "count": len(results),
        "latency_ms": round(elapsed * 1000, 2),
    }
