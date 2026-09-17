# Retrieval Pipeline

How query → embedding → hybrid search → reranking works in `search.py`.

```mermaid
flowchart TD
  Q["Tool call from the agent<br/>dense_search / sparse_search / hybrid_search<br/>request = {collection, query_text, top_k, filters, rerank_pool}"]

  Q --> V{"Pydantic validation"}
  V -->|"invalid collection / extra field / non-int filter op"| VE["ValidationError -> agent sees a tool error"]
  V -->|ok| CAP["api/search.py route only:<br/>top_k <= 100, rerank_pool <= 500"]

  CAP --> PRE
  V --> PRE["_preprocess(text)<br/>camel_tools dediac_ar + normalize_alef_ar + strip U+200F + collapse whitespace"]

  PRE --> F["_build_filter(filters)<br/>list -> MatchAny, IntRangeFilter -> MatchValue + Range,<br/>scalar -> MatchValue, all joined with must"]
  F --> FY{"filters empty?"}
  FY -->|yes| FN["query_filter = None"]
  FY -->|no| FQ["models.Filter(must=[FieldCondition...])"]

  PRE --> D["_dense_query: GATE-AraBert-v1<br/>SentenceTransformer.encode -> float list"]
  PRE --> S["_sparse_query: fastembed Qdrant/bm25<br/>SparseTextEmbedding.embed -> indices + values"]

  D --> DD{"mode"}
  S --> DD
  DD -->|dense_search| D1["client.query_points(collection, using='dense',<br/>limit=rerank_pool, with_payload, no vectors)"]
  DD -->|sparse_search| D2["client.query_points(collection, using='sparse',<br/>query=SparseVector, limit=rerank_pool)"]
  DD -->|hybrid_search| D3["query_points(prefetch=[dense rerank_pool, sparse rerank_pool],<br/>query=FusionQuery(RRF), limit=rerank_pool)"]

  D1 --> RR
  D2 --> RR
  D3 --> RR
  FN --> D1
  FQ --> D1
  FN --> D2
  FQ --> D2
  FN --> D3
  FQ --> D3

  RR["_rerank(query_text, points, top_k)<br/>cross-encoder mmarco-mMiniLMv2-L12-H384-v1<br/>reranker.predict(preprocessed payload text, batch_size=32)"]
  RR --> SORT["attach reranker_score, sort desc, slice top_k"]
  SORT --> OUT["list of {id, version, score, payload, reranker_score}"]

  OUT --> A["Back to the agent as ToolReturnPart"]

  subgraph collections["Qdrant collections"]
    C1["quran"]
    C2["hadith"]
    C3["tafsir"]
    C4["books"]
    C5["sunnah"]
  end
  D1 --- collections
  D2 --- collections
  D3 --- collections

  subgraph getters["Direct-ID getters (no vectors, client.scroll)"]
    G1["get_quran(id)"]
    G2["get_hadith(book, hadith_number) -> 'book:number'"]
    G3["get_tafsir(book, id) -> 'book:id'"]
    G4["get_book(category, book_id, chunk_index) -> 'category:book:chunk'"]
    G5["get_sunnah(category, book_id, chunk_index)"]
    G6["get_books_hadith / get_books_tafsir / get_books_books / get_books_categories<br/>static lists"]
  end
  getters --> IDX["payload index on 'ids' (KEYWORD)"]

  IDX -.->|"created at import by setup_indexes()"| collections
```

Details that matter:

- **Model load is `lru_cache(maxsize=1)`** on `load_model` / `ranker_loader`; the first call prints `Loading Model`. Both are loaded at import.
- **Vectors are never returned** (`with_vectors=False`) — only payloads, which keeps tool results small enough for the context window.
- **`rerank_pool` is the only knob that controls recall** (Qdrant candidate count). `top_k` only slices after reranking, so `top_k` alone cannot improve recall.
- **RRF (Reciprocal Rank Fusion)** is done inside Qdrant via `FusionQuery(fusion=Fusion.RRF)`, not in Python.
- **Errors are swallowed into the result**: every search function catches `Exception` (not `ValueError`) and returns `[{"error": "Type: message"}]` so the agent can react instead of the SSE stream dying.
- `setup_indexes()` (called at `api/chat.py` import) creates the payload indexes for `ids` plus every filterable field per collection — `create_payload_index` is idempotent, so this runs on every boot.
- Collections live in gitignored `qdrant_storage/` and are populated by `notebooks/update_db.ipynb`; there is no seeding code in the running service except `vectordb/seed_*.py` scripts.

Filter field map (from `setup_indexes`):

| Collection | Filterable payload fields |
| --- | --- |
| quran | surah_number, surah |
| hadith | book, grade |
| tafsir | surah_number, surah, ayah_number |
| books | book_id, book_name, category_name, all_authors, author_death, book_date |
| sunnah | book_id, book_name, category_name, all_authors, author_death, book_date, athar_number |
