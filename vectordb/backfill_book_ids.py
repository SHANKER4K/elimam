"""Backfill `category_id`, `book_id` and `chunk_index` on books/sunnah points.

The live payloads carry these only inside `ids` ("category:book:chunk"), so
the `book_id` filter the agent is offered matched nothing (issue #25). This
copies them into payload fields from `ids` -- no re-embedding, vectors are
untouched. Safe to re-run: points that already have `book_id` are skipped.

    python -m vectordb.backfill_book_ids                 # both collections
    python -m vectordb.backfill_book_ids --dry-run books # just count

QDRANT_URL / QDRANT_API_KEY come from the environment (.env), like the app.
"""

import argparse
import logging
import os

from qdrant_client import QdrantClient, models

logger = logging.getLogger("vectordb.backfill_book_ids")

COLLECTIONS = ("books", "sunnah")
PAGE = 1_000


def parse_ids(ids: object) -> dict[str, int] | None:
    """"1:121:0" -> {"category_id": 1, "book_id": 121, "chunk_index": 0}."""
    if not isinstance(ids, str):
        return None
    parts = ids.split(":")
    if len(parts) != 3 or not all(p.strip().isdigit() for p in parts):
        return None
    category_id, book_id, chunk_index = (int(p) for p in parts)
    return {"category_id": category_id, "book_id": book_id, "chunk_index": chunk_index}


def backfill(client: QdrantClient, collection: str, *, dry_run: bool = False) -> int:
    """Set the fields on every point of `collection` that lacks `book_id`.
    Returns how many points were (or, with dry_run, would be) updated."""
    missing = models.Filter(must=[models.IsEmptyCondition(is_empty=models.PayloadField(key="book_id"))])
    updated = skipped = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection,
            scroll_filter=missing,
            limit=PAGE,
            offset=offset,
            with_payload=["ids"],
            with_vectors=False,
        )
        operations = []
        for point in points:
            fields = parse_ids((point.payload or {}).get("ids"))
            if fields is None:
                skipped += 1
                logger.warning("unparseable_ids", extra={"collection": collection, "point": point.id})
                continue
            operations.append(
                models.SetPayloadOperation(
                    set_payload=models.SetPayload(payload=fields, points=[point.id])
                )
            )
        if operations and not dry_run:
            client.batch_update_points(collection, update_operations=operations, wait=True)
        updated += len(operations)
        # The offset is a point id: points updated above it are behind us, so
        # paging stays correct while the "missing" filter shrinks.
        if offset is None:
            break
    logger.info("backfill_done", extra={"collection": collection, "updated": updated, "skipped": skipped})
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("collections", nargs="*", default=list(COLLECTIONS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from dotenv import load_dotenv

    load_dotenv()
    client = QdrantClient(
        url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        api_key=os.environ.get("QDRANT_API_KEY") or None,
        timeout=60,
    )
    for collection in args.collections:
        n = backfill(client, collection, dry_run=args.dry_run)
        print(f"{collection}: {'would update' if args.dry_run else 'updated'} {n} points")


if __name__ == "__main__":
    main()
