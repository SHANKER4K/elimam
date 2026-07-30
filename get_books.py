#!/usr/bin/env python3
"""
Extract all books metadata + content from shameladb/
"""
import sqlite3, json, os, glob
import pandas as pd
from pathlib import Path

BASE = Path("shameladb/database")
MASTER = BASE / "master.db"
BOOK_DIR = BASE / "book"

# ── 1. Books with metadata ──
conn = sqlite3.connect(str(MASTER))

books = pd.read_sql("""
    SELECT b.*, c.category_name, a.author_name AS main_author_name,
           a.death_number AS main_author_death
    FROM book b
    LEFT JOIN category c ON b.book_category = c.category_id
    LEFT JOIN author a ON b.main_author = a.author_id
""", conn)

authors_agg = pd.read_sql("""
    SELECT ab.book_id, GROUP_CONCAT(atr.author_name, ' — ') AS all_authors
    FROM author_book ab
    JOIN author atr ON ab.author_id = atr.author_id
    GROUP BY ab.book_id
""", conn)

coauthors_agg = pd.read_sql("""
    SELECT cab.book_id, GROUP_CONCAT(atr.author_name, ' — ') AS coauthors
    FROM coauthor_book cab
    JOIN author atr ON cab.author_id = atr.author_id
    GROUP BY cab.book_id
""", conn)

books = books.merge(authors_agg, on="book_id", how="left")
books = books.merge(coauthors_agg, on="book_id", how="left")
conn.close()

# Mark downloaded books (those with a local .db in book/)
downloaded_ids = set()
for subdir in BOOK_DIR.iterdir():
    if subdir.is_dir():
        for f in subdir.iterdir():
            if f.suffix == ".db":
                downloaded_ids.add(int(f.stem))

books["is_downloaded"] = books["book_id"].isin(downloaded_ids)

print(f"Total books: {len(books)}")
print(f"Downloaded (local content): {books['is_downloaded'].sum()}")
print()

# ── 2. Extract content via JPype + Lucene ──
def extract_content():
    """Returns {book_id: [page_id, body], ...} from Lucene page/title indexes."""
    import jpype

    jars = glob.glob("shameladb/app/lucene/2/*.jar")
    jvm_path = "shameladb/app/linux/64/jre/2/lib/server/libjvm.so"

    if not jpype.isJVMStarted():
        jpype.startJVM(jvmpath=jvm_path, classpath=[":".join(jars)])

    MMapDirectory = jpype.JClass("org.apache.lucene.store.MMapDirectory")
    DirectoryReader = jpype.JClass("org.apache.lucene.index.DirectoryReader")
    Paths = jpype.JClass("java.nio.file.Paths")

    result = {}
    for store in ["page", "title"]:
        store_path = str(BASE / "store" / store)
        if not os.path.isdir(store_path):
            continue
        directory = MMapDirectory(Paths.get(store_path))
        reader = DirectoryReader.open(directory)
        sf = reader.storedFields()

        for i in range(reader.numDocs()):
            doc = sf.document(i)
            
            id_ = str(doc.get("id") or "")
            body = str(doc.get("body") or "")
            if "-" in id_:
                bid = id_.rsplit("-", 1)[0]
                result.setdefault(bid, []).append({"page_id": id_, "body": body})
        reader.close()

    return result

try:
    content = extract_content()
    books["content"] = books["book_id"].astype(str).map(content)
    books["page_count"] = books["content"].apply(lambda x: len(x) if isinstance(x, list) else 0)

    for _, row in books[books["is_downloaded"]].iterrows():
        print(f"\n  Book {row['book_id']}: {row['book_name']}")
        print(f"    Author: {row['main_author_name']}")
        print(f"    Pages: {row['page_count']}")
        if row["page_count"]:
            print(f"    First page: {row['content'][0]['body'][:120]}")
except Exception as e:
    print(f"Content extraction skipped: {e}")
    print("Install JPype: uv pip install jpype1")

# ── 3. Export ──
books.to_csv("shameladb_books.csv", index=False)
print(f"\nSaved: shamela_books.csv ({len(books)} books)")
print(f"Columns: {list(books.columns)}")
