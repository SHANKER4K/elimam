#!/usr/bin/env python3
"""
Add headers to shameladb_books.csv content pages.

Each content entry gets a 'header' field (most specific section header for that page)
and a 'headers' field (full ordered path of all active headers).

Usage: python get_headers.py
       python get_headers.py --regen
"""
import sqlite3, json, sys, glob, os, ast
import pandas as pd
from pathlib import Path

BASE = Path("shameladb/database")
BOOK_DIR = BASE / "book"
CSV = "shameladb_books.csv"


def load_title_store():
    """Load Lucene store/title into {(book_id, title_id): text}."""
    import jpype
    jars = glob.glob("shameladb/app/lucene/2/*.jar")
    jvm_path = "shameladb/app/linux/64/jre/2/lib/server/libjvm.so"
    if not jpype.isJVMStarted():
        jpype.startJVM(jvmpath=jvm_path, classpath=[":".join(jars)])
    MMapDirectory = jpype.JClass("org.apache.lucene.store.MMapDirectory")
    DirectoryReader = jpype.JClass("org.apache.lucene.index.DirectoryReader")
    Paths = jpype.JClass("java.nio.file.Paths")
    store_path = str(BASE / "store" / "title")
    if not os.path.isdir(store_path):
        return {}
    directory = MMapDirectory(Paths.get(store_path))
    reader = DirectoryReader.open(directory)
    sf = reader.storedFields()
    result = {}
    for i in range(reader.numDocs()):
        doc = sf.document(i)
        doc_id = str(doc.get("id") or "")
        body = str(doc.get("body") or "")
        if "-" in doc_id:
            parts = doc_id.rsplit("-", 1)
            try:
                book_id = int(parts[0])
                title_id = int(parts[1])
                result[(book_id, title_id)] = body
            except ValueError:
                pass
    reader.close()
    return result


def find_book_db(book_id: int) -> Path | None:
    for subdir in BOOK_DIR.iterdir():
        if subdir.is_dir():
            db = subdir / f"{book_id}.db"
            if db.exists():
                return db
    return None


def build_header_tree(title_rows, header_texts):
    """
    Build parent→children map and return (children_map, parent_map).
    children_map: {parent_id: [(title_id, text, page_pid), ...]}
    parent_map: {title_id: parent_id}
    """
    children = {}
    parent_map = {}
    page_of = {}
    for tid, page_pid, parent in title_rows:
        children.setdefault(parent, []).append((tid, page_pid, parent))
        parent_map[tid] = parent
        page_of[tid] = page_pid

    # Sort children of each parent by their page_pid
    for p in children:
        children[p].sort(key=lambda x: x[1])

    return children, parent_map, page_of


def get_header_path(tid, children, parent_map, page_of, header_texts):
    """Get ordered list of header texts from root to this title."""
    # Build path bottom-up, then reverse
    path = []
    cur = tid
    while cur and cur != 0:
        text = header_texts.get(cur, f"[title {cur}]")
        path.append(text)
        cur = parent_map.get(cur, 0)
    path.reverse()

    # Also find siblings that start at the same page (they're alternate titles
    # at the same level, pick the one that's this title)
    return path


def main():
    regen = "--regen" in sys.argv
    df = pd.read_csv(CSV)

    if "headers" in df.columns and not regen:
        print("Already has 'headers' column. Use --regen to rebuild.")
        return

    downloaded = df[df["is_downloaded"] == True]
    print("Loading Lucene title store...")
    title_store = load_title_store()
    print(f"  {len(title_store)} title entries loaded")

    book_headers_list = {}
    updated_content = {}

    for _, row in downloaded.iterrows():
        book_id = int(row["book_id"])
        print(f"  Book {book_id}: {row['book_name'][:50]}...", end=" ")

        db_path = find_book_db(book_id)
        if not db_path:
            print("SKIP (no DB)")
            continue

        conn = sqlite3.connect(str(db_path))
        has_title = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='title'"
        ).fetchone()
        if not has_title:
            print("SKIP (no title table)")
            conn.close()
            continue

        title_rows = conn.execute(
            "SELECT id, page, parent FROM title ORDER BY id"
        ).fetchall()
        conn.close()

        if not title_rows:
            print("SKIP (empty)")
            continue

        # ── Get header texts from title store ──
        header_texts = {}
        for tid, _, _ in title_rows:
            text = title_store.get((book_id, tid))
            if text:
                header_texts[tid] = text

        # ── Build hierarchy ──
        children, parent_map, page_of = build_header_tree(title_rows, header_texts)

        # ── Flat headers list for the book ──
        flat_headers = []
        for tid, _, _ in title_rows:
            if tid in header_texts:
                flat_headers.append(header_texts[tid])
        book_headers_list[book_id] = json.dumps(flat_headers, ensure_ascii=False)

        # ── Parse content ──
        raw = row["content"]
        try:
            content = ast.literal_eval(raw) if isinstance(raw, str) else raw
        except Exception:
            print("SKIP (bad content)")
            continue

        # ── Assign header(s) per page ──
        # Walk content sequentially, tracking which titles are "open"
        # For each content page, determine the most specific header
        # by finding the title whose page_pid is the closest ≤ current page
        
        # Build: for each page that starts a title, what title(s) start there
        title_starts = {}  # {page_pid: [(tid, text), ...]}
        for tid, page_pid, parent in title_rows:
            if tid in header_texts:
                title_starts.setdefault(page_pid, []).append((tid, header_texts[tid]))

        # Active header stack: [(tid, text), ...]
        # As we move through pages, we open new titles and close old ones
        # A title at page P is active from P until the next title at same/higher level starts
        
        # Since page_ids are sequential (book_id-1, book_id-2, ...), we can just walk
        # and find the nearest preceding title for each page
        sorted_start_pages = sorted(title_starts.keys())

        import bisect

        for entry in content:
            pid = entry.get("page_id", "")
            parts = pid.split("-")
            if len(parts) != 2:
                continue
            try:
                page_num = int(parts[1])
            except ValueError:
                continue

            # Find all titles that start on or before this page
            # The most specific one is the last one among them (by depth priority)
            # Actually: among titles that start ≤ page_num, find the one with deepest
            # nesting that is still "active" (no later title at same level has started)
            
            # Simple approach: find the rightmost start page ≤ current page
            idx = bisect.bisect_right(sorted_start_pages, page_num) - 1
            
            active_headers = []
            if idx >= 0:
                start_page = sorted_start_pages[idx]
                titles_here = title_starts[start_page]
                # Take the last title starting at this page (most specific)
                tid, text = titles_here[-1]
                # Build full path
                path = get_header_path(tid, children, parent_map, page_of, header_texts)
                active_headers = path
                most_specific = text
            else:
                most_specific = ""

            entry["header"] = most_specific
            entry["headers"] = active_headers  # full path

        updated_content[book_id] = json.dumps(content, ensure_ascii=False)
        print(f"{len(flat_headers)} headers, {len(content)} pages tagged")

    df["headers"] = df["book_id"].map(book_headers_list)
    df["content"] = df["book_id"].map(updated_content).fillna(df["content"])
    df.to_csv(CSV, index=False)
    total = df["headers"].notna().sum()
    print(f"\nDone. {total} books. Each content entry now has:")
    print("  'header'  → most specific section header for this page")
    print("  'headers' → full ordered path [parent, ..., child]")


if __name__ == "__main__":
    main()
