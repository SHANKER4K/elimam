
# Islamic Knowledge Retrieval (Turath Index)

A Qdrant-backed corpus of five Arabic collections, searched via `dense_search` /
`sparse_search` / `hybrid_search` / `hybrid_search_weighted`, and read directly via
`get_*` functions when the exact reference is already known.

| Collection | Contents |
| --- | --- |
| `quran` | Every ayah, one point per ayah |
| `hadith` | 9 hadith books (see list below — **no Sahih Muslim indexed**) |
| `tafsir` | 5 tafsir books, one point per (book, ayah) |
| `books` | ~260 classical works: aqeedah, fiqh, hadith sciences, tarajim/tabaqat, adab/raqa'iq |
| `sunnah` | آثار السلف — reports *from the Sahaba and Tabi'in*, not from the Prophet ﷺ. Don't confuse this with `hadith`. |

## Step 1: does the user give an exact reference, or a topic?

**Exact reference → call the matching `get_*` function directly.** Don't search first.
**Topic, concept, or "what does Islam say about X" → search** (Step 2).

| User gives | Call |
| --- | --- |
| Surah:ayah (e.g. "آية 255 من سورة البقرة") | `get_quran(id="2:255")` |
| Hadith book + number (e.g. "حديث رقم 1 من صحيح البخاري") | `get_hadith(book="bukhari", hadith_number=1.0)` — **note: float, not int** |
| Tafsir book + ayah (e.g. "تفسير ابن كثير لآية الكرسي") | `get_tafsir(book="katheer", id="2:255")` |
| "the next/previous part" of a book excerpt already shown | `get_book(category, book_id, chunk_index)` or `get_suunah(...)` using numbers parsed from that excerpt's `ids` field — **never invent these numbers**, see Known Gaps |

If a book/slug name isn't obviously valid, call the relevant list function first (`get_books_hadith()`, `get_books_tafsir()`, `get_books_books()`, `get_books_sunnah()`, `get_books_categories()`) rather than guessing — they're static and instant, no reason not to check.

## Step 2: searching by topic

All four search functions share `(collection, query_text, top_k=10, ..., filters=None)`.
`collection` is one of `"quran"`, `"hadith"`, `"tafsir"`, `"books"`, `"sunnah"`.

- **`hybrid_search` — default choice.** Dense (semantic) + sparse (BM25) fused with RRF. Use this unless you have a specific reason to reach for one of the others below.
- **`dense_search`** — pure semantic similarity. Use when the user describes a concept loosely and exact wording doesn't matter (e.g. "ما يعين المسلم على الصبر عند البلاء").
- **`sparse_search`** — exact keyword/phrase match, no stemming. Use when the user quotes specific wording and wants its source (e.g. "من قال هذا الحديث: ... ").
- **`hybrid_search_weighted`** — hybrid_search with tunable `(dense_weight, sparse_weight)`. Only reach for this if plain `hybrid_search` results look clearly too loose or too literal for a given collection — not a first choice.

**`query_text` must always be Arabic**, even when you intend to answer in English — the dense model and BM25 index are both Arabic-only (see Response Language below).

Request `top_k=5` by default for a chat answer; only ask for more if the user wants a broader survey.

### Routing topic questions to a collection

- Ayah meaning/interpretation → `tafsir` (and/or `quran` for the ayah text itself)
- "Is there a hadith about X" / hadith-based evidence → `hadith`
- Fiqh ruling, aqeedah question, adab/manners, hadith-sciences questions → `books`, optionally filtered by `category_name` (call `get_books_categories()` if unsure which of the 8 fits)
- "What did [a Sahabi/Tabi'i] say/do about X" → `sunnah`

### Filters

Pass `filters={key: value}` to narrow before scoring. Scalar = equals; list = OR; `{eq|lt|gt|lte|gte: n}` = range (int fields only). Multiple keys are ANDed.

| Collection | Filter keys |
| --- | --- |
| `quran` | `surah_number` (int), `surah` (str) |
| `hadith` | `book` (str), `grade` (str) |
| `tafsir` | `surah_number` (int), `surah` (str), `ayah_number` (int) |
| `books` | `book_id` (int), `book_name` (str), `category_name` (str), `all_authors` (str), `author_death` (int), `book_date` (int) |
| `sunnah` | same as `books`, plus `athar_number` (int) |

Unknown keys, `bool` values, empty lists, and comparison operators on string fields all raise `ValueError` — stick to the table above rather than guessing a key.

## Known gaps — don't paper over these with confidence you don't have

- **No Sahih Muslim.** `HADITH_BOOKS` = abudawud, bukhari, dehlawi, ibnmajah, malik, nasai, nawawi, qudsi, tirmidhi. If asked specifically about Sahih Muslim, say it isn't in this index rather than substituting another book or answering from memory.
- **`get_book` / `get_suunah` numbering has no exposed lookup.** Nothing maps a category *name* to the integer `category` these functions expect. Only ever call them with `category`/`book_id`/`chunk_index` copied from an `ids` field you already saw in a search hit (format `"category:book_id:chunk_index"`) — never construct these from scratch.
- **Payload fields for `tafsir`, `books`, and `sunnah` are inferred, not confirmed.** Confirmed so far (real samples):
  - `quran` payload: `ids`, `text`, `ayah_number`, `surah_number`, `surah`
  - `hadith` payload: `ids`, `text`, `book`, `book_full`, `section`, `section_number`, `hadith_number`, `source`, and `grade` *when present* (absent for Bukhari — see below)
  For `tafsir`/`books`/`sunnah`, this skill assumes the payload mirrors the filter-schema key names exactly (that pattern held for both confirmed collections) plus `ids` and `text`. If a formatting instruction below references a field that turns out not to exist, fall back to printing whatever descriptive keys the payload actually has rather than failing silently.
- **Bukhari hadiths have no `grade` key at all** — they're unanimously accepted as authentic, so no grade was ever stored. Don't fabricate one; either omit the grade line or note it's from Sahih al-Bukhari.
- **The hadith `text` field already contains the full chain (sanad) and body (matn) together** as one string — there's nothing further to assemble, just print it as-is.

## Response language

Default to Arabic. Reply in English only if the user explicitly asks for it — but `query_text` sent to any search/get function stays Arabic regardless of reply language, since the index is Arabic-only. The "no results" fallback (below) should match whichever language you're replying in.

## Response format

Quran:

```text
> [!quran-ayah] {surah} {surah_number}:{ayah_number}
> {text}
```

Hadith — use the standard Arabic name for the book in the header (map slug → Arabic: bukhari→صحيح البخاري, abudawud→سنن أبي داود, tirmidhi→سنن الترمذي, ibnmajah→سنن ابن ماجه, nasai→السنن الصغرى للنسائي, malik→موطأ الإمام مالك, qudsi→الأحاديث القدسية, nawawi→الأربعون النووية, dehlawi→ use `book_full` if unsure of the Arabic name). Format `hadith_number` without a trailing `.0`:

```text
> [!hadith] {arabic_book_name} - {hadith_number}
> {text}
>
> {grade, or "صحيح بالإجماع" if book is bukhari and grade is absent}
```

Books / Sunnah excerpts — no page numbers exist in the payload, only `chunk_index`, so cite with that:

```text
[{book_name}/{page}]
```

in books the you will sometimes find the page in text like that "<<{page}>> if you find it just cite that the text behind it is from that page

or

```text
[{book_name}/{athar_number}]
```

This should be in sunnah collection

or, if a `source` URL is present in the payload: `[{book_name}/{chunk_index}]({source})`

## Rules

1. Every claim in the response must carry a citation — collection, book, and number/id. No uncited claims.
2. Before sending, re-check that every claim has a citation.
3. Never cite a similarity/relevance score as if it were a source.
4. No results found → say so plainly ("لا أدري" / "I don't know", matching reply language) rather than filling the gap from general knowledge.
5. Citations go in brackets: `[صحيح البخاري/1]`. If a source link exists in the payload, format as `[book_name/number](source)`.
6. Don't answer an Islamic question from memory if this index can answer it — search or look it up first.
7. Provide cite link when you have it
    - Shamela source format is `https://shamela.ws/book/{book_id}/{page}`
    - For hadith use `https://sunnah.com/{book_name}:{hadith_number}`
    - For Quran use `https://quran.com/{surah_number}/{ayah_number}`
8. Generate the source url from the text when page <<page>> is available
9. Reply in Arabic default unless you are asked not to

## Examples

**"أعطني آية الكرسي"** (exact ayah requested by name, not number)
→ Claude knows آية الكرسي is 2:255 (common knowledge) → `get_quran(id="2:255")` → format as quran-ayah block.

**"ما حكم الغيبة؟"** (fiqh topic, no exact reference)
→ `hybrid_search(collection="books", query_text="حكم الغيبة", top_k=5)`, optionally `filters={"category_name": "الفقه الحنبلي"}` if a fiqh-specific angle is wanted → cite each result as `[book_name/chunk_index]`.

**"حديث رقم 1 من صحيح البخاري"** (exact reference)
→ `get_hadith(book="bukhari", hadith_number=1.0)` → hadith block, grade line omitted or noted as unanimously authentic (no `grade` key present for Bukhari).

**"ماذا فعل عمر بن الخطاب عندما..."** (asking about a Sahabi's action/report)
→ `hybrid_search(collection="sunnah", query_text="...", top_k=5)`, not `hadith` — this is an athar, not a Prophetic hadith.

**"من قال: إنما الأعمال بالنيات؟"** (locating the source of an exact quoted phrase)
→ `sparse_search(collection="hadith", query_text="إنما الأعمال بالنيات", top_k=3)` — exact-phrase lookup, not semantic.
