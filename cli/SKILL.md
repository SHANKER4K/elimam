# Turath Index

> **Agentic Islamic Knowledge Search.**
> The Agent manages searches in the Quran, Hadiths, and books via a local Index + API, and intelligently synthesizes the answer.

---

## Overview

**How it works:**

1. The user asks a question.
2. The Agent understands the question and decides: What to search? Where to search? Which endpoint to use?
3. It makes a curl request to the API → retrieves the results.
4. It reads the retrieved texts.
5. It synthesizes an understandable answer, citing the sources.

---

## When to Use

When the user asks about:

- **The Holy Quran** — "What is the interpretation of such-and-such verse?", "Where is such-and-such mentioned in the Quran?"
- **Prophetic Hadith** — "Is this hadith authentic?", "Who narrated the hadith of such-and-such?"
- **Tafsir (Exegesis)** — "What did Ibn Kathir say about this verse?"
- **Fiqh (Jurisprudence)** — "What is the ruling on such-and-such?", "What is the evidence for such-and-such?"
- **Seerah (Biography of the Prophet)** — "When did the Battle of Badr take place?"
- **Aqeedah (Creed)** — "What is the evidence for Tawhid from the Quran?"
- **Arabic Language** — "An attestation from the Quran for this grammatical rule."

**Don't use for:**

- General questions unrelated to Islam (do not answer them).
- A need for contemporary consensus or modern fatwas (use web-search to search fatwa websites).
- Deep linguistic analysis detached from heritage texts.

---

## Core Operations

### A. Start the Server

```bash
cd ~/Documents/projects/python/ML/Summer/projects/islam/cli/
/home/shk/Documents/projects/python/ML/Summer/projects/islam/islamvenv/bin/python3 -m uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

### B. API Server

A **FastAPI server** (`server.py`) runs the semantic search index over Quran, hadith, and tafsir.

**The Agent does the following:**

1. Decides the appropriate keywords for the search (converts the question into search keywords).
2. Makes a curl request to the API.
3. Reviews the results.
4. If results are few or inaccurate, it tries other keywords or adds a `where` filter.

#### Endpoints

| Endpoint | Purpose | Example |
| --- | --- | --- |
| `GET /health` | Check server health | `/health` |
| `GET /get-ayah` | Get verse(s) by number | `?surah=1&ayah=1-5` |
| `GET /get-hadith` | Get a hadith by ID | `?book=bukhari&hadith_number=1` |
| `GET /get-tafsir` | Get tafsir for a verse | `?book=ibnkathir&surah=1&ayah=1` |
| `GET /quran-search` | Semantic search over Quran | `?query=التوكل&k=10` |
| `GET /hadith-search` | Semantic search over hadith | `?query=الصلاة&k=10&where={...}` |
| `GET /tafsir-search` | Semantic search over tafsir | `?query=الرحمن&k=10&where={...}` |

#### Getting a verse

```bash
# Single verse
curl "http://127.0.0.1:8000/get-ayah?surah=1&ayah=1"

# Verse range
curl "http://127.0.0.1:8000/get-ayah?surah=1&ayah=1-3"
```

#### Getting a hadith

```bash
curl "http://127.0.0.1:8000/get-hadith?book=bukhari&hadith_number=1"
```

**Books:** 'abudawud', 'bukhari', 'dehlawi', 'ibnmajah', 'malik', 'nasai', 'nawawi', 'qudsi', 'tirmidhi'

**Full Books Name**: 'Sunan Abu Dawud', 'Sahih al Bukhari', 'Forty Hadith of Shah Waliullah Dehlawi', 'Sunan Ibn Majah', 'Muwatta Malik', 'Sunan an Nasai', 'Forty Hadith of an-Nawawi', 'Forty Hadith Qudsi', 'Jami At Tirmidhi'

In the API you should use **Books**

#### Getting tafsir

```bash
curl "http://127.0.0.1:8000/get-tafsir?book=ibnkathir&surah=1&ayah=1"

# All tafsir books
curl "http://127.0.0.1:8000/get-tafsir?book=*&surah=1&ayah=1"
```

**Books:** `tabari`, `ibnkathir`, `saadi`, `baghawi`, `chengiti`

#### Semantic search

```bash
# Simple search
curl "http://127.0.0.1:8000/quran-search?query=%D8%A7%D9%84%D8%AA%D9%88%D9%83%D9%84&k=10"

# Filter by field (JSON-encoded `where` parameter)
curl "http://127.0.0.1:8000/hadith-search?query=%D8%B5%D9%84%D8%A7%D8%A9&k=10&where=%7B%22book%22%3A%22Bukhari%22%7D"
# where={"book":"Bukhari"} URL-encoded
```

**`where` filter examples (URL-encode the JSON):**

- `{"book": "Bukhari"}` — filter by book
- `{"grade": "صحيح"}` — filter by grade (Sahih)
- `{"surah_number": 1}` — filter by surah number
- `{"tafsir_book_short": "tabari"}` — filter by tafsir book

#### Health check

```bash
curl "http://127.0.0.1:8000/health"

# Returns:
# {
#   "status": "ok",
#   "quran_count": 6236,
#   "tafsir_count": 38520,
#   "hadith_count": 28246
# }
```

#### Ask the agent to do it

```
User: "What are the verses that talk about Tawakkul (reliance on Allah)?"
Agent: curl "http://127.0.0.1:8000/quran-search?query=%D8%AA%D9%88%D9%83%D9%84&k=10"
       -> reads the results
       -> writes: "The word Tawakkul is mentioned in the Quran in several verses, including..."
```

### C. Synthesis (Agent's job — no script)

The Agent **displays the raw results** to the user and synthesizes an answer from them:

❌ Incorrect: "From the results I searched for, this is what I found: [list of results without sources]"

✅ Correct: "Tawakkul (reliance) on Allah is one of the most important traits of a believer. The Almighty said in Surah Ali 'Imran (159): ...
And in Surah At-Tawbah (51): ... As for the Sunnah, the Prophet ﷺ said..."
The Prophet ﷺ said: (mention the Hadith)
From this, it is clear that Tawakkul is an act of worship for Allah, the Glorious and Exalted.
Ibn Abbas said: (mention the report/athar)
Here, it becomes clear to us that Tawakkul brings peace to a person...

---

## Search Strategies for Arabic Texts

### Search Strategy Decision Tree

When the user asks a question, the Agent decides:

1. **Asking about a specific verse** → `curl .../get-ayah?surah=N&ayah=N`
2. **Asking about a specific hadith** → `curl .../get-hadith?book=X&&hadith_number=N`
3. **Semantic search in the Quran** → `curl .../quran-search?query=...&k=10`
4. **Semantic search in Hadith** → `curl .../hadith-search?query=...&k=10`
5. **Tafsir of a verse** → `curl .../get-tafsir?book=X&surah=N&ayah=N`
6. **Insufficient results** → Try synonyms or use the `where` filter.

### Keyword Reformulation Guide (from testing)

The semantic model (GATE-AraBert-v1) works best with **phrase-level** queries that appear in the actual text. Avoid abstract terms that don't literally appear in the Quran (e.g., "العقيدة", "السيرة النبوية", "التوحيد" as a standalone term). Instead, reformulate to concrete textual patterns:

| User says → | Search for → | Why |
| --- | --- | --- |
| **التوحيد** (Monotheism) / **العقيدة** (Creed) | `لا إله إلا الله` or `الإيمان بالله` or `الله أحد` | "Tawhid" is a word not found in the Quran; verbal phrases are present |
| **السيرة النبوية** (Prophetic Biography) | `غزوة بدر`, `فتح مكة`, `الهجرة`, `المهاجرين`, `غار حراء` | "Seerah" is a modern term; use specific events |
| **أسماء الله الحسنى** (Beautiful Names of Allah) | `الرحمن`, `القدير`, etc. | The names themselves appear in the Quran |
| **النبوة / الرسل** (Prophethood / Messengers) | `أرسلنا`, `النبيين`, `المرسلين` | Verbs and plurals are more widespread than the abstract term |
| **الصلاة** (Prayer) | `الصلاة`, `أقيموا الصلاة`, `الزكاة` | The word itself appears, but verbs are better |
| **الإيمان** (Faith) | `آمنوا`, `الذين آمنوا`, `يؤمنون` | Verb forms are more frequent than the verbal noun |
| **الموت / الآخرة** (Death / Hereafter) | `البعث`, `القيامة`, `الموت`, `الحساب` | Diversification yields broader results |

**Best-performing queries from testing:**

- ✅ `لا إله إلا الله` → Brought back Ta-Ha 8, Ali 'Imran 2, Al-Ikhlas 1, An-Naml 26 (very accurate results)
- ✅ `غزوة بدر` → Brought back Ali 'Imran 123 directly
- ✅ `الإيمان بالله` → Brought back Al-Baqarah 8, Al-Buruj 8, Al-An'am 19

**Weaker queries (reformulate these):**

- ❌ `التوحيد` → Only brings back As-Samad (112:2), remaining results are inaccurate
- ❌ `السيرة النبوية` → No accurate results in Hadith
- ❌ `الهجرة` → Failed to bring back At-Tawbah 40 (Cave of Thawr); try `مهاجر` or `المهاجرين` instead

### Distance Metric Awareness

The API returns **cosine distance** values (positive, range `0.0–2.0`), where **lower = more relevant**. Do NOT use `distances` as absolute confidence scores — they depend on the embedding space, not probability. Just use the result ordering.

- ✅ **Positive values (0.0–2.0)**: typical range for cosine distance
- ✅ **Lower is better**: `0.32` is closer than `0.64`
- ❌ **Do not treat as percentages**: these are *distances*, not probabilities
- ❌ **Do not compare across collections**: Quran distances (~0.5) and Hadith distances (~0.3) have different scales

### Metadata Quality

- Most surah names are present, but **some are empty** in the data (e.g., Surah An-Naml 27:30 metadata has an empty `surah` field). When displaying results, always check `meta.get('surah', '')` before showing the surah name — if empty, derive it from `surah_number`.
- Some Quran text entries include the **verse number at the end** in parentheses (e.g., `"...ۖ وَإِلَيْهِ تُرْجَعُونَ (70)"`). Strip trailing verse numbers with regex before display/synthesis if needed.
- Hadith results include: `book`, `hadith_number`, `chapter`, `chapter_number`, `grade`, `sanad`. Always show the grade when presenting a hadith.

---

### Altafsir.com — External Tafsir Source

> **When the local DB doesn't have the tafsir the user wants**, you can fetch it live from altafsir.com (90+ tafsirs available).

Two approaches:

**A. Quick Lookup (single ayah)** — use the curl + extraction method below. Best for answering a single question.

**B. Bulk Scraper (full tafsir)** — use `/home/shk/altafsir-scraper/scraper.py`. Best when the user wants to extract entire tafsirs into JSONL for analysis/training. See `references/altafsir-com.md` → "Bulk Scraper" section for details.

### Lookup: Get a Tafsir from altafsir.com

**Reference:** `references/altafsir-com.md` (full tafsir listing, URL structure, extraction patterns)

**How it works:**

1. The user requests a verse interpretation from a specific scholar (e.g., "Ibn Kathir" or "Al-Tabari").
2. Look up the scholar in the tafsir list in `references/altafsir-com.md` (Full Tafsir List table).
3. Take the `Madh` and `ID` (tTafsirNo) from the table.
4. Construct the URL: `https://www.altafsir.com/Tafasir.asp?tMadhNo={MADH}&tTafsirNo={ID}&tSoraNo={SURAH}&tAyahNo={AYAH}&tDisplay=yes&UserProfile=0&LanguageId=1`
5. Fetch the page using `curl` and decode from `windows-1256`.
6. Extract text from `<div id="DispFrame">` → `<font class='TextResultArabic'>`.
7. Present the answer while mentioning the source.

**Example — Fetching Al-Razi's Tafsir for Al-Fatiha 1:1:**

```bash
curl -sL "https://www.altafsir.com/Tafasir.asp?tMadhNo=1&tTafsirNo=4&tSoraNo=1&tAyahNo=1&tDisplay=yes&UserProfile=0&LanguageId=1" | iconv -f windows-1256 -t utf-8
```

**Important Points:**

- Some commentaries display all verses of a surah on a single page (like Tafsir Ibn Abbas), while others go verse by verse.
- The `windows-1256` encoding must be decoded correctly.
- `tMadhNo` and `tTafsirNo` come from the table in `references/altafsir-com.md`.

### Pitfalls when using altafsir.com

1. **Verify the Encoding** — The site uses windows-1256, not UTF-8. The text will appear garbled if you don't decode it properly.
2. **Script injection in meta tags** — The description has `<script>` tags injected from cdn.jsdelivr.net — ignore them.
3. **Old ASP Site** — It sometimes hangs or slows down. Use a timeout with curl.
4. **Rate Limiting** — Do not make too many rapid requests. Add a 0.5–1 second delay between requests if fetching large amounts of data.
5. **Ensure the Tafsir Exists** — Some commentaries might not cover all verses. If the page comes back empty or lacks commentary text, try another verse or inform the user.

### Hadith Sources (Grades & Bulk Data)

See `references/hadith-sources.md` for:

- **fawazahmed0/hadith-api** — free CDN-hosted hadith data with grades, no API key needed
- **Sunnah.com Official API** — needs API key, has grades in response
- **Collection name mapping** for CSV-to-edition lookup
- **Grade extraction workflow** for batch hadith grade lookups
- **JSON structure** reference for the hadith editions

### Scraping Sunnah.com Live (for data you don't have indexed)

When the local index doesn't have a hadith or you need the full Arabic matn + sanad, scrape sunnah.com live with **Obscura** (installed at `/usr/bin/obscura`). Sunnah.com is behind Cloudflare — Obscura's `--stealth` mode bypasses it.

**Quick lookup:**

```bash
obscura fetch --stealth --dump text --wait 10 "https://sunnah.com/bukhari/1/1"
```

See `references/obscura-sunnah-scraping.md` for full command reference, URL patterns, performance notes, and output processing tips.

---

## API Reference

Server: `http://127.0.0.1:8000`

### `GET /health`

```bash
curl "http://127.0.0.1:8000/health"
# Returns: {"status": "ok", "quran_count": 6236, "tafsir_count": ..., "hadith_count": ...}
```

### `GET /get-ayah`

```bash
curl "http://127.0.0.1:8000/get-ayah?surah=1&ayah=1"
curl "http://127.0.0.1:8000/get-ayah?surah=1&ayah=1-5"
```

Returns `list[dict]` — each dict has `ids`, `documents`, `metadatas`.

### `GET /get-hadith`

```bash
curl "http://127.0.0.1:8000/get-hadith?book=bukhari&hadith_number=1"
```

### `GET /get-tafsir`

```bash
curl "http://127.0.0.1:8000/get-tafsir?book=ibnkathir&surah=1&ayah=1"
curl "http://127.0.0.1:8000/get-tafsir?book=*&surah=1&ayah=1"
```

### `GET /quran-search`

```bash
curl "http://127.0.0.1:8000/quran-search?query=%D8%A7%D9%84%D8%B1%D8%AD%D9%85%D9%86&k=10"
```

### `GET /hadith-search`

```bash
curl "http://127.0.0.1:8000/hadith-search?query=%D8%A7%D9%84%D8%B5%D9%84%D8%A7%D8%A9&k=10&where=%7B%22book%22%3A%22Bukhari%22%7D"
```

### `GET /tafsir-search`

```bash
curl "http://127.0.0.1:8000/tafsir-search?query=%D8%A7%D9%84%D8%B1%D8%AD%D9%85%D9%86&k=10"
```

---

## Common Pitfalls

1. **🔴 Using abstract terms in search** — Terms like "Tawhid", "Aqeedah", "Prophetic Biography" **do not literally exist** in the Quran. Semantic search fails with them because the embedding model is trained on classical Arabic texts. They must be transformed into verbal phrases (see Keyword Reformulation Guide above).
2. **🔴 Trusting distance scores** — The returned `distances` are **cosine distance** values (range `0.0–2.0`, positive), not percentages or probabilities. Do not use them as a confidence metric. Only use the result ordering (smaller = closer). Do not compare values across Quran, hadith, and tafsir — each collection has its own distance distribution.
3. **🔴 Empty surah names in metadata** — Some verses (like 27:30) have an empty `surah` field in the metadata. Always check with `meta.get('surah', '')` before displaying the surah name, and derive it from `surah_number` if it is empty.
4. **🔴 Verse numbers attached to text** — Some Quranic texts include the verse number at the end (e.g., 28:70: "...وَإِلَيْهِ تُرْجَعُونَ (70)"). Use regex to remove them: `re.sub(r'\s*\(\d+\)\s*$', '', text)`.
5. **🔴 Failing to distinguish between Quran and Hadith Qudsi** — Both are the speech of Allah but at different status levels. If a result comes from the Quran and another from Hadith Qudsi, clarify the difference to the user.
6. **🔴 Giving a definitive ruling based on search alone** — Search brings back texts, but jurisprudence (Fiqh) requires deeper understanding. The Agent should say "It is stated in the Sharia that..." "And scholars said...".
7. **🔴 Ignoring Abrogation (Naskh and Mansukh)** — A verse might be abrogated by another verse. The Agent does not know this from the index alone. If the question is about a ruling, look into Tafsir, answer, and then say "It is best to ask a scholar regarding this."
8. **🔴 Searching with diacritics (Tashkeel)** — The user might write "التَّوَكُّل", and diacritics prevent exact matching. The search scripts normalize text, but if the Agent manually writes a search query, it should write without diacritics.
9. **🔴 Ignoring Context** — A verse taken out of context might be misunderstood. Always read the verses before and after it before responding.

- If you bring a verse from, for example: Al-Fatiha 1 (1:5), go to `/get-ayah?surah=1&ayah=4` to grab the context.

1. **🔴 Failing to mention the source** — Every answer must mention: the surah name and verse number (for the Quran) or the collection name, hadith number, and narrator (for the Sunnah).
2. **🔴 Inaccurate translation** — If you need to translate the meaning of a verse, say "In the meaning of the verse..." and do not say "Allah said..." followed by a translation.
3. **🔴 Outdated Index** — If new sources are added and the index is old, results will be incomplete. Run `setup.sh` again.

---

## Verification Checklist

- [ ] Data is present (`data/quran/quran-simple.txt` and `data/hadith/`).
- [ ] The API returns results before synthesis.
- [ ] Results contain: Source (surah/collection), verse/hadith number, text.
- [ ] The final answer clearly mentions sources.
- [ ] The difference between Quran and Hadith is clarified in the answer.
- [ ] Arabic texts are displayed correctly.
- [ ] If the search did not return sufficient results, different keywords were tried.
- [ ] Server is running (`curl http://127.0.0.1:8000/health`).
