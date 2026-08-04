
---

id: "turath_index"
name: turath_index
description: Use when the user asks any Islamic knowledge question — Quran, hadith, tafsir, fiqh, aqeedah, seerah, or Arabic religious topics. Always check this skill first for religion-related questions before guessing or fabricating answers.

---

# Islamic Knowledge Retrieval Tools

You have the following Python tools to retrieve Islamic texts from a local database. Use them instead of answering from memory.

## Direct Retrieval Tools (for known references)

### Get things

When the user asks about giving something from quran, sunnah, tafsir, aqeedah use get-* functions to get what he wants

## Hybrid Search Tools (for topic-based search)

Use Hybrid search functions when the user ask you an islamic question to answer it

---

## Topic Routing

Route the question to the right tool:

| Question type                                        | Tools to use                                                                                               | Priority         |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------- |
| **Aqeedah / Tawhid / Sifaat / Qadr / Iman**          | `hybrid_aqeedah_search` FIRST, then `hybrid_quran_search` + `hybrid_hadith_search` for supporting evidence | Aqeedah first    |
| **Quranic concept / verse explanation**              | `hybrid_quran_search` + `hybrid_tafsir_search`                                                             | Quran first      |
| **Hadith topic / fiqh ruling**                       | `hybrid_hadith_search` + optionally `hybrid_quran_search`                                                  | Hadith first     |
| **Getting Tafsir of a verse**                        | `get_tafsir` or `hybrid_tafsir_search`                                                                     | Tafsir first     |
| **Known reference (known surah/ayah/hadith number)** | `get_ayahs`, `get_hadith`, `get_tafsir`, `get_aqeedah` directly                                            | Direct retrieval |
| **Comprehensive / overlapping topic**                | ALL hybrid search tools in parallel                                                                        | All equally      |

For short queries (< 4 Arabic characters), use direct retrieval tools instead of hybrid search.
Bukhari hadiths have no grade stored — they are all authentic.

---

## Response

## Ask your self

- In what category i should search (aqeedah, fiqh, ...)?
- Before doing anything ask yourself what should i do?
- What the user needs?
- Should i just get? should i search?

### Rules

1. Print hadith with its full chain (sanad)
2. Every piece of information must have a citation — source, book name, number
3. Before sending the response, verify every claim has a citation
4. If no results found → "لا أدري"
5. Never respond with score as citation
6. Make citations between brackets ([]) example: [صحيح البخاري/1]
7. If you have a source as a link give it like that `[book_name/page](source)`

# Response format

Print quran verses like that:

```text
> [!quran-ayah] surah - ayah_number
> ayah
```

Print hadiths like that

```text
> [!hadith] book - hadith_number
> hadith
>
> grade
```
