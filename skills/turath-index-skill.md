
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

---

## Hadith Collection Names

| Tool name | Full name |
| --- | --- |
| `bukhari` | Sahih al Bukhari |
| `muslim` | Sahih Muslim |
| `abudaud` | Sunan Abu Dawud |
| `tirmidhi` | Jami At Tirmidhi |
| `ibnmajah` | Sunan Ibn Majah |
| `nasai` | Sunan an Nasai |
| `malik` | Muwatta Malik |
| `nawawi` | 40 Hadith Nawawi |
| `qudsi` | Hadith Qudsi |

Bukhari hadiths have no grade stored — they are all authentic.

# Aqeedah collection books

الرقم هو book_id

## ابن تيمية (39 كتابًا)

- **170** — الجواب الصحيح لمن بدل دين المسيح
- **264** — تحقيق الإيمان
- **301** — تحقيق الاحتجاج بالقدر
- **792** — القصيدة التائية في القدر
- **892** — رأس الحسين
- **7631** — الرسالة الأكملية في ما يجب لله من صفات الكمال
- **7635** — الرسالة المدنية في تحقيق المجاز والحقيقة في صفات الله
- **8983** — الواسطة بين الحق والخلق
- **9277** — الإخنائية أو الرد على الإخنائي (ت العنزي)
- **10784** — الرسالة العرشية
- **11071** — بغية المرتاد في الرد على المتفلسفة والقرامطة والباطنية
- **11162** — حقوق آل البيت
- **11216** — رسالة في أصول الدين
- **11248** — شرح العقيدة الأصفهانية
- **11258** — شرح حديث النزول
- **11285** — قاعدة جامعة في توحيد الله وإخلاص الوجه والعمل له عبادة واستعانة
- **11620** — اقتضاء الصراط المستقيم لمخالفة أصحاب الجحيم
- **11817** — النبوات
- **12769** — قاعدة عظيمة في الفرق بين عبادات أهل الإسلام والإيمان وعبادات أهل الشرك والنفاق
- **18098** — العقيدة الواسطية (ت ابن مانع)
- **18381** — الفتوى الحموية الكبرى
- **20587** — الإيمان الأوسط (ط ابن الجوزي)
- **21499** — الفرقان بين أولياء الرحمن وأولياء الشيطان
- **21506** — درء تعارض العقل والنقل
- **21512** — مختصر منهاج السنة
- **21565** — جواب في الحلف بغير الله والصلاة إلى القبور، ويليه: فصل في الاستغاثة
- **22647** — العبودية
- **22649** — قاعدة جليلة في التوسل والوسيلة
- **22651** — مسألة في الكنائس
- **22665** — العقيدة الواسطية (ت أشرف عبد المقصود)
- **22666** — التدمرية
- **22667** — تحقيق القول في مسألة: عيسى كلمة الله والقرآن كلام الله
- **22668** — مسألة فيما إذا كان في العبد محبة لما هو خير وحق ومحمود في نفسه
- **22874** — الصفدية
- **37816** — الإخنائية أو الرد على الإخنائي (ت زهوي)
- **147670** — مسألة في توحيد الفلاسفة

## أحمد بن حنبل (3 كتب)

- **6418** — أصول السنة
- **7623** — الرد على الجهمية والزنادقة
- **20879** — الجامع لعلوم الإمام أحمد - العقيدة

## عثمان بن سعيد الدارمي (كتابان)

- **18089** — الرد على الجهمية
- **18091** — نقض الدارمي على المريسي

## عبد الله بن أحمد (كتاب واحد)

- **323** — السنة

## حرب الكرماني (كتاب واحد)

- **121** — إجماع السلف في الاعتقاد كما حكاه حرب الكرماني

## أبو محمد البربهاري (كتاب واحد)

- **8601** — شرح السنة

## محمد بن عبد الوهاب (كتاب واحد)

- **11318** — التوحيد

## ابن القيم (كتاب واحد)

- **11375** — نونية ابن القيم الكافية الشافية
المجموع: 51 كتابًا موزعة على 8 مؤلفين.

---

## Score Interpretation (from hybrid search)

- **Score > 0.7**: Highly relevant
- **Score 0.4–0.7**: Good match, worth citing
- **Score 0.3–0.4**: Possibly relevant
- **Score < 0.3**: Filtered out by default threshold

---

## Response

## Ask your self

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
