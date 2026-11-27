"""Citation tag refill. Pure string/rapidfuzz work with a fake lookup — no Qdrant."""

from retrieval.citations import CitationRenderer, CitationStream, plain_text

VERSE = (
    "اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ ۚ "
    "لَا تَأْخُذُهُ سِنَةٌ وَلَا نَوْمٌ"
)

PAYLOADS = {
    ("quran", "2:255"): {
        "ids": "2:255",
        "text": VERSE,
        "surah": "البقرة",
        "ayah_number": 255,
    },
    ("hadith", "bukhari:1.0"): {
        "ids": "bukhari:1.0",
        "text": "إِنَّمَا الأَعْمَالُ بِالنِّيَّاتِ",
        "book": "bukhari",
        "grade": "صحيح",
    },
    ("books", "0:12:3"): {
        "ids": "0:12:3",
        "text": "باب في الزهد في الدنيا",
        "book_name": "الزهد لأحمد بن حنبل",
    },
}


def lookup(collection: str, ids: str):
    return PAYLOADS.get((collection, ids), [])


def renderer() -> CitationRenderer:
    return CitationRenderer(lookup=lookup)


def test_exact_anchor_returns_the_source_text_with_its_diacritics():
    text, cards = renderer().render('<quran id="2:255">لا تأخذه سنة</quran>')

    assert cards[0]["match"] == "exact"
    assert cards[0]["text"] == "لَا تَأْخُذُهُ سِنَة"
    assert cards[0]["surah"] == "البقرة"
    assert cards[0]["ayah_number"] == 255
    # the model's diacritic-free anchor is replaced by the canonical span
    assert "لَا تَأْخُذُهُ سِنَة" in text
    assert 'surah="البقرة"' in text


def test_typo_anchor_still_lands_on_the_right_span():
    _, cards = renderer().render(
        '<quran id="2:255">لا تأخذه سنه ولا نوم</quran>'
    )

    assert cards[0]["match"] == "fuzzy"
    assert cards[0]["text"] == "لَا تَأْخُذُهُ سِنَةٌ وَلَا نَوْم"


def test_absent_anchor_falls_back_to_the_whole_chunk():
    _, cards = renderer().render('<quran id="2:255">ليس في الآية</quran>')

    assert cards[0]["match"] == "fallback"
    assert cards[0]["text"] == VERSE


def test_unknown_id_drops_the_tag_and_keeps_the_words():
    text, cards = renderer().render('<quran id="9:9">كلام بلا مصدر</quran>')

    assert cards == []
    assert text == "كلام بلا مصدر"
    assert "<" not in text


def test_hadith_and_books_fill_their_own_labels():
    _, hadith = renderer().render('<hadith id="bukhari:1.0">الأعمال بالنيات</hadith>')
    assert hadith[0]["book"] == "bukhari"
    assert hadith[0]["grade"] == "صحيح"
    # the anchor is a fragment of the matn, so the span is that fragment
    assert hadith[0]["text"] == "الأَعْمَالُ بِالنِّيَّات"

    _, book = renderer().render('<book id="0:12:3">الزهد</book>')
    assert book[0]["collection"] == "books"
    assert book[0]["book_name"] == "الزهد لأحمد بن حنبل"


def test_stream_reassembles_a_tag_split_across_deltas():
    stream = CitationStream(renderer())
    pieces = ['الجواب: <qur', 'an id="2:255">لا تأ', "خذه سنة</qur", "an> والله أعلم."]

    text, cards = "", []
    # No chunk may carry half a tag: everything before the closing `>` waits.
    for piece in pieces[:3]:
        chunk, new_cards = stream.feed(piece)
        assert "<" not in chunk, chunk
        text += chunk
        cards += new_cards

    chunk, new_cards = stream.feed(pieces[3])
    text += chunk
    cards += new_cards

    assert text.startswith("الجواب: ")
    assert text.endswith(" والله أعلم.")
    assert "لَا تَأْخُذُهُ سِنَة" in text
    assert len(cards) == 1


def test_stream_holds_text_until_the_tag_closes():
    stream = CitationStream(renderer())

    first, cards = stream.feed('<quran id="2:255">لا تأخذه سنة')
    assert first == "" and cards == []

    tail = stream.flush()
    assert tail.startswith("<quran id=")  # unterminated: model output, verbatim


def test_plain_projection_has_no_markup():
    text, _ = renderer().render('<quran id="2:255">لا تأخذه سنة</quran>')
    plain = plain_text(text)

    assert "<" not in plain and ">" not in plain
    assert "«لَا تَأْخُذُهُ سِنَة»" in plain
    assert "[البقرة: 255]" in plain


def test_rendering_is_idempotent():
    once, _ = renderer().render('<quran id="2:255">لا تأخذه سنة</quran>')
    twice, cards = renderer().render(once)

    assert twice == once
    assert len(cards) == 1


def test_text_outside_tags_is_untouched():
    original = "قال تعالى كذا وكذا. ثم قال: نهاية."
    text, cards = renderer().render(original)

    assert text == original and cards == []
