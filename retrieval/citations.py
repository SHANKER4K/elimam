"""Source citation tags: `<quran id="2:255">لا تأخذه سنة</quran>`.

The model marks where a chunk goes and writes the one or two words it wants
shown. It never transcribes scripture: the inner text is an **anchor**, and this
module replaces it with the canonical span from the retrieved chunk — matched
with rapidfuzz, so a partial or diacritic-free anchor still lands on the right
words, with the source's own diacritics.

Resolution only accepts ids that exist in the collection, so a fabricated id
cannot produce a card; the tag is dropped and the anchor stays as prose.

The same pass exists twice on purpose:

- `CitationStream` rewrites text as it streams, holding back a partial tag;
- `render` rewrites a finished text (used when persisting the message, so
  citation cards survive a reload without persisting tool results).
"""

from __future__ import annotations

import html
import re
from typing import Any, Callable

from pyarabic.araby import ALEFAT_PATTERN, DIACRITICS
from rapidfuzz import fuzz

from .retrieve import retrieval

# Tag name -> collection. `<book>` and `<athar>` are the human names of the
# books/sunnah collections; both spellings are accepted.
TAG_COLLECTIONS = {
    "quran": "quran",
    "tafsir": "tafsir",
    "hadith": "hadith",
    "book": "books",
    "books": "books",
    "athar": "sunnah",
    "sunnah": "sunnah",
}

# Payload fields copied onto the rendered tag so a UI can label the card.
LABEL_FIELDS = (
    "surah",
    "surah_number",
    "ayah_number",
    "tafsir_book",
    "book",
    "grade",
    "book_name",
    "category_name",
)

# Below this the anchor is not in the chunk; emit the whole chunk instead of a
# wrong span. Calibrated on 2:255: exact 100, one-word typo 95, absent < 60.
MATCH_CUTOFF = 80

_TAG_RE = re.compile(
    r"<(?P<name>quran|tafsir|hadith|book|books|athar|sunnah)\b"
    r"(?P<attrs>[^>]*?)(?:/>|>(?P<inner>.*?)</(?P=name)>)",
    re.S,
)
_ATTR_RE = re.compile(r'(\w+)="([^"]*)"')

_DROP = set(DIACRITICS) | {"\u200f"}

Lookup = Callable[[str, str], Any]


def _normalize(text: str) -> tuple[str, list[int]]:
    """Preprocessed text plus a map from normalized to original offsets.

    pyarabic deletes characters, so indices are not comparable across the two
    strings; the map is what lets us slice the original chunk and keep its
    diacritics.
    """
    chars: list[str] = []
    index: list[int] = []

    for position, char in enumerate(text):
        if char in _DROP:
            continue
        chars.append(ALEFAT_PATTERN.sub("ا", char))
        index.append(position)

    return "".join(chars), index


def match_span(chunk: str, anchor: str) -> tuple[str, str]:
    """Locate `anchor` inside `chunk`; returns (canonical text, match kind)."""
    anchor = anchor.strip()
    if not anchor:
        return chunk, "fallback"

    chunk_norm, index = _normalize(chunk)
    anchor_norm, _ = _normalize(anchor)

    alignment = fuzz.partial_ratio_alignment(
        anchor_norm, chunk_norm, score_cutoff=MATCH_CUTOFF
    )
    if alignment is None:
        return chunk, "fallback"

    start = index[alignment.dest_start]
    end = index[alignment.dest_end - 1] + 1

    return chunk[start:end].strip(), ("exact" if alignment.score == 100.0 else "fuzzy")


class CitationRenderer:
    def __init__(self, lookup: Lookup | None = None) -> None:
        self.lookup = lookup or retrieval.by_id

    def resolve(
        self, name: str, attrs: dict[str, str], anchor: str
    ) -> tuple[str, dict] | None:
        """Turn one tag into (rendered tag, card); None when the id is unknown."""
        collection = TAG_COLLECTIONS[name]
        ids = attrs.get("id")

        if not ids:
            return None

        payload = self.lookup(collection, ids)
        if not isinstance(payload, dict) or not payload:
            return None

        chunk = payload.get("text") or ""
        text, match = match_span(chunk, anchor)

        labels = {key: payload[key] for key in LABEL_FIELDS if key in payload}
        rendered = _render_tag(name, {"id": ids, "match": match, **labels}, text)

        card = {
            "collection": collection,
            "ids": ids,
            "match": match,
            "text": text,
            "tag": rendered,
            **labels,
        }
        return rendered, card

    def render(self, text: str) -> tuple[str, list[dict]]:
        """Rewrite every complete tag in a finished text."""
        cards: list[dict] = []

        def replace(match: re.Match) -> str:
            attrs = dict(_ATTR_RE.findall(match.group("attrs")))
            resolved = self.resolve(
                match.group("name"), attrs, match.group("inner") or ""
            )
            if resolved is None:
                return match.group("inner") or ""
            rendered, card = resolved
            cards.append(card)
            return rendered

        return _TAG_RE.sub(replace, text), cards

    def plain(self, text: str) -> str:
        """Strip the tags, keeping readable text and a bracketed reference."""
        return plain_text(text)


def _attrs_to_string(attrs: dict) -> str:
    return " ".join(
        f'{key}="{html.escape(str(value), quote=True)}"'
        for key, value in attrs.items()
    )


def _render_tag(name: str, attrs: dict, text: str) -> str:
    return (
        f"<{name} {_attrs_to_string(attrs)}>"
        f"{html.escape(text, quote=False)}</{name}>"
    )


def _reference(name: str, attrs: dict[str, str]) -> str:
    collection = TAG_COLLECTIONS[name]
    ids = attrs.get("id", "")

    if collection == "quran" and "surah" in attrs:
        ayah = attrs.get("ayah_number", "")
        return f"{attrs['surah']}: {ayah}".strip(": ")
    if attrs.get("book_name"):
        return f"{attrs['book_name']} ({ids})"
    if attrs.get("book"):
        # hadith ids look like "bukhari:1.0"; drop the float noise.
        number = ids.split(":")[-1].removesuffix(".0")
        return f"{attrs['book']}: {number}"
    return f"{collection}: {ids}"


def plain_text(text: str) -> str:
    def replace(match: re.Match) -> str:
        attrs = dict(_ATTR_RE.findall(match.group("attrs")))
        inner = html.unescape(match.group("inner") or "")
        return f"«{inner}» [{_reference(match.group('name'), attrs)}]"

    return _TAG_RE.sub(replace, text)


class CitationStream:
    """Rewrites a text stream tag by tag, holding back a partial tag.

    `feed` returns the text that is safe to send plus any cards resolved since
    the last call. Text outside tags passes through unchanged; only the tail
    starting at an unmatched `<` waits for its `>`.
    """

    def __init__(self, renderer: CitationRenderer | None = None) -> None:
        self.renderer = renderer or CitationRenderer()
        self.buffer = ""
        self.text = ""

    def feed(self, delta: str) -> tuple[str, list[dict]]:
        self.buffer += delta
        out: list[str] = []
        cards: list[dict] = []

        while True:
            match = _TAG_RE.search(self.buffer)

            if match is None:
                # Hold from the earliest `<`: complete tags were already consumed
                # above, so anything left is an unterminated tag. (`rfind` here
                # would hold only from `</...`, leaking the opening half.)
                cut = self.buffer.find("<")
                if cut == -1:
                    out.append(self.buffer)
                    self.buffer = ""
                else:
                    out.append(self.buffer[:cut])
                    self.buffer = self.buffer[cut:]
                break

            out.append(self.buffer[: match.start()])
            attrs = dict(_ATTR_RE.findall(match.group("attrs")))
            resolved = self.renderer.resolve(
                match.group("name"), attrs, match.group("inner") or ""
            )

            if resolved is None:
                # Unknown id: keep the words, drop the claim.
                out.append(match.group("inner") or "")
            else:
                rendered, card = resolved
                out.append(rendered)
                cards.append(card)

            self.buffer = self.buffer[match.end() :]

        chunk = "".join(out)
        self.text += chunk
        return chunk, cards

    def flush(self) -> str:
        """Emit whatever is left (an unterminated tag is model output, keep it)."""
        chunk, self.buffer = self.buffer, ""
        self.text += chunk
        return chunk


renderer = CitationRenderer()


def render_citations(text: str) -> str:
    return renderer.render(text)[0]


def render_plain(text: str) -> str:
    return renderer.plain(text)
