"""Citation record, page locators and the normalization keys for deduplication."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from autoexcerpter.rendering.locators import (
    Locator,
    as_locator,
    format_page_range,
    roman_to_int,
)

# Curly quotes/apostrophes and the several dash characters folded to ASCII so
# that "Müller's" / "Muller's" and en/em dashes do not defeat deduplication.
# Also transliterates the Latin letters NFKD does not decompose, so "Møller"
# and "Moller" share their tokens.
_CURLY_TRANSLATION = {
    ord("‘"): "'",
    ord("’"): "'",
    ord("“"): '"',
    ord("”"): '"',
    ord("–"): "-",
    ord("—"): "-",
    ord("−"): "-",
    ord("­"): "",  # soft hyphen
    ord("ø"): "o",
    ord("Ø"): "O",
    ord("æ"): "ae",
    ord("Æ"): "Ae",
    ord("œ"): "oe",
    ord("Œ"): "Oe",
    ord("ł"): "l",
    ord("Ł"): "L",
    ord("ð"): "d",
    ord("Ð"): "D",
    ord("þ"): "th",
    ord("Þ"): "Th",
}

_YEAR_CORE = r"(1[0-9]{3}|20[0-9]{2})"
_YEAR_RE = re.compile(r"\b" + _YEAR_CORE + r"\b")
# Reprint form: a plain/parenthesized year adjoining a bracketed year, in either
# order ("1976 [1867]" or "[1867] 1976"). Both years are captured so the
# original (earliest) can win over the reprint.
_REPRINT_RE = re.compile(
    _YEAR_CORE + r"\s*\[\s*" + _YEAR_CORE + r"\s*\]"
    r"|\[\s*" + _YEAR_CORE + r"\s*\]\s*" + _YEAR_CORE
)
# Plausible publication-year span (1500-2099) used by the initial-vs-year guard
# below. Narrower than the year pattern itself, which also admits 1000-1499.
PUB_YEAR = r"(?:1[5-9][0-9]{2}|20[0-9]{2})"
# Page-marker spans (English pp./p., German S./SS., folio fol.) stripped before
# year scanning so a page range such as "S. 1066-1071" is never read as a year.
# For the single-letter markers p./s. only, both alternatives are guarded so an
# author initial adjoining a year is NOT swallowed as a page marker:
#   - a lone plausible year ("Sen, S. 1981") is left intact;
#   - a range whose first endpoint is a plausible publication year and whose
#     second endpoint is either a full year or a two-digit abbreviation
#     ("Sraffa, P. 1951-1973" and "Sraffa, P. 1951-73") is left intact.
# Non-year page numbers and ordinary page ranges ("S. 42", "S. 1066-1071",
# whose endpoints fall below 1500) still strip. Multi-letter markers
# (pp./ss./fol.) are never author initials, so they strip any page number,
# including four-digit ones ("pp. 1850", "fol. 1200"). The range alternative
# comes first so "S. 1066-1071" is consumed whole. Trade-off: a genuine
# single-page German cite of a four-digit page ("S. 1815") is read as a year,
# and a page range led by a year-like page ("S. 1518-23") is left intact;
# rare, accepted false positives versus the initial-vs-year collapse.
_PAGE_MARKER_RE = re.compile(
    r"\b(?:(?:pp|ss|fol)\.\s*(?:\d+\s*-\s*\d+|\d+)"
    r"|[ps]\.\s*(?:(?!" + PUB_YEAR + r"\s*-\s*(?:" + PUB_YEAR + r"|\d{2})\b)"
    r"\d+\s*-\s*\d+"
    r"|(?!(?:1[0-9]{3}|20[0-9]{2})\b)\d+))",
    re.IGNORECASE,
)
# Roman numeral in canonical (subtractive) form. A bare character class of
# {i,v,x,l,c,d,m} would accept any letter salad (German "Bd. im" or French
# "t. dix" then parse as volumes 999 and 509 and block legitimate merges), so
# the branch is anchored to real Roman grammar. The leading lookahead forces a
# non-empty match (every quantified group is optional on its own).
_ROMAN_NUMERAL = (
    r"(?=[mdclxvi])m{0,3}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})"
)
# Canonical grammar alone still admits word-shaped numerals such as "dix"
# (= DIX = 509); a volume designator that large is not a real citation, so
# Roman-derived volumes above this bound are rejected as false positives.
MAX_ROMAN_VOLUME = 200
# Volume designators across English/German/French scholarship. The number may
# be Arabic or a Roman numeral. The period is optional for the spelled-out
# designators but required for the bare "t." so ordinary words never match;
# the bare "t." number is further capped at three digits and the "t" is matched
# case-sensitively as lowercase (via the scoped "(?-i:t)" flag) so an author
# initial ("Smith, T. 190. Title" or "Smith, T. 1990. Title") is never read as a
# tome/volume, while the lowercase French "t. II" abbreviation still parses.
_VOLUME_RE = re.compile(
    r"\b(?:vol|volume|bd|band|teil|tome)\.?\s*(\d+|" + _ROMAN_NUMERAL + r")\b"
    r"|\b(?-i:t)\.\s*(\d{1,3}|" + _ROMAN_NUMERAL + r")\b",
    re.IGNORECASE,
)
_UNDATED_RE = re.compile(r"\(\s*n\.\s?d\.\s*\)", re.IGNORECASE)

_TITLE_SPAN_RE = re.compile(r'\*([^*\n]{3,})\*|"([^"\n]{3,})"')
_STOPLIST_RE = re.compile(
    r"\b(?:london|cambridge|oxford|new york|berkeley|chicago"
    r"|press|university|publishers?)\b"
)

# Non-name tokens that must never be taken as a surname.
_SURNAME_STOPWORDS = {"the", "and", "of", "in", "on", "ed", "eds", "trans"}
# Lowercase nobiliary/name particles that precede the actual surname. When a
# name leads with one ("van der Berg"), the first non-particle token is the
# surname, so "van der Berg" and "Berg, J. van der" fold to the same block.
_NAME_PARTICLES = {
    "van",
    "von",
    "der",
    "de",
    "den",
    "du",
    "la",
    "le",
    "ter",
    "ten",
    "da",
    "di",
    "dos",
    "del",
}
# Unicode-aware word token: a letter (any script, incl. Latin Extended) followed
# by at least one more letter/apostrophe/hyphen. The two-character minimum skips
# single-letter initials ("J.").
_NAME_TOKEN_RE = re.compile(r"[^\W\d_](?:[^\W\d_]|['’-])+", re.UNICODE)


def fold(text: str) -> str:
    """Fold text for robust comparison.

    NFKD-normalizes, strips combining marks (so ``Müller`` == ``Muller`` and
    ``Génin`` == ``Genin``), casefolds, unifies curly quotes and dashes to
    ASCII, and rewrites ``&`` as ``and``.
    """
    if not text:
        return ""
    text = text.translate(_CURLY_TRANSLATION)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = text.replace("&", " and ")
    return text


def extract_year(text: str) -> int | None:
    """Return the first 4-digit publication year in *text*, or None.

    URLs and DOIs are stripped first: common DOI registrant prefixes such as
    ``10.1016/`` or ``10.1111/`` otherwise match the ``1[0-9]{3}`` year pattern
    and yield a bogus year that then mis-blocks deduplication.
    """
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"doi:\s*10\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b10\.\d{4,}/\S+", " ", text)
    # A reprint ("1976 [1867]") would split one work across two blocks, so it
    # canonicalizes on the earliest year.
    reprint = _REPRINT_RE.search(text)
    if reprint:
        years = [int(g) for g in reprint.groups() if g]
        if years:
            return min(years)
    # Drop page-marker spans so their numbers cannot be misread as a year.
    text = _PAGE_MARKER_RE.sub(" ", text)
    match = _YEAR_RE.search(text)
    return int(match.group(1)) if match else None


def extract_volume(text: str) -> int | None:
    """Return the volume number in *text*, or None.

    Reads English, German and French designators (``Vol. 3``, ``Bd. 2``,
    ``t. II``) with Arabic or Roman numbers.
    """
    # Scan every match: an implausible first hit is discarded, not treated as
    # the answer, so a real designator later in the string is still found.
    for match in _VOLUME_RE.finditer(text):
        captured = match.group(1) or match.group(2)
        if captured is None:
            continue
        if captured.isdigit():
            arabic = int(captured)
            if 1500 <= arabic <= 2099:
                # A "volume" in the plausible publication-year window is almost
                # always a year following an English word that doubles as a
                # German designator ("The Band 1968 Story"); a spurious volume
                # would poison the dedup key and veto legitimate merges.
                continue
            return arabic
        value = roman_to_int(captured)
        if value is None or value > MAX_ROMAN_VOLUME:
            # An implausibly large Roman "volume" is an ordinary word that
            # happens to be well-formed Roman ("dix"), not a designator.
            continue
        return value
    return None


def cited_years(text: str) -> list[int]:
    """Return every year the citation gives for the work, earliest first.

    A reprint ("1976 [1867]") yields both years; otherwise the single year of
    :func:`extract_year`, or none. An undated citation ("(n.d.)") yields none,
    so a year inside its title ("French cookbooks, 1480-1800") is not taken
    for the publication year.
    """
    if _UNDATED_RE.search(text):
        return []
    reprint = _REPRINT_RE.search(text)
    if reprint:
        return sorted({int(g) for g in reprint.groups() if g})
    year = extract_year(text)
    return [] if year is None else [year]


def title_spans(text: str) -> list[str]:
    """Return italic (``*...*``) or quoted (``"..."``) title spans in *text*."""
    spans: list[str] = []
    for ital, quoted in _TITLE_SPAN_RE.findall(text):
        span = ital or quoted
        if span:
            spans.append(span)
    return spans


def markup_title(text: str) -> str:
    """Return the first italicized or quoted title span in citation order.

    Order matters: in a Chicago-style article citation the quoted article
    title precedes the italicized journal, and in a book citation the
    italicized title comes first. Spans shorter than five characters are
    ignored; returns "" when none is left.
    """
    pairs: list[tuple[str, str]] = _TITLE_SPAN_RE.findall(text)
    for ital, quoted in pairs:
        span = ital or quoted
        if len(span) >= 5:
            return span
    return ""


def first_author_surname(text: str) -> str:
    """Return the folded first-author surname (best-effort), or ``""``.

    Takes the first alphabetic token before the first year or opening
    parenthesis, skipping lowercase name particles so particle surnames fold
    consistently regardless of ordering.
    """
    # An in-text partial like "(Smith, 1990, S. 12)" leads with "(", which would
    # otherwise split to an empty head; drop leading brackets/whitespace first.
    stripped = text.lstrip("([ \t\r\n")
    head = re.split(r"\(|\d{4}", stripped, maxsplit=1)[0]
    tokens = [
        folded
        for tok in _NAME_TOKEN_RE.findall(head)
        if (folded := fold(tok)) not in _SURNAME_STOPWORDS
    ]
    if not tokens:
        return ""
    for tok in tokens:
        if tok not in _NAME_PARTICLES:
            return tok
    return tokens[0]


def token_set(text: str) -> set[str]:
    """Return the set of word tokens (length > 1) in folded *text*."""
    return {t for t in re.findall(r"[a-z0-9]+", text) if len(t) > 1}


def jaccard(a: str, b: str) -> float:
    """Token-set Jaccard similarity of two comparison strings."""
    sa, sb = token_set(a), token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


@dataclass
class Citation:
    """Represents a single citation with metadata and page tracking.

    The deduplication key is *structured*: it folds the text (accents, case,
    quotes, ``&``), keeps the year and volume as discriminators (so different
    editions never collapse), and applies the publisher/city stop-list only
    outside the title span. ``consolidate()`` on the manager performs a later,
    conservative fuzzy merge within (author, year) blocks.
    """

    raw_text: str
    locators: set[Locator] = field(default_factory=set)
    normalized_key: str = ""
    metadata: dict[str, Any] | None = None
    doi: str | None = None
    url: str | None = None
    partial: bool = False
    # Raw texts absorbed by merges; never rendered, kept as a debugger-visible
    # trace of what each surviving citation swallowed.
    variants: list[str] = field(default_factory=list)
    # Structured discriminators and comparison material (set in __post_init__).
    year: int | None = None
    volume: int | None = None
    author: str = ""
    comparison_text: str = ""

    def __post_init__(self) -> None:
        """Derive the discriminators and the normalized key."""
        self.year = extract_year(self.raw_text)
        self.volume = extract_volume(self.raw_text)
        self.author = first_author_surname(self.raw_text)
        if not self.normalized_key:
            self.normalized_key = self._generate_normalized_key()

    def refresh(self) -> None:
        """Re-derive year, volume, author and comparison text from ``raw_text``.

        The normalized key stays: it is the dictionary key of the citation.
        """
        self.year = extract_year(self.raw_text)
        self.volume = extract_volume(self.raw_text)
        self.author = first_author_surname(self.raw_text)
        self._generate_normalized_key()

    def _generate_normalized_key(self) -> str:
        """Create a structured normalized key for citation deduplication."""
        text = fold(self.raw_text.strip())

        # URLs and DOIs do not enter the key.
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"doi:\s*10\.\S+", " ", text)
        text = re.sub(r"\b10\.\d{4,}/\S+", " ", text)

        # Page markers as in _PAGE_MARKER_RE, optionally parenthesized: an
        # author initial before a year or year range ("Sen, S. 1981",
        # "Sraffa, P. 1951-73") stays, so distinct editions keep distinct
        # keys; \b keeps "words." from matching the bare "s.".
        text = re.sub(
            r"\(?\s*\b(?:(?:pp|ss|fol)\.\s*(?:\d+\s*-\s*\d+|\d+)"
            r"|[ps]\.\s*(?:(?!" + PUB_YEAR + r"\s*-\s*(?:" + PUB_YEAR + r"|\d{2})\b)"
            r"\d+\s*-\s*\d+"
            r"|(?!(?:1[0-9]{3}|20[0-9]{2})\b)\d+))"
            r"\s*\)?",
            " ",
            text,
        )

        # Protect the folded title span so the stop-list only strips the
        # non-title parts (keeps "the cambridge world history of food" intact).
        placeholders: dict[str, str] = {}
        for i, span in enumerate(title_spans(self.raw_text)):
            folded_span = fold(span)
            token = f" __title{i}__ "
            if folded_span and folded_span in text:
                text = text.replace(folded_span, token)
                placeholders[token.strip()] = folded_span

        text = _STOPLIST_RE.sub(" ", text)

        for token, folded_span in placeholders.items():
            text = text.replace(token, f" {folded_span} ")

        # Editor / translator markers and bracketed clarifications. The marker
        # must be a standalone token: the (?<![\w-]) / (?![\w-]) guards stop it
        # from firing inside a word or a hyphenated compound, so "Education"
        # keeps its "ed" and "Trans-Atlantic" keeps its "trans" (a bare \b would
        # still strip the latter, since a hyphen is a word boundary).
        text = re.sub(r"\(?\s*(?<![\w-])(?:eds?|trans)(?![\w-])\.?\s*\)?", " ", text)
        text = re.sub(r"\[[^\]]*\]", " ", text)

        text = re.sub(r"[,.:;()\[\]\"'\-*]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        self.comparison_text = text

        # A citation made up entirely of stripped material (a bare URL, say)
        # leaves no comparison text at all; keying on "" would collapse every
        # such citation onto one hash. Fall back to the folded raw text, which
        # still distinguishes two different URLs.
        key_source = text or fold(self.raw_text.strip())

        # Keep year and volume in the key material so different editions never
        # collapse onto the same hash.
        key_material = f"{key_source}|y={self.year}|v={self.volume}"
        return hashlib.md5(key_material.encode("utf-8")).hexdigest()

    def add_page(self, locator: Locator | int | None) -> None:
        """Record where the citation appears.

        A bare int is a printed arabic page number read from the page; None
        records nothing.
        """
        normalized = as_locator(locator)
        if normalized is not None:
            self.locators.add(normalized)

    def get_page_range_str(self) -> str:
        """Return a formatted string of page numbers/ranges."""
        return format_page_range(self.locators)


__all__ = [
    "MAX_ROMAN_VOLUME",
    "PUB_YEAR",
    "Citation",
    "cited_years",
    "extract_volume",
    "extract_year",
    "first_author_surname",
    "fold",
    "jaccard",
    "markup_title",
    "title_spans",
    "token_set",
]
