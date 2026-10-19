"""OpenAlex queries from citation text, and verification of the candidates.

Pure functions without I/O: DOI and title extraction, search terms, the
publication-year filter, the review guard, the strict match check, and the
metadata kept from an OpenAlex work.
"""

from __future__ import annotations

import re
from typing import Any

from autoexcerpter.rendering.citations.model import (
    PUB_YEAR,
    cited_years,
    fold,
    markup_title,
    title_spans,
    token_set,
)

# Minimum fraction of candidate title words that must appear in the citation
# for an OpenAlex link to be considered (strict linking): title-word overlap
# must clear this AND (publication year within +/-1 of a cited year OR the
# candidate author surname appears in the citation).
MATCH_RATIO_THRESHOLD = 0.5
MAX_AUTHORS_TO_EXTRACT = 5
SEARCH_QUERY_MAX_LENGTH = 100

# The text after an author-year citation's "(Year)." -- "(1994).",
# "(1994a).", "(1994, May).", a year range "(1886-87).", a reprint
# "(1976 [1867])." / "([1867] 1976).", or an undated "(n.d.).".
_APA_TITLE_RE = re.compile(
    r"\(\s*(?:(?:\[\s*" + PUB_YEAR + r"\s*\]\s*)?" + PUB_YEAR + r"[a-z]?"
    r"(?:\s*[-–]\s*\d{2,4})?(?:\s*\[\s*" + PUB_YEAR + r"\s*\])?"
    r"|n\.\s?d\.)(?:,[^)]*)?\s*\)\.?\s+(.+)",
    re.DOTALL,
)
# Bracketed asides inside an isolated title ("(M. Grant, Trans.)", "[2nd ed.]")
# and a trailing volume designation (", Vol. IV", "Bd. 2") are not part of the
# title a work is indexed under. An aside cut open by the sentence end
# ("(M. Grant, Trans.") runs to the end.
_TITLE_ASIDE_RE = re.compile(r"\s*(?:\([^)]*(?:\)|$)|\[[^\]]*(?:\]|$))")
_TITLE_VOLUME_RE = re.compile(
    r"[\s,;:]+(?:vols?\.?|bd\.|tome|t\.)(?:\s*(?:\d+|[ivxlc]+)\b.*)?$",
    re.IGNORECASE,
)
# Everything up to the first sentence end (a terminal mark before whitespace
# or the end of the text). A period right after a lone capital letter ("U.S.",
# "J. R.") is an abbreviation or initial, not a sentence end.
_SENTENCE_RE = re.compile(r"(.+?)(?:(?:[?!]|(?<!\b[A-Z])\.)(?:\s|$)|$)", re.DOTALL)

_DOI_PATTERNS = (
    r"doi:\s*(10\.\d{4,}/[^\s]+)",
    r"https?://doi\.org/(10\.\d{4,}/[^\s]+)",
    r"https?://dx\.doi\.org/(10\.\d{4,}/[^\s]+)",
    r"\b(10\.\d{4,}/[^\s,;]+)",
)


def truncate_at_word(text: str, limit: int) -> str:
    """Cut *text* to at most *limit* characters, ending on a whole word."""
    if len(text) <= limit:
        return text
    head = text[: limit + 1].rsplit(" ", 1)[0]
    # A single word longer than the limit has no boundary to cut at.
    return head if len(head) <= limit else text[:limit]


def looks_like_review(citation_text: str, work_data: dict[str, Any]) -> bool:
    """Return True for a candidate that is probably a review of the cited work.

    A book review repeats the book's exact title and appears a year or so
    later, and OpenAlex often lists the reviewed author among its authors, so
    title, year, and author checks all pass. What gives it away is its form: a
    journal article of at most two pages. Such a candidate is rejected unless
    the citation names its journal (then a short article is what was cited).
    """
    if work_data.get("type") == "book-review":
        return True
    if work_data.get("type") != "article":
        return False
    biblio = work_data.get("biblio")
    if not isinstance(biblio, dict):
        return False
    try:
        first = int(str(biblio.get("first_page")))
        last = int(str(biblio.get("last_page")))
    except ValueError:
        return False
    if not 0 <= last - first <= 1:
        return False
    location = work_data.get("primary_location")
    source = location.get("source") if isinstance(location, dict) else None
    journal = source.get("display_name") if isinstance(source, dict) else None
    return not (isinstance(journal, str) and fold(journal) in fold(citation_text))


def publication_year_filter(years: list[int]) -> str:
    """Return an OpenAlex ``publication_year`` filter for *years* +/-1.

    One year gives a range (``publication_year:2019-2021``); a reprint's two
    years give an OR list of the explicit values
    (``publication_year:1866|1867|1868|1975|1976|1977``).
    """
    if not years:
        return ""
    if len(years) == 1:
        return f"publication_year:{years[0] - 1}-{years[0] + 1}"
    window = sorted({y + d for y in years for d in (-1, 0, 1)})
    return "publication_year:" + "|".join(str(y) for y in window)


def extract_doi(citation_text: str) -> str | None:
    """Return the DOI in *citation_text*, or None."""
    for pattern in _DOI_PATTERNS:
        match = re.search(pattern, citation_text, re.IGNORECASE)
        if match:
            doi = match.group(1).rstrip(".,;")
            # A bracketed citation ("(10.1234/abc)", "<https://doi.org/...>")
            # leaves an unmatched trailing ")", "]" or ">" that 404s, while
            # DOIs with balanced inner pairs (10.1016/S0140-6736(00)...) keep
            # theirs. The loop runs to a fixpoint so mixed tails like ")]"
            # and re-exposed ".,;" are stripped too.
            changed = True
            while changed:
                changed = False
                for open_ch, close_ch in (("(", ")"), ("[", "]"), ("<", ">")):
                    while doi.endswith(close_ch) and doi.count(open_ch) < doi.count(
                        close_ch
                    ):
                        doi = doi[:-1]
                        changed = True
                stripped = doi.rstrip(".,;")
                if stripped != doi:
                    doi = stripped
                    changed = True
            return doi
    return None


def extract_title(citation_text: str) -> str:
    """Return the cited work's title, or "" when none can be isolated.

    In an author-year citation the title follows ``(Year).`` (reprints:
    ``(1976 [1867]).``); an italicized or quoted title in that position is
    taken from its markup, a plain one runs to the next sentence end.
    Taking the position first keeps an italicized journal name from
    passing for an article's title. Without ``(Year).`` the first
    italicized, else quoted, span is used. One-word titles are rejected as
    too unspecific for a title search.
    """
    apa = _APA_TITLE_RE.search(citation_text)
    if apa is None:
        title = markup_title(citation_text)
    else:
        rest = apa.group(1)
        if rest.startswith(("*", '"')):
            title = markup_title(rest)
        else:
            sentence = _SENTENCE_RE.match(rest)
            title = sentence.group(1) if sentence else ""
    title = _TITLE_VOLUME_RE.sub("", _TITLE_ASIDE_RE.sub("", title))
    title = title.strip(" .,;:")
    return title if len(title.split()) >= 2 else ""


def extract_search_terms(citation_text: str) -> str:
    """Extract key search terms from citation text.

    Preferred strategy: take up to three author surnames followed by the
    italicized (``*title*``) or quoted (``"title"``) title. Including
    venue/publisher tokens (``Stanford University Press``) dilutes
    relevance on the OpenAlex ``/works`` endpoint and can push the true
    match off the first page of results. When no markup-delimited title
    is present, fall back to the naive cleanup as a last resort.
    """
    title = markup_title(citation_text)

    # Author surnames: the first capitalized words (at least three characters)
    # before the first year or opening parenthesis. The token pattern is
    # Unicode-aware so accented surnames ("Müller") and apostrophe forms
    # ("O'Brien") are kept whole.
    head = re.split(r"\(|\d{4}", citation_text, maxsplit=1)[0]
    surnames = [
        tok
        for tok in re.findall(r"[^\W\d_](?:[^\W\d_]|['’-])*", head)
        if len(tok) >= 3
        and tok[:1].isupper()
        and tok.lower() not in {"ed", "eds", "the", "and", "of", "in", "on"}
    ][:3]

    if title and surnames:
        combined = " ".join(surnames) + " " + title
        # Strip OpenAlex search operators (`*`, `"`, `?`, `!`) that
        # otherwise trigger 500 errors on /works.
        combined = re.sub(r'[*"?!]', " ", combined)
        combined = re.sub(r"\s+", " ", combined).strip()
        return combined[:SEARCH_QUERY_MAX_LENGTH]

    text = re.sub(r"\([^)]*\)", "", citation_text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\d{4}", "", text)
    text = re.sub(r'[*"?!]', " ", text)
    text = re.sub(r"[,.:;]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:SEARCH_QUERY_MAX_LENGTH]


def extract_metadata(work_data: dict[str, Any]) -> dict[str, Any]:
    """Extract the metadata kept for a citation from an OpenAlex work."""
    # Tolerate malformed payloads throughout: any field may arrive as a
    # JSON null or with an unexpected type without crashing the run.
    raw_doi = work_data.get("doi")
    metadata: dict[str, Any] = {
        "title": work_data.get("title"),
        "doi": (
            raw_doi.replace("https://doi.org/", "")
            if isinstance(raw_doi, str) and raw_doi
            else None
        ),
        "publication_year": work_data.get("publication_year"),
        "url": next(
            (
                value
                for value in (work_data.get("doi"), work_data.get("id"))
                if isinstance(value, str) and value
            ),
            None,
        ),
        "authors": [],
        "venue": None,
    }

    authorships = work_data.get("authorships")
    if not isinstance(authorships, list):
        authorships = []
    for authorship in authorships[:MAX_AUTHORS_TO_EXTRACT]:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author", {})
        if isinstance(author, dict) and author.get("display_name"):
            metadata["authors"].append(author["display_name"])

    primary_location = work_data.get("primary_location")
    if isinstance(primary_location, dict):
        source = primary_location.get("source")
        if isinstance(source, dict):
            metadata["venue"] = source.get("display_name")

    return metadata


def verify_citation_match(
    citation_text: str,
    work_data: dict[str, Any],
    title_overlap: float = MATCH_RATIO_THRESHOLD,
) -> bool:
    """Verify that an OpenAlex result matches the citation (strict linking).

    Requires title-word overlap at or above *title_overlap* AND a
    corroborating signal: the candidate's publication year within +/-1 of a
    year cited in the text, or the candidate's author surname appearing in
    the citation. This prefers no link over a wrong one (a permissive
    title-only match assigns wrong DOIs to look-alike titles).
    """
    raw_title = work_data.get("title") or work_data.get("display_name") or ""
    if not isinstance(raw_title, str) or not raw_title:
        # A truthy non-string title (a JSON number) would raise inside
        # fold and abort enrichment for the whole document.
        return False

    citation_folded = fold(citation_text)
    title_words = {w for w in token_set(fold(raw_title)) if len(w) > 3}
    if not title_words:
        return False

    citation_words = token_set(citation_folded)

    # A grossly mismatched known year disqualifies outright: a coincidental
    # author surname must not link a 1950 citation to a 2001 candidate.
    # A reprint cites two years; the candidate may carry either.
    cand_year = work_data.get("publication_year")
    years = cited_years(citation_text)
    year_diff: int | None = None
    if isinstance(cand_year, int) and years:
        year_diff = min(abs(cand_year - year) for year in years)
    if year_diff is not None and year_diff > 2:
        return False

    # Title check. A single substantive title token ("Nations") scores 1.0
    # against any citation containing that word, so demand it appear as an
    # explicit quoted/italic title span; otherwise require at least two
    # substantive title tokens in the overlap plus the ratio threshold.
    if len(title_words) < 2:
        folded_title = fold(raw_title).strip()
        citation_title_spans = {
            fold(span).strip() for span in title_spans(citation_text)
        }
        if folded_title not in citation_title_spans:
            return False
    else:
        overlap_tokens = title_words & citation_words
        if len(overlap_tokens) < 2:
            return False
        if len(overlap_tokens) / len(title_words) < title_overlap:
            return False

    # Corroborating signal 1: publication year within +/-1 of a cited year.
    if year_diff is not None and year_diff <= 1:
        return True

    # Corroborating signal 2: candidate author surname present in citation.
    # A JSON null for "authorships" defeats the .get default; coerce.
    authorships = work_data.get("authorships")
    for authorship in authorships if isinstance(authorships, list) else []:
        author = authorship.get("author", {}) if isinstance(authorship, dict) else {}
        name = author.get("display_name") or ""
        parts = fold(name).split()
        if parts and parts[-1] in citation_words:
            return True

    return False


__all__ = [
    "MATCH_RATIO_THRESHOLD",
    "SEARCH_QUERY_MAX_LENGTH",
    "extract_doi",
    "extract_metadata",
    "extract_search_terms",
    "extract_title",
    "looks_like_review",
    "publication_year_filter",
    "truncate_at_word",
    "verify_citation_match",
]
