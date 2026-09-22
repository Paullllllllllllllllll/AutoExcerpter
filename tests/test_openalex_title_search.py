"""OpenAlex text lookups search by title and cited year when a title is found.

Plain APA citations ("Author (Year). Title. Venue.") used to go to OpenAlex's
full-text ``search`` whole; the lookup now isolates the title and sends a
``title.search`` filter restricted to the cited year +/-1 (both years for a
reprint). Citations without an isolable title keep the free-text search.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rendering.citations import (
    SEARCH_QUERY_MAX_LENGTH,
    CitationManager,
    _looks_like_review,
    _reset_budget_state_for_tests,
)

_MARX = "Marx, K. (1976 [1867]). Capital: A Critique of Political Economy. Penguin."


@pytest.fixture(autouse=True)
def _reset_openalex_budget_latch() -> Generator[None]:
    with patch("rendering.citations._persist_budget_state"):
        _reset_budget_state_for_tests()
        yield
        _reset_budget_state_for_tests()


def _run_text_query(citation: str) -> list[dict[str, Any]]:
    """Run a text lookup against an empty result set; return each call's params."""
    calls: list[dict[str, Any]] = []

    def fake_get(url: str, params: Any = None, timeout: Any = None) -> MagicMock:
        calls.append(dict(params or {}))
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"results": []}
        return response

    with patch("rendering.citations.requests.get", side_effect=fake_get):
        CitationManager()._query_openalex_by_text(citation)
    return calls


class TestExtractTitle:
    @pytest.mark.parametrize(
        ("citation", "title"),
        [
            (
                "de Vries, J. (1994). The Industrial Revolution and the "
                "Industrious Revolution. Journal of Economic History, 54, 249-270.",
                "The Industrial Revolution and the Industrious Revolution",
            ),
            (
                "Noordegraaf, L. (1985). Hollands welvaren? Levensstandaard in "
                "Holland 1450-1650. Bergen: Octavo.",
                "Hollands welvaren",
            ),
            (
                'Smith, J. (2020). "A Sufficiently Long Title." Journal.',
                "A Sufficiently Long Title",
            ),
            (
                "Braudel, F. (1979). *Civilization and Capitalism*. Harper.",
                "Civilization and Capitalism",
            ),
            # The title position wins over an italicized journal name.
            (
                "Smith, J. (2001). Plain article title here. *Journal of Things*, 3.",
                "Plain article title here",
            ),
            (_MARX, "Capital: A Critique of Political Economy"),
            (
                "Braudel, Fernand. *La Mediterranee et le monde*. Paris: Colin.",
                "La Mediterranee et le monde",
            ),
            # Chicago style: the quoted article precedes the italic journal.
            (
                'Smith, John. "A History of Work." *Journal of Labor History* 10 '
                "(2020): 1-20.",
                "A History of Work",
            ),
            # A period after a lone capital is an abbreviation, not the end.
            (
                "Doe, J. (1999). The U.S. Supreme Court and food law. Law Review.",
                "The U.S. Supreme Court and food law",
            ),
            ("Anon. (1901). Recipes. London.", ""),
            ("Braudel Fernand Civilisation materielle economie et capitalisme", ""),
        ],
    )
    def test_extract_title(self, citation: str, title: str) -> None:
        assert CitationManager._extract_title(citation) == title


class TestTitleFilterQuery:
    def test_apa_citation_uses_title_filter_with_year_window(self) -> None:
        (params,) = _run_text_query(
            'Smith, J. (2020). "A Sufficiently Long Title." Journal.'
        )
        assert "search" not in params
        assert params["filter"] == (
            "title.search:A Sufficiently Long Title,publication_year:2019-2021"
        )

    def test_reprint_filters_either_year_with_or_syntax(self) -> None:
        (params,) = _run_text_query(_MARX)
        assert params["filter"] == (
            "title.search:Capital A Critique of Political Economy,"
            "publication_year:1866|1867|1868|1975|1976|1977"
        )

    def test_commas_and_pipes_are_stripped_from_the_title(self) -> None:
        (params,) = _run_text_query(
            "Doe, J. (1990). Bread, Beer | and Wine, 1500-1800. Brill."
        )
        title_filter, year_filter = params["filter"].split(",")
        assert title_filter == "title.search:Bread Beer and Wine 1500 1800"
        assert year_filter == "publication_year:1989-1991"

    def test_long_title_is_cut_at_a_whole_word(self) -> None:
        words = ["Consumption"] * 20
        (params,) = _run_text_query(f"Doe, J. (1990). {' '.join(words)}. Brill.")
        query = params["filter"].split(",")[0].removeprefix("title.search:")
        assert len(query) <= SEARCH_QUERY_MAX_LENGTH
        assert set(query.split()) == {"Consumption"}

    def test_empty_title_result_sends_no_second_query(self) -> None:
        calls = _run_text_query(_MARX)
        assert len(calls) == 1

    def test_citation_without_title_keeps_free_text_search(self) -> None:
        (params,) = _run_text_query(
            "Braudel Fernand Civilisation materielle economie et capitalisme"
        )
        assert "filter" not in params
        assert params["search"]


class TestReprintVerification:
    @staticmethod
    def _candidate(year: int) -> dict[str, Any]:
        return {
            "title": "Capital: A Critique of Political Economy",
            "publication_year": year,
            "authorships": [],
        }

    @pytest.mark.parametrize("year", [1867, 1976, 1977])
    def test_either_cited_year_corroborates(self, year: int) -> None:
        manager = CitationManager()
        assert manager._verify_citation_match(_MARX, self._candidate(year))

    def test_year_far_from_both_is_rejected(self) -> None:
        manager = CitationManager()
        assert not manager._verify_citation_match(_MARX, self._candidate(1920))


_PAWSON = (
    "Pawson, E. (1977). Transport and Economy: The Turnpike Roads of "
    "Eighteenth Century Britain. London: Academic Press."
)


def _work(
    work_type: str = "book",
    pages: tuple[str, str] | None = None,
    journal: str | None = None,
    doi: str = "https://doi.org/10.1/book",
) -> dict[str, Any]:
    """An OpenAlex candidate titled like Pawson's book (a review's authors
    include the reviewed author, as OpenAlex often records them)."""
    work: dict[str, Any] = {
        "title": "Transport and Economy: The Turnpike Roads of Eighteenth-Century "
        "Britain",
        "publication_year": 1978,
        "type": work_type,
        "doi": doi,
        "authorships": [
            {"author": {"display_name": "Robert Smith"}},
            {"author": {"display_name": "Eric Pawson"}},
        ],
    }
    if pages:
        work["biblio"] = {"first_page": pages[0], "last_page": pages[1]}
    if journal:
        work["primary_location"] = {"source": {"display_name": journal}}
    return work


_REVIEW = _work("article", ("346", "346"), "The Journal of Interdisciplinary History",
                "https://doi.org/10.2307/203239")  # fmt: skip


class TestBookReviewsAreSkipped:
    @pytest.mark.parametrize(
        ("work", "is_review"),
        [
            (_REVIEW, True),
            (_work("book-review"), True),
            (_work("article", ("345", "346"), "Some Journal"), True),
            (_work("article", ("340", "360"), "Some Journal"), False),
            (_work("article"), False),
            (_work("book"), False),
            (_work("review", ("1", "1")), False),
        ],
    )
    def test_looks_like_review(self, work: dict[str, Any], is_review: bool) -> None:
        assert _looks_like_review(_PAWSON, work) is is_review

    def test_short_article_in_the_cited_journal_is_kept(self) -> None:
        citation = (
            "Smith, R. (1978). Transport and Economy: The Turnpike Roads of "
            "Eighteenth-Century Britain. The Journal of Interdisciplinary "
            "History, 9, 346."
        )
        assert not _looks_like_review(citation, _REVIEW)

    def test_title_filter_query_skips_the_review(self) -> None:
        response = MagicMock(status_code=200)
        response.json.return_value = {"results": [_REVIEW, _work("book")]}
        with patch("rendering.citations.requests.get", return_value=response):
            hit = CitationManager()._query_openalex_by_text(_PAWSON)
        assert hit is not None
        assert hit["doi"] == "10.1/book"


class TestTitleCleanup:
    @pytest.mark.parametrize(
        ("citation", "title"),
        [
            (
                "Galen. (2000). Galen on Food and Diet (M. Grant, Trans. & Ed.). "
                "London: Routledge.",
                "Galen on Food and Diet",
            ),
            (
                "Temple, R. C. (Ed.). (1925). The Travels of Peter Mundy in Europe "
                "and Asia, 1608-1667, Vol. IV (1639-47). Hakluyt Society.",
                "The Travels of Peter Mundy in Europe and Asia, 1608-1667",
            ),
            (
                "Hyman, P., & Hyman, M. (n.d.). Printing the kitchen: French "
                "cookbooks, 1480-1800. In Flandrin & Montanari.",
                "Printing the kitchen: French cookbooks, 1480-1800",
            ),
            (
                "L'Espinasse, R. de. (1886-87). Les metiers et corporations de la "
                "ville de Paris. Paris.",
                "Les metiers et corporations de la ville de Paris",
            ),
            # A title word that looks like a volume marker stays.
            ("Doe, J. (1990). Till the cows come home. Brill.",
             "Till the cows come home"),
        ],
    )  # fmt: skip
    def test_extract_title(self, citation: str, title: str) -> None:
        assert CitationManager._extract_title(citation) == title

    def test_undated_citation_searches_the_title_without_a_year(self) -> None:
        (params,) = _run_text_query(
            "Hyman, P., & Hyman, M. (n.d.). Printing the kitchen: French "
            "cookbooks, 1480-1800. In Flandrin & Montanari."
        )
        assert params["filter"] == (
            "title.search:Printing the kitchen French cookbooks 1480 1800"
        )
