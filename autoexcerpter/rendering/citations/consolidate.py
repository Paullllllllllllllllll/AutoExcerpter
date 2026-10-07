"""Deduplication and merging of the citations collected from one document."""

from __future__ import annotations

import logging
from collections import defaultdict
from difflib import SequenceMatcher

from autoexcerpter.rendering.citations.model import (
    Citation,
    fold,
    jaccard,
    token_set,
)

logger = logging.getLogger(__name__)

# Conservative fuzzy-merge thresholds (prefer under-merging). Variants merge
# only within a (first-author surname, year) block when EITHER similarity gate
# passes; different years or volumes never merge.
MERGE_RATIO = 0.90  # SequenceMatcher similarity gate
MERGE_JACCARD = 0.85  # token-set Jaccard gate

type _Block = dict[tuple[str, int | None], list[Citation]]


def _blocks(citations: dict[str, Citation]) -> _Block:
    """Group citations by (first-author surname, exact year)."""
    blocks: _Block = defaultdict(list)
    for citation in citations.values():
        blocks[(citation.author, citation.year)].append(citation)
    return blocks


def consolidate(
    citations: dict[str, Citation],
    *,
    merge_ratio: float = MERGE_RATIO,
    merge_jaccard: float = MERGE_JACCARD,
) -> dict[str, Citation]:
    """Return *citations* with near-duplicates merged and partials resolved.

    Blocks citations on (first-author surname, exact year) so different years
    never merge, then within a block merges variants whose SequenceMatcher
    ratio clears *merge_ratio* or whose token-set Jaccard clears
    *merge_jaccard*. Differing volumes never merge. The longest variant
    becomes canonical; page locators union; the first non-null DOI/metadata
    wins; every merge is logged with both variants.
    """
    merged: dict[str, Citation] = {}
    for block in _blocks(citations).values():
        # Deterministic merge order: longest (most complete) variant first,
        # so the survivor set does not depend on document mention order.
        block.sort(key=lambda c: (-len(c.comparison_text), c.comparison_text))
        survivors: list[Citation] = []
        for candidate in block:
            target = find_merge_target(
                candidate,
                survivors,
                merge_ratio=merge_ratio,
                merge_jaccard=merge_jaccard,
            )
            if target is None:
                survivors.append(candidate)
            else:
                merge_into(target, candidate)
        for survivor in survivors:
            merged[survivor.normalized_key] = survivor

    resolve_partials(merged)
    return merged


def resolve_partials(citations: dict[str, Citation]) -> None:
    """Merge or drop partial (in-text-only) citations by containment, in place.

    Within each (first-author surname, exact year) block, a partial citation's
    comparison-text token set is typically just author names + year; a full
    reference for the same work is its token superset. For each partial:

    - exactly one full (non-partial) superset candidate -> merge the partial
      into it (full raw_text stays canonical; page locators union);
    - zero candidates, or more than one (ambiguous) -> drop the partial
      entirely (do not guess), logged at info level.

    Full citations are never dropped or altered by this step.
    """
    partials = [c for c in citations.values() if c.partial]
    if not partials:
        return

    blocks = _blocks(citations)
    for partial in partials:
        block = blocks[(partial.author, partial.year)]
        partial_tokens = token_set(partial.comparison_text)
        if not partial_tokens:
            # An empty token set (bare URL, bracket-only stub) is a subset
            # of every full reference, so containment is zero evidence.
            logger.info(
                "Dropping partial citation (no comparison tokens): %s",
                partial.raw_text,
            )
            citations.pop(partial.normalized_key, None)
            continue
        candidates = [
            full
            for full in block
            if not full.partial
            and full is not partial
            and partial_tokens <= token_set(full.comparison_text)
        ]
        if not candidates and partial.year is not None:
            # An in-text partial cites the reprint year ("(Smith 1990)") while
            # the full reference ("1990 [1890]") canonicalizes to the earliest
            # year, so the two land in different year blocks. All fulls of the
            # same author are candidates then; the subset test still checks
            # the year, whose token must appear among the full's tokens.
            candidates = [
                full
                for (author, _year), same_author in blocks.items()
                if author == partial.author
                for full in same_author
                if not full.partial
                and full is not partial
                and partial_tokens <= token_set(full.comparison_text)
            ]
        if len(candidates) == 1:
            merge_partial_into(candidates[0], partial)
        else:
            reason = (
                "no full match"
                if not candidates
                else f"ambiguous ({len(candidates)} full matches)"
            )
            logger.info("Dropping partial citation (%s): %s", reason, partial.raw_text)
        citations.pop(partial.normalized_key, None)


def merge_partial_into(full: Citation, partial: Citation) -> None:
    """Merge a partial citation into a full one; the full one stays canonical.

    Unlike :func:`merge_into`, the full reference's ``raw_text`` always remains
    canonical regardless of length: a partial stub must never become the
    displayed citation. Locators union.
    """
    logger.info(
        "Merging partial citation into full:\n  - full:    %s\n  - partial: %s",
        full.raw_text,
        partial.raw_text,
    )
    full.locators |= partial.locators


def find_merge_target(
    candidate: Citation,
    survivors: list[Citation],
    *,
    merge_ratio: float = MERGE_RATIO,
    merge_jaccard: float = MERGE_JACCARD,
) -> Citation | None:
    """Return an existing survivor *candidate* should merge into, or None."""
    for survivor in survivors:
        if not candidate.comparison_text or not survivor.comparison_text:
            # A citation stripped down to nothing (a bare URL) carries no
            # comparison material, and SequenceMatcher scores two empty
            # strings as a perfect match, which would silently collapse
            # distinct references. Never fuzzy-merge on absent evidence.
            continue
        if (
            candidate.volume is not None
            and survivor.volume is not None
            and candidate.volume != survivor.volume
        ):
            # Different volumes are genuinely different works.
            continue
        ratio = SequenceMatcher(
            None, candidate.comparison_text, survivor.comparison_text
        ).ratio()
        similarity = jaccard(candidate.comparison_text, survivor.comparison_text)
        if ratio >= merge_ratio or similarity >= merge_jaccard:
            return survivor
    return None


def merge_into(survivor: Citation, other: Citation) -> None:
    """Merge *other* into *survivor* (the longest variant becomes canonical)."""
    logger.info(
        "Merging citation variants:\n  - %s\n  - %s",
        survivor.raw_text,
        other.raw_text,
    )
    survivor.variants.extend(other.variants)
    if other.raw_text != survivor.raw_text:
        survivor.variants.append(other.raw_text)
    if len(other.raw_text) > len(survivor.raw_text):
        survivor.variants.append(survivor.raw_text)
        survivor.raw_text = other.raw_text
        # Later merge, partial-resolution and sorting decisions must use the
        # promoted text.
        survivor.refresh()

    survivor.locators |= other.locators
    # Full wins over partial, exactly as in add_citations: a partial survivor
    # that absorbs a full near-duplicate must stop being partial, or
    # resolve_partials would later drop the merged citation outright.
    survivor.partial = survivor.partial and other.partial
    if survivor.doi is None:
        survivor.doi = other.doi
    if survivor.metadata is None:
        survivor.metadata = other.metadata
    if survivor.url is None:
        survivor.url = other.url


def merge_by_shared_identifier(citations: dict[str, Citation]) -> dict[str, Citation]:
    """Return *citations* with those resolved to the same DOI or work id merged."""
    ident_map: dict[str, Citation] = {}
    merged: dict[str, Citation] = {}
    for citation in list(citations.values()):
        ident: str | None = None
        if citation.doi:
            ident = f"doi:{citation.doi.strip().lower()}"
        elif citation.metadata and citation.metadata.get("url"):
            ident = f"url:{str(citation.metadata['url']).strip().lower()}"

        if ident and ident in ident_map:
            merge_into(ident_map[ident], citation)
            continue

        merged[citation.normalized_key] = citation
        if ident:
            ident_map[ident] = citation
    return merged


def sorted_citations(citations: dict[str, Citation]) -> list[Citation]:
    """Return citations in a stable, accent-folded order.

    Sorts by (folded first-author surname, year, folded title/comparison text)
    so the order is deterministic across runs and does not push accented
    names after ``z`` (as a raw-text sort would).
    """

    def sort_key(c: Citation) -> tuple[str, int, str]:
        return (
            c.author or "~",
            c.year if c.year is not None else 9999,
            c.comparison_text or fold(c.raw_text),
        )

    return sorted(citations.values(), key=sort_key)


__all__ = [
    "MERGE_JACCARD",
    "MERGE_RATIO",
    "consolidate",
    "find_merge_target",
    "merge_by_shared_identifier",
    "merge_into",
    "merge_partial_into",
    "resolve_partials",
    "sorted_citations",
]
