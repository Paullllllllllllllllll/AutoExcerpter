"""Citations: the record and its keys, deduplication, and OpenAlex lookups.

``model`` holds :class:`Citation` and the normalization keys, ``consolidate``
the deduplication and merging, ``matching`` the OpenAlex queries and candidate
checks, ``openalex`` the lookups with their explicit outcomes and the cache,
and ``manager`` the per-document :class:`CitationManager` that ties them
together.
"""

from autoexcerpter.rendering.citations.manager import (
    DEFAULT_MAX_API_REQUESTS,
    CitationManager,
    enrich_if_enabled,
)
from autoexcerpter.rendering.citations.model import Citation
from autoexcerpter.rendering.citations.openalex import (
    LookupOutcome,
    LookupResult,
    OpenAlexCache,
    OpenAlexClient,
)

__all__ = [
    "DEFAULT_MAX_API_REQUESTS",
    "Citation",
    "CitationManager",
    "LookupOutcome",
    "LookupResult",
    "OpenAlexCache",
    "OpenAlexClient",
    "enrich_if_enabled",
]
