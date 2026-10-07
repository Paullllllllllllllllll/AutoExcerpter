"""Fake provider classes and fake OpenAlex service for the characterization tests.

``FakeLLM.constructor`` builds stand-ins for the provider classes the client
factory instantiates (``ChatOpenAI``, ``ChatAnthropic``,
``ChatGoogleGenerativeAI``). Each stand-in records its constructor kwargs and
returns a ``ScriptedChatModel`` that answers from per-page scripts, so the
real factory, managers, request assembly, structured-output routing, retries
and parsing all run. A request with an image block is a transcription
request, matched to a page by the marker in the image; any other request is a
summary request, matched by the page key in its text. Neither depends on the
thread or task that sends the request.

``FakeOpenAlex`` stands in for ``requests.Session`` and ``requests.get`` and
serves the hand-built payloads in ``fixtures/openalex``.
"""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import io
import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import httpx
from langchain_core.messages import AIMessage, BaseMessage
from PIL import Image

from autoexcerpter.common.testing import FakeRequest, ScriptedChatModel
from tests.characterization.inputs import PALETTE, decode_marker

TRANSCRIPTION = "transcription"
SUMMARY = "summary"
ROLE_ORDER = (TRANSCRIPTION, SUMMARY)

CUSTOM_KEY_ENV = "AE_FAKE_CUSTOM_KEY"
DUMMY_KEYS = {
    "OPENAI_API_KEY": "test-openai-key",
    "ANTHROPIC_API_KEY": "test-anthropic-key",
    "GOOGLE_API_KEY": "test-google-key",
    "OPENROUTER_API_KEY": "test-openrouter-key",
    CUSTOM_KEY_ENV: "test-custom-key",
}
_KEY_NAMES = {value: name for name, value in DUMMY_KEYS.items()}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_CLASS_PROVIDERS = {"ChatAnthropic": "anthropic", "ChatGoogleGenerativeAI": "google"}

USAGE: dict[str, dict[str, int]] = {
    TRANSCRIPTION: {"input_tokens": 1500, "output_tokens": 400, "total_tokens": 1900},
    SUMMARY: {"input_tokens": 900, "output_tokens": 250, "total_tokens": 1150},
}

OPENALEX_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "openalex"


# ============================================================================
# Page scripts
# ============================================================================
@dataclass(frozen=True)
class Injection:
    """Replace one response: raise, return raw text, or refuse.

    ``exception`` is raised from ``invoke``. Otherwise the message content is
    ``text`` (raw, unparsed), the refusal lands in
    ``additional_kwargs["refusal"]``, and ``response_metadata`` replaces the
    default metadata (for example a content-filter status).
    """

    exception: BaseException | None = None
    text: str | None = None
    refusal: str | None = None
    response_metadata: Mapping[str, Any] | None = None


@dataclass
class PageScript:
    """Scripted answers for one page.

    ``key`` must occur in the page's transcription text and in no other page's
    text; it identifies the page in summary requests. ``injections`` maps
    ``(role, attempt)`` to an ``Injection``, where *attempt* counts that
    role's calls for this page from 1.
    """

    key: str
    transcription: dict[str, Any]
    summary: dict[str, Any]
    plain_transcription: str | None = None
    plain_summary: str | None = None
    injections: dict[tuple[str, int], Injection] = field(default_factory=dict)

    def text_for(self, role: str) -> str:
        """Return the plain-text answer for *role*."""
        if role == TRANSCRIPTION:
            if self.plain_transcription is not None:
                return self.plain_transcription
            return str(self.transcription.get("transcription") or "")
        if self.plain_summary is not None:
            return self.plain_summary
        bullets = self.summary.get("bullet_points") or []
        return "\n".join(f"- {bullet}" for bullet in bullets) or "No summary."

    def json_for(self, role: str) -> dict[str, Any]:
        """Return the schema-shaped answer for *role*."""
        source = self.transcription if role == TRANSCRIPTION else self.summary
        return copy.deepcopy(source)


ALLEN_FULL = (
    "Allen, R. C. (2001). The great divergence in European wages and prices "
    "from the Middle Ages to the First World War. *Explorations in Economic "
    "History, 38*(4), 411-447. https://doi.org/10.1006/exeh.2001.0775"
)
ALLEN_BIBLIOGRAPHY = (
    "Allen, R. C. (2001). The great divergence in European wages and prices "
    "from the Middle Ages to the First World War. *Explorations in Economic "
    "History, 38*(4), 411-447."
)
BRAUDEL_FULL = (
    "Braudel, F. (1979). *Civilisation matérielle, économie et capitalisme, "
    "XVe-XVIIIe siècle*. Paris: Armand Colin."
)
MONTANARI_FULL = "Montanari, M. (1994). *The Culture of Food*. Oxford: Blackwell."


def _transcription(analysis: str, text: str) -> dict[str, Any]:
    return {
        "image_analysis": analysis,
        "transcription": text,
        "no_transcribable_text": False,
        "transcription_not_possible": False,
    }


def _summary(
    number: int | None,
    number_type: str,
    page_types: list[str],
    bullets: list[str] | None,
    references: list[tuple[str, bool]] | None,
) -> dict[str, Any]:
    return {
        "page_information": {
            "page_number_integer": number,
            "is_two_page_spread": False,
            "page_number_integer_end": None,
            "page_number_type": number_type,
            "page_types": page_types,
        },
        "bullet_points": bullets,
        "references": (
            None
            if references is None
            else [
                {"citation": citation, "is_partial": partial}
                for citation, partial in references
            ]
        ),
    }


def default_page(index: int) -> PageScript:
    """Return the default script for page *index* (0-based).

    Page 0 is a preface numbered ``xi`` with a footnote and a partial
    reference; page 1 is arabic page 1 with markdown, a hyphenated line
    break, an image description, a footnote with a DOI and summary bullets
    holding inline and display LaTeX; page 2 is an unnumbered bibliography;
    later pages are arabic content pages numbered from 2.
    """
    if index == 0:
        text = (
            "# Preface\n\n"
            "This volume brings together **new research** on the *material\n"
            "culture* of eating in early modern Europe. Its chapters trace how\n"
            "prices, wages and diets changed between 1500 and 1800.[^1]\n\n"
            "[^1]: See Braudel (1979) for the classic statement of this view.\n\n"
            "<page_number>xi</page_number>"
        )
        return PageScript(
            key="# Preface",
            transcription=_transcription(
                "Single column; page number centered in the footer; one footnote.",
                text,
            ),
            summary=_summary(
                11,
                "roman",
                ["preface"],
                [
                    "The preface introduces new research on the *material culture*"
                    " of eating in early modern Europe between 1500 and 1800.",
                    "It frames the chapters around prices, wages and diets and"
                    " follows Braudel's long-run view of material life.",
                ],
                [("Braudel (1979)", True)],
            ),
        )
    if index == 1:
        text = (
            "## The Price of Bread\n\n"
            "Bread prices rose faster than wages after 1550, and the eco-\n"
            "nomic burden fell on urban households.[^2] The real wage index\n"
            "is defined as $w_t / p_t$.\n\n"
            "![Image: line chart of bread prices and day wages in Antwerp,"
            " 1500-1800; prices rise steeply after 1550 while wages stay flat]\n\n"
            f"[^2]: {ALLEN_FULL}\n\n"
            "<page_number>1</page_number>"
        )
        return PageScript(
            key="## The Price of Bread",
            transcription=_transcription(
                "Single column; page number in the footer; one figure and one"
                " footnote.",
                text,
            ),
            summary=_summary(
                1,
                "arabic",
                ["content"],
                [
                    "Bread prices rose faster than wages after 1550, shifting the"
                    " economic burden onto **urban households**.",
                    "The real wage index is defined as $w_t / p_t$, nominal wages"
                    " deflated by the bread price.",
                    "A consumer basket weights goods so that"
                    " $$\\sum_{i=1}^{n} \\omega_i = 1$$ holds in every year.",
                ],
                [(ALLEN_FULL, False), (BRAUDEL_FULL, False)],
            ),
        )
    if index == 2:
        text = f"## Bibliography\n\n{ALLEN_BIBLIOGRAPHY}\n\n{MONTANARI_FULL}"
        return PageScript(
            key="## Bibliography",
            transcription=_transcription(
                "Single column bibliography; no page number visible.", text
            ),
            summary=_summary(
                None,
                "none",
                ["bibliography"],
                None,
                [(ALLEN_BIBLIOGRAPHY, False), (MONTANARI_FULL, False)],
            ),
            plain_summary="Bibliography page without summarizable prose.",
        )
    printed = index - 1
    key = f"## Market Records, Part {index:02d}"
    text = (
        f"{key}\n\n"
        f"Grain deliveries to the city market fell in year {1600 + index},\n"
        "and the magistrates fixed the bread assize twice.\n\n"
        f"<page_number>{printed}</page_number>"
    )
    return PageScript(
        key=key,
        transcription=_transcription("Single column; page number in the footer.", text),
        summary=_summary(
            printed,
            "arabic",
            ["content"],
            [
                f"Grain deliveries fell in {1600 + index}, and the magistrates"
                " fixed the bread assize twice."
            ],
            None,
        ),
    )


# ============================================================================
# Request digests
# ============================================================================
def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _text_digest(text: str) -> dict[str, Any]:
    return {"chars": len(text), "digest": _sha(text)}


def summarize_schema(schema: Any) -> dict[str, Any]:
    """Return a compact, stable description of a JSON schema payload."""
    if not isinstance(schema, Mapping):
        return {"value": repr(schema)}
    inner = schema.get("schema", schema)
    properties = inner.get("properties") if isinstance(inner, Mapping) else None
    canonical = json.dumps(schema, sort_keys=True, ensure_ascii=False)
    summary: dict[str, Any] = {
        "properties": sorted(properties) if isinstance(properties, Mapping) else None,
        "digest": _sha(canonical),
    }
    for key in ("type", "name", "strict", "title"):
        if key in schema:
            summary[key] = schema[key]
    return summary


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return f"<{type(value).__name__}>"


def describe_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Return invoke or bound kwargs with schema payloads summarized."""
    described: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key == "text" and isinstance(value, Mapping) and "format" in value:
            text = dict(value)
            text["format"] = summarize_schema(text["format"])
            described[key] = _jsonable(text)
        elif key == "response_format" and isinstance(value, Mapping):
            fmt = dict(value)
            if isinstance(fmt.get("json_schema"), Mapping):
                fmt["json_schema"] = summarize_schema(fmt["json_schema"])
            described[key] = _jsonable(fmt)
        elif key == "response_schema":
            described[key] = summarize_schema(value)
        else:
            described[key] = _jsonable(value)
    return described


def _decode_data_url(url: str) -> tuple[str | None, bytes | None]:
    if not url.startswith("data:") or "," not in url:
        return None, None
    header, data = url.split(",", 1)
    mime = header[5:].split(";", 1)[0] or None
    try:
        return mime, base64.b64decode(data)
    except (binascii.Error, ValueError):
        return mime, None


def _image_from_block(block: Mapping[str, Any]) -> tuple[dict[str, Any], bytes | None]:
    kind = block.get("type")
    if kind == "image_url":
        image_url = block.get("image_url")
        url = image_url.get("url", "") if isinstance(image_url, Mapping) else ""
        mime, data = _decode_data_url(str(url))
        info: dict[str, Any] = {"type": "image_url", "mime": mime}
        if isinstance(image_url, Mapping) and "detail" in image_url:
            info["detail"] = image_url["detail"]
        return info, data
    source = block.get("source")
    if kind == "image" and isinstance(source, Mapping):
        raw = source.get("data")
        try:
            data = base64.b64decode(str(raw)) if raw is not None else None
        except (binascii.Error, ValueError):
            data = None
        return {
            "type": "image",
            "source_type": source.get("type"),
            "mime": source.get("media_type"),
        }, data
    return {"type": str(kind)}, None


_IMAGE_BLOCKS = frozenset({"image_url", "image"})


def _message_shapes(
    messages: list[BaseMessage],
) -> tuple[list[dict[str, Any]], str, int | None, bool]:
    """Return per-message digests, the user text, the decoded page marker and
    whether any message holds an image block."""
    shapes: list[dict[str, Any]] = []
    user_text_parts: list[str] = []
    page: int | None = None
    has_image = False
    for message in messages:
        content = message.content
        entry: dict[str, Any] = {"type": message.type}
        if isinstance(content, str):
            entry["text"] = _text_digest(content)
            if message.type == "human":
                user_text_parts.append(content)
        else:
            described: list[dict[str, Any]] = []
            for block in content:
                if isinstance(block, str):
                    described.append({"type": "str", **_text_digest(block)})
                    user_text_parts.append(block)
                    continue
                if block.get("type") == "text":
                    text = str(block.get("text", ""))
                    described.append({"type": "text", **_text_digest(text)})
                    if message.type == "human":
                        user_text_parts.append(text)
                    continue
                info, data = _image_from_block(block)
                has_image = has_image or info["type"] in _IMAGE_BLOCKS
                if data is not None:
                    with Image.open(io.BytesIO(data)) as image:
                        image.load()
                        info["width"], info["height"] = image.size
                        page = decode_marker(image)
                described.append(info)
            entry["blocks"] = described
        shapes.append(entry)
    return shapes, "\n".join(user_text_parts), page, has_image


def _system_text(messages: list[BaseMessage]) -> str:
    """Return the concatenated text of the system messages."""
    parts: list[str] = []
    for message in messages:
        if message.type != "system":
            continue
        content = message.content
        if isinstance(content, str):
            parts.append(content)
            continue
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif block.get("type") == "text":
                parts.append(str(block.get("text", "")))
    return "\n".join(parts)


# ============================================================================
# Provider-class constructions
# ============================================================================
def redact(value: Any) -> Any:
    """Return a JSON-ready copy with dummy keys named by their variable."""
    if isinstance(value, httpx.Timeout):
        return {"httpx.Timeout": value.as_dict()}
    if isinstance(value, Mapping):
        return {str(key): redact(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [redact(item) for item in value]
    if isinstance(value, str) and value in _KEY_NAMES:
        return f"<env:{_KEY_NAMES[value]}>"
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return f"<{type(value).__name__}>"


def provider_of(constructor: str, base_url: Any) -> str:
    """Return the provider a constructor call addresses."""
    if constructor in _CLASS_PROVIDERS:
        return _CLASS_PROVIDERS[constructor]
    if base_url is None:
        return "openai"
    if str(base_url).rstrip("/") == OPENROUTER_BASE_URL:
        return "openrouter"
    return "custom"


@dataclass
class Construction:
    """One provider-class constructor call.

    ``role`` is the role of the first request the instance serves; it stays
    None for an instance that serves none.
    """

    constructor: str
    kwargs: dict[str, Any]
    provider: str
    model: str
    role: str | None = None

    def record(self) -> dict[str, Any]:
        """Return the golden entry: role, constructor and redacted kwargs."""
        return {
            "role": self.role,
            "constructor": self.constructor,
            "kwargs": redact(self.kwargs),
        }


def _role_rank(construction: Construction) -> int:
    role = construction.role
    return ROLE_ORDER.index(role) if role in ROLE_ORDER else len(ROLE_ORDER)


def _describe_structured(request: FakeRequest) -> dict[str, Any] | None:
    call = request.structured
    if call is None:
        return None
    return {
        "method": call.method,
        "include_raw": call.include_raw,
        "schema_title": call.title,
        "schema": summarize_schema(call.schema),
    }


# ============================================================================
# Fake LLM
# ============================================================================
class FakeLLM:
    """Provider-class stand-ins and the scripts, records and routing behind them.

    ``text_roles`` lists the roles answered in plain text (the text response
    format); other roles answer with schema-shaped JSON. ``calls`` holds one
    digest per request, ``constructions`` one entry per constructed client.
    """

    def __init__(self, pages: Mapping[int, PageScript] | None = None) -> None:
        self.pages: dict[int, PageScript] = dict(pages or {})
        self.text_roles: set[str] = set()
        self.calls: list[dict[str, Any]] = []
        self.constructions: list[Construction] = []
        self.system_prompts: list[tuple[str, int | None, str]] = []
        self._lock = threading.Lock()
        self._attempts: dict[tuple[str, int | None], int] = {}

    # -- configuration ---------------------------------------------------------
    def page(self, index: int) -> PageScript:
        """Return (creating from the default if needed) the script for a page."""
        with self._lock:
            script = self.pages.get(index)
            if script is None:
                script = default_page(index)
                self.pages[index] = script
            return script

    def inject(self, page: int, role: str, attempt: int, injection: Injection) -> None:
        """Replace the *attempt*-th *role* response for *page* (1-based)."""
        self.page(page).injections[(role, attempt)] = injection

    def reset_records(self) -> None:
        """Forget recorded calls, constructions and attempt counters."""
        with self._lock:
            self.calls.clear()
            self.constructions.clear()
            self.system_prompts.clear()
            self._attempts.clear()

    # -- provider classes ------------------------------------------------------
    def constructor(self, name: str) -> Callable[..., ScriptedChatModel]:
        """Return a stand-in for the provider class *name*."""

        def construct(**kwargs: Any) -> ScriptedChatModel:
            construction = Construction(
                constructor=name,
                kwargs=kwargs,
                provider=provider_of(name, kwargs.get("base_url")),
                model=str(kwargs.get("model", "")),
            )
            with self._lock:
                self.constructions.append(construction)
            return ScriptedChatModel(
                responder=self.respond, recorder=self.record, tag=construction
            )

        return construct

    @property
    def models(self) -> list[dict[str, Any]]:
        """Return the golden entry of every recorded construction."""
        with self._lock:
            return [construction.record() for construction in self.constructions]

    def constructions_by_role(self) -> list[Construction]:
        """Return the recorded constructions in role order."""
        with self._lock:
            return sorted(self.constructions, key=_role_rank)

    # -- answering -------------------------------------------------------------
    def _page_for_summary(self, text: str) -> int | None:
        candidates = set(self.pages) | set(range(len(PALETTE)))
        for index in sorted(candidates):
            if self.page(index).key in text:
                return index
        return None

    def record(self, request: FakeRequest) -> None:
        """Record the digest of one request; note its role, page and attempt."""
        construction: Construction = request.tag
        shapes, user_text, marker, has_image = _message_shapes(request.messages)
        role = TRANSCRIPTION if has_image else SUMMARY
        page = marker if role == TRANSCRIPTION else self._page_for_summary(user_text)
        with self._lock:
            if construction.role is None:
                construction.role = role
            attempt = self._attempts.get((role, page), 0) + 1
            self._attempts[(role, page)] = attempt
            self.calls.append(
                {
                    "role": role,
                    "page": page,
                    "attempt": attempt,
                    "provider": construction.provider,
                    "model": construction.model,
                    "kwargs": describe_kwargs(request.kwargs),
                    "bound": describe_kwargs(request.bound),
                    "structured": _describe_structured(request),
                    "messages": shapes,
                }
            )
            self.system_prompts.append((role, page, _system_text(request.messages)))
        request.notes.update(role=role, page=page, attempt=attempt)

    def respond(self, request: FakeRequest) -> AIMessage:
        """Return the scripted answer to a recorded request."""
        construction: Construction = request.tag
        role: str = request.notes["role"]
        page: int | None = request.notes["page"]
        attempt: int = request.notes["attempt"]
        script = self.page(page) if page is not None else None
        injection = script.injections.get((role, attempt)) if script else None
        metadata: dict[str, Any] = {
            "model_name": construction.model,
            "status": "completed",
        }
        additional: dict[str, Any] = {}
        if injection is not None:
            if injection.exception is not None:
                raise injection.exception
            if injection.response_metadata is not None:
                metadata = dict(injection.response_metadata)
            if injection.refusal is not None:
                additional["refusal"] = injection.refusal
            content = injection.text if injection.text is not None else ""
        elif script is None:
            content = self._unknown_answer(role)
        elif role in self.text_roles:
            content = script.text_for(role)
        else:
            content = json.dumps(script.json_for(role), ensure_ascii=False)
        return AIMessage(
            content=content,
            usage_metadata=dict(USAGE.get(role, USAGE[SUMMARY])),
            response_metadata=metadata,
            additional_kwargs=additional,
        )

    def _unknown_answer(self, role: str) -> str:
        if role in self.text_roles:
            return "Unrecognized page."
        if role == TRANSCRIPTION:
            payload = _transcription("Unrecognized page.", "Unrecognized page.")
        else:
            payload = _summary(None, "none", ["other"], None, None)
        return json.dumps(payload, ensure_ascii=False)

    def system_prompts_for(self, role: str) -> list[str]:
        """Return the raw system prompt text of every *role* request."""
        with self._lock:
            return [text for r, _page, text in self.system_prompts if r == role]

    def sorted_calls(self) -> list[dict[str, Any]]:
        """Return the call digests in a concurrency-independent order."""

        def key(call: dict[str, Any]) -> tuple[int, int, int]:
            page = call["page"] if isinstance(call["page"], int) else 10_000
            role = ROLE_ORDER.index(call["role"]) if call["role"] in ROLE_ORDER else 9
            return page, role, call["attempt"]

        with self._lock:
            return sorted((copy.deepcopy(call) for call in self.calls), key=key)

    def calls_for(self, role: str) -> list[dict[str, Any]]:
        """Return the sorted call digests of one role."""
        return [call for call in self.sorted_calls() if call["role"] == role]


# ============================================================================
# Fake OpenAlex
# ============================================================================
_OPENALEX_BASE = "https://api.openalex.org"
_DOI_PREFIX = f"{_OPENALEX_BASE}/works/https://doi.org/"


@dataclass(frozen=True)
class OpenAlexFixture:
    """One recorded OpenAlex answer and the request it answers."""

    name: str
    match: Mapping[str, str]
    status: int
    body: Any


def load_openalex_fixtures(folder: Path = OPENALEX_FIXTURES) -> list[OpenAlexFixture]:
    """Load every ``*.json`` fixture in *folder*, sorted by file name."""
    fixtures = []
    for path in sorted(folder.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        fixtures.append(
            OpenAlexFixture(
                name=path.stem,
                match=data["match"],
                status=int(data.get("status", 200)),
                body=data["body"],
            )
        )
    return fixtures


class FakeResponse:
    """The subset of ``requests.Response`` that ``rendering.citations`` reads."""

    def __init__(self, url: str, status_code: int, body: Any) -> None:
        self.url = url
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self._body = body

    def json(self) -> Any:
        return copy.deepcopy(self._body)


class FakeOpenAlex:
    """Answer OpenAlex GET requests from fixtures; never opens a socket.

    A DOI lookup without a fixture answers 404; a search without a fixture
    answers an empty result list. ``requests`` records every request without
    the ``mailto`` and ``api_key`` parameters.
    """

    def __init__(self, fixtures: list[OpenAlexFixture] | None = None) -> None:
        self.fixtures = load_openalex_fixtures() if fixtures is None else list(fixtures)
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def get(
        self, url: str, params: Mapping[str, Any] | None = None, **_: Any
    ) -> FakeResponse:
        """Serve one GET request."""
        params = dict(params or {})
        recorded = {
            key: value
            for key, value in params.items()
            if key not in ("mailto", "api_key")
        }
        fixture, kind = self._match(url, params)
        status = fixture.status if fixture else (404 if kind == "doi" else 200)
        body: Any
        if fixture is not None:
            body = fixture.body
        elif kind == "doi":
            body = {"error": "not found"}
        else:
            body = {"meta": {"count": 0}, "results": []}
        with self._lock:
            self.requests.append(
                {
                    "url": url,
                    "params": _jsonable(recorded),
                    "fixture": fixture.name if fixture else None,
                    "status": status,
                }
            )
        return FakeResponse(url, status, body)

    def _match(
        self, url: str, params: Mapping[str, Any]
    ) -> tuple[OpenAlexFixture | None, str]:
        if url.startswith(_DOI_PREFIX):
            doi = unquote(url[len(_DOI_PREFIX) :]).casefold()
            for fixture in self.fixtures:
                if str(fixture.match.get("doi", "")).casefold() == doi:
                    return fixture, "doi"
            return None, "doi"
        filters = str(params.get("filter", "")).casefold()
        search = str(params.get("search", "")).casefold()
        for fixture in self.fixtures:
            title = str(fixture.match.get("title_search", "")).casefold()
            if title and f"title.search:{title}" in filters:
                return fixture, "search"
            query = str(fixture.match.get("search", "")).casefold()
            if query and query in search:
                return fixture, "search"
        return None, "search"

    def session(self) -> _FakeSession:
        """Return a session object bound to this service."""
        return _FakeSession(self)


class _FakeSession:
    """Context-managed stand-in for ``requests.Session``."""

    def __init__(self, service: FakeOpenAlex) -> None:
        self._service = service

    def __enter__(self) -> _FakeSession:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        return self._service.get(url, **kwargs)

    def close(self) -> None:
        """Nothing to release."""
