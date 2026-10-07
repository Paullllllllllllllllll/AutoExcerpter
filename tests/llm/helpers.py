"""Builders for LLM-layer tests: scripted chat models, phases and call envs.

``install`` replaces the client factory used by ``ModelCaller`` with one that
returns a :class:`ScriptedChatModel`, so managers run their real request
assembly, structured-output routing, retry ladder and parsing offline.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterable
from typing import Any

import pytest
from langchain_core.messages import AIMessage

import autoexcerpter.llm.caller as caller_module
from autoexcerpter import ext
from autoexcerpter.common.rate_limit import RateLimiterRegistry
from autoexcerpter.common.retry import Decision
from autoexcerpter.common.testing import FakeRequest, ScriptedChatModel
from autoexcerpter.common.usage import Usage
from autoexcerpter.events import RunEvents
from autoexcerpter.llm import CallEnv, PhaseModel
from autoexcerpter.llm.client import default_key_env
from autoexcerpter.providers import infer_provider
from autoexcerpter.settings import RetryRule, RetrySettings, Settings

USAGE = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}

TRANSCRIPTION_ANSWER: dict[str, Any] = {
    "image_analysis": "Single column.",
    "transcription": "Bread prices rose.",
    "no_transcribable_text": False,
    "transcription_not_possible": False,
}

SUMMARY_ANSWER: dict[str, Any] = {
    "page_information": {
        "page_number_integer": 3,
        "is_two_page_spread": False,
        "page_number_integer_end": None,
        "page_number_type": "arabic",
        "page_types": ["content"],
    },
    "bullet_points": ["Bread prices rose."],
    "references": None,
}


def message(content: str = "", **fields: Any) -> AIMessage:
    """Return an answer message with the default usage."""
    fields.setdefault("usage_metadata", dict(USAGE))
    return AIMessage(content=content, **fields)


class Script:
    """Answers handed out in order; records every request.

    An answer is a string (message content), a dict (sent as JSON), an
    ``AIMessage``, an exception to raise, or a callable that turns the
    request into one of those. The last answer repeats.
    """

    def __init__(self, answers: Iterable[Any]) -> None:
        self.answers = list(answers)
        self.requests: list[FakeRequest] = []
        self.constructions: list[dict[str, Any]] = []

    def record(self, request: FakeRequest) -> None:
        self.requests.append(request)

    def respond(self, request: FakeRequest) -> AIMessage:
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if callable(answer) and not isinstance(answer, type):
            answer = answer(request)
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, AIMessage):
            return answer
        if isinstance(answer, dict):
            return message(json.dumps(answer))
        return message(str(answer))

    def model(self, **kwargs: Any) -> ScriptedChatModel:
        self.constructions.append(kwargs)
        return ScriptedChatModel(responder=self.respond, recorder=self.record)


def install(monkeypatch: pytest.MonkeyPatch, answers: Iterable[Any]) -> Script:
    """Make every client built by ``ModelCaller`` answer from *answers*."""
    script = Script(answers)

    def build(phase: PhaseModel, timeouts: Any, **kwargs: Any) -> ScriptedChatModel:
        return script.model(phase=phase, **kwargs)

    monkeypatch.setattr(caller_module, "build_chat_model", build)
    return script


class StatusError(Exception):
    """A provider error with an HTTP status and, optionally, usage in its body."""

    def __init__(self, status: int, usage: dict[str, int] | None = None) -> None:
        super().__init__(f"status {status}")
        self.status_code = status
        self.body = {"usage": usage} if usage else None


class RecordingExt:
    """Wrap ``ext.attempt`` and record attempts, commits and ``on_error`` calls.

    ``attempts`` holds the attempts that were entered, ``opened`` every
    attempt, including those only built. *decision*, when set, replaces the
    decision ``on_error`` returns.
    """

    def __init__(self, decision: Decision | None = None) -> None:
        self.decision = decision
        self.opened: list[ext.Attempt[Any]] = []
        self.attempts: list[ext.Attempt[Any]] = []
        self.commits: list[Usage | None] = []
        self.errors: list[BaseException] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> RecordingExt:
        real = ext.attempt

        def attempt(
            role: str, provider: str, estimate: int | None = None, **kwargs: Any
        ) -> ext.Attempt[Any]:
            opened: ext.Attempt[Any] = real(role, provider, estimate, **kwargs)
            commit, on_error, refresh = opened.commit, opened.on_error, opened.refresh

            async def recording_refresh() -> Any:
                if opened not in self.attempts:
                    self.attempts.append(opened)
                return await refresh()

            def recording_commit(usage: Usage | None) -> None:
                self.commits.append(usage)
                commit(usage)

            def recording_on_error(exc: BaseException) -> Decision:
                self.errors.append(exc)
                decision = on_error(exc)
                return self.decision or decision

            opened.commit = recording_commit  # type: ignore[method-assign]
            opened.on_error = recording_on_error  # type: ignore[method-assign]
            opened.refresh = recording_refresh  # type: ignore[method-assign]
            self.opened.append(opened)
            return opened

        monkeypatch.setattr(ext, "attempt", attempt)
        return self

    @property
    def roles(self) -> list[str]:
        """The role of every attempt, in order."""
        return [attempt.identity.role for attempt in self.attempts]


def phase(
    name: str = "gpt-5.6-terra", provider: str | None = "openai", **fields: Any
) -> PhaseModel:
    """Return a phase model with the provider's default key variable.

    *provider* None takes the provider the model name implies.
    """
    resolved = provider or infer_provider(name)
    fields.setdefault("api_key_env", default_key_env(resolved))
    return PhaseModel(name, resolved, **fields)


def rules(phase_name: str, **changes: tuple[bool, int, float, float]) -> RetrySettings:
    """Return the retry settings with some output rules of a phase replaced."""
    base = RetrySettings()
    merged = dict(base.schema_retries)
    phase_rules = dict(merged[phase_name])
    for name, (enabled, attempts, start, factor) in changes.items():
        phase_rules[name] = RetryRule(enabled, attempts, start, factor)
    merged[phase_name] = phase_rules
    return dataclasses.replace(base, schema_retries=merged)


def env(
    retry: RetrySettings | None = None,
    *,
    clock: Callable[[], float] | None = None,
    events: RunEvents | None = None,
    **settings: Any,
) -> CallEnv:
    """Return a call env with wide rate limits.

    *clock* drives the rate limiters, so their error penalty ends in fake
    time when the test also fakes the sleeps.
    """
    values: dict[str, Any] = {"rate_limits": ((100_000, 1),)}
    if retry is not None:
        values["retry"] = retry
    values.update(settings)
    created = CallEnv.create(dataclasses.replace(Settings(), **values), events)
    if clock is None:
        return created
    limiters = RateLimiterRegistry(created.settings.rate_limits, clock=clock)
    return dataclasses.replace(created, limiters=limiters)


__all__ = [
    "SUMMARY_ANSWER",
    "TRANSCRIPTION_ANSWER",
    "USAGE",
    "RecordingExt",
    "Script",
    "StatusError",
    "env",
    "install",
    "message",
    "phase",
    "rules",
]
