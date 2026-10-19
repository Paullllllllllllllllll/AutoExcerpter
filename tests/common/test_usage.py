"""Tests for common/usage.py: usage values, extraction and per-role totals."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, PropertyMock

import pytest
from langchain_core.messages import AIMessage

from autoexcerpter.common.usage import (
    Usage,
    UsageTotals,
    coerce_token_count,
    usage_from_exception,
    usage_from_response,
    usage_metadata_folds_cache,
)


def _total(usage_metadata: dict[str, Any]) -> int | None:
    usage = usage_from_response(SimpleNamespace(usage_metadata=usage_metadata))
    return None if usage is None else usage.total_tokens


class TestUsageValue:
    def test_addition(self) -> None:
        first = Usage(10, 5, 15, 2, 1)
        second = Usage(1, 2, 3, 0, 4)
        assert first + second == Usage(11, 7, 18, 2, 5)

    def test_empty_and_dict(self) -> None:
        assert Usage().is_empty
        assert not Usage(total_tokens=1).is_empty
        assert Usage(1, 2, 3).as_dict() == {
            "input_tokens": 1,
            "output_tokens": 2,
            "total_tokens": 3,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
        }


class TestCoerceTokenCount:
    def test_values(self) -> None:
        assert coerce_token_count(123.0) == 123
        assert coerce_token_count(122.4) == 122
        assert coerce_token_count(50) == 50
        assert coerce_token_count(True) is None
        assert coerce_token_count("x") is None
        assert coerce_token_count(None) is None


class TestUsageFromResponse:
    @pytest.mark.parametrize("metadata", [None, "not a dict", {}])
    def test_missing_metadata(self, metadata: Any) -> None:
        response = MagicMock()
        response.usage_metadata = metadata
        assert usage_from_response(response) is None

    def test_total_tokens_preferred(self) -> None:
        assert _total(
            {"total_tokens": 200, "input_tokens": 100, "output_tokens": 50}
        ) == (200)

    def test_input_output_fallback(self) -> None:
        assert _total({"input_tokens": 100, "output_tokens": 50}) == 150

    def test_float_counts(self) -> None:
        assert _total({"total_tokens": 123.0}) == 123
        assert _total({"input_tokens": 100.0, "output_tokens": 50.0}) == 150

    def test_no_usable_counts(self) -> None:
        assert _total({"model": "gpt-5"}) is None
        assert (
            _total({"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}) is None
        )

    def test_raw_anthropic_shape_adds_cache_at_full_weight(self) -> None:
        usage = usage_from_response(
            SimpleNamespace(
                usage_metadata={
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_read_input_tokens": 800,
                    "cache_creation_input_tokens": 100,
                }
            )
        )
        assert usage == Usage(
            input_tokens=1000, output_tokens=50, total_tokens=1050, cached_tokens=800
        )

    def test_raw_anthropic_cache_writes_are_not_cached(self) -> None:
        usage = usage_from_response(
            SimpleNamespace(
                usage_metadata={
                    "input_tokens": 9,
                    "output_tokens": 50,
                    "cache_creation_input_tokens": 63_378,
                }
            )
        )
        assert usage == Usage(
            input_tokens=63_387, output_tokens=50, total_tokens=63_437, cached_tokens=0
        )

    @pytest.mark.parametrize("prefix", ["", "flex_", "priority_"])
    def test_langchain_cache_writes_are_not_cached(self, prefix: str) -> None:
        metadata = {
            "input_tokens": 63_387,
            "output_tokens": 20,
            "total_tokens": 63_407,
            "input_token_details": {
                f"{prefix}cache_read": 0,
                f"{prefix}cache_creation": 63_378,
            },
        }
        usage = usage_from_response(SimpleNamespace(usage_metadata=metadata))
        assert usage == Usage(63_387, 20, 63_407, cached_tokens=0)

    def test_langchain_normalized_shape_no_addition(self) -> None:
        metadata = {
            "input_tokens": 1000,
            "output_tokens": 200,
            "total_tokens": 1200,
            "input_token_details": {"cache_read": 800, "cache_creation": 150},
        }
        assert usage_metadata_folds_cache(metadata)
        usage = usage_from_response(SimpleNamespace(usage_metadata=metadata))
        assert usage is not None
        assert usage.total_tokens == 1200
        assert usage.cached_tokens == 800

    @pytest.mark.parametrize(
        ("input_details", "output_details"),
        [
            ({"cache_read": 10}, {"reasoning": 5}),
            ({"flex_cache_read": 10, "flex": 90}, {"flex_reasoning": 5, "flex": 15}),
            (
                {"priority_cache_read": 10, "priority": 90},
                {"priority_reasoning": 5, "priority": 15},
            ),
        ],
        ids=["default", "flex", "priority"],
    )
    def test_service_tier_prefixed_details(
        self, input_details: dict[str, int], output_details: dict[str, int]
    ) -> None:
        """langchain-openai prefixes details with a non-default service tier."""
        metadata = {
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "input_token_details": input_details,
            "output_token_details": output_details,
        }
        usage = usage_from_response(SimpleNamespace(usage_metadata=metadata))
        assert usage == Usage(100, 20, 120, cached_tokens=10, reasoning_tokens=5)

    def test_openai_cached_tokens_shape_no_addition(self) -> None:
        assert (
            _total(
                {
                    "input_tokens": 1500,
                    "output_tokens": 100,
                    "total_tokens": 1600,
                    "input_token_details": {"cache_read": 1200},
                }
            )
            == 1600
        )

    def test_cache_on_unwrapped_response_metadata(self) -> None:
        response = SimpleNamespace(
            usage_metadata={"input_tokens": 100, "output_tokens": 50},
            response_metadata={
                "usage": {
                    "cache_read_input_tokens": 400,
                    "cache_creation_input_tokens": 0,
                }
            },
        )
        usage = usage_from_response(response)
        assert usage is not None
        assert usage.total_tokens == 550

    def test_cache_only_usage_is_counted(self) -> None:
        assert (
            _total(
                {
                    "total_tokens": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 500,
                }
            )
            == 500
        )

    def test_reasoning_tokens(self) -> None:
        message = AIMessage(
            content="x",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 30,
                "total_tokens": 40,
                "output_token_details": {"reasoning": 20},
            },
        )
        usage = usage_from_response(message)
        assert usage is not None
        assert usage.reasoning_tokens == 20

    def test_include_raw_dict_is_unwrapped(self) -> None:
        raw = AIMessage(
            content="x",
            usage_metadata={"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        )
        usage = usage_from_response({"raw": raw, "parsed": {}})
        assert usage == Usage(3, 4, 7)


def _error(**attributes: Any) -> Exception:
    exc = Exception("API error")
    for name, value in attributes.items():
        setattr(exc, name, value)
    return exc


class TestUsageFromException:
    def test_body_total(self) -> None:
        usage = usage_from_exception(_error(body={"usage": {"total_tokens": 300}}))
        assert usage is not None
        assert usage.total_tokens == 300

    def test_body_prompt_completion(self) -> None:
        body = {"usage": {"prompt_tokens": 200, "completion_tokens": 100}}
        assert usage_from_exception(_error(body=body)) == Usage(200, 100, 300)

    def test_body_input_output(self) -> None:
        body = {"usage": {"input_tokens": 150, "output_tokens": 75}}
        assert usage_from_exception(_error(body=body)) == Usage(150, 75, 225)

    def test_float_counts(self) -> None:
        body = {"usage": {"input_tokens": 100.0, "output_tokens": 50.0}}
        usage = usage_from_exception(_error(body=body))
        assert usage is not None
        assert usage.total_tokens == 150
        usage = usage_from_exception(_error(body={"usage": {"total_tokens": 123.0}}))
        assert usage is not None
        assert usage.total_tokens == 123

    def test_response_json(self) -> None:
        response = MagicMock()
        response.json.return_value = {"usage": {"total_tokens": 500}}
        usage = usage_from_exception(_error(response=response))
        assert usage is not None
        assert usage.total_tokens == 500

    def test_response_json_failure_is_none(self) -> None:
        response = MagicMock()
        response.json.side_effect = ValueError("not json")
        assert usage_from_exception(_error(response=response)) is None

    def test_raw_anthropic_cache_added(self) -> None:
        body = {
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 100,
                "cache_creation_input_tokens": 40,
            }
        }
        assert usage_from_exception(_error(body=body)) == Usage(
            input_tokens=150, output_tokens=5, total_tokens=155, cached_tokens=100
        )

    def test_openai_details(self) -> None:
        body = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "total_tokens": 140,
                "prompt_tokens_details": {"cached_tokens": 60},
                "completion_tokens_details": {"reasoning_tokens": 30},
            }
        }
        assert usage_from_exception(_error(body=body)) == Usage(100, 40, 140, 60, 30)

    def test_attached_usage(self) -> None:
        attached = Usage(total_tokens=9)
        assert usage_from_exception(_error(usage=attached)) == attached
        assert usage_from_exception(_error(usage=Usage())) is None

    def test_discarded_total(self) -> None:
        assert usage_from_exception(_error(discarded_total=42)) == Usage(
            total_tokens=42
        )

    def test_no_usage(self) -> None:
        assert usage_from_exception(Exception("plain error")) is None
        assert usage_from_exception(_error(body={"usage": {"total_tokens": 0}})) is None

    def test_never_raises(self) -> None:
        exc = _error(body=PropertyMock(side_effect=RuntimeError("boom")))
        assert usage_from_exception(exc) is None

    def test_attribute_access_error_is_none(self) -> None:
        class Hostile(Exception):
            @property
            def body(self) -> Any:
                raise RuntimeError("boom")

        assert usage_from_exception(Hostile("x")) is None


class TestUsageTotals:
    def test_per_role_accumulation(self) -> None:
        totals = UsageTotals()
        totals.add("transcription", Usage(10, 5, 15))
        totals.add("summary", Usage(4, 1, 5))
        totals.add("transcription", Usage(1, 1, 2))
        totals.add("summary", None)

        assert totals.roles == ("transcription", "summary")
        assert totals.get("transcription") == Usage(11, 6, 17)
        assert totals.get("missing") == Usage()
        assert totals.total() == Usage(15, 7, 22)
        assert totals.as_dict()["summary"]["total_tokens"] == 5

    def test_empty(self) -> None:
        totals = UsageTotals()
        assert totals.total() == Usage()
        assert totals.as_dict() == {}
