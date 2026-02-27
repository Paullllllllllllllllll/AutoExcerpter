"""Extended tests for modules/concurrency_helper.py - coverage gap filling.

Covers:
- get_api_concurrency: missing api_type in config, empty config dict
- get_transcription_concurrency: delegates to get_api_concurrency
- get_summary_concurrency: delegates to get_api_concurrency
- get_image_processing_concurrency: missing config, error handling
- get_service_tier: tier present, tier missing, error branch
- get_api_timeout: missing key, error branch
- get_rate_limits: valid list, invalid items, empty list, error branch
- get_target_dpi: normal, missing key, error branch
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import modules.concurrency_helper as ch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mock_loader(concurrency_cfg=None, image_cfg=None):
    """Build a mock config loader returning the given configs."""
    loader = MagicMock()
    if concurrency_cfg is not None:
        loader.get_concurrency_config.return_value = concurrency_cfg
    else:
        loader.get_concurrency_config.return_value = {}
    if image_cfg is not None:
        loader.get_image_processing_config.return_value = image_cfg
    else:
        loader.get_image_processing_config.return_value = {}
    return loader


# ============================================================================
# get_api_concurrency
# ============================================================================
class TestGetApiConcurrency:
    """Tests for get_api_concurrency()."""

    def test_missing_api_type_falls_back_to_defaults(self, monkeypatch):
        """When the requested api_type is absent, defaults are returned."""
        cfg = {"api_requests": {}}
        monkeypatch.setattr(
            ch, "get_config_loader", lambda: _mock_loader(concurrency_cfg=cfg)
        )

        workers, delay = ch.get_api_concurrency("transcription")
        assert workers == ch.DEFAULT_CONCURRENT_REQUESTS
        assert delay == 0.05

    def test_completely_empty_config(self, monkeypatch):
        """Empty concurrency config returns defaults."""
        monkeypatch.setattr(
            ch, "get_config_loader", lambda: _mock_loader(concurrency_cfg={})
        )

        workers, delay = ch.get_api_concurrency("transcription")
        assert workers == ch.DEFAULT_CONCURRENT_REQUESTS
        assert delay == 0.05

    def test_summary_api_type(self, monkeypatch):
        """Correctly reads 'summary' api_type settings."""
        cfg = {
            "api_requests": {
                "summary": {"concurrency_limit": 20, "delay_between_tasks": 0.1},
            }
        }
        monkeypatch.setattr(
            ch, "get_config_loader", lambda: _mock_loader(concurrency_cfg=cfg)
        )

        workers, delay = ch.get_api_concurrency("summary")
        assert workers == 20
        assert delay == 0.1


# ============================================================================
# get_transcription_concurrency / get_summary_concurrency
# ============================================================================
class TestConvenienceWrappers:
    """Tests for transcription/summary concurrency convenience functions."""

    def test_get_transcription_concurrency(self, monkeypatch):
        """get_transcription_concurrency delegates to get_api_concurrency('transcription')."""
        cfg = {
            "api_requests": {
                "transcription": {"concurrency_limit": 8, "delay_between_tasks": 0.02},
            }
        }
        monkeypatch.setattr(
            ch, "get_config_loader", lambda: _mock_loader(concurrency_cfg=cfg)
        )

        workers, delay = ch.get_transcription_concurrency()
        assert workers == 8
        assert delay == 0.02

    def test_get_summary_concurrency(self, monkeypatch):
        """get_summary_concurrency delegates to get_api_concurrency('summary')."""
        cfg = {
            "api_requests": {
                "summary": {"concurrency_limit": 12, "delay_between_tasks": 0.03},
            }
        }
        monkeypatch.setattr(
            ch, "get_config_loader", lambda: _mock_loader(concurrency_cfg=cfg)
        )

        workers, delay = ch.get_summary_concurrency()
        assert workers == 12
        assert delay == 0.03

    def test_get_transcription_concurrency_error_fallback(self, monkeypatch):
        """get_transcription_concurrency returns defaults on error."""
        loader = MagicMock()
        loader.get_concurrency_config.side_effect = RuntimeError("fail")
        monkeypatch.setattr(ch, "get_config_loader", lambda: loader)

        workers, delay = ch.get_transcription_concurrency()
        assert workers == ch.DEFAULT_CONCURRENT_REQUESTS
        assert delay == 0.05


# ============================================================================
# get_image_processing_concurrency
# ============================================================================
class TestGetImageProcessingConcurrency:
    """Tests for get_image_processing_concurrency()."""

    def test_reads_config(self, monkeypatch):
        """Reads concurrency_limit and delay from image_processing section."""
        cfg = {
            "image_processing": {"concurrency_limit": 16, "delay_between_tasks": 0.01}
        }
        monkeypatch.setattr(
            ch, "get_config_loader", lambda: _mock_loader(concurrency_cfg=cfg)
        )

        workers, delay = ch.get_image_processing_concurrency()
        assert workers == 16
        assert delay == 0.01

    def test_missing_image_processing_section(self, monkeypatch):
        """Returns defaults when image_processing section is absent."""
        monkeypatch.setattr(
            ch, "get_config_loader", lambda: _mock_loader(concurrency_cfg={})
        )

        workers, delay = ch.get_image_processing_concurrency()
        assert workers == 24
        assert delay == 0

    def test_error_fallback(self, monkeypatch):
        """Returns defaults on exception."""
        loader = MagicMock()
        loader.get_concurrency_config.side_effect = ValueError("broken")
        monkeypatch.setattr(ch, "get_config_loader", lambda: loader)

        workers, delay = ch.get_image_processing_concurrency()
        assert workers == 24
        assert delay == 0


# ============================================================================
# get_service_tier
# ============================================================================
class TestGetServiceTier:
    """Tests for get_service_tier()."""

    def test_tier_present(self, monkeypatch):
        """Returns the configured service tier."""
        cfg = {
            "api_requests": {
                "transcription": {"service_tier": "priority"},
            }
        }
        monkeypatch.setattr(
            ch, "get_config_loader", lambda: _mock_loader(concurrency_cfg=cfg)
        )

        assert ch.get_service_tier("transcription") == "priority"

    def test_tier_missing_returns_flex(self, monkeypatch):
        """Returns 'flex' when service_tier is not specified."""
        cfg = {"api_requests": {"transcription": {}}}
        monkeypatch.setattr(
            ch, "get_config_loader", lambda: _mock_loader(concurrency_cfg=cfg)
        )

        assert ch.get_service_tier("transcription") == "flex"

    def test_tier_none_returns_flex(self, monkeypatch):
        """Returns 'flex' when service_tier is explicitly None."""
        cfg = {"api_requests": {"transcription": {"service_tier": None}}}
        monkeypatch.setattr(
            ch, "get_config_loader", lambda: _mock_loader(concurrency_cfg=cfg)
        )

        assert ch.get_service_tier("transcription") == "flex"

    def test_tier_empty_string_returns_flex(self, monkeypatch):
        """Returns 'flex' when service_tier is an empty string."""
        cfg = {"api_requests": {"transcription": {"service_tier": ""}}}
        monkeypatch.setattr(
            ch, "get_config_loader", lambda: _mock_loader(concurrency_cfg=cfg)
        )

        assert ch.get_service_tier("transcription") == "flex"

    def test_error_returns_flex(self, monkeypatch):
        """Returns 'flex' on exception."""
        loader = MagicMock()
        loader.get_concurrency_config.side_effect = RuntimeError("fail")
        monkeypatch.setattr(ch, "get_config_loader", lambda: loader)

        assert ch.get_service_tier("transcription") == "flex"

    def test_summary_tier(self, monkeypatch):
        """Reads service tier for 'summary' api_type."""
        cfg = {
            "api_requests": {
                "summary": {"service_tier": "auto"},
            }
        }
        monkeypatch.setattr(
            ch, "get_config_loader", lambda: _mock_loader(concurrency_cfg=cfg)
        )

        assert ch.get_service_tier("summary") == "auto"


# ============================================================================
# get_api_timeout
# ============================================================================
class TestGetApiTimeout:
    """Tests for get_api_timeout()."""

    def test_reads_timeout_from_config(self, monkeypatch):
        """Returns the configured timeout value."""
        cfg = {"api_requests": {"api_timeout": 600}}
        monkeypatch.setattr(
            ch, "get_config_loader", lambda: _mock_loader(concurrency_cfg=cfg)
        )

        assert ch.get_api_timeout() == 600

    def test_missing_timeout_returns_default(self, monkeypatch):
        """Returns 900 when api_timeout is not in config."""
        cfg = {"api_requests": {}}
        monkeypatch.setattr(
            ch, "get_config_loader", lambda: _mock_loader(concurrency_cfg=cfg)
        )

        assert ch.get_api_timeout() == 900

    def test_error_returns_default(self, monkeypatch):
        """Returns 900 on exception."""
        loader = MagicMock()
        loader.get_concurrency_config.side_effect = RuntimeError("fail")
        monkeypatch.setattr(ch, "get_config_loader", lambda: loader)

        assert ch.get_api_timeout() == 900

    def test_string_timeout_converted_to_int(self, monkeypatch):
        """String timeout values are converted to int."""
        cfg = {"api_requests": {"api_timeout": "450"}}
        monkeypatch.setattr(
            ch, "get_config_loader", lambda: _mock_loader(concurrency_cfg=cfg)
        )

        assert ch.get_api_timeout() == 450


# ============================================================================
# get_rate_limits
# ============================================================================
class TestGetRateLimits:
    """Tests for get_rate_limits()."""

    def test_valid_limits_parsed(self, monkeypatch):
        """Valid rate limits are parsed into tuples of ints."""
        cfg = {"api_requests": {"rate_limits": [[100, 1], [5000, 60]]}}
        monkeypatch.setattr(
            ch, "get_config_loader", lambda: _mock_loader(concurrency_cfg=cfg)
        )

        limits = ch.get_rate_limits()
        assert limits == [(100, 1), (5000, 60)]

    def test_invalid_items_skipped(self, monkeypatch):
        """Invalid items in the list are skipped."""
        cfg = {"api_requests": {"rate_limits": [[100, 1], "bad", [3], [200, 2]]}}
        monkeypatch.setattr(
            ch, "get_config_loader", lambda: _mock_loader(concurrency_cfg=cfg)
        )

        limits = ch.get_rate_limits()
        assert (100, 1) in limits
        assert (200, 2) in limits

    def test_empty_list_returns_default(self, monkeypatch):
        """Empty rate_limits list returns the default limits."""
        cfg = {"api_requests": {"rate_limits": []}}
        monkeypatch.setattr(
            ch, "get_config_loader", lambda: _mock_loader(concurrency_cfg=cfg)
        )

        limits = ch.get_rate_limits()
        # Default limits from the module
        assert len(limits) == 3

    def test_all_invalid_items_returns_default(self, monkeypatch):
        """When all items are invalid, returns the default limits."""
        cfg = {"api_requests": {"rate_limits": [["bad", "worse"], "string"]}}
        monkeypatch.setattr(
            ch, "get_config_loader", lambda: _mock_loader(concurrency_cfg=cfg)
        )

        limits = ch.get_rate_limits()
        assert len(limits) == 3  # default

    def test_not_a_list_returns_default(self, monkeypatch):
        """Non-list rate_limits returns the default limits."""
        cfg = {"api_requests": {"rate_limits": "not_a_list"}}
        monkeypatch.setattr(
            ch, "get_config_loader", lambda: _mock_loader(concurrency_cfg=cfg)
        )

        limits = ch.get_rate_limits()
        assert isinstance(limits, list)
        assert len(limits) == 3

    def test_missing_rate_limits_returns_default(self, monkeypatch):
        """Missing rate_limits key returns the default limits."""
        cfg = {"api_requests": {}}
        monkeypatch.setattr(
            ch, "get_config_loader", lambda: _mock_loader(concurrency_cfg=cfg)
        )

        limits = ch.get_rate_limits()
        assert isinstance(limits, list)
        assert len(limits) == 3

    def test_error_returns_default(self, monkeypatch):
        """Exception returns the default limits."""
        loader = MagicMock()
        loader.get_concurrency_config.side_effect = RuntimeError("fail")
        monkeypatch.setattr(ch, "get_config_loader", lambda: loader)

        limits = ch.get_rate_limits()
        assert isinstance(limits, list)
        assert len(limits) == 3

    def test_tuple_items_accepted(self, monkeypatch):
        """Tuple items in the list are also accepted."""
        cfg = {"api_requests": {"rate_limits": [(50, 1), (1000, 60)]}}
        monkeypatch.setattr(
            ch, "get_config_loader", lambda: _mock_loader(concurrency_cfg=cfg)
        )

        limits = ch.get_rate_limits()
        assert limits == [(50, 1), (1000, 60)]


# ============================================================================
# get_target_dpi
# ============================================================================
class TestGetTargetDpi:
    """Tests for get_target_dpi()."""

    def test_reads_dpi_from_config(self, monkeypatch):
        """Returns the configured target DPI."""
        img_cfg = {"api_image_processing": {"target_dpi": 600}}
        monkeypatch.setattr(
            ch, "get_config_loader", lambda: _mock_loader(image_cfg=img_cfg)
        )

        assert ch.get_target_dpi() == 600

    def test_missing_target_dpi_returns_default(self, monkeypatch):
        """Returns 300 when target_dpi is not in config."""
        img_cfg = {"api_image_processing": {}}
        monkeypatch.setattr(
            ch, "get_config_loader", lambda: _mock_loader(image_cfg=img_cfg)
        )

        assert ch.get_target_dpi() == 300

    def test_missing_section_returns_default(self, monkeypatch):
        """Returns 300 when api_image_processing section is missing."""
        monkeypatch.setattr(ch, "get_config_loader", lambda: _mock_loader(image_cfg={}))

        assert ch.get_target_dpi() == 300

    def test_error_returns_default(self, monkeypatch):
        """Returns 300 on exception."""
        loader = MagicMock()
        loader.get_image_processing_config.side_effect = RuntimeError("fail")
        monkeypatch.setattr(ch, "get_config_loader", lambda: loader)

        assert ch.get_target_dpi() == 300

    def test_string_dpi_converted_to_int(self, monkeypatch):
        """String DPI values are converted to int."""
        img_cfg = {"api_image_processing": {"target_dpi": "450"}}
        monkeypatch.setattr(
            ch, "get_config_loader", lambda: _mock_loader(image_cfg=img_cfg)
        )

        assert ch.get_target_dpi() == 450
