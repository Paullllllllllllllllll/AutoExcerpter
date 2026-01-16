"""Tests for modules/concurrency_helper.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import modules.concurrency_helper as ch


class TestConcurrencyHelper:
    def test_get_api_concurrency_reads_config(self, monkeypatch):
        cfg = {
            "api_requests": {
                "transcription": {"concurrency_limit": 10, "delay_between_tasks": 0.25},
            }
        }
        loader = MagicMock()
        loader.get_concurrency_config.return_value = cfg
        monkeypatch.setattr(ch, "get_config_loader", lambda: loader)

        max_workers, delay = ch.get_api_concurrency("transcription")
        assert max_workers == 10
        assert delay == 0.25

    def test_get_api_concurrency_falls_back_on_error(self, monkeypatch):
        loader = MagicMock()
        loader.get_concurrency_config.side_effect = RuntimeError("boom")
        monkeypatch.setattr(ch, "get_config_loader", lambda: loader)

        max_workers, delay = ch.get_api_concurrency("transcription")
        assert isinstance(max_workers, int)
        assert delay == 0.05

    def test_get_image_processing_concurrency_defaults(self, monkeypatch):
        cfg = {"image_processing": {"concurrency_limit": 7, "delay_between_tasks": 0.0}}
        loader = MagicMock()
        loader.get_concurrency_config.return_value = cfg
        monkeypatch.setattr(ch, "get_config_loader", lambda: loader)

        max_workers, delay = ch.get_image_processing_concurrency()
        assert max_workers == 7
        assert delay == 0.0

    def test_get_service_tier_default_flex_when_missing(self, monkeypatch):
        cfg = {"api_requests": {"transcription": {}}}
        loader = MagicMock()
        loader.get_concurrency_config.return_value = cfg
        monkeypatch.setattr(ch, "get_config_loader", lambda: loader)

        assert ch.get_service_tier("transcription") == "flex"

    def test_get_api_timeout_reads_config(self, monkeypatch):
        cfg = {"api_requests": {"api_timeout": 123}}
        loader = MagicMock()
        loader.get_concurrency_config.return_value = cfg
        monkeypatch.setattr(ch, "get_config_loader", lambda: loader)

        assert ch.get_api_timeout() == 123

    def test_get_rate_limits_parsing(self, monkeypatch):
        cfg = {"api_requests": {"rate_limits": [[120, 1], ["15000", "60"], ["bad"]]}}
        loader = MagicMock()
        loader.get_concurrency_config.return_value = cfg
        monkeypatch.setattr(ch, "get_config_loader", lambda: loader)

        limits = ch.get_rate_limits()
        assert limits[0] == (120, 1)
        assert (15000, 60) in limits

    def test_get_rate_limits_default_on_invalid(self, monkeypatch):
        cfg = {"api_requests": {"rate_limits": "not-a-list"}}
        loader = MagicMock()
        loader.get_concurrency_config.return_value = cfg
        monkeypatch.setattr(ch, "get_config_loader", lambda: loader)

        limits = ch.get_rate_limits()
        assert isinstance(limits, list)
        assert all(isinstance(x, tuple) and len(x) == 2 for x in limits)
