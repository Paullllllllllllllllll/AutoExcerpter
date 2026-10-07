"""Capability edge cases the recorded family table cannot express.

Per-model values live in ``data/capability_expectations.json`` and are checked
by test_capability_families.py.
"""

import pytest

from autoexcerpter.llm.capabilities import (
    CapabilityError,
    ProviderCapabilities,
    detect_capabilities,
    ensure_image_support,
)


@pytest.mark.parametrize("model", ["gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.4-pro"])
def test_longer_prefix_wins_over_shorter_family(model: str) -> None:
    """A variant id matches its own family, not the 'gpt-5.4' prefix."""
    assert detect_capabilities(model).family == model
    assert detect_capabilities("gpt-5.4").family == "gpt-5.4"


class TestCapabilityError:
    """Test CapabilityError and ensure_image_support."""

    def test_ensure_image_support_raises(self) -> None:
        caps = ProviderCapabilities(
            provider_name="openai",
            model_name="text-only",
            supports_vision=False,
        )
        with pytest.raises(CapabilityError, match="does not support image inputs"):
            ensure_image_support("text-only", caps)

    def test_ensure_image_support_passes(self) -> None:
        caps = ProviderCapabilities(
            provider_name="openai",
            model_name="gpt-4o",
            supports_vision=True,
        )
        ensure_image_support("gpt-4o", caps)
