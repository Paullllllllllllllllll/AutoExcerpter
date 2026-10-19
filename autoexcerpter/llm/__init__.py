"""The LLM layer: client factory, model calls and the two page roles.

- ``TranscriptionManager`` and ``SummaryManager``: one page in, one result
  entry out (async).
- ``CallEnv`` and ``ModelCaller``: per-run collaborators and the structured
  call with its retry ladder and attempt contexts; ``collect_usage``: the
  usage of the calls made in one block, per role.
- ``PhaseModel``: the model of one phase.
- ``build_chat_model``, ``get_chat_model``, ``LLMConfig``: the client factory.
- ``detect_capabilities``, ``detect_provider``: the capability table.
"""

from autoexcerpter.llm.caller import CallEnv, ModelCaller, collect_usage
from autoexcerpter.llm.capabilities import (
    CapabilityError,
    ProviderCapabilities,
    detect_capabilities,
    detect_provider,
    ensure_image_support,
)
from autoexcerpter.llm.client import LLMConfig, build_chat_model, get_chat_model
from autoexcerpter.llm.summary import SummaryManager
from autoexcerpter.llm.transcription import TranscriptionManager
from autoexcerpter.llm.types import PhaseModel

__all__ = [
    "CallEnv",
    "CapabilityError",
    "LLMConfig",
    "ModelCaller",
    "PhaseModel",
    "ProviderCapabilities",
    "SummaryManager",
    "TranscriptionManager",
    "build_chat_model",
    "collect_usage",
    "detect_capabilities",
    "detect_provider",
    "ensure_image_support",
    "get_chat_model",
]
