# Release Notes - AutoExcerpter v2.0

**Release Date:** November 2025

---

## Overview

AutoExcerpter v2.0 marks a major architectural shift from single-provider OpenAI-only support to a **multi-provider LLM architecture** powered by **LangChain**. This release enables users to choose from OpenAI, Anthropic, Google, and OpenRouter providers while maintaining full backward compatibility with existing configurations.

---

## Highlights

- **Multi-Provider Support**: Switch between OpenAI (GPT-5.1, GPT-5, GPT-4, o-series), Anthropic (Claude 4.5, Claude 4, Claude 3.x), Google (Gemini 3, Gemini 2.5), and OpenRouter with configuration changes
- **LangChain Integration**: Unified interface for all providers with built-in retry logic and exponential backoff
- **Capability Guarding**: Automatic parameter filtering prevents API errors from unsupported model parameters
- **Simplified Retry System**: LangChain handles API-level retries automatically; schema-specific retries remain configurable
- **Python 3.10+ Required**: Minimum Python version updated for LangChain v1.0 compatibility

---

## New Features

### Multi-Provider Architecture

AutoExcerpter now supports multiple LLM providers through LangChain's unified interface:

| Provider | Models | Package |
|----------|--------|---------|
| **OpenAI** | GPT-5.1, GPT-5, GPT-4.1, GPT-4o, o-series | `langchain-openai` |
| **Anthropic** | Claude 4.5, Claude 4, Claude 3.x | `langchain-anthropic` |
| **Google** | Gemini 3, Gemini 2.5, Gemini 2.0, Gemini 1.5 | `langchain-google-genai` |
| **OpenRouter** | All providers via unified API | `langchain-openai` |

**Provider Auto-Detection**: The system automatically infers the provider from model names:
- Models starting with `gpt-`, `o1`, `o3`, `o4` → OpenAI
- Models starting with `claude-` → Anthropic
- Models starting with `gemini-` → Google

### Model Capability Profiles

New `MODEL_CAPABILITIES` dictionary in `api/llm_client.py` defines supported parameters for each model family:

**Tracked Capabilities:**
- `reasoning` - Supports reasoning_effort parameter (GPT-5, o-series)
- `text_verbosity` - Supports text.verbosity parameter (GPT-5)
- `temperature` - Supports temperature control (most models; o-series excluded)
- `max_tokens` - Supports output length control
- `structured_output` - Supports JSON schema output
- `multimodal` - Supports image input
- `thinking` - Supports thinking parameter (Gemini)
- `extended_thinking` - Supports extended thinking (Claude)

The `get_model_capabilities()` function matches model names to profiles using prefix matching, returning conservative defaults for unknown models.

### LangChain Built-in Retry

API-level retries are now handled by LangChain's built-in retry mechanism:
- Configured via `max_retries` parameter (default: 5)
- Automatic exponential backoff with jitter
- Handles rate limits, timeouts, and server errors

**Schema-specific retries** remain configurable for content flags:
- **Transcription**: `no_transcribable_text`, `transcription_not_possible`
- **Summary**: `contains_no_semantic_content`, `contains_no_page_number`

### Provider Configuration

Configure your preferred provider in `modules/config/app.yaml`:

```yaml
# OpenAI (default)
openai:
  model: 'gpt-5-mini'
  transcription_model: 'gpt-5-mini'
  provider: 'openai'  # Optional - auto-detected

# Anthropic Claude
openai:
  model: 'claude-sonnet-4-5'
  transcription_model: 'claude-sonnet-4-5'
  provider: 'anthropic'

# Google Gemini
openai:
  model: 'gemini-2.5-pro'
  transcription_model: 'gemini-2.5-pro'
  provider: 'google'

# OpenRouter
openai:
  model: 'anthropic/claude-3-opus'
  transcription_model: 'anthropic/claude-3-opus'
  provider: 'openrouter'
```

---

## Supported Models (November 2025)

### OpenAI

| Model Family | Models | Capabilities |
|--------------|--------|--------------|
| GPT-5.1 | gpt-5.1, gpt-5.1-instant, gpt-5.1-thinking | Reasoning, text verbosity, multimodal |
| GPT-5 | gpt-5, gpt-5-mini, gpt-5-nano | Reasoning, text verbosity, multimodal |
| O-series | o4, o4-mini, o3, o3-mini, o1, o1-mini | Reasoning (no temperature control) |
| GPT-4.1 | gpt-4.1, gpt-4.1-mini, gpt-4.1-nano | Multimodal |
| GPT-4o | gpt-4o, gpt-4o-mini | Multimodal |

### Anthropic Claude

| Model Family | Models | Capabilities |
|--------------|--------|--------------|
| Claude 4.5 | claude-opus-4-5, claude-sonnet-4-5, claude-haiku-4-5 | Multimodal, extended thinking (opus/sonnet) |
| Claude 4 | claude-opus-4, claude-sonnet-4 | Multimodal, extended thinking |
| Claude 3.x | claude-3-7-sonnet, claude-3-5-sonnet, claude-3-5-haiku | Multimodal |

### Google Gemini

| Model Family | Models | Capabilities |
|--------------|--------|--------------|
| Gemini 3 | gemini-3-pro | Thinking, multimodal |
| Gemini 2.5 | gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite | Thinking, multimodal |
| Gemini 2.0 | gemini-2.0-flash, gemini-2.0-flash-lite | Multimodal |
| Gemini 1.5 | gemini-1.5-pro, gemini-1.5-flash | Multimodal |

---

## Breaking Changes

### Python Version Requirement

**Python 3.10 or higher is now required** for LangChain v1.0 compatibility. Python 3.9 is no longer supported.

### Environment Variables

Additional API keys are required for non-OpenAI providers:

```bash
OPENAI_API_KEY="..."      # OpenAI (GPT-5, GPT-4, o-series)
ANTHROPIC_API_KEY="..."   # Anthropic (Claude models)
GOOGLE_API_KEY="..."      # Google (Gemini models)
OPENROUTER_API_KEY="..."  # OpenRouter (multi-provider)
```

Only the API key for your selected provider is required.

---

## Dependency Changes

### Updated Packages

| Package | Previous | New |
|---------|----------|-----|
| langchain-core | 1.0.5 | 1.1.0 |
| langchain-openai | 1.0.3 | 1.1.0 |
| langsmith | 0.4.43 | 0.4.49 |
| pydantic | 2.12.4 | 2.12.5 |

### Added Packages

| Package | Version | Purpose |
|---------|---------|---------|
| langchain-anthropic | 1.2.0 | Anthropic Claude models support |
| langchain-google-genai | 3.2.0 | Google Gemini models support |

### Removed Packages

- `langchain>=1.0.0` - Not directly used; `langchain-core` is sufficient

### requirements.txt

Dependencies are now pinned to exact versions for reproducibility:

```
# LangChain Ecosystem
langchain-core==1.1.0
langchain-openai==1.1.0
langchain-anthropic==1.2.0
langchain-google-genai==3.2.0
```

---

## Architecture Changes

### Before (v1.x - Single Provider)

- OpenAI-only support
- Custom retry logic with exponential backoff
- Manual error classification
- No parameter validation

### After (v2.0 - Multi-Provider)

- Support for OpenAI, Anthropic, Google, OpenRouter via LangChain
- LangChain handles API retries automatically (`max_retries=5`)
- Capability guarding prevents unsupported parameter errors
- Provider auto-detection from model names
- Schema-specific retries retained for content flags

### New Files

| File | Purpose |
|------|---------|
| `api/llm_client.py` | Multi-provider LLM client with capability profiles |
| `api/transcribe_api.py` | Provider-agnostic transcription manager |
| `api/summary_api.py` | Provider-agnostic summary manager |

### Deprecated Files

| File | Status |
|------|--------|
| `api/base_openai_client.py` | Retained for backward compatibility |
| `api/openai_api.py` | Retained for backward compatibility |
| `api/openai_transcribe_api.py` | Retained for backward compatibility |

---

## API Changes

### LLMConfig Dataclass

New configuration class in `api/llm_client.py`:

```python
@dataclass
class LLMConfig:
    model: str                          # Model identifier
    provider: Optional[ProviderType]    # "openai", "anthropic", "google", "openrouter"
    api_key: Optional[str]              # Optional (defaults to env var)
    timeout: int = 900                  # Request timeout
    max_retries: int = 5                # LangChain retry count
    temperature: Optional[float]        # Model temperature
    max_tokens: Optional[int]           # Max output tokens
    service_tier: Optional[str]         # OpenAI: "flex", "default", "auto"
    extra_kwargs: Dict[str, Any]        # Provider-specific parameters
```

### New Functions

- `get_chat_model(config: LLMConfig) -> BaseChatModel` - Create provider-appropriate chat model
- `get_model_capabilities(model_name: str) -> Dict[str, bool]` - Get capability profile
- `get_provider_for_model(model: str) -> ProviderType` - Infer provider from model name
- `is_provider_available(provider: ProviderType) -> bool` - Check if API key is set
- `get_available_providers() -> list[ProviderType]` - List providers with API keys

---

## Migration Guide

### From v1.x to v2.0

1. **Update Python**: Ensure Python 3.10+ is installed

2. **Update dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set API keys**: Add environment variables for your provider(s)

4. **Update configuration**: No changes required for OpenAI users. For other providers, set the `provider` field in `app.yaml`

5. **Review model selection**: Verify your selected model is supported by your provider

### Configuration Compatibility

Existing OpenAI configurations work without modification. The `provider` field is auto-detected from the model name.

---

## Documentation Updates

### README.md

- **New sections**: Supported Models, Prerequisites, Installation, Quick Start, Provider Configuration
- **Updated sections**: Overview (LangChain mention), Key Features (Multi-Provider Architecture), How It Works (Transcription via LangChain)
- **Troubleshooting**: Multi-provider API key issues, unsupported parameter errors

---

## Testing

All changes verified:
- All imports successful
- Main module loads correctly
- CLI `--help` works
- No dependency conflicts
- Provider switching tested

---

## Contributors

This release represents a significant architectural evolution of AutoExcerpter, enabling flexibility in LLM provider choice while maintaining the robust transcription and summarization capabilities of previous versions.
