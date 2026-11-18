"""Provider-agnostic LLM client that returns validated Pydantic objects.

Two backends cover the provider landscape:

* ``gemini``  - native google-genai SDK, structured JSON output (v1beta).
* ``openai``  - any OpenAI-compatible ``/chat/completions`` endpoint. That covers
  OpenAI, OpenRouter, Groq, DeepSeek, Mistral, Together, xAI, Ollama, LM Studio,
  vLLM and llama.cpp servers. Anthropic works through its OpenAI-compatible
  endpoint (``https://api.anthropic.com/v1/``).

Every reply is validated against the requested schema. On a validation failure
the model is re-prompted with the error, so bad JSON never reaches the pipeline.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_TRANSIENT_MARKERS = (
    "429", "500", "502", "503", "504",
    "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED",
    "overloaded", "timed out", "timeout", "Connection",
)


class LLMError(RuntimeError):
    """Raised when a provider cannot produce a valid answer."""


class LLM(Protocol):
    name: str

    def generate(self, prompt: str, schema: type[T], *, temperature: float = 0.3) -> T: ...


def _is_transient(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}"
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def _with_retries(call, *, attempts: int = 4, base_delay: float = 2.0):
    """Retry transient provider failures with exponential backoff."""
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:  # provider SDKs raise many unrelated types
            if attempt == attempts - 1 or not _is_transient(exc):
                raise
            delay = base_delay * (2**attempt)
            log.warning("LLM transient error (%s); retrying in %.0fs", exc, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


def extract_json(text: str) -> str:
    """Pull the first JSON object out of a model reply (handles ```json fences)."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in model reply")
    return text[start : end + 1]


class _ValidatingMixin:
    """Shared retry-on-invalid-output loop. Subclasses implement ``_call``."""

    name: str

    def _call(self, prompt: str, schema: type[T], temperature: float) -> str:  # pragma: no cover
        raise NotImplementedError

    def generate(self, prompt: str, schema: type[T], *, temperature: float = 0.3) -> T:
        current = prompt
        last_error: Exception | None = None
        for _ in range(3):
            raw = self._call(current, schema, temperature)
            try:
                return schema.model_validate_json(extract_json(raw))
            except (ValidationError, ValueError) as exc:
                last_error = exc
                log.warning("LLM reply failed validation, re-prompting: %s", str(exc)[:200])
                current = (
                    f"{prompt}\n\nYour previous reply was invalid: {str(exc)[:500]}\n"
                    "Reply again with ONLY a valid JSON object matching the schema."
                )
        raise LLMError(f"{self.name} never returned valid {schema.__name__}: {last_error}")


class GeminiLLM(_ValidatingMixin):
    def __init__(self, api_key: str, model: str, timeout_s: int = 120):
        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version="v1beta", timeout=timeout_s * 1000),
        )
        self.model = model
        self.name = f"gemini:{model}"

    def _call(self, prompt: str, schema: type[BaseModel], temperature: float) -> str:
        config = self._types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=temperature,
        )
        # The JSON schema goes in the prompt rather than response_schema: it keeps the
        # same prompt working on every backend and avoids SDK-specific schema limits.
        full = f"{prompt}\n\nJSON schema for the reply:\n{json.dumps(schema.model_json_schema())}"
        response = _with_retries(
            lambda: self._client.models.generate_content(model=self.model, contents=full, config=config)
        )
        return response.text or ""


class OpenAICompatibleLLM(_ValidatingMixin):
    def __init__(self, base_url: str, api_key: str | None, model: str, timeout_s: int = 120):
        from openai import OpenAI

        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key or "not-needed",  # local servers (Ollama, LM Studio) ignore it
            timeout=timeout_s,
            max_retries=0,  # _with_retries owns retry policy
        )
        self.model = model
        self.name = f"openai-compatible:{model}"

    def _call(self, prompt: str, schema: type[BaseModel], temperature: float) -> str:
        messages = [
            {
                "role": "system",
                "content": "Reply with a single JSON object that matches this JSON Schema. "
                f"No prose, no markdown.\n{json.dumps(schema.model_json_schema())}",
            },
            {"role": "user", "content": prompt},
        ]
        response = _with_retries(
            lambda: self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
        )
        return response.choices[0].message.content or ""


def build_llm(settings) -> LLM:
    """Create the configured LLM backend from ``config.Settings``."""
    if settings.llm_provider == "gemini":
        key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else ""
        if not key:
            raise LLMError("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set")
        return GeminiLLM(key, settings.llm_model)
    if settings.llm_provider == "openai":
        if not settings.llm_base_url:
            raise LLMError("LLM_PROVIDER=openai requires LLM_BASE_URL")
        key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
        return OpenAICompatibleLLM(settings.llm_base_url, key, settings.llm_model)
    raise LLMError(f"unknown LLM_PROVIDER: {settings.llm_provider!r}")
