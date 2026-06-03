"""
LLM layer — pluggable generation backend.

Two providers behind one `.generate(system, user)` interface:

  * Anthropic Messages API  (claude-sonnet-4-6 default, claude-opus-4-8 heavy)
  * OpenAI Chat Completions (gpt-5.4-mini default, gpt-5.4 heavy)

Pick the backend with FDI_LLM_PROVIDER=anthropic|openai. Use `get_llm()` to
obtain the right client; both implement the same interface so the RAG pipeline
doesn't care which one is active.
"""
from __future__ import annotations

from config import settings


class BaseLLM:
    model: str

    def generate(self, system: str, user: str,
                 max_tokens: int | None = None,
                 temperature: float | None = None) -> str:
        raise NotImplementedError


class AnthropicLLM(BaseLLM):
    def __init__(self, model: str | None = None):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("anthropic is required. Run: pip install anthropic") from e
        if not settings.llm.anthropic_api_key:
            raise RuntimeError(
                "Set ANTHROPIC_API_KEY to use the Anthropic backend.")
        self._client = anthropic.Anthropic(api_key=settings.llm.anthropic_api_key)
        self.model = model or settings.llm.model

    def generate(self, system: str, user: str,
                 max_tokens: int | None = None,
                 temperature: float | None = None) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens or settings.llm.max_tokens,
            temperature=(settings.llm.temperature
                         if temperature is None else temperature),
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(
            block.text for block in resp.content
            if getattr(block, "type", None) == "text"
        ).strip()


class OpenAILLM(BaseLLM):
    def __init__(self, model: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("openai is required. Run: pip install openai") from e
        if not settings.llm.openai_api_key:
            raise RuntimeError("Set OPENAI_API_KEY to use the OpenAI backend.")
        self._client = OpenAI(api_key=settings.llm.openai_api_key)
        self.model = model or settings.llm.openai_model

    def generate(self, system: str, user: str,
                 max_tokens: int | None = None,
                 temperature: float | None = None) -> str:
        temp = settings.llm.temperature if temperature is None else temperature
        # GPT-5.x reasoning models use `max_completion_tokens` and may reject a
        # non-default temperature. Build kwargs optimistically, then strip any
        # parameter the API complains about and retry — keeps us compatible
        # across both reasoning and classic chat models.
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": max_tokens or settings.llm.max_tokens,
        }
        if temp not in (None, 1, 1.0):
            kwargs["temperature"] = temp

        for _ in range(3):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001 - inspect message to adapt
                msg = str(e).lower()
                if "temperature" in msg and "temperature" in kwargs:
                    kwargs.pop("temperature")
                    continue
                if "max_completion_tokens" in msg and \
                        "max_completion_tokens" in kwargs:
                    kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                    continue
                raise
        raise RuntimeError("OpenAI generation failed after parameter retries.")


def get_llm(model: str | None = None) -> BaseLLM:
    if settings.llm.provider == "openai":
        return OpenAILLM(model)
    return AnthropicLLM(model)


# Backwards-compatible alias.
LLM = AnthropicLLM
