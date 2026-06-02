from __future__ import annotations
import logging
from typing import Any


class LLMClient:
    """Unified LLM client supporting OpenAI and Anthropic backends.

    User brings their own API key via config.yaml.
    """

    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini", api_key: str = "") -> None:
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._logger = logging.getLogger("isonome.llm.client")
        self._client: Any = None

    async def _ensure_client(self) -> None:
        if self._client is not None:
            return
        if self._provider == "openai":
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self._api_key)
        elif self._provider == "anthropic":
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self._provider}")

    async def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        timeout: float = 30.0,
    ) -> str:
        """Send a completion request and return the text response."""
        await self._ensure_client()
        self._logger.info("llm_request", extra={"provider": self._provider, "model": self._model})

        if self._provider == "openai":
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return response.choices[0].message.content or ""

        if self._provider == "anthropic":
            response = await self._client.messages.create(
                model=self._model,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            return response.content[0].text if response.content else ""

        return ""
