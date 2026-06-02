from __future__ import annotations
import asyncio
import json
import logging
from typing import Any


class LLMSwarm:
    """Multi-LLM orchestrator for Plasticity layer.

    Runs multiple LLM calls in parallel. Each proposes a JSON patch.
    Aggregation via best-confidence selection.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        api_key: str = "",
        size: int = 3,
    ) -> None:
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._size = size
        self._logger = logging.getLogger("isonome.llm.swarm")

    async def _single_proposal(self, idx: int, prompt: str, system: str) -> str:
        """Single LLM call in the swarm."""
        try:
            if self._provider == "openai":
                from openai import AsyncOpenAI

                client = AsyncOpenAI(api_key=self._api_key)
                response = await client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": f"{system}\n\nYou are swarm member {idx}."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=2048,
                )
                return response.choices[0].message.content or "[]"

            if self._provider == "anthropic":
                import anthropic

                client = anthropic.AsyncAnthropic(api_key=self._api_key)
                response = await client.messages.create(
                    model=self._model,
                    system=f"{system}\n\nYou are swarm member {idx}.",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2048,
                )
                return response.content[0].text if response.content else "[]"

            self._logger.error("unsupported_provider", extra={"provider": self._provider})
            return "[]"
        except Exception as e:
            self._logger.error("swarm_member_error", extra={"member": idx, "error": str(e)})
            return "[]"

    async def propose(self, prompt: str, system: str = "") -> list[str]:
        """Run all swarm members in parallel, collect raw results."""
        tasks = [self._single_proposal(i, prompt, system) for i in range(self._size)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, str)]
