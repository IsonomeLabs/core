from __future__ import annotations

import asyncio
import logging
from isonome.core.layers.base import LayerBase
from isonome.core.state import WorldModel, CortexAdvice
from isonome.llm.client import LLMClient
from isonome.llm.cache import SemanticCache
from isonome.utils.logging import get_layer_logger


CORTEX_SYSTEM_PROMPT = """You are the Prefrontal Cortex of a robot. You observe the robot's world model \
and provide advice to the JEPA (predictive) layer. You NEVER control motors directly.

Analyze the world model state and provide:
1. A brief summary of what you observe
2. Specific suggestions for the JEPA layer to improve predictions
3. A priority level (low/medium/high/critical)

Respond in structured form:
- summary: one sentence
- suggestions: list of actionable suggestions
- priority: low/medium/high/critical
"""


class CortexLayer(LayerBase):
    """LLM-driven observer that watches JEPA's world model and advises.

    Never touches motors directly. Only advises JEPA via natural-language prompts.
    Triggers at low frequency (~0.1-0.5Hz) or on anomaly detection.
    """

    def __init__(
        self,
        frequency_hz: float = 0.5,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
        sandbox_timeout_s: float = 30.0,
    ) -> None:
        super().__init__(name="cortex", frequency_hz=frequency_hz)
        self._llm: LLMClient | None = None
        self._provider = provider
        self._model = model
        self._api_key_env = api_key_env
        self._sandbox_timeout_s = sandbox_timeout_s
        self._cache = SemanticCache()
        self._logger = get_layer_logger("cortex")

    async def on_boot(self) -> None:
        import os

        api_key = os.environ.get(self._api_key_env)
        if api_key:
            self._llm = LLMClient(
                provider=self._provider,
                model=self._model,
                api_key=api_key,
            )
        self._logger.info(
            "cortex_layer_booting",
            extra={"provider": self._provider, "model": self._model},
        )

    async def on_tick(self) -> None:
        pass  # tick logic driven externally via advise()

    async def on_shutdown(self) -> None:
        self._logger.info("cortex_layer_shutdown")

    async def advise(self, world_model: WorldModel) -> CortexAdvice:
        """Observe world model and produce advice for JEPA.

        Uses LLM client with semantic caching. Falls back to no-op if no LLM
        configured.
        """
        cache_key = f"wm:{hash(world_model.model_dump_json())}"
        cached = self._cache.get(cache_key)
        if cached:
            self._logger.info("cortex_cache_hit")
            return cached

        if not self._llm:
            return CortexAdvice(summary="No LLM configured", suggestions=[])

        prompt = f"World Model State:\n{world_model.model_dump_json(indent=2)}"

        try:
            response = await asyncio.wait_for(
                self._llm.complete(
                    prompt=prompt, system=CORTEX_SYSTEM_PROMPT
                ),
                timeout=self._sandbox_timeout_s,
            )
            advice = CortexAdvice(
                summary=response[:200],
                suggestions=[
                    s.strip()
                    for s in response.split("\n")
                    if s.strip().startswith(("-", "*", "•"))
                ],
                priority="low",
            )
        except asyncio.TimeoutError:
            self._logger.warning("cortex_timeout")
            advice = CortexAdvice(summary="LLM call timed out", priority="low")
        except Exception as e:
            self._logger.error("cortex_error", extra={"error": str(e)})
            advice = CortexAdvice(summary=f"LLM error: {e}", priority="low")

        self._cache.put(cache_key, advice)
        return advice
