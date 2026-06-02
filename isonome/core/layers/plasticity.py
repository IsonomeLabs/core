from __future__ import annotations

import json
import logging
from isonome.core.layers.base import LayerBase
from isonome.core.state import Patch, ErrorEvent, PatchType
from isonome.llm.swarm import LLMSwarm
from isonome.utils.logging import get_layer_logger


PLASTICITY_SYSTEM_PROMPT = """You are part of the Neuroplasticity layer of a robot. \
Analyze error logs and layer states, then propose patches to fix issues.

Each patch must be a JSON object with:
- patch_type: "hyperparameter" | "code" | "behavior_tree" | "config"
- target_layer: "reflex" | "jepa" | "cortex"
- description: what the patch does
- changes: dict of parameter/code changes
- confidence: 0.0-1.0

Respond with a JSON array of patches.
"""


class PlasticityLayer(LayerBase):
    """Swarm of LLMs that rewrite kernels, tune hyperparameters, adjust behavior trees.

    Safety-critical: only runs when SafetyGovernor permits.
    Default: only when robot is powered off or in guaranteed idle/safe state.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        api_key_env: str = "OPENAI_API_KEY",
        swarm_size: int = 3,
    ) -> None:
        super().__init__(name="plasticity", frequency_hz=0.0)  # not scheduled
        self._swarm: LLMSwarm | None = None
        self._provider = provider
        self._model = model
        self._api_key_env = api_key_env
        self._swarm_size = swarm_size
        self._logger = get_layer_logger("plasticity")

    async def on_boot(self) -> None:
        import os

        api_key = os.environ.get(self._api_key_env)
        if api_key:
            self._swarm = LLMSwarm(
                provider=self._provider,
                model=self._model,
                api_key=api_key,
                size=self._swarm_size,
            )
        self._logger.info("plasticity_layer_booting")

    async def on_tick(self) -> None:
        pass  # Not scheduled -- triggered by adapt()

    async def on_shutdown(self) -> None:
        self._logger.info("plasticity_layer_shutdown")

    async def generate_patches(
        self, error_log: list[ErrorEvent], layer_states: dict
    ) -> list[Patch]:
        """Generate patches via LLM swarm.

        Each LLM in the swarm independently proposes patches.
        The framework validates and applies them via SafetyGovernor.
        """
        if not self._swarm:
            self._logger.warning("plasticity_no_swarm")
            return []

        prompt = (
            f"Error Log ({len(error_log)} events):\n"
            + "\n".join(
                f"- [{e.severity}] {e.error_class}: {e.message}"
                for e in error_log[-20:]
            )
            + f"\n\nLayer States:\n{layer_states}"
        )

        try:
            raw_results = await self._swarm.propose(
                prompt=prompt, system=PLASTICITY_SYSTEM_PROMPT
            )
        except Exception as e:
            self._logger.error(
                "plasticity_swarm_error", extra={"error": str(e)}
            )
            return []

        patches: list[Patch] = []
        for i, result in enumerate(raw_results):
            try:
                parsed = json.loads(result)
                items = parsed if isinstance(parsed, list) else [parsed]
                for item in items:
                    patches.append(
                        Patch(
                            patch_id=f"p_{i}_{len(patches)}",
                            patch_type=PatchType(
                                item.get("patch_type", "hyperparameter")
                            ),
                            target_layer=item.get("target_layer", ""),
                            description=item.get("description", ""),
                            changes=item.get("changes", {}),
                            confidence=float(item.get("confidence", 0.5)),
                            proposer=f"swarm_{i}",
                        )
                    )
            except (json.JSONDecodeError, ValueError) as e:
                self._logger.warning(
                    "plasticity_parse_error",
                    extra={"error": str(e), "proposer": i},
                )
        return patches
