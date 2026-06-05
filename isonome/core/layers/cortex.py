"""Cortex Layer — discrepancy watcher and advice generator.

Watches the gap between what JEPA intended and what the body actually did,
then produces natural-language advice for JEPA on the next tick.

Cortex does NOT touch motor commands. It only produces text advice for JEPA.
"""
from __future__ import annotations

from typing import List

from isonome.core.layers.base import LayerBase
from isonome.core.state import (
    CanonicalActionChunk,
    CortexAdvice,
    Discrepancy,
    ExecutionResult,
    RawSensorState,
)
from isonome.utils.logging import get_layer_logger


class DiscrepancyBuffer:
    """Ring buffer of observed intent-vs-outcome discrepancies."""

    def __init__(self, max_size: int = 10) -> None:
        self.buffer: List[Discrepancy] = []
        self._max_size = max_size

    def add(
        self,
        intended: CanonicalActionChunk,
        actual: ExecutionResult,
        raw_state: RawSensorState,
    ) -> None:
        """Store the difference between what JEPA intended and what actually happened."""
        self.buffer.append(
            Discrepancy(
                intended=intended,
                actual=actual,
                raw_state=raw_state,
            )
        )
        if len(self.buffer) > self._max_size:
            self.buffer.pop(0)

    def last(self) -> Discrepancy | None:
        return self.buffer[-1] if self.buffer else None

    def clear(self) -> None:
        self.buffer.clear()


class CortexLayer(LayerBase):
    """Natural-language advice generator for JEPA.

    Watches the DiscrepancyBuffer and generates CortexAdvice strings.
    Never touches motor commands.
    """

    def __init__(self, frequency_hz: float = 0.5) -> None:
        super().__init__(name="cortex", frequency_hz=frequency_hz)
        self.buffer = DiscrepancyBuffer()
        self._logger = get_layer_logger("cortex")

    async def on_boot(self) -> None:
        self._logger.info("cortex_layer_booting")

    async def on_tick(self) -> None:
        pass  # tick logic driven externally by agent.py

    async def on_shutdown(self) -> None:
        self._logger.info("cortex_layer_shutdown")

    def advise(self) -> List[CortexAdvice]:
        """Generate advice for JEPA based on the latest discrepancies."""
        last = self.buffer.last()
        if last is None:
            return []

        advice_list: List[CortexAdvice] = []

        # Compute simple discrepancy metrics
        intended_actions = last.intended.actions
        final_state = last.actual.final_proprioception

        if intended_actions.numel() > 0 and final_state.numel() > 0:
            # Example heuristic: compare first DOF
            delta = final_state[0].item() - intended_actions[0, 0].item()
            if abs(delta) > 0.05:
                advice_list.append(
                    CortexAdvice(
                        text=f"Motor 0 overshot by {delta:.2f}m. "
                        f"Reduce effective gain by {min(abs(delta) * 100, 50):.0f}%.",
                        priority="high" if abs(delta) > 0.15 else "medium",
                    )
                )

        if not last.actual.success:
            advice_list.append(
                CortexAdvice(
                    text="Last execution failed. Consider a more conservative action.",
                    priority="critical",
                )
            )

        if advice_list:
            self._logger.info(
                "cortex_advice_generated",
                extra={"count": len(advice_list)},
            )
        return advice_list

    def build_prompt(
        self, raw_state: RawSensorState, advice: List[CortexAdvice]
    ) -> str:
        """Build a task prompt for JEPA from raw state and advice."""
        prompt_lines = [
            "You are a generalist visuomotor policy.",
            f"Current proprioception shape: {list(raw_state.proprioception.shape)}",
            f"Timestamp: {raw_state.timestamp:.3f}",
        ]
        if raw_state.camera_frames:
            prompt_lines.append(
                f"Camera frames: {len(raw_state.camera_frames)} available"
            )
        return "\n".join(prompt_lines)
