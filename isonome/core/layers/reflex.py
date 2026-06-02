from __future__ import annotations

import logging
from isonome.core.layers.base import LayerBase
from isonome.core.state import SensorState, MotorCommand
from isonome.utils.logging import get_layer_logger


class ReflexLayer(LayerBase):
    """Hard real-time reactive control. Runs at ~100Hz.

    Ingests raw sensor streams and emits motor primitives with minimal latency.
    No reasoning -- pure stimulus-response with learned weights.
    """

    def __init__(self, frequency_hz: float = 100.0) -> None:
        super().__init__(name="reflex", frequency_hz=frequency_hz)
        self._logger = get_layer_logger("reflex")

    async def on_boot(self) -> None:
        self._logger.info("reflex_layer_booting")

    async def on_tick(self) -> None:
        pass  # tick logic driven externally via react()

    async def on_shutdown(self) -> None:
        self._logger.info("reflex_layer_shutdown")

    async def react(self, sensors: SensorState) -> MotorCommand:
        """Process sensor input and produce motor command.

        Override in user's layers/reflex.py for real behavior.
        Default: zero-effort hold position.
        """
        positions = {j.name: j.position for j in sensors.joints}
        return MotorCommand(joint_positions=positions)
