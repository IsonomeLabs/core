from __future__ import annotations

import logging
from isonome.core.layers.base import LayerBase
from isonome.core.state import (
    SensorState,
    MotorCommand,
    Adjustment,
    WorldModel,
    CortexAdvice,
    PredictedState,
)
from isonome.utils.logging import get_layer_logger


class JEPALayer(LayerBase):
    """World Model / Predictive Reasoning layer.

    Runs in parallel with Reflex on the same sensory input.
    Maintains a predictive world model. Slightly longer-horizon than Reflex:
    predicts outcomes 0.5-2s ahead. Can modulate Reflex output.
    Does not replace Reflex -- supplements it.
    """

    def __init__(
        self,
        frequency_hz: float = 10.0,
        prediction_horizon_s: float = 1.0,
        model_path: str | None = None,
    ) -> None:
        super().__init__(name="jepa", frequency_hz=frequency_hz)
        self._prediction_horizon_s = prediction_horizon_s
        self._model_path = model_path
        self._world_model = WorldModel()
        self._pending_advice: list[CortexAdvice] = []
        self._logger = get_layer_logger("jepa")

    @property
    def world_model(self) -> WorldModel:
        return self._world_model

    async def on_boot(self) -> None:
        self._logger.info(
            "jepa_layer_booting", extra={"model_path": self._model_path}
        )
        # Placeholder: load pre-existing JEPA model
        if self._model_path:
            self._logger.info(
                "jepa_model_loading", extra={"path": self._model_path}
            )

    async def on_tick(self) -> None:
        pass  # tick logic driven externally via predict_and_adjust()

    async def on_shutdown(self) -> None:
        self._logger.info("jepa_layer_shutdown")

    async def predict_and_adjust(
        self, sensors: SensorState, reflex_cmd: MotorCommand
    ) -> Adjustment:
        """Predict future state and compute adjustment to Reflex output.

        Override to integrate a real JEPA model.
        Default: no adjustment (identity pass-through).
        """
        # Update internal world model with current sensor data
        current = PredictedState(
            timestamp=sensors.timestamp,
            joint_positions={j.name: j.position for j in sensors.joints},
            joint_velocities={j.name: j.velocity for j in sensors.joints},
        )
        self._world_model.current_state = current

        # Incorporate any pending Cortex advice
        if self._pending_advice:
            self._logger.info(
                "jepa_processing_advice",
                extra={"count": len(self._pending_advice)},
            )
            self._pending_advice.clear()

        # Default: no adjustment
        return Adjustment()

    async def inject_advice(self, advice: CortexAdvice) -> None:
        """Receive advice from Cortex layer."""
        self._pending_advice.append(advice)
        self._logger.info(
            "jepa_advice_injected", extra={"summary": advice.summary}
        )
