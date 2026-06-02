from __future__ import annotations

import asyncio
import logging
import time
from isonome.core.config import AppConfig
from isonome.core.state import (
    SensorState,
    MotorCommand,
    Adjustment,
    CortexAdvice,
    Patch,
    ErrorEvent,
)
from isonome.core.layers.base import LayerBase
from isonome.core.layers.reflex import ReflexLayer
from isonome.core.layers.jepa import JEPALayer
from isonome.core.layers.cortex import CortexLayer
from isonome.core.layers.plasticity import PlasticityLayer
from isonome.core.bus import MessageBus, Channel
from isonome.safety.governor import SafetyGovernor, RobotState
from isonome.utils.logging import get_layer_logger


class Agent:
    """Four-layer orchestrator -- the framework kernel.

    Manages parallel Reflex + JEPA dispatch, Cortex observation, and gated
    Plasticity.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._logger = get_layer_logger(f"agent.{config.agent_name}")
        self._state = RobotState.OFFLINE

        # Core layers
        self.reflex = ReflexLayer(frequency_hz=config.reflex.frequency_hz)
        self.jepa = JEPALayer(
            frequency_hz=config.jepa.frequency_hz,
            prediction_horizon_s=config.jepa.prediction_horizon_s,
            model_path=config.jepa.model_path,
        )
        self.cortex = CortexLayer(
            frequency_hz=config.cortex.frequency_hz,
            provider=config.cortex.provider,
            model=config.cortex.model,
            api_key_env=config.cortex.api_key_env,
            sandbox_timeout_s=config.cortex.sandbox_timeout_s,
        )
        self.plasticity = PlasticityLayer(
            provider=config.plasticity.provider,
            model=config.plasticity.model,
            api_key_env=config.plasticity.api_key_env,
            swarm_size=config.plasticity.swarm_size,
        )

        # Infrastructure
        self.bus = MessageBus()
        self.safety = SafetyGovernor(config.safety, self)

        # Timing state
        self._last_reflex_tick = 0.0
        self._last_jepa_tick = 0.0
        self._last_cortex_tick = 0.0
        self._error_buffer: list[ErrorEvent] = []
        self._running = False

    @property
    def state(self) -> RobotState:
        return self._state

    def set_state(self, state: RobotState) -> None:
        self._state = state
        self._logger.info("agent_state_change", extra={"state": state.value})

    async def boot(self) -> None:
        self.set_state(RobotState.BOOTING)
        await self.bus.start()
        for layer in (self.reflex, self.jepa, self.cortex, self.plasticity):
            await layer.boot()
        self._running = True
        self.set_state(RobotState.IDLE)

    async def shutdown(self) -> None:
        self._running = False
        for layer in (self.plasticity, self.cortex, self.jepa, self.reflex):
            await layer.shutdown()
        await self.bus.stop()
        self.set_state(RobotState.OFFLINE)

    async def sense(self) -> SensorState:
        """Override or set via bridge to provide sensor data."""
        return SensorState()

    async def act(self, command: MotorCommand) -> None:
        """Override or set via bridge to execute motor commands."""

    async def tick(self) -> None:
        """Single tick of the agent loop.

        Parallel Reflex + JEPA, conditional Cortex.
        """
        if not self._running:
            return

        now = time.monotonic()
        raw = await self.sense()

        # Reflex: produce motor command from raw sensors
        reflex_cmd = await self.reflex.react(raw)

        # JEPA: predict and compute adjustment
        jepa_adjustment = await self.jepa.predict_and_adjust(raw, reflex_cmd)

        # Merge Reflex + JEPA
        modulated = self._merge_reflex_jepa(reflex_cmd, jepa_adjustment)

        # Cortex observation (low frequency)
        if now - self._last_cortex_tick >= self.cortex.tick_period:
            self._last_cortex_tick = now
            try:
                advice = await self.cortex.advise(self.jepa.world_model)
                await self.jepa.inject_advice(advice)
                await self.bus.publish(Channel.CORTEX_ADVICE, advice)
            except Exception as e:
                self._logger.error(
                    "cortex_tick_error", extra={"error": str(e)}
                )

        await self.act(modulated)

    def _merge_reflex_jepa(
        self, reflex_cmd: MotorCommand, adjustment: Adjustment
    ) -> MotorCommand:
        """Apply JEPA's additive adjustment to Reflex output."""
        merged_positions = dict(reflex_cmd.joint_positions)
        for k, v in adjustment.position_deltas.items():
            merged_positions[k] = merged_positions.get(k, 0.0) + v

        merged_velocities = dict(reflex_cmd.joint_velocities)
        for k, v in adjustment.velocity_deltas.items():
            merged_velocities[k] = merged_velocities.get(k, 0.0) + v

        merged_efforts = dict(reflex_cmd.joint_efforts)
        for k, v in adjustment.effort_deltas.items():
            merged_efforts[k] = merged_efforts.get(k, 0.0) + v

        return MotorCommand(
            joint_positions=merged_positions,
            joint_velocities=merged_velocities,
            joint_efforts=merged_efforts,
        )

    async def adapt(self) -> None:
        """Trigger Plasticity -- gated by SafetyGovernor."""
        if not self.safety.can_adapt():
            return
        patches = await self.plasticity.generate_patches(
            error_log=self._error_buffer,
            layer_states={},
        )
        await self.safety.apply_patches(patches)
        self._error_buffer.clear()

    def record_error(self, event: ErrorEvent) -> None:
        self._error_buffer.append(event)
        self._logger.error(
            "agent_error",
            extra={
                "error_class": event.error_class,
                "message": event.message,
            },
        )

    async def run(self, duration_s: float | None = None) -> None:
        """Main agent loop."""
        await self.boot()
        self.set_state(RobotState.RUNNING)
        start = time.monotonic()
        try:
            while self._running:
                if duration_s and (time.monotonic() - start) >= duration_s:
                    break
                await self.tick()
                await asyncio.sleep(self.reflex.tick_period)
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()
