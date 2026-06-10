"""Agent — Four-layer orchestrator for v0.2 Frozen Brain + Learned Nervous System.

Tick enforces the one-frame delay invariant:
  1. Read RAW state
  2. Cortex builds prompt from previous tick's discrepancy
  3. JEPA generates canonical intent from RAW state
  4. SomaKernel corrects canonical intent for THIS body
  5. Reflex interpolates to control frequency and enforces safety
  6. Execute
  7. Observe result for NEXT tick's discrepancy analysis
"""
from __future__ import annotations

import asyncio
from pathlib import Path
import time
from typing import Optional

from isonome.bridge.factory import build_body_bridge
from isonome.core.config import AppConfig
from isonome.core.safety import AgentMode, EmergencyStop, SafetyGovernor
from isonome.core.state import (
    CanonicalActionChunk,
    CorrectedMotorCommand,
    ExecutionResult,
    RawSensorState,
)
from isonome.core.layers.soma import SomaLayer
from isonome.core.layers.jepa import JEPALayer
from isonome.core.layers.cortex import CortexLayer
from isonome.core.layers.reflex import ReflexLayer
from isonome.core.layers.plasticity import PlasticityLayer
from isonome.utils.logging import get_layer_logger


class Agent:
    """Four-layer orchestrator — the framework kernel.

    Manages Soma → JEPA → Cortex → Reflex pipeline with strict invariants.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._logger = get_layer_logger(f"agent.{config.agent_name}")
        self._mode = AgentMode.BOOT

        # Body bridge — the integration point between core and sim/hardware
        self._body_bridge = build_body_bridge(config)

        # Core layers
        self.soma = SomaLayer(
            urdf_path=Path(config.soma.urdf_path),
            frequency_hz=config.reflex.frequency_hz,
            body_bridge=self._body_bridge,
        )
        self.jepa = JEPALayer(
            frequency_hz=config.jepa.frequency_hz,
            backend=config.jepa.backend,
            model_id=config.jepa.model_id,
        )
        self.cortex = CortexLayer(frequency_hz=config.cortex.frequency_hz)
        self.reflex = ReflexLayer(
            frequency_hz=config.reflex.frequency_hz,
            control_freq=config.reflex.control_freq_hz,
            policy_freq=config.reflex.policy_freq_hz,
        )
        self.plasticity = PlasticityLayer(
            kernel_dir=config.plasticity.kernel_dir,
        )

        # Safety
        self.safety = SafetyGovernor(self)

        # Timing state
        self._last_cortex_tick = 0.0
        self._running = False

    @property
    def mode(self) -> AgentMode:
        return self._mode

    @mode.setter
    def mode(self, value: AgentMode) -> None:
        self._logger.info("agent_mode_change", extra={"from": self._mode.value, "to": value.value})
        self._mode = value

    async def boot(self) -> None:
        self.mode = AgentMode.BOOT
        for layer in (self.soma, self.jepa, self.cortex, self.reflex, self.plasticity):
            await layer.boot()
        self._running = True
        self.mode = AgentMode.IDLE

    async def shutdown(self) -> None:
        self._running = False
        for layer in (self.plasticity, self.cortex, self.jepa, self.reflex, self.soma):
            await layer.shutdown()
        self.mode = AgentMode.IDLE

    async def load_kernel(self, path: Optional[Path] = None) -> None:
        """Load a calibrated kernel, transitioning through CALIBRATING."""
        if not self.safety.can_load_kernel():
            raise RuntimeError(f"Cannot load kernel in mode {self.mode.value}")
        self.mode = AgentMode.CALIBRATING
        if path is None:
            if self._config.soma.kernel_path:
                path = Path(self._config.soma.kernel_path)
            else:
                robot_hash = self.soma._robot_hash()
                path = self.plasticity.kernel_path(robot_hash)
        self.soma.load_kernel(path)
        self.mode = AgentMode.RUNTIME

    async def tick(self) -> None:
        """Single tick of the agent loop.

        Enforces the invariant: JEPA never sees corrected or post-execution state.
        """
        if not self._running:
            return
        if not self.safety.can_execute():
            return

        # 1. Read RAW state (never corrected)
        raw_state = await self._async_perceive()

        # 2. Cortex builds prompt from previous tick's discrepancy
        advice = self.cortex.advise()
        prompt = self.cortex.build_prompt(raw_state, advice)

        # 3. JEPA generates canonical intent from RAW state
        #    INVARIANT: JEPA never sees corrected or post-execution state
        canonical_chunk = await self.jepa.deliberate(raw_state, prompt, advice)

        # 4. SomaKernel corrects canonical intent for THIS body
        #    If no calibrated kernel exists, fall back to naive mapping
        if self.soma.has_calibrated_kernel:
            corrected_chunk = self.soma.apply_kernel(canonical_chunk, raw_state)
        else:
            corrected_chunk = self._naive_map(canonical_chunk)

        # 5. Reflex interpolates to control frequency and enforces safety
        safe_commands = self.reflex.process(corrected_chunk)

        # 6. Execute
        await self._async_act(safe_commands)

        # 7. Observe result for NEXT tick's discrepancy analysis
        execution_result = await self._async_observe_result()
        self.cortex.buffer.add(canonical_chunk, execution_result, raw_state)

    async def _async_perceive(self) -> RawSensorState:
        """Async wrapper around soma.perceive() or the body bridge."""
        if self._body_bridge is not None and self._body_bridge.is_connected:
            return await self._body_bridge.perceive()
        return self.soma.perceive()

    def _naive_map(self, canonical_chunk: CanonicalActionChunk) -> CorrectedMotorCommand:
        """Fallback naive mapping when no calibrated kernel is loaded."""
        mapped = self.soma.naive_mapper.map(canonical_chunk.actions)
        return CorrectedMotorCommand(
            commands=mapped,
            robot_hash=self.soma._robot_hash(),
        )

    async def _async_act(self, safe_commands: list) -> None:
        """Async wrapper around soma.act() or the body bridge."""
        if not safe_commands:
            return
        # For now, execute the first command in the interpolated chunk.
        # In a real system, this would stream at control frequency.
        cmd = CorrectedMotorCommand(
            commands=safe_commands[0].command.unsqueeze(0),
            robot_hash=self.soma._robot_hash(),
        )
        if self._body_bridge is not None and self._body_bridge.is_connected:
            await self._body_bridge.act(cmd)
        else:
            self.soma.act(cmd)

    async def _async_observe_result(self) -> ExecutionResult:
        """Async wrapper around soma.observe_result() or the body bridge."""
        if self._body_bridge is not None and self._body_bridge.is_connected:
            return await self._body_bridge.observe_result()
        return self.soma.observe_result()

    async def run(self, duration_s: float | None = None) -> None:
        """Main agent loop."""
        await self.boot()
        self.mode = AgentMode.RUNTIME
        start = time.monotonic()
        try:
            while self._running:
                if duration_s and (time.monotonic() - start) >= duration_s:
                    break
                await self.tick()
                await asyncio.sleep(self.reflex.tick_period)
        except EmergencyStop:
            self._logger.critical("agent_emergency_stop")
            self.mode = AgentMode.SAFE_STOP
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()
