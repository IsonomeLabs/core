"""Coordinator — multi-agent composition layer for Chamber 3.

The coordinator owns:
  * N ``SubAgentSlot`` entries (each wraps an ``Agent`` + metadata)
  * one ``FSMExecutor``          (global phase machine)
  * one ``ActionMerger``         (merge strategy)

On each ``tick()``:
  1. FSM executor updates global phase.
  2. Each sub-agent runs its own tick pipeline (perceive → … → act).
  3. Partial actions are collected from sub-agents.
  4. Merger produces a ``FullAction``.
  5. Full action is executed via the coordinator's body bridge.
  6. FSM ``during`` actions run.

This closes architecture gap #2: previously ``Agent.tick()`` was a single
linear pipeline for one body with no multi-agent composition, no FSM, and
no merge strategy.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from isonome.core.agent import Agent
from isonome.core.config import AppConfig
from isonome.core.coordination.fsm import FSMContext, FSMExecutor
from isonome.core.coordination.merger import ActionMerger
from isonome.core.layers.reflex import ReflexLayer
from isonome.core.ports.body_bridge import BodyBridge
from isonome.core.state import FullAction, PartialAction
from isonome.utils.logging import get_layer_logger


@dataclass
class SubAgentSlot:
    """A single sub-agent inside the coordination graph.

    ``dof_slice`` tells the merger where this agent's commands map into
    the full robot joint vector.  ``phase_filter`` is an optional set of
    FSM state names; when the global phase is NOT in this set, the agent's
    ``partial_action`` is marked inactive.
    """

    agent: Agent
    agent_id: str
    dof_slice: slice = field(default_factory=lambda: slice(0, 0))
    priority: int = 0
    weight: float = 1.0
    phase_filter: Optional[set[str]] = None

    def is_active_in_phase(self, phase: str) -> bool:
        if self.phase_filter is None:
            return True
        return phase in self.phase_filter


class Coordinator:
    """Multi-agent coordinator with FSM + ActionMerger.

    Usage:
        coord = Coordinator(
            slots=[
                SubAgentSlot(agent=loco_agent, agent_id="locomotion",
                             dof_slice=slice(0, 4), priority=1),
                SubAgentSlot(agent=arm_agent, agent_id="arm",
                             dof_slice=slice(4, 11), priority=2),
            ],
            fsm_executor=fsm_exec,
            merger=ActionMerger.create(MergeStrategy.PRIORITY),
            body_bridge=bridge,
            total_dof=11,
        )
        await coord.boot()
        await coord.tick()
        await coord.shutdown()
    """

    def __init__(
        self,
        *,
        slots: List[SubAgentSlot],
        fsm_executor: FSMExecutor,
        merger: ActionMerger,
        body_bridge: Optional[BodyBridge] = None,
        total_dof: int,
        frequency_hz: float = 200.0,
    ) -> None:
        self._slots = list(slots)
        self._fsm = fsm_executor
        self._merger = merger
        self._bridge = body_bridge
        self._total_dof = total_dof
        self._frequency_hz = frequency_hz
        self._logger = get_layer_logger("coordination.coordinator")
        self._running = False

        # Validate unique agent IDs
        ids = [s.agent_id for s in slots]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate agent_ids in coordinator slots: {ids}")

    # -- properties ----------------------------------------------------------

    @property
    def slots(self) -> List[SubAgentSlot]:
        return list(self._slots)

    @property
    def current_phase(self) -> str:
        return self._fsm.current_state

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def tick_period(self) -> float:
        return 1.0 / self._frequency_hz if self._frequency_hz > 0 else float("inf")

    # -- lifecycle -----------------------------------------------------------

    async def boot(self) -> None:
        self._logger.info(
            "coordinator_booting",
            extra={
                "agents": [s.agent_id for s in self._slots],
                "phase": self._fsm.current_state,
                "strategy": self._merger.strategy.value,
            },
        )

        # Boot body bridge first (if present)
        if self._bridge is not None:
            await self._bridge.boot()

        # Boot all sub-agents concurrently
        await asyncio.gather(
            *[s.agent.boot() for s in self._slots],
            return_exceptions=True,
        )

        self._running = True
        self._logger.info("coordinator_ready")

    async def shutdown(self) -> None:
        self._running = False
        self._logger.info("coordinator_shutting_down")

        # Shutdown all sub-agents concurrently
        await asyncio.gather(
            *[s.agent.shutdown() for s in self._slots],
            return_exceptions=True,
        )

        if self._bridge is not None:
            await self._bridge.shutdown()

        self._logger.info("coordinator_shutdown_complete")

    # -- tick ----------------------------------------------------------------

    async def tick(self) -> FullAction:
        """Execute one coordinator tick.

        1. Update FSM phase.
        2. Run each sub-agent tick (concurrently where possible).
        3. Collect partial actions.
        4. Merge into full action.
        5. Execute via body bridge.
        """
        if not self._running:
            raise RuntimeError("Coordinator is not running. Call boot() first.")

        # 1. FSM tick — updates global phase
        phase = self._fsm.tick()

        # 2. Run sub-agent ticks concurrently
        #    Each agent already runs perceive → advise → deliberate → ... → act
        #    internally.  We let them tick independently.
        results = await asyncio.gather(
            *[self._safe_agent_tick(s) for s in self._slots],
            return_exceptions=True,
        )

        # 3. Collect partial actions
        partials: List[PartialAction] = []
        for slot, result in zip(self._slots, results):
            if isinstance(result, Exception):
                self._logger.warning(
                    "sub_agent_tick_failed",
                    extra={"agent_id": slot.agent_id, "error": str(result)},
                )
                continue

            active = slot.is_active_in_phase(phase)
            # Extract the command from the agent's last tick.
            # Agent.tick() writes to the body bridge internally, but we also
            # need the *intended* command for merging.  We read from the
            # agent's ReflexLayer output buffer (if present) or fall back
            # to zero.
            cmd = self._extract_command(slot.agent)

            partials.append(
                PartialAction(
                    agent_id=slot.agent_id,
                    commands=cmd,
                    dof_slice=slot.dof_slice,
                    priority=slot.priority,
                    weight=slot.weight,
                    active=active,
                )
            )

        # 4. Merge
        full_action = self._merger.merge(partials, self._total_dof)

        # 5. Execute merged action via coordinator body bridge
        await self._execute_full_action(full_action)

        self._logger.debug(
            "coordinator_tick",
            extra={
                "phase": phase,
                "agents": full_action.merged_from,
                "strategy": full_action.strategy.value,
            },
        )

        return full_action

    async def run(self, duration_s: float | None = None) -> None:
        """Run the coordinator loop continuously."""
        await self.boot()
        import time

        start = time.monotonic()
        try:
            while self._running:
                if duration_s and (time.monotonic() - start) >= duration_s:
                    break
                await self.tick()
                await asyncio.sleep(self.tick_period)
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()

    # -- helpers -------------------------------------------------------------

    async def _safe_agent_tick(self, slot: SubAgentSlot) -> None:
        """Tick a sub-agent, swallowing exceptions so one failure doesn't
        kill the whole coordinator."""
        if slot.agent.mode.value not in ("runtime", "idle"):
            # Ensure agent is in RUNTIME before ticking
            from isonome.core.safety import AgentMode

            slot.agent.mode = AgentMode.RUNTIME
        await slot.agent.tick()

    def _extract_command(self, agent: Agent) -> torch.Tensor:
        """Extract the latest motor command from an agent's pipeline.

        Currently we approximate by taking the agent's most recent
        ``CorrectedMotorCommand`` from the SomaLayer's last act call.
        In future iterations this can be formalised as an output buffer.
        """
        # Try to read from soma's most recent act state.
        # Since Agent._async_act builds a CorrectedMotorCommand and sends it
        # to the bridge, we can look at agent.soma for any cached command.
        # For now we fall back to zero so the merge is well-defined even when
        # no command has been generated yet.
        joint_count = agent.soma.naive_mapper.joint_count
        if hasattr(agent.soma, "_last_command") and agent.soma._last_command is not None:
            return agent.soma._last_command
        return torch.zeros(joint_count)

    async def _execute_full_action(self, full_action: FullAction) -> None:
        """Send the merged full action to the coordinator body bridge."""
        if self._bridge is None or not self._bridge.is_connected:
            return
        from isonome.core.state import CorrectedMotorCommand

        cmd = CorrectedMotorCommand(
            commands=full_action.commands,
            robot_hash="coordinator",
        )
        await self._bridge.act(cmd)
