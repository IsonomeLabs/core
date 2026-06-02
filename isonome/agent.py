"""The IsonomeAgent — the top-level agent orchestration class.

Ties together the three pillars and the equilibrium engine into
a single coordinated agent loop. The agent follows a tick-based
architecture:

    1. Drain signals from pillars → route to targets
    2. Drain feedback from pillars → apply to equilibrium engine
    3. Let each pillar process its queued signals
    4. Pillars read current tension profile to modulate behavior
    5. Repeat

This is inspired by homeostatic biological systems — continuous
adjustment toward equilibrium, not discrete decision points.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Sequence

from isonome.base import BasePillar
from isonome.equilibrium import EquilibriumEngine
from isonome.types import (
    AgentIdentity,
    AgentLifecycle,
    AgentState,
    Feedback,
    Pillar,
    Signal,
    Task,
    TaskStatus,
    TensionAxis,
    TensionSnapshot,
    now,
)

logger = logging.getLogger(__name__)


class IsonomeAgent:
    """An autonomous agent regulated by dynamic equilibrium.

    The agent maintains three pillars (Cognition, Praxis, Mneme)
    and an EquilibriumEngine that continuously balances competing
    tensions. Each tick of the agent loop:
    - Collects feedback from pillars
    - Applies tension adjustments
    - Routes inter-pillar signals
    - Lets pillars process their queues

    Usage:
        agent = IsonomeAgent(
            name="research-agent",
            cognition=MyCognition(),
            praxis=MyPraxis(),
            mneme=MyMneme(),
        )
        agent.start()
        agent.submit_task(task)
        while agent.has_work():
            agent.tick()
        agent.stop()
    """

    def __init__(
        self,
        *,
        name: str,
        cognition: BasePillar | None = None,
        praxis: BasePillar | None = None,
        mneme: BasePillar | None = None,
        axes: Sequence[TensionAxis] | None = None,
    ):
        self.identity = AgentIdentity(name=name)
        self.engine = EquilibriumEngine(axes=axes)
        self.state = AgentState(
            identity=self.identity,
            tensions=self.engine.snapshot(agent_id=self.identity.id),
        )

        # Wire up pillars
        self._cognition = cognition
        self._praxis = praxis
        self._mneme = mneme

        self._pillar_map: dict[Pillar, BasePillar] = {}
        for p in [cognition, praxis, mneme]:
            if p is not None:
                self._pillar_map[p.pillar] = p

        self._signals_sent: int = 0
        self._feedback_applied: int = 0
        self._tick_count: int = 0
        self._task_queue: deque[Task] = deque()

    # ── Properties ──────────────────────────────────────────────

    @property
    def cognition(self) -> BasePillar | None:
        return self._cognition

    @property
    def praxis(self) -> BasePillar | None:
        return self._praxis

    @property
    def mneme(self) -> BasePillar | None:
        return self._mneme

    @property
    def lifecycle(self) -> AgentLifecycle:
        return self.state.lifecycle

    # ── Lifecycle ───────────────────────────────────────────────

    def start(self) -> None:
        """Initialize all pillars and begin the agent loop."""
        logger.info(f"Agent '{self.identity.name}' starting (id={self.identity.id})")
        self.state.lifecycle = AgentLifecycle.BOOTSTRAPPING

        for pillar in self._pillar_map.values():
            pillar.initialize(self.state)

        self.state.lifecycle = AgentLifecycle.IDLE
        logger.info(
            f"Agent '{self.identity.name}' ready — {len(self._pillar_map)} pillars active"
        )

    def stop(self) -> None:
        """Gracefully shut down all pillars."""
        logger.info(f"Agent '{self.identity.name}' stopping")
        self.state.lifecycle = AgentLifecycle.TERMINATED

        for pillar in self._pillar_map.values():
            pillar.shutdown()

    # ── Task Management ──────────────────────────────────────────

    def submit_task(self, task: Task) -> None:
        """Enqueue a task for the agent to work on.

        Tasks flow through all three pillars: Cognition plans,
        Praxis executes, and Mneme learns from the result.
        """
        self._task_queue.append(task)
        logger.info(f"Task '{task.description}' submitted (id={task.id})")

    def has_work(self) -> bool:
        """Whether the agent has pending tasks to process."""
        return len(self._task_queue) > 0

    def tick(self) -> TensionSnapshot:
        """Execute one cycle of the agent loop.

        Returns the tension snapshot AFTER this tick's adjustments.
        """
        self._tick_count += 1
        self.state.lifecycle = AgentLifecycle.REASONING

        # 1. Collect feedback from all pillars
        all_feedback: list[Feedback] = []
        for pillar in self._pillar_map.values():
            all_feedback.extend(pillar.drain_feedback())

        # 2. Apply feedback to equilibrium engine
        if all_feedback:
            snapshot = self.engine.apply_feedback_batch(all_feedback)
            self._feedback_applied += len(all_feedback)
        else:
            snapshot = self.engine.snapshot()

        # Update agent state
        self.state.tensions = snapshot
        self.state.last_active = now()

        # 3. Route inter-pillar signals
        for pillar in self._pillar_map.values():
            signals = pillar.drain_signals()
            for sig in signals:
                target = self._pillar_map.get(sig.target)
                if target is not None:
                    target.receive_signal(sig)
                    self._signals_sent += 1

        # 4. Let pillars process their queues
        for pillar in self._pillar_map.values():
            pillar.process_queued()

        self.state.lifecycle = AgentLifecycle.IDLE
        return snapshot

    def send_signal(self, signal: Signal) -> None:
        """Route a signal to the appropriate pillar."""
        target = self._pillar_map.get(signal.target)
        if target is None:
            logger.warning(f"No pillar registered for target {signal.target}")
            return
        target.receive_signal(signal)

    # ── Information ─────────────────────────────────────────────

    def get_tension_profile(self) -> dict[str, float]:
        """Current equilibrium state as a flat dict."""
        return self.engine.get_behavior_profile()

    # Alias for compatibility
    def get_behavior_profile(self) -> dict[str, float]:
        """Alias for get_tension_profile."""
        return self.get_tension_profile()

    def get_stress_level(self) -> float:
        """How far the agent has drifted from homeostasis (0-1)."""
        return self.engine.tension_distance()

    @property
    def stats(self) -> dict:
        return {
            "tick_count": self._tick_count,
            "signals_sent": self._signals_sent,
            "feedback_applied": self._feedback_applied,
            "oscillation_events": self.engine.total_oscillation_events,
            "pillars_active": len(self._pillar_map),
            "task_queue_depth": len(self._task_queue),
            "lifecycle": self.state.lifecycle,
            "stress": round(self.get_stress_level(), 4),
        }
