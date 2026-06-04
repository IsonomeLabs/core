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
from typing import Any, Sequence

from isonome.base import BasePillar
from isonome.equilibrium import EquilibriumEngine
from isonome.equilibrium.task_type_homeostasis import (
    TaskTypeHomeostasis,
    infer_task_type,
)
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

        # Task-type adaptive homeostasis — learns tension default profiles
        # per task type, enabling pre-adaptation on new tasks
        axis_order = tuple(
            a.id for a in (axes or EquilibriumEngine.DEFAULT_AXES)
        )
        self._task_type_homeostasis = TaskTypeHomeostasis(
            axis_order=axis_order,
        )
        self._current_task_type: str | None = None

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
    def task_type_homeostasis(self) -> TaskTypeHomeostasis:
        """The task-type adaptive homeostasis system."""
        return self._task_type_homeostasis

    @property
    def current_task_type(self) -> str | None:
        """The task type of the most recently submitted task, or None."""
        return self._current_task_type

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

        Additionally, the agent infers the task type from the
        description and pre-adapts its tension defaults if the
        task type has a learned homeostatic profile.
        """
        self._task_queue.append(task)

        # ── Task-type adaptive homeostasis ──
        inferred_type = infer_task_type(task.description)
        self._current_task_type = inferred_type

        logger.info(
            f"Task '{task.description}' submitted (id={task.id}, "
            f"type={inferred_type})"
        )

        # Pre-adapt defaults if we have a converged profile for this type
        profile = self._task_type_homeostasis.get_profile(inferred_type)
        if profile is not None and profile.is_converged:
            adjusted = self._task_type_homeostasis.soft_pre_adapt(
                self.engine, inferred_type,
            )
            if adjusted > 0:
                logger.info(
                    f"TaskTypeHomeostasis: pre-adapted {adjusted} axes for "
                    f"task type '{inferred_type}'"
                )

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

        # 5. Process execution outcomes: feed calibrator, adapt defaults
        self._process_execution_outcomes()

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

    # ── Outcome Processing ─────────────────────────────────────────

    def _process_execution_outcomes(self) -> None:
        """Process execution outcomes from Praxis: feed calibrator and adapt defaults.

        This closes the outermost homeostatic learning loop:

            Praxis executes → outcomes observed → calibrator learns →
            calibrator trends → default positions adapt → behavior shifts

        Two mechanisms:

        1. **Calibrator recording**: Each action in the ExecutionReport
           generates a confidence-outcome pair. The calibrator learns
           to distinguish well-confident from overconfident predictions.

        2. **Default position adaptation**: When outcomes consistently
           deviate from expectations, the equilibrium engine's set points
           shift — the agent learns that certain tension configurations
           produce better or worse outcomes in the current environment.

        Both mechanisms require the Praxis pillar to be wired with a
        confidence_calibrator (shared with Cognition) and to have
        produced at least one ExecutionReport.
        """
        # Only process if we have a fully wired set of pillars
        praxis = self._praxis
        cognition = self._cognition
        if praxis is None or cognition is None:
            return

        # Get the last execution report
        if not hasattr(praxis, "last_report") or praxis.last_report is None:
            return
        report = praxis.last_report

        if report.actions_total == 0:
            return

        # ── Mechanism 1: Feed calibrator from execution outcomes ──
        calibrator = None
        if hasattr(cognition, "reasoning") and cognition.reasoning is not None:
            calibrator = getattr(cognition.reasoning, "calibrator", None)

        if calibrator is not None and calibrator.total_predictions > 0:
            # Record aggregate confidence-assessment from the execution report.
            # The success rate is the observed accuracy; the calibrator can
            # track whether the system's own confidence estimates were right.
            conf_signal = min(1.0, max(0.0, report.success_rate))
            calibrator.record(predicted_confidence=conf_signal, actual_success=True)

            # Each failed action becomes a negative calibration signal
            # — the system overestimated its ability to execute.
            for _ in range(report.actions_failed):
                calibrator.record(predicted_confidence=0.8, actual_success=False)

            # Record a follow-up calibration adjustment
            if hasattr(cognition, "reasoning") and cognition.reasoning is not None:
                try:
                    cognition.reasoning.calibrate(
                        predicted_confidence=report.success_rate,
                        actual_success=(report.success_rate >= 0.8),
                    )
                except Exception:
                    pass

        # ── Mechanism 2: Adapt default positions from outcome trends ──
        engine = self.engine

        # autonomy_safety: Low success → push default toward safe
        if report.actions_total > 2:
            # Compute an outcome signal for each relevant axis
            if report.success_rate < 0.5:
                # Many failures — default should shift toward safe
                engine.adjust_default("autonomy_safety", outcome_signal=-0.5 * report.actions_failed / max(1, report.actions_total))
                engine.adjust_default("verify_execute", outcome_signal=-0.4)  # More verification

                # Low success also shifts explore_exploit toward explore
                # (need more information before committing)
                engine.adjust_default("explore_exploit", outcome_signal=-0.2)

                # And consolidate_prune toward consolidate
                # (failures mean learn more from experiences)
                engine.adjust_default("consolidate_prune", outcome_signal=-0.15)
            elif report.success_rate > 0.95:
                # Smooth sailing — default shifts toward autonomous/fast
                engine.adjust_default("autonomy_safety", outcome_signal=0.3)
                engine.adjust_default("verify_execute", outcome_signal=0.3)

                # High success shifts explore_exploit toward exploit
                # (current approach is working)
                engine.adjust_default("explore_exploit", outcome_signal=0.15)

            # Gate blocks → reinforce current safety posture
            if report.gate_blocks > 2:
                engine.adjust_default("autonomy_safety", outcome_signal=-0.1 * report.gate_blocks)

            # Retry rate > 0.3 → push toward verify_heavy
            if report.actions_retried > 0 and report.actions_retried / max(1, report.actions_total) > 0.3:
                engine.adjust_default("verify_execute", outcome_signal=-0.25)

        # ── Mechanism 3: Record defaults for task-type homeostasis ──
        # After processing outcomes, record the updated default positions
        # for the current task type. This builds up the profile that
        # enables pre-adaptation on future tasks.
        if self._current_task_type is not None and report.actions_total > 2:
            self._task_type_homeostasis.record_defaults(
                engine, task_type=self._current_task_type,
            )

    # ── Serialization ──────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full agent state for cross-session persistence.

        Saves the agent identity, equilibrium engine state, all three
        pillar states (if available), task queue, and statistics.
        The agent can be reconstructed with identical state.

        Schema version is included for forward/backward compatibility —
        from_dict() uses it to detect and gracefully handle format
        changes across releases.

        Returns:
            A JSON-serializable dict of the entire agent state.
        """
        from isonome import SERIALIZATION_SCHEMA_VERSION

        result: dict[str, Any] = {
            "schema_version": SERIALIZATION_SCHEMA_VERSION,
            "agent": {
                "name": self.identity.name,
                "id": str(self.identity.id),
                "version": self.identity.version,
                "created_at": self.identity.created_at.isoformat(),
            },
            "engine": self.engine.to_dict(),
            "task_type_homeostasis": self._task_type_homeostasis.to_dict(),
            "current_task_type": self._current_task_type,
            "tick_count": self._tick_count,
            "signals_sent": self._signals_sent,
            "feedback_applied": self._feedback_applied,
        }

        # Serialize pillar states (each includes its own _pillar_config
        # with name, config params, and callable presence flags)
        if self._cognition is not None:
            result["cognition"] = self._cognition.serialize()
        if self._praxis is not None:
            result["praxis"] = self._praxis.serialize()
        if self._mneme is not None:
            result["mneme"] = self._mneme.serialize()

        # Serialize calibrator (from reasoning engine)
        if self._cognition is not None and hasattr(self._cognition, "reasoning"):
            r = getattr(self._cognition, "reasoning", None)
            if r is not None:
                result["calibrator"] = r.calibrator.to_dict()

        # Serialize task queue
        result["task_queue"] = [
            {
                "id": str(t.id),
                "description": t.description,
                "complexity": t.complexity.value,
                "status": t.status.value,
            }
            for t in self._task_queue
        ]

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IsonomeAgent:
        """Deserialize full agent state from saved data.

        Reconstructs the agent with all three pillars and equilibrium
        engine in their saved state. Pillars are restored via their
        restore() methods:

        - **CognitionPillar**: restore() reconstructs both Attention
          and Reasoning subsystems, including calibrator state and
          tension profiles.
        - **PraxisPillar**: restore() reconstructs the
          ActionOrchestrator from its serialized state, including
          pillar config (max_parallel, retry policy).
        - **MnemePillar**: restore() reconstructs the
          HierarchicalMneme from its serialized state, including
          consolidation/promotion thresholds.

        After pillar restoration, the pillar_map and internal
        references (_cognition, _praxis, _mneme) are re-wired so
        the agent is fully operational. Restored pillars are bound
        to the restored equilibrium engine for auto-sync.

        Schema version validation: if the saved data has a newer
        schema version than the current code, a warning is logged
        but deserialization still proceeds (best-effort forward
        compatibility). Data from older schemas is accepted
        silently — missing fields fall back to defaults.

        Args:
            data: A dict produced by to_dict().

        Returns:
            A reconstructed IsonomeAgent with full state.
        """
        from uuid import UUID

        from isonome import SERIALIZATION_SCHEMA_VERSION
        from isonome.types import AgentIdentity, Task, TaskComplexity, TaskStatus

        # Schema version validation
        saved_version = data.get("schema_version", 0)
        if saved_version > SERIALIZATION_SCHEMA_VERSION:
            logger.warning(
                f"Deserializing schema v{saved_version} data with "
                f"v{SERIALIZATION_SCHEMA_VERSION} code — some fields "
                f"may be ignored. Consider upgrading isonome."
            )

        agent_data = data.get("agent", {})

        agent = cls(
            name=agent_data.get("name", "restored-agent"),
        )

        # Restore agent identity id
        saved_id = agent_data.get("id")
        if saved_id:
            agent.identity = AgentIdentity(
                id=UUID(saved_id),
                name=agent_data.get("name", "restored-agent"),
                version=agent_data.get("version", "0.1.0"),
            )

        # Restore equilibrium engine
        engine_data = data.get("engine", {})
        if engine_data:
            from isonome.equilibrium import EquilibriumEngine
            agent.engine = EquilibriumEngine.from_dict(engine_data)

        # Restore task-type homeostasis
        ht_data = data.get("task_type_homeostasis", {})
        if ht_data:
            agent._task_type_homeostasis = TaskTypeHomeostasis.from_dict(ht_data)
        agent._current_task_type = data.get("current_task_type")

        # Restore counters
        agent._tick_count = int(data.get("tick_count", 0))
        agent._signals_sent = int(data.get("signals_sent", 0))
        agent._feedback_applied = int(data.get("feedback_applied", 0))

        # ── Restore pillars ──────────────────────────────────────────
        # Each pillar is reconstructed and restored from its serialized
        # data, then re-registered in the pillar map.
        # Pillar names are read from the serialized _pillar_config
        # (schema v1+) so custom names survive round-trips.

        # Cognition pillar
        cog_data = data.get("cognition")
        if cog_data is not None:
            from isonome.cognition.pillar import CognitionPillar
            cog = CognitionPillar(
                name="thinker",
                engine=agent.engine,
                token_capacity=int(cog_data.get("token_capacity", 128_000)),
                compress_ratio=float(cog_data.get("compress_ratio", 0.20)),
                auto_gc=bool(cog_data.get("auto_gc", True)),
                gc_utilization_threshold=float(cog_data.get("gc_utilization_threshold", 0.80)),
            )

            cog.restore(cog_data)
            agent._cognition = cog
            agent._pillar_map[Pillar.COGNITION] = cog

        # Praxis pillar
        praxis_data = data.get("praxis")
        if praxis_data is not None:
            from isonome.praxis.pillar import PraxisPillar
            praxis = PraxisPillar(name="executor")
            praxis.restore(praxis_data)
            agent._praxis = praxis
            agent._pillar_map[Pillar.PRAXIS] = praxis

        # Mneme pillar
        mneme_data = data.get("mneme")
        if mneme_data is not None:
            from isonome.mneme.pillar import MnemePillar
            mneme = MnemePillar(name="memory")
            mneme.restore(mneme_data)
            agent._mneme = mneme
            agent._pillar_map[Pillar.MNEME] = mneme

        # ── Post-restore wiring ──────────────────────────────────────

        # Re-wire cross-pillar references after all pillars are restored
        if agent._cognition is not None and agent._mneme is not None:
            agent._cognition._engine = agent.engine
            agent._cognition._mneme_pillar = agent._mneme

        # Bind all restored pillars to the restored engine so they
        # receive auto-sync equilibrium views on each process_queued()
        for pillar in agent._pillar_map.values():
            pillar.bind_engine(agent.engine)

        # Restore task queue
        for t_data in data.get("task_queue", []):
            try:
                task = Task(
                    id=UUID(t_data.get("id", "")),
                    description=t_data.get("description", ""),
                    complexity=TaskComplexity(t_data.get("complexity", "simple")),
                    status=TaskStatus(t_data.get("status", "pending")),
                )
                agent._task_queue.append(task)
            except Exception:
                pass

        return agent
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
            "task_type_homeostasis": self._task_type_homeostasis.summary(),
        }
