"""PraxisPillar — BasePillar wrapper for the ActionOrchestrator system.

Integrates the action orchestration system into the agent lifecycle:
    - initialize: Creates the ActionOrchestrator instance
    - tick: Runs execution batch, emits feedback about success/failure rates
    - signals: Handles 'execute_plan', 'execute_single', 'import_plan', 'cancel_action'
    - shutdown: Serializes execution state for cross-session persistence

Design: The PraxisPillar wraps an ActionOrchestrator and acts as the bridge
between the equilibrium engine (tension modulation of execution behavior)
and the actual tool execution. It receives plans from Cognition, executes
actions with appropriate safety/parallelism/verification, and provides
execution results to Mneme for learning.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from isonome.base import BasePillar
from isonome.praxis.delegation import DelegationGate
from isonome.praxis.orchestrator import (
    Action,
    ActionOrchestrator,
    ActionRisk,
    ActionState,
    ExecutionReport,
    RetryPolicy,
)
from isonome.types import (
    AgentState,
    Feedback,
    Pillar,
    Signal,
)

logger = logging.getLogger(__name__)


class PraxisPillar(BasePillar):
    """The Praxis pillar — wraps ActionOrchestrator for agent integration.

    On each tick (via process_queued → _on_signal), the pillar:
    1. Reads the tension profile from the agent state
    2. Sets it on the orchestrator so execution behavior is modulated
    3. Processes any incoming signals (execute_plan, import_plan, etc.)
    4. Emits Feedback to the equilibrium engine about execution outcomes

    Usage:
        praxis = PraxisPillar(
            name="executor",
            executor_fn=my_tool_runner,
            validator_fn=my_validator,
        )
        # Set tension profile each tick:
        praxis.orchestrator.set_tension_profile(agent.get_tension_profile())
        # Import a plan from cognition:
        praxis.import_plan(cognition_tasks)
        # Execute pending actions:
        report = praxis.execute_pending()
        # Get results for mneme:
        results = praxis.get_execution_memories()
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        executor_fn: Callable[[Action], Any] | None = None,
        validator_fn: Callable[[Action, Any], tuple[bool, float]] | None = None,
        approve_fn: Callable[[Action], bool] | None = None,
        max_parallel: int = 8,
        default_retry_policy: RetryPolicy | None = None,
        confidence_calibrator: Any = None,  # ConfidenceCalibrator for safety gating
    delegation_gate: DelegationGate | None = None, # Calibration-gated delegation
    ):
        """Initialize the Praxis pillar.

        Args:
            name: Pillar display name.
            executor_fn: Called for each action — actually executes the tool.
            validator_fn: Optional post-execution validation.
            approve_fn: Optional callback to approve safety-gated actions.
            max_parallel: Maximum concurrent executions.
            default_retry_policy: Fallback retry policy.
    delegation_gate: Optional DelegationGate for calibration-gated delegation.
        """
        super().__init__(name=name)
        self._executor_fn = executor_fn
        self._validator_fn = validator_fn
        self._approve_fn = approve_fn
        self._max_parallel = max_parallel
        self._default_retry = default_retry_policy
        self._confidence_calibrator = confidence_calibrator
        self._delegation_gate = delegation_gate
        self.orchestrator: ActionOrchestrator | None = None
        self._last_report: ExecutionReport | None = None

    # ── Abstract interface ──────────────────────────────────────

    @property
    def pillar(self) -> Pillar:
        return Pillar.PRAXIS

    def _on_initialize(self, state: AgentState) -> None:
        """Create the ActionOrchestrator system."""
        self.orchestrator = ActionOrchestrator(
            max_parallel=self._max_parallel,
            default_retry_policy=self._default_retry,
            confidence_calibrator=self._confidence_calibrator,
        )
        # Wire delegation gate if provided or if calibrator is available
        if self._delegation_gate is not None:
            self.orchestrator.set_delegation_gate(self._delegation_gate)
        elif self._confidence_calibrator is not None:
            self.orchestrator.set_delegation_gate(calibrator=self._confidence_calibrator)
        # Set initial tension profile from agent state
        if state.tensions is not None:
            profile = {}
            for axis in state.tensions.axes:
                profile[axis.id] = axis.position
            self.orchestrator.set_tension_profile(profile)
        logger.info(f"{self.name}: ActionOrchestrator initialized")

    def _on_signal(self, signal: Signal) -> None:
        """Handle incoming signals from other pillars.

        Supported signal kinds:
            - 'import_plan': Convert cognition plan into actions.
              payload: {tasks: [{description, tool_name, ...}]}
            - 'execute_plan': Import and immediately execute.
              payload: {tasks: [...]}
            - 'execute_single': Execute a single action.
              payload: {description, tool_name, params?, risk?, ...}
            - 'execute_pending': Run all pending actions.
              payload: {} (no extra data needed)
            - 'cancel_action': Cancel a specific action.
              payload: {action_id: str}
            - 'cancel_all': Cancel all non-completed actions.
              payload: {}
        """
        if self.orchestrator is None:
            logger.warning(f"{self.name}: not initialized, ignoring signal")
            return

        kind = signal.kind
        payload = signal.payload

        try:
            if kind == "import_plan":
                tasks = payload.get("tasks", [])
                if tasks:
                    ids = self.orchestrator.import_from_cognition(tasks)
                    logger.info(
                        f"{self.name}: imported {len(ids)} actions from plan"
                    )

            elif kind == "execute_plan":
                tasks = payload.get("tasks", [])
                if tasks:
                    self.orchestrator.import_from_cognition(tasks)
                self._run_execution_batch()

            elif kind == "execute_single":
                action = Action(
                    description=payload.get("description", ""),
                    tool_name=payload.get("tool_name", "unknown"),
                    params=payload.get("params", {}),
                    risk=ActionRisk[payload.get("risk", "LOW").upper()],
                    preconditions=tuple(payload.get("preconditions", ())),
                    tags=tuple(payload.get("tags", ())),
                )
                self.orchestrator.register_action(action)
                self._run_execution_batch()

            elif kind == "execute_pending":
                self._run_execution_batch()

            elif kind == "cancel_action":
                from uuid import UUID
                action_id = UUID(payload.get("action_id", ""))
                if self.orchestrator.cancel_action(action_id):
                    logger.info(f"{self.name}: cancelled action {action_id}")
                else:
                    logger.warning(f"{self.name}: could not cancel action {action_id}")

            elif kind == "cancel_all":
                count = self.orchestrator.cancel_all()
                logger.info(f"{self.name}: cancelled {count} actions")

            else:
                logger.debug(f"{self.name}: unknown signal kind '{kind}'")

        except Exception:
            logger.exception(f"{self.name}: error handling signal {kind}")

    def _on_shutdown(self) -> None:
        """Serialize execution state for cross-session persistence."""
        if self.orchestrator is not None:
            try:
                self.orchestrator.to_dict()  # validate serialization works
                total = self.orchestrator.total_actions
                completed = len(self.orchestrator.completed_actions)
                logger.info(
                    f"{self.name}: shutting down — "
                    f"{total} actions registered, {completed} completed, "
                    f"{len(self.orchestrator.failed_actions)} failed"
                )
            except Exception:
                logger.exception(f"{self.name}: error serializing state")

    # ── Equilibrium pull integration ──────────────────────────────

    def _on_equilibrium_sync(self, view) -> None:
        """Auto-sync tension state from the equilibrium view.

        When bound to an engine, this is called automatically at the
        start of each process_queued() tick. It replaces the need
        for external update_tension_profile() calls.

        Applies the view's all_positions to the orchestrator and
        reads cross-pillar influence (e.g., Cognition's shallow_deep
        position affects execution planning depth).
        """
        if self.orchestrator is not None:
            self.orchestrator.set_tension_profile(view.all_positions)

        # Cross-pillar modulation: if Cognition is in shallow mode,
        # prefer sequential execution for simpler coordination
        shallow_deep = view.cross_axes.get("shallow_deep", 0.0)
        if shallow_deep < -0.5 and self.orchestrator is not None:
            # Shallow cognition → limit parallelism
            self.orchestrator._max_parallel = max(
                1, self._max_parallel // 2
            )

    # ── Execution ─────────────────────────────────────────────────

    def _run_execution_batch(self) -> None:
        """Run a full execution batch on the orchestrator.

        After execution, emits feedback to the equilibrium engine
        based on success/failure rates.
        """
        if self._executor_fn is None:
            logger.warning(f"{self.name}: no executor_fn configured, cannot execute")
            return

        self._last_report = self.orchestrator.execute_batch(
            executor_fn=self._executor_fn,
            validator_fn=self._validator_fn,
            approve_fn=self._approve_fn,
        )

        # Emit feedback for each Praxis tension axis
        self._emit_execution_feedback(self._last_report)

    # ── Feedback ──────────────────────────────────────────────────

    def _emit_execution_feedback(self, report: ExecutionReport) -> None:
        """Emit equilibrium feedback based on execution outcomes.

        Three feedback axes:
        1. autonomy_safety: High success → push autonomous; high failures → push safe
        2. sequential_parallel: DAG parallelism in use → reinforce parallel if successful
        3. verify_execute: Validation failures → push toward verify_heavy;
           smooth execution → push toward execute_fast
        """
        # ── autonomy_safety feedback ──────────────────────────
        if report.actions_total == 0:
            return

        # Success rate feedback: high success → autonomous, low → safe
        if report.success_rate >= 0.95:
            auto_signal = 0.15  # Push toward autonomous
        elif report.success_rate >= 0.80:
            auto_signal = 0.05
        elif report.success_rate < 0.50:
            auto_signal = -0.20  # Push toward safe
        elif report.success_rate < 0.70:
            auto_signal = -0.08
        else:
            auto_signal = 0.0

        # Gate blocks intensify safe push
        if report.gate_blocks > 0:
            auto_signal -= 0.05 * report.gate_blocks

        self.emit_feedback(
            Feedback(
                source=self.pillar,
                tension_axis_id="autonomy_safety",
                signal=max(-1.0, min(1.0, auto_signal)),
                confidence=0.7,
                reason=f"success_rate={report.success_rate:.2f}, "
                       f"blocks={report.gate_blocks}",
            )
        )

        # ── sequential_parallel feedback ──────────────────────
        # If we used parallelism and it succeeded, reinforce parallel
        if report.parallelism_level > 1 and report.success_rate >= 0.80:
            para_signal = 0.10
        elif report.parallelism_level == 1 and report.actions_total > 5:
            para_signal = -0.08  # Sequential with many actions → push parallel
        else:
            para_signal = 0.0

        self.emit_feedback(
            Feedback(
                source=self.pillar,
                tension_axis_id="sequential_parallel",
                signal=max(-1.0, min(1.0, para_signal)),
                confidence=0.6,
                reason=f"parallelism={report.parallelism_level}, "
                       f"actions={report.actions_total}",
            )
        )

        # ── verify_execute feedback ───────────────────────────
        # Low validation scores → push toward verify_heavy
        if report.avg_validation_score < 0.3 and report.actions_total > 0:
            verify_signal = -0.12  # Push toward verify_heavy
        elif report.avg_validation_score >= 0.8:
            verify_signal = 0.08  # Push toward execute_fast
        else:
            verify_signal = 0.0

        self.emit_feedback(
            Feedback(
                source=self.pillar,
                tension_axis_id="verify_execute",
                signal=max(-1.0, min(1.0, verify_signal)),
                confidence=0.65,
                reason=f"avg_validation={report.avg_validation_score:.2f}",
            )
        )

    # ── Calibrator wiring ──────────────────────────────────────────────────────

    def set_confidence_calibrator(self, calibrator: Any) -> None:
        """Set or replace the confidence calibrator for safety gating.

        Passes through to the underlying orchestrator. Set to None
        to disable confidence-based gating. Call this after wiring
        the Cognition pillar's calibrator during agent setup.

        Also updates the delegation gate if one is present.
        """
        self._confidence_calibrator = calibrator
        if self.orchestrator is not None:
            self.orchestrator.set_confidence_calibrator(calibrator)
            # Auto-wire delegation gate to use the new calibrator
            if self.orchestrator._delegation_gate is not None:
                self.orchestrator._delegation_gate.set_calibrator(calibrator)

    def set_delegation_gate(self, gate: DelegationGate | None = None, *, calibrator: Any = None) -> None:
        """Set or replace the delegation gate.

        Passes through to the underlying orchestrator. When a calibrator
        is set on the pillar, it is automatically wired to the gate.
        """
        if calibrator is None and self._confidence_calibrator is not None:
            calibrator = self._confidence_calibrator
        if self.orchestrator is not None:
            self.orchestrator.set_delegation_gate(gate, calibrator=calibrator)
        self._delegation_gate = self.orchestrator._delegation_gate if self.orchestrator else gate

    # ── Convenience methods ────────────────────────────────────────

    def update_tension_profile(self, profile: dict) -> None:
        """Update the orchestrator's tension profile (call each tick)."""
        if self.orchestrator is not None:
            self.orchestrator.set_tension_profile(profile)

    def import_plan(self, tasks: list[dict[str, Any]]) -> list:
        """Import a cognition plan as executable actions."""
        if self.orchestrator is None:
            return []
        return self.orchestrator.import_from_cognition(tasks)

    def execute_pending(self) -> ExecutionReport | None:
        """Execute all pending actions and return the report."""
        if self.orchestrator is None or self._executor_fn is None:
            return None
        self._run_execution_batch()
        return self._last_report

    def get_execution_memories(self) -> list[dict[str, Any]]:
        """Get execution log entries for mneme persistence (πρᾶξις → μνήμη)."""
        if self.orchestrator is None:
            return []
        return self.orchestrator.export_to_mneme()

    def serialize(self) -> dict | None:
        """Get the full serializable execution state.

        Includes both the orchestrator state and the pillar configuration
        (max_parallel, default_retry_policy, calibrator reference) so that
        restore() can faithfully reconstruct the full pillar.
        """
        if self.orchestrator is None:
            return None

        from isonome import SERIALIZATION_SCHEMA_VERSION

        result = self.orchestrator.to_dict()
        # Layer pillar config on top of orchestrator state
        result["_pillar_config"] = {
            "name": self.name,
            "max_parallel": self._max_parallel,
            "has_executor_fn": self._executor_fn is not None,
            "has_validator_fn": self._validator_fn is not None,
            "has_approve_fn": self._approve_fn is not None,
            "default_retry_policy": (
                {
                    "max_retries": self._default_retry.max_retries,
                    "base_delay": self._default_retry.base_delay,
                    "backoff_factor": self._default_retry.backoff_factor,
                    "max_delay": self._default_retry.max_delay,
                }
                if self._default_retry is not None
                else None
            ),
        }
        result["_schema_version"] = SERIALIZATION_SCHEMA_VERSION
        return result

    def restore(self, data: dict) -> None:
        """Restore execution state from serialized data.

        Reconstructs both the ActionOrchestrator and the pillar's config
        parameters (max_parallel, default_retry_policy). Executor/validator/
        approve callables cannot be serialized — the caller must re-wire
        them after restore().
        """
        from isonome import SERIALIZATION_SCHEMA_VERSION

        # Validate schema version for forward-compat detection
        saved_version = data.get("_schema_version", 0)
        if saved_version > SERIALIZATION_SCHEMA_VERSION:
            logger.warning(
                f"{self.name}: serialized with schema v{saved_version}, "
                f"current is v{SERIALIZATION_SCHEMA_VERSION} — "
                f"some fields may be ignored"
            )

        self.orchestrator = ActionOrchestrator.from_dict(data)

        # Restore pillar config if present (schema v1+)
        config = data.get("_pillar_config", {})
        if config:
            self.name = config.get("name", self.name)
            self._max_parallel = int(config.get("max_parallel", self._max_parallel))
            # Restore retry policy from pillar config (preferred)
            # or fall back to orchestrator-level data
            retry_data = config.get("default_retry_policy")
            if retry_data is not None:
                self._default_retry = RetryPolicy(
                    max_retries=retry_data.get("max_retries", 3),
                    base_delay=retry_data.get("base_delay", 1.0),
                    backoff_factor=retry_data.get("backoff_factor", 2.0),
                    max_delay=retry_data.get("max_delay", 300.0),
                )
            # Sync max_parallel to the orchestrator
            if self.orchestrator is not None:
                self.orchestrator._max_parallel = self._max_parallel

        # Callables cannot be serialized — log a reminder
        if config.get("has_executor_fn"):
            logger.info(
                f"{self.name}: executor_fn was present at serialize time; "
                f"re-wire with set_executor_fn() after restore"
            )

        logger.info(
            f"{self.name}: restored {self.orchestrator.total_actions} actions"
        )

    @property
    def last_report(self) -> ExecutionReport | None:
        """Most recent execution report, or None."""
        return self._last_report
