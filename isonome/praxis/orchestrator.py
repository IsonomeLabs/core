"""Action Orchestration System — the execution heart of the Praxis pillar.

Core insight: Every agent action is a node in a directed acyclic graph (DAG).
The orchestrator schedules these nodes respecting dependencies while modulating
execution behavior via three equilibrium tension axes:

    autonomy_safety:  Safe (<0) → gate risky actions; Autonomous (>0) → execute freely
    sequential_parallel: Sequential (<0) → strict serial; Parallel (>0) → concurrent execution
    verify_execute:   Verify (<0) → exhaustive validation; Execute (>0) → minimal checks

The system treats execution as a flow through these stages:
    PENDING → QUEUED → EXECUTING → VERIFYING → COMPLETED
                                           └→ FAILED → RETRYING → EXECUTING

Cross-pillar pipelines:
    νοῦς → πρᾶξις: import_from_cognition() converts plans into action DAGs
    πρᾶξις → μνήμη: export_to_mneme() persists execution results for learning
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Sequence
from uuid import UUID, uuid4

from isonome.types import TensionID

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════


class ActionState(Enum):
    """Lifecycle states for an action in the orchestrator."""
    PENDING = auto()
    QUEUED = auto()
    EXECUTING = auto()
    VERIFYING = auto()
    COMPLETED = auto()
    FAILED = auto()
    RETRYING = auto()
    CANCELLED = auto()
    BLOCKED = auto()  # Waiting for dependencies


class ActionRisk(Enum):
    """Risk classification for safety gating.

    The autonomy_safety tension axis gates execution: at safe extremes,
    HIGH and CRITICAL actions are blocked unless explicitly approved."""
    TRIVIAL = 0   # No side effects, idempotent (e.g., read a file)
    LOW = 1       # Minor side effects, reversible (e.g., create temp file)
    MODERATE = 2  # Notable effects, somewhat reversible (e.g., modify config)
    HIGH = 3      # Significant effects, hard to reverse (e.g., deploy code)
    CRITICAL = 4  # Irreversible, catastrophic if wrong (e.g., delete database)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential backoff retry configuration.

    Max delay is capped at 300s to prevent unbounded waits.
    Formula: delay = base_delay × backoff_factor^attempt
    """
    max_retries: int = 3
    base_delay: float = 1.0
    backoff_factor: float = 2.0
    max_delay: float = 300.0

    def delay_for_attempt(self, attempt: int) -> float:
        if attempt < 0:
            return 0.0
        raw = self.base_delay * (self.backoff_factor ** attempt)
        return min(raw, self.max_delay)


@dataclass(frozen=True, slots=True)
class Action:
    """A single unit of executable work in the DAG.

    Fields WITHOUT defaults must precede fields WITH defaults
    (Python dataclass constraint with frozen=True).
    """
    description: str = field(repr=True)
    tool_name: str = field(repr=True)
    params: dict[str, Any] = field(default_factory=dict, repr=False)
    risk: ActionRisk = field(default=ActionRisk.LOW)
    preconditions: tuple[str, ...] = field(
        default_factory=tuple,
        repr=False,
    )
    expected_outcome: str = field(default="", repr=False)
    dependencies: tuple[UUID, ...] = field(default_factory=tuple)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    tags: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)
    confidence_required: float = field(default=0.0, repr=False)
    id: UUID = field(default_factory=uuid4)

    def dependency_count(self) -> int:
        return len(self.dependencies)

    def is_ready(self, completed: frozenset[UUID]) -> bool:
        """True when all dependencies are in the completed set."""
        return completed.issuperset(self.dependencies)


@dataclass
class ExecutionResult:
    """The outcome of running a single action."""
    action_id: UUID
    success: bool
    output: Any = None
    error: str = ""
    duration_ms: float = 0.0
    validation_passed: bool = False
    validation_score: float = 0.0
    attempt: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExecutionReport:
    """Summary of a batch execution cycle.

    Carries aggregate statistics for feedback routing to the
    equilibrium engine and for mneme persistence."""
    actions_total: int
    actions_completed: int
    actions_failed: int
    actions_blocked: int
    actions_retried: int
    total_duration_ms: float
    success_rate: float
    avg_validation_score: float
    parallelism_level: int
    gate_blocks: int  # How many blocked by risk-based safety gate
    tension_profile: dict[TensionID, float]
    confidence_blocks: int = 0  # How many blocked by confidence-based safety gate
    calibration_applied: bool = False  # Whether the calibrator was used


# ═══════════════════════════════════════════════════════════════════
# Utility: frozendict (immutable dict, hashable)
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class frozendict:
    """Immutable dictionary — usable in frozenset and as dict keys."""
    _data: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def __getitem__(self, key: str) -> Any:
        for k, v in self._data:
            if k == key:
                return v
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return any(k == key for k, _ in self._data)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return (k for k, _ in self._data)

    def values(self):
        return (v for _, v in self._data)

    def items(self):
        return iter(self._data)

    def __iter__(self):
        return (k for k, _ in self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return "frozendict({" + ", ".join(f"{k!r}: {v!r}" for k, v in self._data) + "})"


# ═══════════════════════════════════════════════════════════════════
# The Action Orchestrator
# ═══════════════════════════════════════════════════════════════════


class ActionOrchestrator:
    """DAG-based action scheduler with equilibrium-tension modulation.

    This is THE key system within the Praxis pillar. It:
    1. Registers actions with dependencies → builds a DAG
    2. Determines topological levels for parallel scheduling
    3. Applies risk-based safety gating (autonomy_safety tension)
    4. Schedules execution with variable parallelism (sequential_parallel)
    5. Validates results with configurable depth (verify_execute)
    6. Retries failures with exponential backoff
    7. Emits execution reports for feedback and memory persistence

    Mathematical foundation:
        Given N actions with dependency DAG G = (V, E):
        - Topological level ℓ(v) = max(ℓ(u) for (u,v) ∈ E) + 1, or 0 if no deps
        - Max concurrency C = max(1, floor(4 × (p_parallel + 1) / 2 + 1))
          where p_parallel ∈ [-1, 1] is the sequential_parallel tension position
        - Safety gate threshold τ = 0.5 + p_autonomy × 0.5
          Action allowed if risk.value / max_risk ≤ 1 − τ
        - Verification depth ω = 0.5 × (1 − p_verify)
          Validation score threshold = 0.3 + ω × 0.7

    Usage:
        orchestrator = ActionOrchestrator(engine=equilibrium_engine)
        orchestrator.register_action(Action(...))
        orchestrator.register_action(Action(...))
        report = orchestrator.execute_batch(executor_fn=my_tool_executor)
        print(report.success_rate)
    """

    # Default tension profile (used when no engine is available)
    _DEFAULT_PROFILE: dict[TensionID, float] = {
        "autonomy_safety": -0.4,
        "sequential_parallel": 0.1,
        "verify_execute": 0.0,
    }

    def __init__(
        self,
        *,
        max_parallel: int = 8,
        default_retry_policy: RetryPolicy | None = None,
        engine: Any = None,  # EquilibriumEngine — optional for standalone use
        confidence_calibrator: Any = None,  # ConfidenceCalibrator for calibrated safety gating
    ):
        """Initialize the action orchestrator.

        Args:
            max_parallel: Hard cap on concurrent executions.
            default_retry_policy: Fallback retry policy for actions without one.
            engine: EquilibriumEngine for tension-driven modulation.
        """
        self._actions: dict[UUID, Action] = {}
        self._states: dict[UUID, ActionState] = {}
        self._results: dict[UUID, list[ExecutionResult]] = {}
        self._completed: set[UUID] = set()
        self._topological_levels: dict[UUID, int] = {}
        self._max_parallel = max_parallel
        self._default_retry = default_retry_policy or RetryPolicy()
        self._engine = engine
        self._confidence_calibrator = confidence_calibrator

        # Statistics
        self._total_executed: int = 0
        self._total_completed: int = 0
        self._total_failed: int = 0
        self._total_retried: int = 0
        self._total_blocked: int = 0
        self._batch_count: int = 0

        # Execution log for mneme export
        self._execution_log: list[dict[str, Any]] = []

    # ── Calibrator ─────────────────────────────────────────────
    
    def set_confidence_calibrator(self, calibrator: Any) -> None:
        """Set or replace the confidence calibrator for safety gating.

        When a calibrator is set, the orchestrator uses calibrated
        (observed) confidence rather than raw confidence to gate
        actions. This closes the metacognitive loop: the Cognition
        pillar's calibrator learns from execution outcomes, and the
        Praxis pillar uses those learnings to make smarter safety decisions.

        Set to None to disable confidence-based gating.
        """
        self._confidence_calibrator = calibrator

    # ── Action Registration ──────────────────────────────────────

    def register_action(self, action: Action) -> UUID:
        """Register an action in the DAG.

        Automatically sets the action to PENDING and computes
        topological levels for all registered actions.

        Args:
            action: The action to register.

        Returns:
            The action's UUID for dependency referencing.
        """
        self._actions[action.id] = action
        self._states[action.id] = ActionState.PENDING
        self._results[action.id] = []
        self._recompute_topological_levels()
        return action.id

    def register_batch(self, actions: Sequence[Action]) -> list[UUID]:
        """Register multiple actions in one call.

        Levels are recomputed only once after all registrations.
        """
        ids = []
        for action in actions:
            ids.append(action.id)
            self._actions[action.id] = action
            self._states[action.id] = ActionState.PENDING
            self._results[action.id] = []
        self._recompute_topological_levels()
        return ids

    # ── Execution ────────────────────────────────────────────────

    def execute_batch(
        self,
        executor_fn: Callable[[Action], Any],
        *,
        validator_fn: Callable[[Action, Any], tuple[bool, float]] | None = None,
        approve_fn: Callable[[Action], bool] | None = None,
    ) -> ExecutionReport:
        """Execute all PENDING actions respecting DAG dependencies and tensions.

        This is THE central method. It:
        1. Reads the current tension profile
        2. Computes safety-gate thresholds
        3. Schedules actions by topological level, respecting parallelism
        4. Executes, validates, and retries as needed
        5. Returns an ExecutionReport for feedback routing

        Args:
            executor_fn: Called with an Action, returns the action's output.
            validator_fn: Optional. Called with (Action, output) → (passed, score).
            approve_fn: Optional. Called for gated actions → True if approved.

        Returns:
            ExecutionReport with aggregate statistics.
        """
        self._batch_count += 1
        profile = self._get_tension_profile()
        t_start = time.time()

        # ── Phase 1: Compute safety gate ────────────────────────
        autonomy = profile.get("autonomy_safety", 0.0)
        # High autonomy → high gate threshold (more actions pass)
        # Low autonomy (safe) → low gate threshold (fewer risky actions allowed)
        # τ ∈ [0, 1]: autonomy=-1 → τ=0 (only TRIVIAL passes), autonomy=+1 → τ=1 (all pass)
        tau = 0.5 + autonomy * 0.5  # Gate threshold
        # Risk value 0-4 mapped to [0, 1]; blocked if risk_q > τ
        blocked_ids: set[UUID] = set()

        for aid, action in self._actions.items():
            if self._states[aid] != ActionState.PENDING:
                continue
            risk_q = action.risk.value / 4.0  # max risk = 4 → q=1.0
            if risk_q > tau:
                # Action exceeds autonomy threshold
                if approve_fn is not None and approve_fn(action):
                    self._states[aid] = ActionState.QUEUED
                else:
                    self._states[aid] = ActionState.BLOCKED
                    blocked_ids.add(aid)
                    self._total_blocked += 1
            else:
                self._states[aid] = ActionState.QUEUED

        # ── Phase 1.5: Confidence-based safety gate ──────────────
        # When a calibrator is available, this extends the risk gate
        # with confidence awareness. Actions that require high confidence
        # are gated by calibrated (observed) confidence, not raw estimates.
        # This closes the metacognitive loop: Cognition learns calibration,
        # Praxis uses it to make smarter safety decisions.
        confidence_blocks = 0
        if self._confidence_calibrator is not None:
            # Confidence threshold modulated by autonomy tension:
            # autonomy=-1 (safe):    θ = 0.9 → very high bar, block uncertain actions
            # autonomy= 0 (neutral): θ = 0.7 → moderate bar
            # autonomy=+1 (auto):    θ = 0.5 → low bar, allow even uncertain actions
            conf_threshold = max(0.3, min(0.95, 0.7 - autonomy * 0.2))
            for aid, action in self._actions.items():
                if aid in blocked_ids:
                    continue  # Already blocked by risk gate
                if self._states[aid] not in (ActionState.QUEUED, ActionState.PENDING):
                    continue
                if action.confidence_required <= 0:
                    continue  # No confidence requirement — skip
                try:
                    calibrated = self._confidence_calibrator.calibrate_confidence(
                        action.confidence_required
                    )
                except Exception:
                    calibrated = action.confidence_required  # Fallback to raw
                if calibrated < conf_threshold:
                    if approve_fn is not None and approve_fn(action):
                        self._states[aid] = ActionState.QUEUED
                    else:
                        self._states[aid] = ActionState.BLOCKED
                        blocked_ids.add(aid)
                        self._total_blocked += 1
                        confidence_blocks += 1

        # ── Phase 2: Compute parallelism level ──────────────────
        parallel = profile.get("sequential_parallel", 0.0)
        max_concurrent = max(1, int(4 * (parallel + 1) / 2 + 1))
        max_concurrent = min(max_concurrent, self._max_parallel)

        # ── Phase 3: Schedule by topological levels ─────────────
        verify = profile.get("verify_execute", 0.0)
        omega = 0.5 * (1.0 - verify)  # verification depth

        completed_this_batch = 0
        failed_this_batch = 0
        retried_this_batch = 0

        # Build level groups of QUEUED actions
        level_groups: dict[int, list[UUID]] = defaultdict(list)
        for aid in self._actions:
            if self._states[aid] == ActionState.QUEUED:
                level_groups[self._topological_levels.get(aid, 0)].append(aid)

        # Process levels in ascending order
        for level in sorted(level_groups):
            queue = deque(level_groups[level])
            running: set[UUID] = set()

            while queue or running:
                # Fill running set up to max_concurrent
                while len(running) < max_concurrent and queue:
                    aid = queue.popleft()
                    # Skip actions already in terminal states (completed/failed/cancelled)
                    if self._states.get(aid) in (
                        ActionState.COMPLETED, ActionState.FAILED, ActionState.CANCELLED
                    ):
                        continue
                    action = self._actions[aid]
                    if not action.is_ready(frozenset(self._completed)):
                        # Not ready yet — push back (dependency not done)
                        # This should be rare with level-based scheduling
                        continue
                    running.add(aid)
                    self._states[aid] = ActionState.EXECUTING

                if not running:
                    break

                # Execute one running action (simulate concurrent with serial loop)
                aid = running.pop()
                action = self._actions[aid]
                policy = action.retry_policy

                # Execute with retry
                result = self._execute_single(
                    action, executor_fn, validator_fn, omega, policy
                )

                self._states[aid] = (
                    ActionState.COMPLETED if result.success else ActionState.FAILED
                )
                self._results[aid].append(result)
                self._total_executed += 1

                # Count retries (attempt > 0 means at least one failure was recovered from)
                if result.attempt > 0:
                    retried_this_batch += 1
                    self._total_retried += 1

                if result.success:
                    self._completed.add(aid)
                    completed_this_batch += 1
                    self._total_completed += 1
                    # Re-enqueue dependents that are now unblocked
                    for dep_id, dep_action in self._actions.items():
                        dep_state = self._states.get(dep_id)
                        dep_level = self._topological_levels.get(dep_id, 0)
                        # Only re-enqueue if dependent is at a HIGHER level
                        # (same-level actions are already in the queue being processed)
                        if (
                            dep_level > level
                            and dep_state in (ActionState.QUEUED, ActionState.BLOCKED)
                            and dep_action.is_ready(frozenset(self._completed))
                            and dep_id not in blocked_ids
                        ):
                            self._states[dep_id] = ActionState.QUEUED
                            queue.append(dep_id)
                else:
                    failed_this_batch += 1
                    self._total_failed += 1
                    if result.attempt < policy.max_retries:
                        # Re-queue for retry
                        self._states[aid] = ActionState.QUEUED
                        queue.append(aid)

                # Log for mneme
                self._execution_log.append({
                    "action_id": str(aid),
                    "description": action.description,
                    "tool_name": action.tool_name,
                    "success": result.success,
                    "error": result.error,
                    "attempt": result.attempt,
                    "duration_ms": result.duration_ms,
                    "validation_score": result.validation_score,
                    "batch": self._batch_count,
                    "tension_profile": dict(profile),
                })

        t_end = time.time()
        total_actions = len(self._actions) - len(blocked_ids)
        success_rate = completed_this_batch / max(1, total_actions)

        return ExecutionReport(
            actions_total=total_actions,
            actions_completed=completed_this_batch,
            actions_failed=failed_this_batch,
            actions_blocked=len(blocked_ids),
            actions_retried=retried_this_batch,
            total_duration_ms=(t_end - t_start) * 1000,
            success_rate=success_rate,
            avg_validation_score=self._compute_avg_validation(),
            parallelism_level=max_concurrent,
            gate_blocks=len(blocked_ids),
            confidence_blocks=confidence_blocks,
            calibration_applied=self._confidence_calibrator is not None,
            tension_profile=dict(profile),
        )

    def _execute_single(
        self,
        action: Action,
        executor_fn: Callable[[Action], Any],
        validator_fn: Callable[[Action, Any], tuple[bool, float]] | None,
        omega: float,
        policy: RetryPolicy,
    ) -> ExecutionResult:
        """Execute one action with retry and validation.

        Validation depth (omega) controls:
        - omega < 0.2: No validation (execute_fast)
        - 0.2 ≤ omega < 0.6: Basic validation (check output type)
        - omega ≥ 0.6: Full validation (validator_fn required)
        """
        last_error = ""
        output = None

        for attempt in range(policy.max_retries + 1):
            t0 = time.time()
            try:
                output = executor_fn(action)
                duration_ms = (time.time() - t0) * 1000
            except Exception as exc:
                last_error = str(exc)
                duration_ms = (time.time() - t0) * 1000
                if attempt < policy.max_retries:
                    delay = policy.delay_for_attempt(attempt)
                    if delay > 0:
                        time.sleep(delay)
                continue

            # Validation
            validation_passed = True
            validation_score = 1.0

            if omega < 0.2:
                # Fast mode — skip validation entirely, always pass
                validation_passed = True
                validation_score = 0.0
            elif omega < 0.6 and validator_fn is not None:
                # Light validation
                try:
                    validation_passed, validation_score = validator_fn(action, output)
                    validation_score = min(validation_score, omega * 2.0)
                except Exception:
                    validation_passed = False
                    validation_score = 0.0
            elif validator_fn is not None:
                # Full validation
                try:
                    validation_passed, validation_score = validator_fn(action, output)
                except Exception:
                    validation_passed = False
                    validation_score = 0.0

            # Check validation threshold (skip for fast mode)
            if omega >= 0.2:
                val_threshold = 0.3 + omega * 0.7
                if not validation_passed or validation_score < val_threshold:
                    if attempt < policy.max_retries:
                        delay = policy.delay_for_attempt(attempt)
                        if delay > 0:
                            time.sleep(delay)
                    continue

            return ExecutionResult(
                action_id=action.id,
                success=True,
                output=output,
                duration_ms=duration_ms,
                validation_passed=validation_passed,
                validation_score=validation_score,
                attempt=attempt,
            )

        # All retries exhausted
        return ExecutionResult(
            action_id=action.id,
            success=False,
            error=last_error,
            duration_ms=0.0,
            attempt=policy.max_retries,
        )

    # ── Cross-pillar pipelines ───────────────────────────────────

    def import_from_cognition(
        self,
        tasks: Sequence[dict[str, Any]],
    ) -> list[UUID]:
        """Convert cognition plan items into executable actions.

        Expected task dict format:
            {description, tool_name, params?, risk?, preconditions?,
             dependencies?, expected_outcome?, tags?, confidence_required?}

        This is the νοῦς → πρᾶξις pipeline entry point. The cognition
        pillar plans work; Praxis executes it.

        Args:
            tasks: Sequence of task specifications from cognition.

        Returns:
            List of registered action UUIDs.
        """
        # Build a mapping for resolving string dependency refs
        id_map: dict[str, UUID] = {}
        actions: list[Action] = []

        for task in tasks:
            risk_str = task.get("risk", "low").upper()
            try:
                risk = ActionRisk[risk_str]
            except KeyError:
                risk = ActionRisk.LOW

            action = Action(
                description=task["description"],
                tool_name=task.get("tool_name", "unknown"),
                params=task.get("params", {}),
                risk=risk,
                preconditions=tuple(task.get("preconditions", ())),
                expected_outcome=task.get("expected_outcome", ""),
                dependencies=(),  # Resolved below
                tags=tuple(task.get("tags", ())),
                metadata=task.get("metadata", {}),
                confidence_required=float(task.get("confidence_required", 0.0)),
            )
            ref = task.get("ref")
            if ref:
                id_map[ref] = action.id
            actions.append(action)

        # Resolve dependencies: string refs → UUIDs
        for i, task in enumerate(tasks):
            dep_refs = task.get("dependencies", ())
            if dep_refs:
                resolved = tuple(
                    id_map.get(r, actions[i].id) for r in dep_refs
                )
                # Rebuild with resolved deps
                actions[i] = Action(
                    description=actions[i].description,
                    tool_name=actions[i].tool_name,
                    params=actions[i].params,
                    risk=actions[i].risk,
                    preconditions=actions[i].preconditions,
                    expected_outcome=actions[i].expected_outcome,
                    dependencies=resolved,
                    tags=actions[i].tags,
                    metadata=actions[i].metadata,
                    confidence_required=actions[i].confidence_required,
                )

        return self.register_batch(actions)

    def export_to_mneme(self) -> list[dict[str, Any]]:
        """Export execution log entries for Mneme persistence.

        Each entry is a dict suitable for storage as a working→episodic
        memory. The mneme pillar can call this to persist execution
        results for cross-session learning.

        This is the πρᾶξις → μνήμη pipeline.

        Returns:
            List of log entries ready for memory storage.
        """
        result = list(self._execution_log)
        return result

    # ── Serialization ────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize the orchestrator state for cross-session persistence.

        Captures all actions, states, completed set, statistics, and
        recent execution log.
        """
        actions_dict = {
            str(aid): {
                "id": str(action.id),
                "description": action.description,
                "tool_name": action.tool_name,
                "params": action.params,
                "risk": action.risk.name,
                "preconditions": list(action.preconditions),
                "expected_outcome": action.expected_outcome,
                "dependencies": [str(d) for d in action.dependencies],
                "tags": list(action.tags),
                "metadata": action.metadata,
                "retry_max": action.retry_policy.max_retries,
                "retry_base_delay": action.retry_policy.base_delay,
                "retry_backoff": action.retry_policy.backoff_factor,
                "confidence_required": action.confidence_required,
            }
            for aid, action in self._actions.items()
        }

        return {
            "actions": actions_dict,
            "states": {str(aid): s.name for aid, s in self._states.items()},
            "completed": [str(aid) for aid in self._completed],
            "stats": {
                "total_executed": self._total_executed,
                "total_completed": self._total_completed,
                "total_failed": self._total_failed,
                "total_retried": self._total_retried,
                "total_blocked": self._total_blocked,
                "batch_count": self._batch_count,
            },
            "execution_log": self._execution_log[-100:],  # Keep last 100
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs: Any) -> ActionOrchestrator:
        """Deserialize from a to_dict() output.

        Rebuilds the full orchestrator state including action DAG,
        completion tracking, and statistics. Recomputes topological
        levels from the reconstructed DAG.

        Stats counters and execution log are rebuilt from deserialized
        actions, not from saved stats (stats dict may be stale).
        """
        orch = cls(**kwargs)

        # Rebuild actions
        for aid_str, adata in data.get("actions", {}).items():
            aid = UUID(aid_str)
            policy = RetryPolicy(
                max_retries=adata.get("retry_max", 3),
                base_delay=adata.get("retry_base_delay", 1.0),
                backoff_factor=adata.get("retry_backoff", 2.0),
            )
            action = Action(
                id=aid,
                description=adata["description"],
                tool_name=adata["tool_name"],
                params=adata.get("params", {}),
                risk=ActionRisk[adata.get("risk", "LOW")],
                preconditions=tuple(adata.get("preconditions", ())),
                expected_outcome=adata.get("expected_outcome", ""),
                dependencies=tuple(
                    UUID(d) for d in adata.get("dependencies", [])
                ),
                retry_policy=policy,
                tags=tuple(adata.get("tags", ())),
                metadata=adata.get("metadata", {}),
                confidence_required=float(adata.get("confidence_required", 0.0)),
            )
            orch._actions[aid] = action

        # Rebuild states
        for aid_str, state_name in data.get("states", {}).items():
            aid = UUID(aid_str)
            orch._states[aid] = ActionState[state_name]

        # Rebuild completed
        orch._completed = {
            UUID(aid_str) for aid_str in data.get("completed", [])
        }

        # Rebuild stats from deserialized data
        stats = data.get("stats", {})
        orch._total_executed = stats.get("total_executed", 0)
        orch._total_completed = stats.get("total_completed", 0)
        orch._total_failed = stats.get("total_failed", 0)
        orch._total_retried = stats.get("total_retried", 0)
        orch._total_blocked = stats.get("total_blocked", 0)
        orch._batch_count = stats.get("batch_count", 0)

        # Rebuild execution log
        orch._execution_log = list(data.get("execution_log", []))

        # Recompute topological levels from the DAG
        orch._recompute_topological_levels()

        return orch

    # ── Properties ───────────────────────────────────────────────

    @property
    def actions(self) -> tuple[Action, ...]:
        """All registered actions (immutable view)."""
        return tuple(self._actions.values())

    @property
    def pending_actions(self) -> tuple[Action, ...]:
        """Actions still pending or queued (not yet executed)."""
        return tuple(
            a for a in self._actions.values()
            if self._states.get(a.id) in (ActionState.PENDING, ActionState.QUEUED)
        )

    @property
    def completed_actions(self) -> tuple[Action, ...]:
        """Actions that have completed successfully."""
        return tuple(
            a for a in self._actions.values()
            if self._states.get(a.id) == ActionState.COMPLETED
        )

    @property
    def failed_actions(self) -> tuple[Action, ...]:
        """Actions that failed after all retries."""
        return tuple(
            a for a in self._actions.values()
            if self._states.get(a.id) == ActionState.FAILED
        )

    @property
    def blocked_actions(self) -> tuple[Action, ...]:
        """Actions blocked by the safety gate."""
        return tuple(
            a for a in self._actions.values()
            if self._states.get(a.id) == ActionState.BLOCKED
        )

    @property
    def total_actions(self) -> int:
        return len(self._actions)

    @property
    def dag_depth(self) -> int:
        """Maximum topological level (length of longest dependency chain)."""
        if not self._topological_levels:
            return 0
        return max(self._topological_levels.values())

    @property
    def stats(self) -> dict[str, Any]:
        """Aggregate statistics for monitoring and feedback."""
        return {
            "total_actions": self.total_actions,
            "total_executed": self._total_executed,
            "total_completed": self._total_completed,
            "total_failed": self._total_failed,
            "total_retried": self._total_retried,
            "total_blocked": self._total_blocked,
            "batch_count": self._batch_count,
            "dag_depth": self.dag_depth,
            "pending_count": len(self.pending_actions),
            "success_rate": (
                self._total_completed / max(1, self._total_executed)
            ),
        }

    @property
    def action_states(self) -> dict[UUID, ActionState]:
        """Current state of every action (mutable view for inspection)."""
        return dict(self._states)

    @property
    def execution_log(self) -> tuple[dict[str, Any], ...]:
        """Recent execution log entries (immutable view)."""
        return tuple(self._execution_log)

    def set_tension_profile(self, profile: dict[TensionID, float]) -> None:
        """Update the tension profile for modulation (called each tick).

        When an engine is available, profile is read from it automatically.
        This method is for standalone/test use.
        """
        self._tension_override = dict(profile)

    # ── Internal ─────────────────────────────────────────────────

    def _recompute_topological_levels(self) -> None:
        """Recompute topological levels for all actions in the DAG.

        Uses iterative DFS with memoization to compute the length of the
        longest dependency chain for each node.

        Level ℓ(v) = 0 if no dependencies, else max(ℓ(u) for (u→v) ∈ E) + 1.
        """
        self._topological_levels.clear()

        def compute_level(aid: UUID, visited: set[UUID] | None = None) -> int:
            if aid in self._topological_levels:
                return self._topological_levels[aid]
            if visited is None:
                visited = set()
            if aid in visited:
                return 0  # Cycle guard
            visited.add(aid)

            action = self._actions.get(aid)
            if action is None:
                return 0

            if not action.dependencies:
                level = 0
            else:
                max_dep = 0
                for dep_id in action.dependencies:
                    dep_level = compute_level(dep_id, visited.copy())
                    max_dep = max(max_dep, dep_level)
                level = max_dep + 1

            self._topological_levels[aid] = level
            return level

        for aid in self._actions:
            compute_level(aid)

    def _get_tension_profile(self) -> dict[TensionID, float]:
        """Get current tension profile (from engine or override or defaults)."""
        if hasattr(self, "_tension_override") and self._tension_override:
            return dict(self._tension_override)
        if self._engine is not None:
            try:
                return dict(self._engine.get_behavior_profile())
            except Exception:
                pass
        return dict(self._DEFAULT_PROFILE)

    def _compute_avg_validation(self) -> float:
        """Compute average validation score across completed actions."""
        scores = []
        for results in self._results.values():
            for r in results:
                if r.success and r.validation_score > 0:
                    scores.append(r.validation_score)
        if not scores:
            return 0.0
        return sum(scores) / len(scores)
