"""Task-Type Adaptive Homeostasis — curriculum learning for tension defaults.

Each task type (analysis, coding, research, etc.) produces a characteristic
equilibrium profile — the default positions that settle after repeated
exposure. This module records those trajectories and enables pre-adaptation:
when a new task of type "research" arrives, the agent immediately shifts
its defaults toward the configuration that worked best for previous
"research" tasks.

Architecture:

    submit_task(task_type="analysis")
                │
                ▼
    ┌───────────────────────────┐
    │  TaskTypeHomeostasis      │
    │                          │
    │  profiles:               │
    │    analysis → [v0, v1...]│  ← trajectories per task type
    │    coding  → [v0, v1...] │
    │    research→ [v0, v1...] │
    │                          │
    │  get_norm(task_type)     │  ← returns the converged default profile
    │  record(task_type, vec)  │  ← records a new observation
    │  apply_to(engine)        │  ← shifts engine defaults
    │                          │
    └───────────────────────────┘
                │
                ▼
    EquilibriumEngine.adjust_default() per axis
        → set points shift toward learned profile for this task type

Integration:
    - Agent.tick() records default positions at intervals after outcome processing
    - Agent.submit_task() or a pre_hook checks for learned task-type profile
    - Serialized via to_dict()/from_dict() on the equilibrium engine or agent

Theoretical foundation:
    Each task type has a homeostatic attractor — a set of tension defaults
    that minimize action failure rate. The system doesn't learn task-specific
    *behavior*; it learns what *equilibrium configuration* works best for
    each class of problem.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

import numpy as np

from isonome.equilibrium import EquilibriumEngine

logger = logging.getLogger(__name__)

# ── Default task-type taxonomy ──────────────────────────────────
# The built-in types the agent can recognize from task descriptions.
# Custom types can be added dynamically.

BUILTIN_TASK_TYPES: tuple[str, ...] = (
    "analysis",
    "coding",
    "research",
    "writing",
    "planning",
    "debugging",
    "design",
    "data_processing",
)


def infer_task_type(task_description: str) -> str:
    """Infer a task type from its description using keyword heuristics.

    This is intentionally simple — a production system could use an LLM
    or embedding classifier. The heuristic is rule-based and deterministic
    so the homeostatic learning loop is always reproducible.

    Returns one of the BUILTIN_TASK_TYPES, or 'general' if no match.
    """
    lower = task_description.lower()

    # Check from most specific to least specific to avoid false positives
    if any(kw in lower for kw in ("debug", "bug", "fix", "error", "crash", "traceback")):
        return "debugging"

    if any(kw in lower for kw in ("code", "implement", "program", "function", "class", "api")):
        return "coding"

    if any(kw in lower for kw in ("analy", "investigat", "evaluat", "assess", "examine")):
        return "analysis"

    if any(kw in lower for kw in ("research", "literature", "paper", "survey", "study", "find")):
        return "research"

    if any(kw in lower for kw in ("write", "draft", "compose", "essay", "report", "document")):
        return "writing"

    if any(kw in lower for kw in ("plan", "strategy", "roadmap", "schedule", "organize")):
        return "planning"

    if any(kw in lower for kw in ("design", "architecture", "layout", "ui", "mockup")):
        return "design"

    if any(kw in lower for kw in ("data", "transform", "extract", "load", "process", "pipeline", "etl")):
        return "data_processing"

    return "general"


class TaskTypeProfile:
    """A learned homeostatic profile for one task type.

    Stores the trajectory of default positions observed over N executions
    of tasks with this type. The profile converges as more observations
    accumulate, and the system can query whether the profile is stable
    enough to pre-adapt defaults.

    Attributes:
        task_type: The task type identifier.
        observations: List of default-position vectors, one per recording.
        axis_order: Canonical axis order (the same as the engine's DEFAULT_AXES).
        _converged: Whether the profile has reached convergence.
    """

    def __init__(
        self,
        task_type: str,
        axis_order: tuple[str, ...],
        observations: list[list[float]] | None = None,
    ):
        self.task_type = task_type
        self.axis_order = axis_order
        self._observations: list[list[float]] = observations or []

    @property
    def total_observations(self) -> int:
        return len(self._observations)

    @property
    def is_converged(self) -> bool:
        """Profile is converged when at least 3 observations and stable."""
        if len(self._observations) < 3:
            return False
        return self._compute_convergence_ratio() < 0.05

    @property
    def convergence_ratio(self) -> float:
        """How much the last observation moved the norm (0=perfectly stable).

        Returns np.inf when too few observations exist to compute
        convergence. This preserves the invariant:
            is_converged == (convergence_ratio < 0.05)
        for all observation counts, since np.inf is never < 0.05.
        """
        if len(self._observations) < 3:
            return float(np.inf)
        return self._compute_convergence_ratio()

    def _compute_convergence_ratio(self) -> float:
        """Compute the RMS change between the last two observations.

        A low value means the profile is stable — the system is consistently
        converging to the same defaults for this task type.

        Must not be called with fewer than 3 observations — convergence_ratio
        guards against that and returns np.inf instead.
        """
        if len(self._observations) < 3:
            return float(np.inf)

        # Use the average of the first half as the "initial" profile
        # and the average of the second half as the "current" profile
        mid = len(self._observations) // 2
        initial_avg = np.mean(self._observations[:mid], axis=0)
        recent_avg = np.mean(self._observations[mid:], axis=0)

        diff = recent_avg - initial_avg
        rms = float(np.sqrt(np.mean(diff ** 2)))
        return rms

    def get_norm(self) -> dict[str, float]:
        """Return the converged norm as a dict of axis_id → default_position.

        If the profile hasn't converged yet, returns None.
        """
        if not self._observations:
            return {}

        # The norm is the mean of all observations
        mean_vec = np.mean(self._observations, axis=0)

        return dict(zip(self.axis_order, [float(v) for v in mean_vec]))

    def record(self, default_vector: list[float]) -> None:
        """Record a new observation of default positions for this task type.

        Args:
            default_vector: Current default positions in axis_order.
        """
        self._observations.append(default_vector)
        logger.debug(
            f"TaskTypeProfile[{self.task_type}]: recorded observation "
            f"#{len(self._observations)} — convergence_ratio={self.convergence_ratio:.4f}, "
            f"converged={self.is_converged}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "task_type": self.task_type,
            "axis_order": list(self.axis_order),
            "observations": [[float(v) for v in obs] for obs in self._observations],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskTypeProfile:
        """Deserialize from a dict."""
        return cls(
            task_type=data.get("task_type", "unknown"),
            axis_order=tuple(data.get("axis_order", [])),
            observations=[[float(v) for v in obs] for obs in data.get("observations", [])],
        )


class TaskTypeHomeostasis:
    """Manages homeostatic profiles across all task types.

    This is the central coordinator. It:
    1. Maintains a profile per task type (including 'general')
    2. Knows the canonical axis order from the equilibrium engine
    3. Provides methods to record observations and query pre-adaptation
    4. Applies learned profiles to an equilibrium engine
    """

    def __init__(self, axis_order: tuple[str, ...] | None = None):
        """Initialize the homeostasis tracker.

        Args:
            axis_order: Canonical axis IDs in the order they appear in
                TensionAxis. If None, uses the standard 8-axis order.
        """
        self._axis_order = axis_order or (
            "explore_exploit",
            "shallow_deep",
            "divergent_convergent",
            "autonomy_safety",
            "sequential_parallel",
            "verify_execute",
            "consolidate_prune",
            "specific_general",
        )
        self._profiles: dict[str, TaskTypeProfile] = {}
        self._total_recordings: int = 0
        self._pre_adaptations_applied: int = 0

    # ── Properties ──────────────────────────────────────────────────

    @property
    def profile_count(self) -> int:
        """Number of task types with at least one observation."""
        return len(self._profiles)

    @property
    def known_task_types(self) -> tuple[str, ...]:
        """All task types that have at least one recorded observation."""
        return tuple(self._profiles.keys())

    @property
    def converged_task_types(self) -> tuple[str, ...]:
        """Task types whose profile has converged (stable enough to use)."""
        return tuple(
            t for t, p in self._profiles.items() if p.is_converged
        )

    @property
    def total_recordings(self) -> int:
        return self._total_recordings

    @property
    def pre_adaptations_applied(self) -> int:
        return self._pre_adaptations_applied

    # ── Core API ────────────────────────────────────────────────────

    def get_profile(self, task_type: str) -> TaskTypeProfile | None:
        """Get the profile for a task type, creating it if it doesn't exist."""
        return self._profiles.get(task_type)

    def get_or_create_profile(self, task_type: str) -> TaskTypeProfile:
        """Get an existing profile or create a new one."""
        if task_type not in self._profiles:
            self._profiles[task_type] = TaskTypeProfile(
                task_type=task_type,
                axis_order=self._axis_order,
            )
        return self._profiles[task_type]

    def record_defaults(self, engine: EquilibriumEngine, task_type: str | None = None) -> None:
        """Record the current engine default positions for a task type.

        This should be called after outcome processing in the agent loop,
        so the recorded defaults reflect any adjustment_default calls.

        Args:
            engine: The equilibrium engine to read defaults from.
            task_type: The task type to associate with this recording.
                If None, uses 'general'.
        """
        task_type = task_type or "general"

        # Extract default positions in canonical order
        default_vector: list[float] = []
        for axis_id in self._axis_order:
            axis = engine.get_axis(axis_id)
            if axis is not None:
                default_vector.append(axis.default_position)

        if not default_vector:
            return

        profile = self.get_or_create_profile(task_type)
        profile.record(default_vector)
        self._total_recordings += 1

    def apply_task_type_profile(self, engine: EquilibriumEngine, task_type: str) -> int:
        """Apply a learned profile to the engine's defaults.

        Only applies if:
        1. The task type has a profile
        2. The profile has converged (≥3 observations, stable)
        3. The learned defaults differ from current defaults

        Returns the number of axes that were adjusted (0 if none).
        """
        profile = self._profiles.get(task_type)
        if profile is None:
            return 0

        if not profile.is_converged:
            logger.debug(
                f"TaskTypeHomeostasis: skipping pre-adaptation for "
                f"'{task_type}' — not converged ({profile.total_observations} obs, "
                f"ratio={profile.convergence_ratio:.4f})"
            )
            return 0

        norm = profile.get_norm()
        if not norm:
            return 0

        adjusted_count = 0
        for axis_id, learned_default in norm.items():
            axis = engine.get_axis(axis_id)
            if axis is None:
                continue

            current_default = axis.default_position

            # Only adjust if the learned default is meaningfully different
            if abs(current_default - learned_default) < 0.001:
                continue

            # Compute the signal that moves current_default → learned_default
            # via adjust_default's formula: new = current + signal × learning_rate
            # ⇒ signal = (learned_default - current_default) / learning_rate
            learning_rate = axis.learning_rate
            if learning_rate > 0:
                needed_signal = (learned_default - current_default) / learning_rate
                engine.adjust_default(axis_id, outcome_signal=needed_signal)
                adjusted_count += 1

        if adjusted_count > 0:
            self._pre_adaptations_applied += 1
            logger.info(
                f"TaskTypeHomeostasis: pre-adapted {adjusted_count} axes for "
                f"task type '{task_type}'"
            )

        return adjusted_count

    def soft_pre_adapt(self, engine: EquilibriumEngine, task_type: str) -> int:
        """Apply a softened version of the learned profile.

        Unlike apply_task_type_profile which moves to the learned default
        in a single step, soft_pre_adapt moves one-third of the distance.
        This is gentler when the profile may be outdated or the task type
        is only loosely matched.

        Returns the number of axes adjusted.
        """
        profile = self._profiles.get(task_type)
        if profile is None:
            return 0

        if not profile.is_converged:
            return 0

        norm = profile.get_norm()
        if not norm:
            return 0

        adjusted_count = 0
        for axis_id, learned_default in norm.items():
            axis = engine.get_axis(axis_id)
            if axis is None:
                continue

            current_default = axis.default_position
            if abs(current_default - learned_default) < 0.001:
                continue

            # Move one-third of the way to the learned default
            target_default = current_default + (learned_default - current_default) / 3.0
            learning_rate = axis.learning_rate
            if learning_rate > 0:
                needed_signal = (target_default - current_default) / learning_rate
                engine.adjust_default(axis_id, outcome_signal=needed_signal)
                adjusted_count += 1

        if adjusted_count > 0:
            self._pre_adaptations_applied += 1
            logger.info(
                f"TaskTypeHomeostasis: soft pre-adapted {adjusted_count} axes for "
                f"task type '{task_type}'"
            )
        return adjusted_count

    def get_profile_similarity(
        self, engine: EquilibriumEngine, task_type: str
    ) -> float:
        """How similar the current engine defaults are to the learned profile.

        Returns 1.0 if identical, 0.0 if orthogonal, or 0.0 if no profile exists.
        Uses cosine similarity on the default-position vector.
        """
        profile = self._profiles.get(task_type)
        if profile is None or not profile._observations:
            return 0.0

        norm = profile.get_norm()
        if not norm:
            return 0.0

        current_vec: list[float] = []
        learned_vec: list[float] = []
        for axis_id in self._axis_order:
            axis = engine.get_axis(axis_id)
            if axis is not None and axis_id in norm:
                current_vec.append(axis.default_position)
                learned_vec.append(norm[axis_id])

        if not current_vec:
            return 0.0

        c = np.array(current_vec)
        l = np.array(learned_vec)
        norm_c = np.linalg.norm(c)
        norm_l = np.linalg.norm(l)
        if norm_c == 0 or norm_l == 0:
            return 0.0
        return float(np.dot(c, l) / (norm_c * norm_l))

    # ── Statistics ──────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return a summary dict for reporting."""
        return {
            "profile_count": self.profile_count,
            "known_task_types": self.known_task_types,
            "converged_task_types": self.converged_task_types,
            "total_recordings": self._total_recordings,
            "pre_adaptations_applied": self._pre_adaptations_applied,
            "profiles": {
                ttype: {
                    "observations": p.total_observations,
                    "converged": p.is_converged,
                    "convergence_ratio": round(p.convergence_ratio, 4),
                }
                for ttype, p in self._profiles.items()
            },
        }

    # ── Serialization ───────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize all profiles for cross-session persistence."""
        return {
            "axis_order": list(self._axis_order),
            "profiles": {
                ttype: p.to_dict() for ttype, p in self._profiles.items()
            },
            "total_recordings": self._total_recordings,
            "pre_adaptations_applied": self._pre_adaptations_applied,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskTypeHomeostasis:
        """Deserialize from a dict produced by to_dict()."""
        ht = cls(
            axis_order=tuple(data.get("axis_order", (
                "explore_exploit",
                "shallow_deep",
                "divergent_convergent",
                "autonomy_safety",
                "sequential_parallel",
                "verify_execute",
                "consolidate_prune",
                "specific_general",
            ))),
        )
        for ttype, pdata in data.get("profiles", {}).items():
            ht._profiles[ttype] = TaskTypeProfile.from_dict(pdata)
        ht._total_recordings = int(data.get("total_recordings", 0))
        ht._pre_adaptations_applied = int(data.get("pre_adaptations_applied", 0))
        return ht
