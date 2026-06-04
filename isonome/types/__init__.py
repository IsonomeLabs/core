"""Core type system for the isonome framework.

All types are frozen Pydantic models — immutable, hashable, serializable.
This enables safe sharing across concurrent agent processes and
building consistent audit trails.
"""

from __future__ import annotations

from abc import ABC, abstractmethod  # noqa: F401 — re-exports
from dataclasses import dataclass, field  # noqa: F401 — re-exports
from datetime import datetime, timezone
from enum import Enum, StrEnum  # noqa: F401 — Enum re-export
from typing import (
    Any,
    Callable,  # noqa: F401
    Generic,  # noqa: F401
    Literal,  # noqa: F401
    Mapping,
    MutableMapping,  # noqa: F401
    Protocol,
    Sequence,
    TypeAlias,
    TypeVar,  # noqa: F401
)
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# ═══════════════════════════════════════════════════════════════════
# Fundamental types
# ═══════════════════════════════════════════════════════════════════

AgentID: TypeAlias = UUID
TensionID: TypeAlias = str
Token: TypeAlias = int
Timestamp: TypeAlias = datetime


def now() -> Timestamp:
    """UTC now — single source of truth for all timestamps."""
    return datetime.now(timezone.utc)


# ═══════════════════════════════════════════════════════════════════
# The Three Pillars (domain classification)
# ═══════════════════════════════════════════════════════════════════

class Pillar(StrEnum):
    """The three foundational domains of agent operation."""

    COGNITION = "cognition"  # νοῦς — reasoning, planning, context
    PRAXIS = "praxis"  # πρᾶξις — tool use, execution, orchestration
    MNEME = "mneme"  # μνήμη — memory, learning, persistence


# ═══════════════════════════════════════════════════════════════════
# Tension system — the heart of equilibrium
# ═══════════════════════════════════════════════════════════════════

class TensionAxis(BaseModel):
    """A single continuous dimension that the equilibrium engine balances.

    Each axis has a current position [-1.0, 1.0] representing where the
    agent sits between two competing poles. Zero is perfect balance.

    The engine adjusts positions dynamically based on feedback signals,
    task characteristics, and learned priors.

    Examples:
        explore(<0) ← → exploit(>0)
        fast(<0)    ← → thorough(>0)
        safe(<0)    ← → autonomous(>0)
        shallow(<0) ← → deep(>0)
    """

    model_config = ConfigDict(frozen=True)

    id: TensionID = Field(description="Unique identifier, e.g. 'explore_exploit'")
    pillar: Pillar = Field(description="Which pillar this tension belongs to")
    pole_left: str = Field(description="Name of the negative pole, e.g. 'explore'")
    pole_right: str = Field(description="Name of the positive pole, e.g. 'exploit'")
    position: float = Field(
        default=0.0, ge=-1.0, le=1.0, description="Current position on the axis"
    )
    default_position: float = Field(
        default=0.0, ge=-1.0, le=1.0, description="Homeostasis target"
    )
    damping: float = Field(
        default=0.3, ge=0.0, le=1.0, description="How resistant to change (0=fluid, 1=rigid)"
    )
    learning_rate: float = Field(
        default=0.1, ge=0.0, le=1.0, description="How fast the default_position adapts"
    )
    clip: tuple[float, float] = Field(
        default=(-1.0, 1.0), description="Hard bounds for position"
    )

    def adjust(self, delta: float, *, clip: bool = True) -> TensionAxis:
        """Return a new TensionAxis with position shifted by delta.

        Args:
            delta: Raw shift amount. Damping is applied internally.
            clip: Whether to enforce hard bounds.
        """
        effective_delta = delta * (1.0 - self.damping)
        new_pos = self.position + effective_delta
        if clip:
            new_pos = max(self.clip[0], min(self.clip[1], new_pos))
        return self.model_copy(update={"position": new_pos})

    def distance_from_default(self) -> float:
        """How far the current position has drifted from homeostasis."""
        return abs(self.position - self.default_position)

    def __repr__(self) -> str:
        """Compact debug representation showing the tension position."""
        return (
            f"TensionAxis({self.id!r}, "
            f"pos={self.position:+.3f}, "
            f"default={self.default_position:+.3f}, "
            f"damping={self.damping:.2f})"
        )


class TensionSnapshot(BaseModel):
    """A point-in-time capture of all tension axes.

    Used for audit trails, learning, and cross-session state transfer.
    """

    model_config = ConfigDict(frozen=True)

    timestamp: Timestamp = Field(default_factory=now)
    axes: frozenset[TensionAxis] = Field(description="All axis states at this moment")
    agent_id: AgentID | None = Field(default=None)
    trigger: str | None = Field(
        default=None, description="What caused this snapshot (task_complete, manual, etc.)"
    )

    def get(self, axis_id: TensionID) -> TensionAxis | None:
        for axis in self.axes:
            if axis.id == axis_id:
                return axis
        return None

    def to_vector(self, axis_ids: Sequence[TensionID]) -> np.ndarray:
        """Extract positions as a numpy vector for ML operations."""
        lookup = {a.id: a.position for a in self.axes}
        return np.array([lookup.get(aid, 0.0) for aid in axis_ids], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
# Agent identity & lifecycle
# ═══════════════════════════════════════════════════════════════════

class AgentLifecycle(StrEnum):
    """States in the agent lifecycle."""

    CREATED = "created"
    BOOTSTRAPPING = "bootstrapping"  # Loading memories, warming caches
    IDLE = "idle"
    REASONING = "reasoning"  # Cognition active
    ACTING = "acting"  # Praxis active
    CONSOLIDATING = "consolidating"  # Mneme active (learning)
    PAUSED = "paused"
    TERMINATED = "terminated"


class AgentIdentity(BaseModel):
    """Immutable identity for an agent instance."""

    model_config = ConfigDict(frozen=True)

    id: AgentID = Field(default_factory=uuid4)
    name: str = Field(min_length=1, max_length=128)
    version: str = Field(default="0.1.0")
    created_at: Timestamp = Field(default_factory=now)
    parent_id: AgentID | None = Field(
        default=None, description="For delegated/spawned sub-agents"
    )


class AgentState(BaseModel):
    """Mutable state that changes as the agent runs."""

    model_config = ConfigDict(frozen=False)

    identity: AgentIdentity
    lifecycle: AgentLifecycle = Field(default=AgentLifecycle.CREATED)
    tensions: TensionSnapshot | None = Field(default=None)
    task_count: int = Field(default=0, ge=0)
    error_count: int = Field(default=0, ge=0)
    tokens_consumed: int = Field(default=0, ge=0)
    last_active: Timestamp = Field(default_factory=now)
    metadata: Mapping[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# Message types for inter-pillar communication
# ═══════════════════════════════════════════════════════════════════

class Signal(BaseModel):
    """A message passed between pillars or from the equilibrium engine."""

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    source: Pillar
    target: Pillar
    kind: str = Field(description="Signal type, e.g. 'tension_adjusted', 'plan_ready'")
    payload: Mapping[str, Any] = Field(default_factory=dict)
    timestamp: Timestamp = Field(default_factory=now)
    priority: int = Field(default=0, ge=0, le=10)


class Feedback(BaseModel):
    """Structured feedback from any pillar back to the equilibrium engine.

    Feedback signals cause tension adjustments. Positive feedback
    pushes toward the right pole; negative toward the left.
    """

    model_config = ConfigDict(frozen=True)

    source: Pillar
    tension_axis_id: TensionID
    signal: float = Field(ge=-1.0, le=1.0, description="Direction and magnitude")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    timestamp: Timestamp = Field(default_factory=now)


# ═══════════════════════════════════════════════════════════════════
# Task representation
# ═══════════════════════════════════════════════════════════════════

class TaskStatus(StrEnum):
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    AWAITING_FEEDBACK = "awaiting_feedback"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskComplexity(StrEnum):
    TRIVIAL = "trivial"  # < 3 steps, no dependencies
    SIMPLE = "simple"  # 3-10 steps
    MODERATE = "moderate"  # 10-30 steps, some parallelism
    COMPLEX = "complex"  # 30-100 steps, significant planning
    WICKED = "wicked"  # 100+ steps, emergent, requires replanning


class Task(BaseModel):
    """A unit of work that flows through all three pillars."""

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    description: str
    complexity: TaskComplexity = Field(default=TaskComplexity.SIMPLE)
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    parent_id: UUID | None = Field(default=None)
    subtasks: tuple[UUID, ...] = Field(default_factory=tuple)
    created_at: Timestamp = Field(default_factory=now)
    deadline: Timestamp | None = Field(default=None)
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    def is_atomic(self) -> bool:
        return len(self.subtasks) == 0


# ═══════════════════════════════════════════════════════════════════
# Core exceptions
# ═══════════════════════════════════════════════════════════════════

class IsonomeError(Exception):
    """Base exception for all isonome errors."""


class EquilibriumError(IsonomeError):
    """Error in the equilibrium engine."""


class TensionOscillationError(EquilibriumError):
    """Tension axis is oscillating beyond acceptable bounds."""


class PillarError(IsonomeError):
    """Error originating from one of the three pillars."""


class CognitionError(PillarError):
    """Error in the νοῦς (cognition) pillar."""


class PraxisError(PillarError):
    """Error in the πρᾶξις (praxis) pillar."""


class MnemeError(PillarError):
    """Error in the μνήμη (mneme) pillar."""


# ═══════════════════════════════════════════════════════════════════
# Protocol interfaces for pillars
# ═══════════════════════════════════════════════════════════════════

class PillarProtocol(Protocol):
    """Structural protocol that all pillar implementations must satisfy."""

    pillar: Pillar

    def initialize(self, agent_state: AgentState) -> None: ...

    def receive_signal(self, signal: Signal) -> None: ...

    def shutdown(self) -> None: ...
