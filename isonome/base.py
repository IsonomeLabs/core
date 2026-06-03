"""Base pillar implementation.

Each pillar (Cognition, Praxis, Mneme) extends BasePillar and
implements the three lifecycle methods: initialize, receive_signal, shutdown.

Pillars communicate via Signal messages and provide Feedback to the
equilibrium engine to adjust tension positions.
"""

from __future__ import annotations

import abc
import logging
import math
from typing import Sequence

from isonome.types import (
    AgentState,
    Feedback,
    IsonomeError,
    Pillar,
    PillarProtocol,
    Signal,
)

# Import at module level for type hints; the actual class lives in
# equilibrium/__init__.py to avoid circular imports at runtime.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isonome.equilibrium import EquilibriumEngine, PillarEquilibriumView

logger = logging.getLogger(__name__)


class BasePillar(PillarProtocol, abc.ABC):
    """Abstract base for all three pillars.

    Provides:
    - Signal queue management
    - Feedback pipeline to the equilibrium engine
    - Lifecycle state tracking
    - Structured logging

    Subclasses must implement:
    - pillar (property): which pillar this is
    - _on_signal(signal): handle incoming signals
    - _on_initialize(state): setup
    - _on_shutdown(): teardown
    """

    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.__name__
        self._signal_queue: list[Signal] = []
        self._pending_feedback: list[Feedback] = []
        self._initialized = False
        self._state: AgentState | None = None

        # ── Equilibrium pull mechanism ────────────────────────────
        # When bound to an engine, the pillar auto-syncs its tension
        # view on each process_queued() call — no external push needed.
        self._engine: EquilibriumEngine | None = None  # type: ignore[name-defined]
        self._equilibrium_view: PillarEquilibriumView | None = None  # type: ignore[name-defined]
        self._stress_feedback_enabled: bool = True

    # ── Abstract interface ──────────────────────────────────────

    @property
    @abc.abstractmethod
    def pillar(self) -> Pillar:
        """Which of the three pillars this implementation belongs to."""
        ...

    @abc.abstractmethod
    def _on_signal(self, signal: Signal) -> None:
        """Handle an incoming signal from another pillar."""
        ...

    @abc.abstractmethod
    def _on_initialize(self, state: AgentState) -> None:
        """Initialize pillar-specific resources."""
        ...

    @abc.abstractmethod
    def _on_shutdown(self) -> None:
        """Release pillar-specific resources."""
        ...

    # ── Public API ───────────────────────────────────────────────

    def initialize(self, agent_state: AgentState) -> None:
        """Called once when the agent boots."""
        if self._initialized:
            logger.warning(f"{self.name}: already initialized")
            return
        self._state = agent_state
        self._on_initialize(agent_state)
        self._initialized = True
        logger.info(f"{self.name}: initialized (pillar={self.pillar})")

    def receive_signal(self, signal: Signal) -> None:
        """Queue a signal for processing."""
        if signal.target != self.pillar:
            logger.warning(
                f"{self.name}: received signal targeting {signal.target}, "
                f"but I am {self.pillar}"
            )
        self._signal_queue.append(signal)

    def shutdown(self) -> None:
        """Called when the agent terminates."""
        logger.info(f"{self.name}: shutting down")
        self._on_shutdown()
        self._initialized = False

    # ── Helpers for subclasses ───────────────────────────────────

    def emit_feedback(self, feedback: Feedback) -> None:
        """Add feedback to the pending queue for the equilibrium engine."""
        if feedback.source != self.pillar:
            raise IsonomeError(
                f"Feedback source {feedback.source} doesn't match pillar {self.pillar}"
            )
        self._pending_feedback.append(feedback)

    def drain_feedback(self) -> list[Feedback]:
        """Pop and return all pending feedback (up to limit)."""
        result = self._pending_feedback[:]
        self._pending_feedback.clear()
        return result

    def drain_signals(self) -> list[Signal]:
        """Pop and return all queued signals."""
        result = self._signal_queue[:]
        self._signal_queue.clear()
        return result

    def process_queued(self) -> None:
        """Process all queued signals (call during the tick loop).

        When the pillar is bound to an engine, this also:
        1. Auto-syncs the equilibrium view (pull mechanism)
        2. Generates stress-reactive feedback if under pressure
        3. Calls the pillar's _on_equilibrium_sync() hook if defined
        """
        # ── Auto-sync equilibrium view ────────────────────────────
        if self._engine is not None:
            self._equilibrium_view = self._engine.view_for(self.pillar)
            # Stress-reactive feedback: if the agent is highly stressed,
            # emit a gentle pull-toward-homeostasis signal on the
            # most-drifted own axis. This is the self-regulation loop —
            # the pillar doesn't just read tension, it actively
            # resists excessive drift from homeostasis.
            if self._stress_feedback_enabled:
                self._emit_stress_feedback()
            # Hook for pillar-specific equilibrium sync
            self._on_equilibrium_sync(self._equilibrium_view)

        signals = self.drain_signals()
        for signal in signals:
            try:
                self._on_signal(signal)
            except Exception:
                logger.exception(f"{self.name}: error processing signal {signal.id}")

    @property
    def state(self) -> AgentState | None:
        return self._state

    @property
    def initialized(self) -> bool:
        return self._initialized

    # ── Equilibrium pull mechanism ────────────────────────────────

    def bind_engine(self, engine: EquilibriumEngine) -> None:  # type: ignore[name-defined]
        """Bind this pillar to an EquilibriumEngine for auto-sync.

        After binding, process_queued() automatically pulls a fresh
        PillarEquilibriumView on each tick — no external
        update_tension_profile() calls needed.

        Args:
            engine: The shared equilibrium engine instance.

        Raises:
            IsonomeError: If the pillar is already bound to a different engine.
        """
        if self._engine is not None and self._engine is not engine:
            raise IsonomeError(
                f"{self.name}: already bound to a different engine. "
                f"Call unbind_engine() first."
            )
        self._engine = engine
        # Immediately create an initial view
        self._equilibrium_view = engine.view_for(self.pillar)
        logger.info(
            f"{self.name}: bound to equilibrium engine "
            f"(stress={self._equilibrium_view.stress_level:.3f})"
        )

    def unbind_engine(self) -> None:
        """Disconnect this pillar from its bound equilibrium engine."""
        self._engine = None
        self._equilibrium_view = None
        logger.info(f"{self.name}: unbound from equilibrium engine")

    @property
    def equilibrium_view(self) -> PillarEquilibriumView | None:  # type: ignore[name-defined]
        """Current equilibrium view, or None if not bound to an engine.

        Updated automatically on each process_queued() call when
        bound to an engine. Pillars can also read this at any time
        between ticks for the latest view.
        """
        return self._equilibrium_view

    @property
    def engine(self) -> EquilibriumEngine | None:  # type: ignore[name-defined]
        """The bound equilibrium engine, or None."""
        return self._engine

    # ── Hooks for subclasses ──────────────────────────────────────

    def _on_equilibrium_sync(self, view: PillarEquilibriumView) -> None:  # type: ignore[name-defined]
        """Hook called after each equilibrium view sync in process_queued().

        Override in subclasses to implement pillar-specific tension
        modulation behavior. For example:
        - Cognition: adjust reasoning depth based on shallow_deep
        - Praxis: modulate parallelism based on sequential_parallel
        - Mneme: adjust consolidation aggressiveness based on consolidate_prune

        The default implementation does nothing — pillars that don't
        need auto-modulation can skip overriding.

        Args:
            view: The freshly synced equilibrium view for this tick.
        """
        # Default: no-op. Subclasses override for pillar-specific behavior.
        pass

    def _emit_stress_feedback(self) -> None:
        """Emit stress-reactive feedback toward homeostasis on own axes.

        When the agent is highly stressed (far from equilibrium), this
        generates a gentle pull-toward-default signal on the most-drifted
        own axis. The signal strength scales with stress level:

        - stress_level ≤ 0.3: no feedback (healthy drift is normal)
        - 0.3 < stress_level ≤ 0.5: mild feedback (σ × 0.05 toward default)
        - stress_level > 0.5: moderate feedback (σ × 0.10 toward default)

        The feedback is intentionally weak — it's a homeostatic pull,
        not a forceful correction. It coexists with the pillar's
        own task-driven feedback and simply biases toward stability.

        This is the self-regulation complement to the external feedback
        loop: pillars push Feedback to the engine (changing state),
        and the engine's stress level pulls them back toward homeostasis.
        """
        if self._equilibrium_view is None:
            return

        stress = self._equilibrium_view.stress_level
        if stress <= 0.3:
            return  # Healthy — no auto-correction needed

        # Find the most-drifted own axis
        own_axes = self._equilibrium_view.own_axes
        if not own_axes:
            return  # No own axes (shouldn't happen in practice)

        drift = self._equilibrium_view.drift
        own_drift = {k: v for k, v in drift.items() if k in own_axes}
        if not own_drift:
            return

        max_axis = max(own_drift, key=own_drift.get)  # type: ignore[arg-type]
        max_drift_val = own_drift[max_axis]

        if max_drift_val < 0.1:
            return  # Drift is tiny even if stress is high — skip

        # Compute the direction toward homeostasis
        current_pos = own_axes[max_axis]
        default_pos = self._equilibrium_view.all_defaults.get(max_axis, 0.0)
        direction = default_pos - current_pos  # Positive = toward default
        if abs(direction) < 1e-6:
            return

        # Scale signal: gentle pull, stronger when more stressed
        if stress > 0.5:
            magnitude = min(abs(direction), stress * 0.10)
        else:
            magnitude = min(abs(direction), stress * 0.05)

        signal_val = math.copysign(magnitude, direction)

        self.emit_feedback(
            Feedback(
                source=self.pillar,
                tension_axis_id=max_axis,
                signal=max(-1.0, min(1.0, signal_val)),
                confidence=0.4,  # Low confidence — this is a gentle nudge
                reason=f"stress-reactive homeostasis pull "
                       f"(stress={stress:.3f}, drift={max_drift_val:.3f} "
                       f"on {max_axis})",
            )
        )
