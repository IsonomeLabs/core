"""Base pillar implementation.

Each pillar (Cognition, Praxis, Mneme) extends BasePillar and
implements the three lifecycle methods: initialize, receive_signal, shutdown.

Pillars communicate via Signal messages and provide Feedback to the
equilibrium engine to adjust tension positions.
"""

from __future__ import annotations

import abc
import logging
from typing import Sequence

from isonome.types import (
    AgentState,
    Feedback,
    IsonomeError,
    Pillar,
    PillarProtocol,
    Signal,
)

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
        """Process all queued signals (call during the tick loop)."""
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
