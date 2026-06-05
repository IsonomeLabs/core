"""Tests for isonome.base — BasePillar lifecycle, equilibrium pull, stress feedback."""
from __future__ import annotations

import pytest
from collections import deque

from isonome.base import BasePillar
from isonome.types import (
    AgentIdentity,
    AgentLifecycle,
    AgentState,
    Feedback,
    IsonomeError,
    Pillar,
    Signal,
    TensionAxis,
    TensionID,
)
from isonome.equilibrium import EquilibriumEngine


class CognitionPillar(BasePillar):
    """Minimal concrete BasePillar for testing."""

    _signals_received: list[Signal]

    @property
    def pillar(self) -> Pillar:
        return Pillar.COGNITION

    def _on_signal(self, signal: Signal) -> None:
        self._signals_received.append(signal)

    def _on_initialize(self, state: AgentState) -> None:
        self._signals_received = []

    def _on_shutdown(self) -> None:
        self._signals_received = []


class PraxisPillar(BasePillar):
    """Minimal Praxis pillar for testing."""

    @property
    def pillar(self) -> Pillar:
        return Pillar.PRAXIS

    def _on_signal(self, signal: Signal) -> None:
        pass

    def _on_initialize(self, state: AgentState) -> None:
        pass

    def _on_shutdown(self) -> None:
        pass


def _make_state() -> AgentState:
    identity = AgentIdentity(name="test_agent")
    return AgentState(identity=identity, lifecycle=AgentLifecycle.CREATED)


class TestBasePillarLifecycle:
    def test_initialize(self):
        p = CognitionPillar()
        assert not p.initialized
        p.initialize(_make_state())
        assert p.initialized
        assert p.state is not None

    def test_double_initialize_is_noop(self):
        p = CognitionPillar()
        p.initialize(_make_state())
        p.initialize(_make_state())  # Should not raise
        assert p.initialized

    def test_shutdown(self):
        p = CognitionPillar()
        p.initialize(_make_state())
        p.shutdown()
        assert not p.initialized

    def test_pillar_property(self):
        p = CognitionPillar()
        assert p.pillar == Pillar.COGNITION


class TestBasePillarSignals:
    def test_receive_signal(self):
        p = CognitionPillar()
        p.initialize(_make_state())
        sig = Signal(source=Pillar.PRAXIS, target=Pillar.COGNITION, kind="test")
        p.receive_signal(sig)
        assert len(p._signal_queue) == 1

    def test_drain_signals(self):
        p = CognitionPillar()
        p.initialize(_make_state())
        sig1 = Signal(source=Pillar.PRAXIS, target=Pillar.COGNITION, kind="test1")
        sig2 = Signal(source=Pillar.MNEME, target=Pillar.COGNITION, kind="test2")
        p.receive_signal(sig1)
        p.receive_signal(sig2)
        drained = p.drain_signals()
        assert len(drained) == 2
        assert len(p._signal_queue) == 0  # queue is empty after drain

    def test_process_queued_calls_on_signal(self):
        p = CognitionPillar()
        p.initialize(_make_state())
        sig = Signal(source=Pillar.PRAXIS, target=Pillar.COGNITION, kind="test")
        p.receive_signal(sig)
        p.process_queued()
        assert len(p._signals_received) == 1

    def test_process_queued_handles_exception(self):
        """process_queued should not propagate exceptions from _on_signal."""
        class BrokenPillar(BasePillar):
            @property
            def pillar(self) -> Pillar:
                return Pillar.COGNITION

            def _on_signal(self, signal: Signal) -> None:
                raise RuntimeError("boom")

            def _on_initialize(self, state: AgentState) -> None:
                pass

            def _on_shutdown(self) -> None:
                pass

        p = BrokenPillar()
        p.initialize(_make_state())
        sig = Signal(source=Pillar.PRAXIS, target=Pillar.COGNITION, kind="test")
        p.receive_signal(sig)
        # Should not raise
        p.process_queued()


class TestBasePillarFeedback:
    def test_emit_feedback(self):
        p = CognitionPillar()
        p.initialize(_make_state())
        fb = Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.5,
            reason="test feedback",
        )
        p.emit_feedback(fb)
        assert len(p._pending_feedback) == 1

    def test_emit_feedback_wrong_source_raises(self):
        p = CognitionPillar()
        p.initialize(_make_state())
        fb = Feedback(
            source=Pillar.PRAXIS,  # Wrong source
            tension_axis_id="explore_exploit",
            signal=0.5,
            reason="wrong source",
        )
        with pytest.raises(IsonomeError):
            p.emit_feedback(fb)

    def test_drain_feedback(self):
        p = CognitionPillar()
        p.initialize(_make_state())
        fb = Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3,
            reason="test1",
        )
        p.emit_feedback(fb)
        drained = p.drain_feedback()
        assert len(drained) == 1
        assert len(p._pending_feedback) == 0


class TestBasePillarEquilibrium:
    def test_bind_engine(self):
        p = CognitionPillar()
        engine = EquilibriumEngine()
        p.bind_engine(engine)
        assert p.engine is engine
        assert p.equilibrium_view is not None

    def test_bind_engine_twice_same_is_ok(self):
        p = CognitionPillar()
        engine = EquilibriumEngine()
        p.bind_engine(engine)
        p.bind_engine(engine)  # Same engine — no error
        assert p.engine is engine

    def test_bind_engine_twice_different_raises(self):
        p = CognitionPillar()
        e1 = EquilibriumEngine()
        e2 = EquilibriumEngine()
        p.bind_engine(e1)
        with pytest.raises(IsonomeError):
            p.bind_engine(e2)

    def test_unbind_engine(self):
        p = CognitionPillar()
        engine = EquilibriumEngine()
        p.bind_engine(engine)
        p.unbind_engine()
        assert p.engine is None
        assert p.equilibrium_view is None

    def test_process_queued_syncs_view(self):
        p = CognitionPillar()
        p.initialize(_make_state())
        engine = EquilibriumEngine()
        p.bind_engine(engine)
        p.process_queued()
        # After process_queued, equilibrium_view should still be set
        assert p.equilibrium_view is not None

    def test_process_queued_without_engine(self):
        """process_queued should work fine without an engine bound."""
        p = CognitionPillar()
        p.initialize(_make_state())
        sig = Signal(source=Pillar.PRAXIS, target=Pillar.COGNITION, kind="test")
        p.receive_signal(sig)
        p.process_queued()  # Should not raise
        assert len(p._signals_received) == 1


class TestBasePillarStressFeedback:
    def test_no_stress_feedback_when_healthy(self):
        """When stress_level <= 0.3, no auto-feedback is emitted."""
        p = CognitionPillar()
        p.initialize(_make_state())
        engine = EquilibriumEngine()
        p.bind_engine(engine)
        # Default axes start at their default positions, so drift=0, stress=0
        p.process_queued()
        drained = p.drain_feedback()
        # Only stress feedback would appear; none expected since stress=0
        stress_fb = [f for f in drained if "stress-reactive" in f.reason]
        assert len(stress_fb) == 0

    def test_stress_feedback_when_drifted(self):
        """When the agent is stressed (high drift), stress feedback should appear."""
        p = CognitionPillar()
        p.initialize(_make_state())
        # Create axes — the engine resets position to default_position in __init__,
        # so we must apply feedback to create actual drift.
        axes = [
            TensionAxis(
                id="explore_exploit",
                pillar=Pillar.COGNITION,
                pole_left="explore",
                pole_right="exploit",
                default_position=0.0,
                position=0.0,  # Will be set to default_position by engine
                damping=0.4,
                learning_rate=0.05,
            ),
        ]
        engine = EquilibriumEngine(axes=axes)
        # Push the axis far from its default to create high stress
        engine.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.8,  # Large push toward exploit
            confidence=1.0,  # Full confidence for maximum effect
            reason="test: push axis far from default",
        ))
        p.bind_engine(engine)
        p.process_queued()
        drained = p.drain_feedback()
        stress_fb = [f for f in drained if "stress-reactive" in f.reason]
        assert len(stress_fb) >= 1

    def test_stress_feedback_disabled(self):
        """When _stress_feedback_enabled is False, no auto-feedback."""
        p = CognitionPillar()
        p.initialize(_make_state())
        axes = [
            TensionAxis(
                id="explore_exploit",
                pillar=Pillar.COGNITION,
                pole_left="explore",
                pole_right="exploit",
                default_position=0.0,
                position=0.0,  # Will be set to default_position by engine
                damping=0.4,
                learning_rate=0.05,
            ),
        ]
        engine = EquilibriumEngine(axes=axes)
        # Push axis far from default (same as drifted test)
        engine.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.8,
            confidence=1.0,
            reason="test: push axis far from default",
        ))
        p.bind_engine(engine)
        p._stress_feedback_enabled = False
        p.process_queued()
        drained = p.drain_feedback()
        stress_fb = [f for f in drained if "stress-reactive" in f.reason]
        assert len(stress_fb) == 0
