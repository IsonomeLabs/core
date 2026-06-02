"""Integration tests for the IsonomeAgent lifecycle.

These tests exercise the full agent loop with concrete
(but minimal) pillar implementations.
"""

from __future__ import annotations

import pytest

from isonome.agent import IsonomeAgent
from isonome.base import BasePillar
from isonome.equilibrium import EquilibriumEngine
from isonome.types import (
    AgentState,
    Feedback,
    Pillar,
    Signal,
    Task,
    TaskComplexity,
    TensionAxis,
)


# ── Minimal concrete pillars for testing ──────────────────────────

class CognitionTest(BasePillar):
    """A cognition pillar that pushes toward exploration."""

    @property
    def pillar(self) -> Pillar:
        return Pillar.COGNITION

    def _on_initialize(self, state: AgentState) -> None:
        pass

    def _on_signal(self, signal: Signal) -> None:
        self.emit_feedback(
            Feedback(
                source=self.pillar,
                tension_axis_id="explore_exploit",
                signal=-0.3,
                confidence=0.8,
                reason="novel approach warranted",
            )
        )

    def _on_shutdown(self) -> None:
        pass


class PraxisTest(BasePillar):
    """A praxis pillar that pushes toward fast execution."""

    @property
    def pillar(self) -> Pillar:
        return Pillar.PRAXIS

    def _on_initialize(self, state: AgentState) -> None:
        pass

    def _on_signal(self, signal: Signal) -> None:
        self.emit_feedback(
            Feedback(
                source=self.pillar,
                tension_axis_id="verify_execute",
                signal=+0.4,
                confidence=0.7,
                reason="low-risk action, skip heavy verification",
            )
        )

    def _on_shutdown(self) -> None:
        pass


class MnemeTest(BasePillar):
    """A mneme pillar that pushes toward consolidation."""

    @property
    def pillar(self) -> Pillar:
        return Pillar.MNEME

    def _on_initialize(self, state: AgentState) -> None:
        pass

    def _on_signal(self, signal: Signal) -> None:
        self.emit_feedback(
            Feedback(
                source=self.pillar,
                tension_axis_id="consolidate_prune",
                signal=-0.5,
                confidence=0.9,
                reason="new knowledge worth consolidating",
            )
        )

    def _on_shutdown(self) -> None:
        pass


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def agent():
    """Create a fully-wired IsonomeAgent with all three test pillars."""
    return IsonomeAgent(
        name="test-agent",
        cognition=CognitionTest(),
        praxis=PraxisTest(),
        mneme=MnemeTest(),
    )


@pytest.fixture
def sample_task():
    """A simple task for testing."""
    return Task(
        description="analyze data and produce report",
        complexity=TaskComplexity.SIMPLE,
    )


# ── Tests ─────────────────────────────────────────────────────────


class TestAgentLifecycle:
    """Test the full agent boot → work → shutdown cycle."""

    def test_agent_initialization(self, agent):
        """Agent starts with correct identity and default tensions."""
        assert agent.identity.name == "test-agent"
        assert agent.cognition is not None
        assert agent.praxis is not None
        assert agent.mneme is not None
        profile = agent.get_tension_profile()
        assert len(profile) == 8  # 3 cognition + 3 praxis + 2 mneme
        assert "explore_exploit" in profile
        assert "autonomy_safety" in profile

    def test_full_agent_loop(self, agent, sample_task):
        """Boot → submit task → tick → check results → stop.

        Note: feedback is drained at the *start* of tick() and emitted
        during process_queued() at the *end*. So a signal sent before
        tick 1 produces feedback only visible after tick 1 drains it at
        the start of tick 2.
        """
        agent.start()
        assert agent.lifecycle.name == "IDLE"

        # Submit a task
        agent.submit_task(sample_task)
        assert agent.has_work()
        assert agent.stats["task_queue_depth"] == 1

        # Send a signal to cognition to trigger feedback emission
        signal = Signal(
            source=Pillar.PRAXIS,
            target=Pillar.COGNITION,
            kind="plan_ready",
            payload={"plan": "test"},
        )
        agent.send_signal(signal)

        # Tick 1: routes signal to cognition, cognition processes it
        # and emits feedback, but feedback is drained at tick start
        snapshot1 = agent.tick()

        # Tick 2: now the feedback from tick 1 is drained and applied
        snapshot2 = agent.tick()

        assert agent.stats["feedback_applied"] >= 1
        assert agent.stats["tick_count"] == 2

        # Tension should have moved from default (0.15) toward explore
        explore_axis = snapshot2.get("explore_exploit")
        assert explore_axis is not None
        assert explore_axis.position < explore_axis.default_position

        assert agent.stats["pillars_active"] == 3

        agent.stop()
        assert agent.lifecycle.name == "TERMINATED"

    def test_multiple_ticks_accumulate(self, agent):
        """Feedback accumulates across ticks without oscillation."""
        agent.start()

        for i in range(10):
            signal = Signal(
                source=Pillar.MNEME,
                target=Pillar.COGNITION,
                kind=f"tick_{i}",
            )
            agent.send_signal(signal)
            agent.tick()

        # One more tick to drain the last batch of feedback
        agent.tick()

        assert agent.stats["tick_count"] == 11
        assert agent.stats["feedback_applied"] >= 10

        stress = agent.get_stress_level()
        assert 0.0 < stress < 0.8, f"Stress {stress:.3f} outside expected range"

        agent.stop()

    def test_no_work_without_task(self, agent):
        """has_work() should be false when no tasks are submitted."""
        assert not agent.has_work()
        agent.start()
        assert not agent.has_work()
        agent.tick()
        assert not agent.has_work()
        agent.stop()

    def test_stress_level_starts_zero(self, agent):
        """Fresh agent has zero stress — all axes start at defaults."""
        assert agent.get_stress_level() == 0.0

    def test_tension_profile_readable(self, agent):
        """get_tension_profile() returns a flat dict consumable by pillars."""
        agent.start()
        profile = agent.get_tension_profile()
        assert isinstance(profile, dict)
        for axis_id, pos in profile.items():
            assert isinstance(axis_id, str)
            assert isinstance(pos, float)
            assert -1.0 <= pos <= 1.0, f"{axis_id} out of bounds: {pos}"


class TestTensionAxisRepr:
    """Test the TensionAxis __repr__ for debugging."""

    def test_repr_format(self):
        axis = TensionAxis(
            id="test_axis",
            pillar=Pillar.COGNITION,
            pole_left="left",
            pole_right="right",
            position=0.42,
            default_position=-0.15,
            damping=0.75,
        )
        rep = repr(axis)
        assert "test_axis" in rep
        assert "+0.420" in rep
        assert "-0.150" in rep
        assert "0.75" in rep


class TestEquilibriumEngine:
    """Test the equilibrium engine standalone."""

    def test_feedback_moves_tension(self):
        engine = EquilibriumEngine()
        initial = engine.get_behavior_profile()["explore_exploit"]
        fb = Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=-0.5,
            confidence=0.8,
            reason="test",
        )
        engine.apply_feedback(fb)
        new_pos = engine.get_behavior_profile()["explore_exploit"]
        assert new_pos < initial

    def test_adjust_default_learns(self):
        engine = EquilibriumEngine()
        axis = engine.get_axis("explore_exploit")
        original_default = axis.default_position
        engine.adjust_default("explore_exploit", outcome_signal=0.5)
        new_axis = engine.get_axis("explore_exploit")
        assert new_axis.default_position > original_default

    def test_snapshot_is_immutable_frozen(self):
        engine = EquilibriumEngine()
        snap = engine.snapshot(trigger="test")
        assert snap.trigger == "test"
        assert isinstance(snap.axes, frozenset)
