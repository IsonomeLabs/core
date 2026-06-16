"""Tests for dashboard cross-pillar interaction data in server state."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from isonome.agent import IsonomeAgent
from isonome.cognition.pillar import CognitionPillar
from isonome.praxis.pillar import PraxisPillar
from isonome.mneme.pillar import MnemePillar
from isonome.equilibrium.velocity import TensionVelocityTracker
from isonome.equilibrium import AdaptiveDampingController
from isonome.equilibrium.event_log import TensionEventLog
from isonome.types import Feedback, Pillar


@pytest.fixture
def agent_with_event_log():
    """Create a real agent with event logging enabled."""
    agent = IsonomeAgent(
        name="test-cross-pillar-agent",
        cognition=CognitionPillar(),
        praxis=PraxisPillar(),
        mneme=MnemePillar(),
    )
    agent.engine._event_log = TensionEventLog()
    agent.engine._velocity_tracker = TensionVelocityTracker()
    for axis in agent.engine._axes.values():
        agent.engine._velocity_tracker.register_axis(axis.id)
    agent.start()
    return agent


@pytest.fixture
def agent_with_cross_pillar_feedback(agent_with_event_log):
    """Apply feedback from different pillars to the same axis to create cross-pillar interaction."""
    agent = agent_with_event_log
    axes = list(agent.engine._axes.keys())
    # Apply cognition feedback
    for _ in range(5):
        axis_id = axes[0]
        fb = Feedback(
            source=Pillar.COGNITION,
            tension_axis_id=axis_id,
            signal=0.3,
            confidence=0.8,
            reason="test cognition feedback",
        )
        agent.engine.apply_feedback(fb)
    # Apply praxis feedback to same axis
    for _ in range(5):
        axis_id = axes[0]
        fb = Feedback(
            source=Pillar.PRAXIS,
            tension_axis_id=axis_id,
            signal=-0.2,
            confidence=0.7,
            reason="test praxis counter-feedback",
        )
        agent.engine.apply_feedback(fb)
    return agent


class TestCrossPillarInteractionData:
    """Test that state extraction provides sufficient data for cross-pillar visualization."""

    def test_pillar_activity_has_cross_axes(self, agent_with_event_log):
        """pillar_activity should include cross_axes for each pillar."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_event_log)
        pa = state["pillar_activity"]
        for pillar_name, pdata in pa.items():
            assert "own_axes" in pdata, f"{pillar_name} missing own_axes"
            assert "cross_axes" in pdata, f"{pillar_name} missing cross_axes"
            assert isinstance(pdata["cross_axes"], list), f"{pillar_name} cross_axes should be a list"

    def test_pillar_activity_has_stress_and_drift(self, agent_with_event_log):
        """pillar_activity should include stress_level and drift."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_event_log)
        pa = state["pillar_activity"]
        for pillar_name, pdata in pa.items():
            assert "stress_level" in pdata, f"{pillar_name} missing stress_level"
            assert "drift" in pdata, f"{pillar_name} missing drift"
            assert isinstance(pdata["stress_level"], (int, float)), f"{pillar_name} stress_level should be numeric"
            assert isinstance(pdata["drift"], (int, float)), f"{pillar_name} drift should be numeric"

    def test_event_log_has_pillar_stress_scores(self, agent_with_cross_pillar_feedback):
        """Event log analysis should include pillar_stress_scores."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_cross_pillar_feedback)
        el = state.get("event_log")
        if el is not None:
            assert "pillar_stress_scores" in el, "event_log missing pillar_stress_scores"
            pss = el["pillar_stress_scores"]
            assert isinstance(pss, dict), "pillar_stress_scores should be a dict"
            # Should have entries for each pillar
            for pillar_name in ["cognition", "praxis", "mneme"]:
                assert pillar_name in pss, f"pillar_stress_scores missing {pillar_name}"

    def test_event_log_has_cross_pillar_conflicts(self, agent_with_cross_pillar_feedback):
        """Event log analysis should include cross_pillar_conflicts."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_cross_pillar_feedback)
        el = state.get("event_log")
        if el is not None:
            assert "cross_pillar_conflicts" in el, "event_log missing cross_pillar_conflicts"
            assert isinstance(el["cross_pillar_conflicts"], list), "cross_pillar_conflicts should be a list"

    def test_tensions_axes_include_drift(self, agent_with_event_log):
        """Each tension axis should include drift and pillar info for cross-pillar visualization."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_event_log)
        axes = state["tensions"]["axes"]
        assert len(axes) > 0, "Should have tension axes"
        for axis in axes:
            assert "pillar" in axis, "Axis missing pillar info"
            assert "drift" in axis, "Axis missing drift"
            assert "id" in axis, "Axis missing id"
            assert isinstance(axis["drift"], (int, float)), "drift should be numeric"

    def test_pillar_activity_cross_axes_nonempty_for_each_pillar(self, agent_with_event_log):
        """Each pillar should have at least some cross-axes (8 axes total, each pillar owns 2-3)."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_event_log)
        pa = state["pillar_activity"]
        # With 8 axes across 3 pillars, each pillar reads ~5-6 as cross-axes
        for pillar_name, pdata in pa.items():
            cross_axes = pdata["cross_axes"]
            # Each pillar should have cross axes (since 8 axes total, each pillar owns ~3)
            assert len(cross_axes) > 0, f"{pillar_name} should have cross-axes"

    def test_pillar_activity_initialized_flag(self, agent_with_event_log):
        """Each pillar should report its initialization status."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_event_log)
        pa = state["pillar_activity"]
        for pillar_name, pdata in pa.items():
            assert "initialized" in pdata, f"{pillar_name} missing initialized flag"
            assert isinstance(pdata["initialized"], bool), f"{pillar_name} initialized should be bool"

