"""Tests for dashboard health score and convergence detection state extraction."""

from __future__ import annotations

import json

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
from isonome.equilibrium.health import EquilibriumHealthScore
from isonome.equilibrium.convergence import ConvergenceDetector


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def agent_with_health_and_convergence():
    """Create a real agent with health scoring and convergence detection enabled."""
    agent = IsonomeAgent(
        name="test-health-agent",
        cognition=CognitionPillar(),
        praxis=PraxisPillar(),
        mneme=MnemePillar(),
    )
    # Enable velocity tracking (needed for convergence detection)
    vt = TensionVelocityTracker()
    agent.engine._velocity_tracker = vt
    for axis in agent.engine._axes.values():
        vt.register_axis(axis.id)
    # Enable adaptive damping
    ad = AdaptiveDampingController()
    for axis in agent.engine._axes.values():
        ad.register_axis(axis.id, base_damping=axis.damping)
    ad.velocity_tracker = vt
    agent.engine._adaptive_damping = ad
    # Enable health scoring
    agent.engine._health_scorer = EquilibriumHealthScore()
    # Enable convergence detection
    agent.engine._convergence_detector = ConvergenceDetector()
    agent.start()
    return agent


# ── Tests for health state extraction ───────────────────────────

class TestHealthStateExtraction:
    """Test health score state extraction in dashboard."""

    def test_state_includes_health_key(self, agent_with_health_and_convergence):
        """Verify extracted state includes health top-level key."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_health_and_convergence)
        assert "health" in state, "health should be in state when scorer is enabled"

    def test_health_has_overall_score(self, agent_with_health_and_convergence):
        """Verify health state includes overall score."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_health_and_convergence)
        health = state["health"]
        assert health is not None
        assert "overall" in health
        assert isinstance(health["overall"], (int, float))
        assert 0.0 <= health["overall"] <= 1.0

    def test_health_has_level(self, agent_with_health_and_convergence):
        """Verify health state includes level classification."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_health_and_convergence)
        health = state["health"]
        assert "level" in health
        assert health["level"] in ("critical", "poor", "fair", "good", "excellent")

    def test_health_has_all_component_scores(self, agent_with_health_and_convergence):
        """Verify health state includes all 5 component scores."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_health_and_convergence)
        health = state["health"]
        assert "components" in health
        components = health["components"]
        # Must have all 5 components matching the EquilibriumHealthScore model
        assert "drift" in components, "Missing drift component"
        assert "oscillation" in components, "Missing oscillation component"
        assert "cooldown" in components, "Missing cooldown component"
        assert "velocity" in components, "Missing velocity component"
        assert "convergence" in components, "Missing convergence component"
        # Each component should be numeric in [0, 1]
        for key in ("drift", "oscillation", "cooldown", "velocity", "convergence"):
            val = components[key]
            assert isinstance(val, (int, float)), f"{key} should be numeric"
            assert 0.0 <= val <= 1.0, f"{key} should be in [0, 1], got {val}"

    def test_health_has_per_axis_scores(self, agent_with_health_and_convergence):
        """Verify health state includes per-axis health scores."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_health_and_convergence)
        health = state["health"]
        assert "per_axis" in health
        per_axis = health["per_axis"]
        assert isinstance(per_axis, dict)
        assert len(per_axis) > 0, "Should have per-axis health for all axes"
        # All values should be numeric in [0, 1]
        for axis_id, score in per_axis.items():
            assert isinstance(score, (int, float)), f"{axis_id} score should be numeric"
            assert 0.0 <= score <= 1.0, f"{axis_id} score in [0,1], got {score}"

    def test_health_none_when_scorer_disabled(self):
        """Verify health is None when EquilibriumHealthScore is not enabled."""
        from dashboard.server import extract_agent_state

        agent = IsonomeAgent(
            name="no-health-agent",
            cognition=CognitionPillar(),
            praxis=PraxisPillar(),
            mneme=MnemePillar(),
        )
        agent.start()
        state = extract_agent_state(agent)
        assert state.get("health") is None, "Health should be None when scorer disabled"


# ── Tests for convergence state extraction ─────────────────────

class TestConvergenceStateExtraction:
    """Test convergence detection state extraction in dashboard."""

    def test_state_includes_convergence_key(self, agent_with_health_and_convergence):
        """Verify extracted state includes convergence top-level key."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_health_and_convergence)
        assert "convergence" in state, "convergence should be in state when detector enabled"

    def test_convergence_has_overall_status(self, agent_with_health_and_convergence):
        """Verify convergence state includes overall status."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_health_and_convergence)
        conv = state["convergence"]
        assert conv is not None
        assert "overall_status" in conv
        assert conv["overall_status"] in ("converging", "diverging", "stable", "unknown")

    def test_convergence_has_per_axis_status(self, agent_with_health_and_convergence):
        """Verify convergence state includes per-axis convergence status."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_health_and_convergence)
        conv = state["convergence"]
        assert "per_axis" in conv
        per_axis = conv["per_axis"]
        assert isinstance(per_axis, dict)
        # Each value should be a valid ConvergenceStatus
        for axis_id, status in per_axis.items():
            assert status in ("converging", "diverging", "stable", "unknown"), \
                f"{axis_id} has invalid status: {status}"

    def test_convergence_has_detection_stats(self, agent_with_health_and_convergence):
        """Verify convergence state includes counts and rates."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_health_and_convergence)
        conv = state["convergence"]
        assert "convergence_rate" in conv
        assert isinstance(conv["convergence_rate"], (int, float))
        assert "n_converging" in conv
        assert isinstance(conv["n_converging"], int)
        assert "n_diverging" in conv
        assert isinstance(conv["n_diverging"], int)
        assert "n_stable" in conv
        assert isinstance(conv["n_stable"], int)
        assert "total_detections" in conv
        assert isinstance(conv["total_detections"], int)

    def test_convergence_has_trend(self, agent_with_health_and_convergence):
        """Verify convergence state includes trend history."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_health_and_convergence)
        conv = state["convergence"]
        assert "trend" in conv
        assert isinstance(conv["trend"], list)

    def test_convergence_none_when_detector_disabled(self):
        """Verify convergence is None when detector is not enabled."""
        from dashboard.server import extract_agent_state

        agent = IsonomeAgent(
            name="no-conv-agent",
            cognition=CognitionPillar(),
            praxis=PraxisPillar(),
            mneme=MnemePillar(),
        )
        agent.start()
        state = extract_agent_state(agent)
        assert state.get("convergence") is None, "Convergence should be None when detector disabled"


# ── Tests for LiveAgent with health+convergence ──────────────────

class TestLiveAgentHealthConvergence:
    """Test that LiveAgent enables health and convergence detection."""

    def test_live_agent_includes_health(self):
        """Verify LiveAgent state includes health data."""
        from dashboard.server import LiveAgent

        live = LiveAgent()
        state = live.get_state()
        assert "health" in state, "LiveAgent should include health in state"
        assert state["health"] is not None, "LiveAgent health should not be None"

    def test_live_agent_includes_convergence(self):
        """Verify LiveAgent state includes convergence data."""
        from dashboard.server import LiveAgent

        live = LiveAgent()
        state = live.get_state()
        assert "convergence" in state, "LiveAgent should include convergence in state"
        assert state["convergence"] is not None, "LiveAgent convergence should not be None"

    def test_live_agent_health_is_valid(self):
        """Verify LiveAgent health data has all expected fields."""
        from dashboard.server import LiveAgent

        live = LiveAgent()
        state = live.get_state()
        health = state["health"]
        assert "overall" in health
        assert "level" in health
        assert "components" in health
        assert "per_axis" in health

    def test_live_agent_convergence_is_valid(self):
        """Verify LiveAgent convergence data has all expected fields."""
        from dashboard.server import LiveAgent

        live = LiveAgent()
        state = live.get_state()
        conv = state["convergence"]
        assert "overall_status" in conv
        assert "per_axis" in conv

    def test_live_agent_state_json_serializable(self):
        """Verify LiveAgent state with health+convergence serializes to JSON."""
        from dashboard.server import LiveAgent

        live = LiveAgent()
        state = live.get_state()
        # Should not raise
        json_str = json.dumps(state, default=str)
        parsed = json.loads(json_str)
        assert "health" in parsed
        assert "convergence" in parsed

