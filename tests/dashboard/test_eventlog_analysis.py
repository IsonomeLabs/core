"""Tests for dashboard event log analysis state extraction.

Verify that extract_agent_state includes all event log analysis fields:
pillar_stress_scores, axis_volatility, feedback_bursts,
dominant_feedback_source, convergence_from_events, cross_pillar_conflicts.
"""

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
from isonome.equilibrium.event_log import TensionEventLog, TensionEventType
from isonome.equilibrium.health import EquilibriumHealthScore
from isonome.equilibrium.convergence import ConvergenceDetector
from isonome.types import Feedback, Pillar


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def agent_with_event_log():
    """Create a real agent with tension event logging enabled and seeded events."""
    agent = IsonomeAgent(
        name="test-el-analysis-agent",
        cognition=CognitionPillar(),
        praxis=PraxisPillar(),
        mneme=MnemePillar(),
    )
    vt = TensionVelocityTracker()
    agent.engine._velocity_tracker = vt
    for axis in agent.engine._axes.values():
        vt.register_axis(axis.id)
    ad = AdaptiveDampingController()
    for axis in agent.engine._axes.values():
        ad.register_axis(axis.id, base_damping=axis.damping)
    ad.velocity_tracker = vt
    agent.engine._adaptive_damping = ad
    agent.engine._event_log = TensionEventLog()
    agent.engine._health_scorer = EquilibriumHealthScore()
    agent.engine._convergence_detector = ConvergenceDetector()
    agent.start()
    # Seed a few feedback events
    axes = list(agent.engine._axes.keys())
    for i in range(10):
        axis_id = axes[i % len(axes)]
        pillar = agent.engine._axes[axis_id].pillar
        signal = 0.1 * (1 if i % 2 == 0 else -1)
        fb = Feedback(
            source=pillar,
            tension_axis_id=axis_id,
            signal=signal,
            confidence=0.8,
            reason="test feedback for event log analysis",
        )
        agent.engine.apply_feedback(fb)
    return agent


# ── Tests for event log analysis fields ──────────────────────────

class TestEventLogAnalysisStateExtraction:
    """Test that event log analysis fields are present in extracted state."""

    def test_event_log_has_pillar_stress_scores(self, agent_with_event_log):
        """Verify event_log includes pillar_stress_scores after feedback."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_event_log)
        el = state.get("event_log")
        assert el is not None, "event_log should be present"
        assert "pillar_stress_scores" in el, "pillar_stress_scores should be in event_log"

    def test_event_log_has_axis_volatility(self, agent_with_event_log):
        """Verify event_log includes axis_volatility after feedback."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_event_log)
        el = state.get("event_log")
        assert el is not None
        assert "axis_volatility" in el, "axis_volatility should be in event_log"

    def test_event_log_has_feedback_bursts(self, agent_with_event_log):
        """Verify event_log includes feedback_bursts."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_event_log)
        el = state.get("event_log")
        assert el is not None
        assert "feedback_bursts" in el, "feedback_bursts should be in event_log"
        assert isinstance(el["feedback_bursts"], list), "feedback_bursts should be a list"

    def test_event_log_has_dominant_feedback_source(self, agent_with_event_log):
        """Verify event_log includes dominant_feedback_source."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_event_log)
        el = state.get("event_log")
        assert el is not None
        assert "dominant_feedback_source" in el, "dominant_feedback_source should be in event_log"

    def test_event_log_has_convergence_from_events(self, agent_with_event_log):
        """Verify event_log includes convergence_from_events."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_event_log)
        el = state.get("event_log")
        assert el is not None
        assert "convergence_from_events" in el, "convergence_from_events should be in event_log"

    def test_event_log_has_cross_pillar_conflicts(self, agent_with_event_log):
        """Verify event_log includes cross_pillar_conflicts."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_event_log)
        el = state.get("event_log")
        assert el is not None
        assert "cross_pillar_conflicts" in el, "cross_pillar_conflicts should be in event_log"
        assert isinstance(el["cross_pillar_conflicts"], list), "cross_pillar_conflicts should be a list"


class TestEventLogAnalysisValueTypes:
    """Test value types and ranges for event log analysis data."""

    def test_pillar_stress_scores_are_numeric(self, agent_with_event_log):
        """Verify pillar_stress_scores values are numeric."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_event_log)
        el = state["event_log"]
        pss = el["pillar_stress_scores"]
        assert isinstance(pss, dict)
        for pillar, score in pss.items():
            assert isinstance(score, (int, float)), f"{pillar} stress score should be numeric"
            assert score >= 0, f"{pillar} stress score should be non-negative"

    def test_axis_volatility_are_numeric(self, agent_with_event_log):
        """Verify axis_volatility values are numeric."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_event_log)
        el = state["event_log"]
        av = el["axis_volatility"]
        assert isinstance(av, dict)
        for axis_id, vol in av.items():
            assert isinstance(vol, (int, float)), f"{axis_id} volatility should be numeric"
            assert vol >= 0, f"{axis_id} volatility should be non-negative"

    def test_convergence_from_events_structure(self, agent_with_event_log):
        """Verify convergence_from_events has expected structure."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_event_log)
        el = state["event_log"]
        cfe = el["convergence_from_events"]
        assert isinstance(cfe, dict)
        assert "direction" in cfe, "convergence_from_events should have direction"
        assert "confidence" in cfe, "convergence_from_events should have confidence"
        assert "trend_slope" in cfe, "convergence_from_events should have trend_slope"
        assert cfe["direction"] in ("converging", "diverging", "stable", "unknown")
        assert 0 <= cfe["confidence"] <= 1, "confidence should be in [0, 1]"

    def test_dominant_feedback_source_structure(self, agent_with_event_log):
        """Verify dominant_feedback_source entries have expected structure."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_event_log)
        el = state["event_log"]
        dfs = el["dominant_feedback_source"]
        assert isinstance(dfs, dict)
        for axis_id, info in dfs.items():
            assert "pillar" in info, f"{axis_id} should have pillar"
            assert "total_weight" in info, f"{axis_id} should have total_weight"
            assert "event_count" in info, f"{axis_id} should have event_count"

    def test_cross_pillar_conflicts_structure(self, agent_with_event_log):
        """Verify cross_pillar_conflicts entries have expected structure when present."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_event_log)
        el = state["event_log"]
        cpc = el["cross_pillar_conflicts"]
        # Conflicts may be empty depending on feedback patterns
        for conflict in cpc:
            assert "pillars" in conflict, "conflict should have pillars"
            assert "axis" in conflict, "conflict should have axis"
            assert "opposing_deltas" in conflict, "conflict should have opposing_deltas"
            assert "conflict_intensity" in conflict, "conflict should have conflict_intensity"


class TestEventLogAnalysisJSONSerializable:
    """Test that event log analysis data is JSON-serializable."""

    def test_full_state_with_event_log_analysis_serializable(self, agent_with_event_log):
        """Verify the entire state including event log analysis serializes to JSON."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_event_log)
        # Should not raise
        json_str = json.dumps(state, default=str)
        parsed = json.loads(json_str)
        assert "event_log" in parsed
        el = parsed["event_log"]
        assert "pillar_stress_scores" in el
        assert "axis_volatility" in el
        assert "convergence_from_events" in el
        assert "dominant_feedback_source" in el
        assert "cross_pillar_conflicts" in el
        assert "feedback_bursts" in el


class TestLiveAgentEventLogAnalysis:
    """Test that LiveAgent includes event log analysis data after ticks."""

    def test_live_agent_event_log_includes_analysis_fields(self):
        """Verify LiveAgent produces event log with analysis fields after ticks."""
        from dashboard.server import LiveAgent

        live = LiveAgent()
        # Tick several times to generate event log data
        for _ in range(20):
            live.tick()

        state = live.get_state()
        el = state.get("event_log")
        assert el is not None, "LiveAgent should have event_log"
        assert "pillar_stress_scores" in el
        assert "axis_volatility" in el
        assert "convergence_from_events" in el
        assert "dominant_feedback_source" in el
        assert "cross_pillar_conflicts" in el
        assert "feedback_bursts" in el

    def test_live_agent_analysis_data_json_serializable(self):
        """Verify LiveAgent event log analysis data serializes cleanly."""
        from dashboard.server import LiveAgent

        live = LiveAgent()
        for _ in range(10):
            live.tick()

        state = live.get_state()
        # Should not raise
        json_str = json.dumps(state, default=str)
        parsed = json.loads(json_str)
        assert parsed["event_log"]["pillar_stress_scores"] is not None
