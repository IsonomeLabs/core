"""Tests for dashboard server state extraction and chunk/enforcement data."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on path
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


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def agent_with_attention():
    """Create a real agent with seeded attention chunks."""
    agent = IsonomeAgent(
        name="test-dashboard-agent",
        cognition=CognitionPillar(),
        praxis=PraxisPillar(),
        mneme=MnemePillar(),
    )
    agent.engine._velocity_tracker = TensionVelocityTracker()
    for axis in agent.engine._axes.values():
        agent.engine._velocity_tracker.register_axis(axis.id)
    agent.start()
    return agent


@pytest.fixture
def agent_with_adaptive_damping():
    """Create a real agent with adaptive damping and velocity tracker enabled."""
    agent = IsonomeAgent(
        name="test-ad-agent",
        cognition=CognitionPillar(),
        praxis=PraxisPillar(),
        mneme=MnemePillar(),
    )
    vt = TensionVelocityTracker()
    agent.engine._velocity_tracker = vt
    ad = AdaptiveDampingController()
    for axis in agent.engine._axes.values():
        vt.register_axis(axis.id)
        ad.register_axis(axis.id, base_damping=axis.damping)
    ad.velocity_tracker = vt
    agent.engine._adaptive_damping = ad
    agent.start()
    return agent


# ── Tests for extract_agent_state ────────────────────────────────

class TestExtractAgentState:
    """Test the state extraction function used by the dashboard."""

    def test_state_has_required_top_level_keys(self, agent_with_attention):
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_attention)
        assert "agent" in state
        assert "tensions" in state
        assert "pillar_activity" in state
        assert "attention" in state
        assert "mneme" in state
        assert "calibration" in state

    def test_attention_includes_enforcement_data(self, agent_with_attention):
        """Verify attention state includes enforcement and rejected queue stats."""
        from dashboard.server import extract_agent_state

        att = agent_with_attention.cognition.attention
        # Seed some chunks
        att.add_chunk("Test chunk one", token_count=200, mutual_info=0.6, task_relevance=0.7)
        att.add_chunk("Test chunk two", token_count=300, mutual_info=0.5, task_relevance=0.8)

        state = extract_agent_state(agent_with_attention)
        attention_state = state.get("attention")
        assert attention_state is not None, "Attention state should not be None"

        # Check core fields
        assert "token_capacity" in attention_state
        assert "tokens_used" in attention_state
        assert "utilization" in attention_state
        assert "chunks_active" in attention_state

        # Check enforcement fields
        assert "enforcement" in attention_state, "Enforcement data should be included"
        enforcement = attention_state["enforcement"]
        assert "policy" in enforcement
        assert "threshold" in enforcement
        assert "auto_gc_triggered" in enforcement
        assert "rejections" in enforcement
        assert "auto_compressions" in enforcement
        assert "oversized_rejections" in enforcement
        assert "post_gc_rejections" in enforcement

        # Check rejected queue fields
        assert "rejected_queue" in attention_state, "Rejected queue stats should be included"
        rq = attention_state["rejected_queue"]
        assert "current_size" in rq
        assert "max_size" in rq
        assert "total_enqueued" in rq
        assert "total_dequeued" in rq
        assert "total_evicted" in rq
        assert "total_dropped" in rq

        # Check splitting fields
        assert "splitting" in attention_state, "Splitting stats should be included"
        splitting = attention_state["splitting"]
        assert "total_splits" in splitting
        assert "total_fragments_produced" in splitting
        assert "total_fragments_dropped" in splitting

    def test_attention_includes_top_chunks(self, agent_with_attention):
        """Verify attention state includes top chunks with scores and metadata."""
        from dashboard.server import extract_agent_state

        att = agent_with_attention.cognition.attention
        # Add several chunks
        for i in range(5):
            att.add_chunk(
                f"Chunk content {i}",
                token_count=100 * (i + 1),
                mutual_info=0.5 + i * 0.1,
                task_relevance=0.6 + i * 0.05,
            )

        state = extract_agent_state(agent_with_attention)
        attention_state = state.get("attention")
        assert attention_state is not None

        # Check top_chunks field
        assert "top_chunks" in attention_state, "top_chunks should be present in attention state"
        top_chunks = attention_state["top_chunks"]
        assert isinstance(top_chunks, list)

        # Each chunk should have key metadata
        if len(top_chunks) > 0:
            chunk = top_chunks[0]
            assert "content" in chunk
            assert "token_count" in chunk
            assert "attention_score" in chunk
            assert "mutual_info" in chunk
            assert "task_relevance" in chunk
            assert "recency" in chunk
            assert "surprisal" in chunk
            assert "importance_tags" in chunk

    def test_state_serializable_to_json(self, agent_with_attention):
        """Verify the entire state can be serialized to JSON (dashboard API requirement)."""
        from dashboard.server import extract_agent_state

        att = agent_with_attention.cognition.attention
        att.add_chunk("Serializable test", token_count=150, mutual_info=0.7, task_relevance=0.8)

        state = extract_agent_state(agent_with_attention)
        # This should not raise
        json_str = json.dumps(state, default=str)
        assert len(json_str) > 0

        # Parse back and verify structure
        parsed = json.loads(json_str)
        assert parsed["attention"]["top_chunks"] is not None


class TestExtractAgentStateNoAttention:
    """Test state extraction when attention is not available."""

    def test_attention_state_handles_none_gracefully(self):
        """Verify extract_agent_state handles missing attention without error."""
        from dashboard.server import extract_agent_state

        agent = IsonomeAgent(
            name="no-attention-agent",
            cognition=CognitionPillar(),
            praxis=PraxisPillar(),
            mneme=MnemePillar(),
        )
        agent.engine._velocity_tracker = TensionVelocityTracker()
        for axis in agent.engine._axes.values():
            agent.engine._velocity_tracker.register_axis(axis.id)
        agent.start()

        # Manually remove attention to test graceful handling
        agent.cognition._attention = None

        state = extract_agent_state(agent)
        # Should not crash; attention may be None
        assert state is not None


class TestLiveAgentState:
    """Test the LiveAgent demo wrapper used by the dashboard server."""

    def test_live_agent_produces_state_with_chunks(self):
        from dashboard.server import LiveAgent

        live = LiveAgent()
        state = live.get_state()

        assert "attention" in state
        att = state["attention"]
        assert att is not None, "Seeded attention should not be None"
        assert "top_chunks" in att
        assert "enforcement" in att
        assert "splitting" in att
        assert "rejected_queue" in att

    def test_live_agent_tick_updates_state(self):
        from dashboard.server import LiveAgent

        live = LiveAgent()
        tick0 = live.get_state()["agent"]["tick_count"]

        live.tick()
        tick1 = live.get_state()["agent"]["tick_count"]

        assert tick1 > tick0, "Tick should increment tick count"


class TestAdaptiveDampingState:
    """Test adaptive damping state extraction in dashboard."""

    def test_state_includes_adaptive_damping_key(self, agent_with_adaptive_damping):
        """Verify extracted state includes adaptive_damping top-level key."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_adaptive_damping)
        assert "adaptive_damping" in state, "adaptive_damping should be in state"

    def test_adaptive_damping_has_required_fields(self, agent_with_adaptive_damping):
        """Verify adaptive damping state contains all expected controller-level fields."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_adaptive_damping)
        ad = state["adaptive_damping"]
        assert isinstance(ad, dict)
        # Core boolean flag
        assert "enabled" in ad
        assert ad["enabled"] is True
        # Controller-level stats
        assert "total_adaptations" in ad
        assert "preemptive_oscillation_count" in ad
        # Configuration
        assert "damping_min" in ad
        assert "damping_max" in ad
        assert "boost_rate" in ad
        assert "decay_rate" in ad
        assert "stability_window" in ad
        assert "preemptive_boost_rate" in ad
        # Per-axis detail
        assert "axis_detail" in ad

    def test_adaptive_damping_axis_detail_structure(self, agent_with_adaptive_damping):
        """Verify each axis in axis_detail has the expected sub-fields."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_adaptive_damping)
        ad = state["adaptive_damping"]
        axis_detail = ad.get("axis_detail", {})
        assert isinstance(axis_detail, dict)
        assert len(axis_detail) > 0, "axis_detail should have at least one axis"

        for axis_id, detail in axis_detail.items():
            assert "effective_damping" in detail, f"{axis_id}: missing effective_damping"
            assert "base_damping" in detail, f"{axis_id}: missing base_damping"
            assert "damping_delta" in detail, f"{axis_id}: missing damping_delta"
            assert "oscillation_severity" in detail, f"{axis_id}: missing oscillation_severity"
            assert "stability_counter" in detail, f"{axis_id}: missing stability_counter"
            assert "pillar" in detail, f"{axis_id}: missing pillar"
            # Values should be numeric
            assert isinstance(detail["effective_damping"], (int, float)), f"{axis_id}: effective_damping not numeric"
            assert isinstance(detail["base_damping"], (int, float)), f"{axis_id}: base_damping not numeric"
            assert isinstance(detail["damping_delta"], (int, float)), f"{axis_id}: damping_delta not numeric"
            assert isinstance(detail["oscillation_severity"], (int, float)), f"{axis_id}: oscillation_severity not numeric"
            assert isinstance(detail["stability_counter"], int), f"{axis_id}: stability_counter not int"
            assert isinstance(detail["pillar"], str), f"{axis_id}: pillar not str"

    def test_adaptive_damping_enabled_in_live_agent(self):
        """Verify LiveAgent enables adaptive damping and produces valid state."""
        from dashboard.server import LiveAgent

        live = LiveAgent()
        state = live.get_state()
        ad = state.get("adaptive_damping")
        assert ad is not None, "LiveAgent state should include adaptive_damping"
        assert ad.get("enabled") is True, "Adaptive damping should be enabled in LiveAgent"
        assert len(ad.get("axis_detail", {})) > 0, "LiveAgent should have axis_detail entries"

    def test_adaptive_damping_state_json_serializable(self, agent_with_adaptive_damping):
        """Verify adaptive damping state can be serialized to JSON."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_adaptive_damping)
        # Should not raise
        json_str = json.dumps(state, default=str)
        parsed = json.loads(json_str)
        assert "adaptive_damping" in parsed
        assert "axis_detail" in parsed["adaptive_damping"]

    def test_adaptive_damping_none_when_not_enabled(self, agent_with_attention):
        """Verify adaptive_damping is None when no controller is attached."""
        from dashboard.server import extract_agent_state

        state = extract_agent_state(agent_with_attention)
        assert state.get("adaptive_damping") is None, "Should be None without controller"
