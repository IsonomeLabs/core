"""Isonome Framework Developer Dashboard — HTTP Server.

Serves a real-time visualization dashboard for the isonome agent framework.
Provides:
  /            — Dashboard HTML
  /api/state   — Full agent state as JSON (tensions, lifecycle, stats)
  /api/demo    — Simulated agent with ticking for demo purposes

Run: python server.py [--port 8420] [--demo]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any

# ── Isonome imports ──────────────────────────────────────────────

from isonome.agent import IsonomeAgent
from isonome.cognition.pillar import CognitionPillar
from isonome.praxis.pillar import PraxisPillar
from isonome.mneme.pillar import MnemePillar
from isonome.types import Task


# ── Agent State Extractor ────────────────────────────────────────

def extract_agent_state(agent: IsonomeAgent) -> dict[str, Any]:
    """Extract all dashboard-relevant state from a live agent."""
    engine = agent.engine
    axes = []
    for axis_id, axis in engine._axes.items():
        axes.append({
            "id": axis.id,
            "pillar": axis.pillar.value,
            "pole_left": axis.pole_left,
            "pole_right": axis.pole_right,
            "position": axis.position,
            "default_position": axis.default_position,
            "damping": axis.damping,
            "learning_rate": axis.learning_rate,
            "drift": abs(axis.position - axis.default_position),
        })

    # Compute stress level
    if axes:
        squared = sum(a["drift"] ** 2 for a in axes)
        stress = math.sqrt(squared / len(axes))
    else:
        stress = 0.0

    # Oscillation info
    oscillating = []
    for axis_id, hist in engine._history.items():
        if len(hist) >= 4:
            n = len(hist)
            mean = sum(hist) / n
            variance = sum((x - mean) ** 2 for x in hist) / n
            stddev = math.sqrt(variance)
            if stddev > engine._oscillation_threshold:
                oscillating.append(axis_id)

    # Behavior profile
    bp = engine.get_behavior_profile()

    # Adaptive damping
    ad = engine.adaptive_damping
    ad_state = None
    if ad is not None:
        ad_state = {
            "enabled": ad._enabled,
            "axis_damping": {
                k: {
                    "damping_boost": v.damping_boost,
                    "oscillation_count": v.oscillation_count,
                    "severity": v.severity,
                }
                for k, v in ad._axis_states.items()
            } if hasattr(ad, '_axis_states') else {},
        }

    # Pillar activity
    pillar_activity = {}
    for p_type, pillar in agent._pillar_map.items():
        view = engine.view_for(p_type)
        pillar_activity[p_type.value] = {
            "own_axes": view.own_axes,
            "cross_axes": view.cross_axes,
            "stress_level": view.stress_level,
            "oscillating": list(view.oscillating),
            "drift": view.drift,
            "initialized": pillar.initialized,
        }

    # Attention budget (if cognition pillar has attention)
    attention = None
    if agent.cognition is not None and hasattr(agent.cognition, 'attention') and agent.cognition.attention is not None:
        att = agent.cognition.attention
        att_stats = att.stats if isinstance(att.stats, dict) else att.stats.__dict__
        budget = att.budget
        attention = {
            "token_capacity": budget.token_capacity,
            "tokens_used": budget.tokens_used,
            "utilization": att_stats.get("utilization", 0.0),
            "chunks_active": att_stats.get("chunks_active", 0),
            "gc_cycles": att_stats.get("gc_cycles", 0),
            "entropy_estimate": att_stats.get("entropy_estimate", 0.0),
            "total_kept": att_stats.get("total_kept", 0),
            "total_compressed": att_stats.get("total_compressed", 0),
            "total_pruned": att_stats.get("total_pruned", 0),
        }

    # Mneme stats
    mneme_stats = None
    if agent.mneme is not None and hasattr(agent.mneme, 'mneme') and agent.mneme.mneme is not None:
        mm = agent.mneme.mneme
        ms = mm._stats
        mneme_stats = {
            "working_count": len(mm._working) if hasattr(mm, '_working') else ms.working_count,
            "episodic_count": len(mm._episodic) if hasattr(mm, '_episodic') else ms.episodic_count,
            "semantic_count": len(mm._semantic) if hasattr(mm, '_semantic') else ms.semantic_count,
            "total_consolidations": ms.total_consolidations,
            "total_pruned": ms.total_pruned,
            "total_rehearsals": ms.total_rehearsals,
            "total_retrievals": ms.total_retrievals,
        }

    # Calibration
    calibration = None
    if agent.cognition is not None and hasattr(agent.cognition, 'reasoning') and agent.cognition.reasoning is not None:
        cal = agent.cognition.reasoning.calibrator
        calibration = {
            "ece": cal.compute_ece(),
            "bias": cal.compute_bias(),
            "mce": cal.compute_mce(),
            "total_predictions": cal.total_predictions,
        }

    # Task type homeostasis
    tth = agent.task_type_homeostasis
    profiles = {}
    for ptype, profile in tth._profiles.items():
        profiles[ptype] = {
            "observation_count": profile.observation_count,
            "is_converged": profile.is_converged,
            "convergence_ratio": float(profile.convergence_ratio) if not math.isinf(profile.convergence_ratio) else None,
            "norm": [float(x) for x in profile.norm()],
        }

    return {
        "agent": {
            "name": agent.identity.name,
            "id": str(agent.identity.id),
            "lifecycle": agent.lifecycle.value,
            "tick_count": agent._tick_count,
            "task_count": agent.state.task_count,
            "error_count": agent.state.error_count,
            "current_task_type": agent.current_task_type,
        },
        "tensions": {
            "axes": axes,
            "stress": stress,
            "oscillating": oscillating,
            "behavior_profile": bp,
        },
        "adaptive_damping": ad_state,
        "pillar_activity": pillar_activity,
        "attention": attention,
        "mneme": mneme_stats,
        "calibration": calibration,
        "task_type_profiles": profiles,
        "feedback_count": engine.total_feedback_received,
        "oscillation_events": engine.total_oscillation_events,
    }


# ── Demo Agent (simulates ticks for visual demo) ────────────────

class DemoAgent:
    """Wraps a real IsonomeAgent and simulates activity for the dashboard."""

    def __init__(self):
        self.agent = IsonomeAgent(
            name="demo-agent",
            cognition=CognitionPillar(),
            praxis=PraxisPillar(),
            mneme=MnemePillar(),
        )
        self.agent.start()
        self.tick_count = 0
        self._lock = threading.Lock()

        # Seed some data
        self._seed_attention()
        self._seed_mneme()

    def _seed_attention(self):
        """Add some attention chunks."""
        if self.agent.cognition and self.agent.cognition.attention:
            topics = [
                ("Analyzing equilibrium drift patterns in multi-axis systems", 0.85, 0.9),
                ("Task decomposition strategy for complex workflows", 0.72, 0.75),
                ("Memory consolidation threshold adaptation", 0.65, 0.6),
                ("Safety constraint verification in autonomous mode", 0.55, 0.8),
                ("Context pruning for efficient token usage", 0.45, 0.5),
                ("Recursive reasoning depth calibration", 0.40, 0.7),
                ("Cross-pillar feedback loop optimization", 0.35, 0.4),
                ("Episodic memory retrieval for similar tasks", 0.28, 0.3),
                ("Parallel execution DAG scheduling", 0.20, 0.65),
                ("Semantic pattern extraction from working memory", 0.15, 0.35),
            ]
            for content, mi, tr in topics:
                self.agent.cognition.attention.add_chunk(
                    content,
                    token_count=random.randint(100, 800),
                    mutual_info=mi,
                    task_relevance=tr,
                )

    def _seed_mneme(self):
        """Add some memory entries."""
        if self.agent.mneme and self.agent.mneme.mneme:
            entries = [
                ("Deployed model v0.3.2 with improved calibration", 0.8, ("deploy", "model")),
                ("Fixed DAG cycle detection in orchestrator", 0.7, ("bug", "orchestrator")),
                ("Research: homeostatic regulation bounds", 0.75, ("research", "equilibrium")),
                ("Refactored attention chunk scoring", 0.6, ("refactor", "attention")),
                ("Added cross-pillar integration tests", 0.65, ("test", "integration")),
                ("Explored multi-agent coordination", 0.5, ("research", "multi-agent")),
                ("Calibration ECE at 0.08 after 50 predictions", 0.55, ("calibration", "metrics")),
            ]
            for content, sig, tags in entries:
                self.agent.mneme.mneme.store(content, significance=sig, tags=tags)

    def tick(self):
        """Simulate one agent tick with random feedback."""
        with self._lock:
            # Apply random feedback to simulate activity
            from isonome.types import Feedback, Pillar
            axes = list(self.agent.engine._axes.keys())
            # 2-4 random feedbacks per tick
            for _ in range(random.randint(1, 3)):
                axis_id = random.choice(axes)
                pillar = self.agent.engine._axes[axis_id].pillar
                signal = random.gauss(0, 0.15)
                signal = max(-1.0, min(1.0, signal))
                fb = Feedback(
                    source=pillar,
                    tension_axis_id=axis_id,
                    signal=signal,
                    confidence=random.uniform(0.3, 0.9),
                    reason=f"demo tick feedback",
                )
                self.agent.engine.apply_feedback(fb)

            self.agent._tick_count += 1
            self.tick_count += 1

            # Occasionally submit a task
            if self.tick_count % 15 == 0:
                tasks = [
                    "Analyze the stability of equilibrium convergence",
                    "Debug the memory consolidation threshold",
                    "Research optimal damping parameters",
                    "Deploy the updated calibration model",
                    "Plan the multi-agent coordination strategy",
                    "Write documentation for the attention system",
                ]
                task = Task(description=random.choice(tasks))
                self.agent.submit_task(task)

            # Process pillars
            for pillar in self.agent._pillar_map.values():
                pillar.process_queued()

    def get_state(self) -> dict:
        with self._lock:
            return extract_agent_state(self.agent)


# ── HTTP Handler ─────────────────────────────────────────────────

class DashboardHandler(SimpleHTTPRequestHandler):
    """Serves dashboard files and API endpoints."""

    demo: DemoAgent | None = None
    agent: IsonomeAgent | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)

    def do_GET(self):
        if self.path == "/api/state":
            self._serve_state()
        elif self.path == "/" or self.path == "/index.html":
            self.path = "/index.html"
            super().do_GET()
        else:
            super().do_GET()

    def _serve_state(self):
        """Return current agent state as JSON."""
        try:
            if self.demo is not None:
                state = self.demo.get_state()
            elif self.agent is not None:
                state = extract_agent_state(self.agent)
            else:
                state = {"error": "No agent configured"}

            payload = json.dumps(state, default=str).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:
            payload = json.dumps({"error": str(e)}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


# ── Demo ticker thread ───────────────────────────────────────────

def run_demo_ticker(demo: DemoAgent, interval: float = 1.5):
    """Background thread that ticks the demo agent."""
    while True:
        demo.tick()
        time.sleep(interval)


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Isonome Framework Dashboard")
    parser.add_argument("--port", type=int, default=8420, help="HTTP port (default: 8420)")
    parser.add_argument("--demo", action="store_true", help="Run with simulated agent activity")
    parser.add_argument("--tick-interval", type=float, default=1.5, help="Demo tick interval in seconds")
    args = parser.parse_args()

    if args.demo:
        print(f"Starting demo agent (tick interval: {args.tick_interval}s)")
        demo = DemoAgent()
        DashboardHandler.demo = demo
        t = threading.Thread(target=run_demo_ticker, args=(demo, args.tick_interval), daemon=True)
        t.start()
        print("Demo ticker running in background")
    else:
        # Create a basic agent without demo
        agent = IsonomeAgent(
            name="dashboard-agent",
            cognition=CognitionPillar(),
            praxis=PraxisPillar(),
            mneme=MnemePillar(),
        )
        agent.start()
        DashboardHandler.agent = agent
        print("Agent initialized (no demo — use --demo for simulated activity)")

    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print(f"Dashboard running at http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
