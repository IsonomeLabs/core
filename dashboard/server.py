"""Isonome Framework Developer Dashboard — HTTP Server.

Serves a real-time visualization dashboard for the isonome agent framework.
Provides:
  /              — Dashboard HTML
  /api/state     — Full agent state as JSON (tensions, lifecycle, stats)
  /api/demo      — Simulated agent with ticking for demo purposes
  /sim           — URDF Sim Scaffold (new)
  /api/upload-urdf — Upload URDF or ZIP bundle
  /api/sim/stream — MJPEG proxy to Isaac Sim / mock bridge

Run: python server.py [--port 8420] [--demo]
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import shutil
import socket
import sys
import tempfile
import threading
import time
import zipfile
from email.parser import BytesParser
from email.policy import default
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any

# Ensure project root is on path for isonome imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Sim Scaffold Constants ───────────────────────────────────────

MJPEG_PROXY_HOST = "localhost"
MJPEG_PROXY_PORT = 8766  # Isaac Sim or mock bridge MJPEG port
SIM_UPLOAD_DIR = Path(tempfile.gettempdir()) / "isonome_uploads"
SIM_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Isonome imports ──────────────────────────────────────────────

from isonome.agent import IsonomeAgent
from isonome.cognition.pillar import CognitionPillar
from isonome.praxis.pillar import PraxisPillar
from isonome.mneme.pillar import MnemePillar
from isonome.equilibrium.velocity import TensionVelocityTracker
from isonome.equilibrium.event_log import TensionEventLog, TensionEventType
from isonome.types import Task


# ── Agent State Extractor ────────────────────────────────────────

def extract_agent_state(agent: IsonomeAgent) -> dict[str, Any]:
    """Extract all dashboard-relevant state from a live agent."""
    engine = agent.engine
    axes = []
    vt = engine._velocity_tracker
    for axis_id, axis in engine._axes.items():
        axis_data = {
            "id": axis.id,
            "pillar": axis.pillar.value,
            "pole_left": axis.pole_left,
            "pole_right": axis.pole_right,
            "position": axis.position,
            "default_position": axis.default_position,
            "damping": axis.damping,
            "learning_rate": axis.learning_rate,
            "drift": abs(axis.position - axis.default_position),
        }
        # Add velocity/momentum data if tracker is available
        if vt is not None:
            axis_data["velocity"] = vt.get_velocity(axis_id)
            axis_data["momentum_score"] = vt.get_momentum_score(axis_id)
            axis_data["reversal_count"] = vt.get_reversal_count(axis_id)
            axis_data["reversal_rate"] = round(vt.get_reversal_rate(axis_id), 3)
            # Position history for sparkline (most recent 30 points)
            hist = vt._position_history.get(axis_id)
            if hist is not None:
                axis_data["position_history"] = list(hist)[-30:]
            else:
                axis_data["position_history"] = []
        else:
            # Fallback: get history from engine if no velocity tracker
            hist = engine._history.get(axis_id)
            if hist is not None and len(hist) > 0:
                axis_data["position_history"] = list(hist)[-30:]
            else:
                axis_data["position_history"] = []
            axis_data["velocity"] = 0.0
            axis_data["momentum_score"] = 0.0
            axis_data["reversal_count"] = 0
            axis_data["reversal_rate"] = 0.0
        axes.append(axis_data)

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
    if (
        agent.cognition is not None
        and hasattr(agent.cognition, 'attention')
        and agent.cognition.attention is not None
    ):
        att = agent.cognition.attention
        att_stats = att.stats if isinstance(att.stats, dict) else att.stats.__dict__
        budget = att.budget

        # Top chunks with scores and metadata
        top_chunks_list = []
        try:
            for chunk in att.get_top_chunks(10):
                chunk_score = chunk.attention_score()
                top_chunks_list.append({
                    "content": chunk.content[:120],  # truncate for dashboard
                    "token_count": chunk.token_count,
                    "attention_score": round(chunk_score, 4),
                    "mutual_info": round(chunk.mutual_info, 4),
                    "task_relevance": round(chunk.task_relevance, 4),
                    "surprisal": round(chunk.surprisal, 4),
                    "recency": round(chunk.recency, 4),
                    "importance_tags": list(chunk.importance_tags),
                })
        except Exception:
            pass  # Graceful fallback if scoring fails

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
            "enforcement": att_stats.get("enforcement", {
                "policy": "reject",
                "threshold": 0.9,
                "auto_gc_triggered": 0,
                "rejections": 0,
                "auto_compressions": 0,
                "oversized_rejections": 0,
                "post_gc_rejections": 0,
            }),
            "rejected_queue": att_stats.get("rejected_queue", {
                "current_size": 0,
                "max_size": 64,
                "total_enqueued": 0,
                "total_dequeued": 0,
                "total_evicted": 0,
                "total_dropped": 0,
            }),
            "splitting": att_stats.get("splitting", {
                "total_splits": 0,
                "total_fragments_produced": 0,
                "total_fragments_dropped": 0,
            }),
            "top_chunks": top_chunks_list,
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

    # Velocity summary
    velocity_summary = None
    if vt is not None:
        velocity_summary = {
            "total_reversals": vt.total_reversals,
            "total_updates": vt.total_updates,
        }

    # Tension event log
    event_log_data = None
    el = engine._event_log
    if el is not None:
        # Event type breakdown
        counts_by_type = el.count_by_type()
        type_breakdown = {
            et.value: count for et, count in counts_by_type.items()
        }

        # Pillar breakdown
        counts_by_source = el.count_by_source()
        pillar_breakdown = {
            p.value: count for p, count in counts_by_source.items()
        }

        # Axis breakdown (top 10 by activity)
        counts_by_axis = el.count_by_axis()
        axis_items = sorted(counts_by_axis.items(), key=lambda x: x[1], reverse=True)[:10]
        axis_breakdown = {k: v for k, v in axis_items}

        # Recent events (last 25)
        recent = el.events()[-25:]
        recent_list = [e.to_dict() for e in recent]

        # Feedback density
        feedback_density = el.feedback_density(window=10)

        # Most active axis
        most_active = el.most_active_axis()

        event_log_data = {
            "total_events": el.total_events,
            "current_size": len(el._events),
            "max_events": el.max_events,
            "type_breakdown": type_breakdown,
            "pillar_breakdown": pillar_breakdown,
            "axis_breakdown": axis_breakdown,
            "recent_events": recent_list,
            "feedback_density": round(feedback_density, 3),
            "most_active_axis": most_active or "",
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
        "velocity": velocity_summary,
        "event_log": event_log_data,
        "feedback_count": engine.total_feedback_received,
        "oscillation_events": engine.total_oscillation_events,
    }


# ── Demo Agent (simulates ticks for visual demo) ────────────────

class LiveAgent:
    """Wraps a real IsonomeAgent and drives it with feedback for live dashboard data.

    The agent is real — all tension dynamics, stress computation, oscillation
    detection, and pillar state are computed by the actual framework. The only
    synthetic input is the random feedback signal that perturbs the engine.
    """

    def __init__(self):
        self.agent = IsonomeAgent(
            name="dashboard-agent",
            cognition=CognitionPillar(),
            praxis=PraxisPillar(),
            mneme=MnemePillar(),
        )
        # Enable velocity tracking for momentum/velocity dashboard panel
        self.agent.engine._velocity_tracker = TensionVelocityTracker()
        for axis in self.agent.engine._axes.values():
            self.agent.engine._velocity_tracker.register_axis(axis.id)
        # Enable tension event logging for the event log dashboard panel
        self.agent.engine._event_log = TensionEventLog()
        self.agent.start()
        self.tick_count = 0
        self._lock = threading.Lock()

        # Seed initial attention and memory state
        self._seed_attention()
        self._seed_mneme()

    def _seed_attention(self):
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
        """Execute one agent tick: apply feedback, process pillars."""
        with self._lock:
            from isonome.types import Feedback, Pillar
            axes = list(self.agent.engine._axes.keys())
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
                    reason="live tick feedback",
                )
                self.agent.engine.apply_feedback(fb)

            self.agent._tick_count += 1
            self.tick_count += 1

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

            for pillar in self.agent._pillar_map.values():
                pillar.process_queued()

    def get_state(self) -> dict:
        with self._lock:
            return extract_agent_state(self.agent)


# ── HTTP Handler ─────────────────────────────────────────────────

class DashboardHandler(SimpleHTTPRequestHandler):
    """Serves dashboard files and API endpoints."""

    live: LiveAgent | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)

    def do_GET(self):
        if self.path == "/api/state":
            self._serve_state()
        elif self.path == "/sim" or self.path == "/sim/":
            self.path = "/sim.html"
            super().do_GET()
        elif self.path == "/api/sim/stream":
            self._proxy_mjpeg()
        elif self.path.startswith("/api/sim/snapshot"):
            self._snapshot_mjpeg()
        elif self.path == "/" or self.path == "/index.html":
            self.path = "/index.html"
            super().do_GET()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/upload-urdf":
            self._upload_urdf()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_state(self):
        """Return current agent state as JSON."""
        try:
            if self.live is not None:
                state = self.live.get_state()
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

    # ── Sim Scaffold Handlers ────────────────────────────────────

    def _upload_urdf(self) -> None:
        """Handle URDF or ZIP upload, extract, return path."""
        try:
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                self._send_json({"error": "Expected multipart/form-data"}, 400)
                return

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            # Parse multipart using stdlib email parser
            msg = BytesParser(policy=default).parsebytes(
                b"Content-Type: " + content_type.encode() + b"\n\n" + body
            )

            file_bytes = None
            filename = None
            for part in msg.iter_parts():
                if part.get_filename():
                    file_bytes = part.get_payload(decode=True)
                    filename = part.get_filename()
                    break

            if file_bytes is None or filename is None:
                self._send_json({"error": "No file found in upload"}, 400)
                return

            # Save to temp dir
            upload_path = SIM_UPLOAD_DIR / filename
            upload_path.write_bytes(file_bytes)

            # If ZIP, extract
            if filename.lower().endswith(".zip"):
                extract_dir = SIM_UPLOAD_DIR / filename[:-4]
                extract_dir.mkdir(exist_ok=True)
                with zipfile.ZipFile(upload_path, "r") as zf:
                    zf.extractall(extract_dir)
                # Find URDF inside
                urdf_files = list(extract_dir.rglob("*.urdf"))
                if not urdf_files:
                    self._send_json({"error": "No .urdf found in ZIP archive"}, 400)
                    return
                urdf_path = urdf_files[0]
            else:
                urdf_path = upload_path

            self._send_json({"ok": True, "path": str(urdf_path)})

        except Exception as exc:
            self._send_json({"error": str(exc)}, 500)

    def _proxy_mjpeg(self) -> None:
        """Proxy MJPEG stream from Isaac Sim / mock bridge."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((MJPEG_PROXY_HOST, MJPEG_PROXY_PORT))

            # Request the MJPEG stream
            request = (
                f"GET / HTTP/1.1\r\n"
                f"Host: {MJPEG_PROXY_HOST}:{MJPEG_PROXY_PORT}\r\n"
                f"Connection: close\r\n\r\n"
            )
            sock.sendall(request.encode())

            # Read response headers
            header_data = b""
            while b"\r\n\r\n" not in header_data:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                header_data += chunk

            # Extract Content-Type from proxied response
            headers_end = header_data.find(b"\r\n\r\n")
            proxied_headers = header_data[:headers_end].decode("utf-8", errors="ignore")

            content_type = "multipart/x-mixed-replace; boundary=frame"
            for line in proxied_headers.split("\r\n"):
                if line.lower().startswith("content-type:"):
                    content_type = line.split(":", 1)[1].strip()
                    break

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            # Forward any remaining body data from headers read
            body_remainder = header_data[headers_end + 4:]
            if body_remainder:
                self.wfile.write(body_remainder)

            # Stream the rest
            while True:
                chunk = sock.recv(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)

        except (socket.error, ConnectionRefusedError) as exc:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Sim bridge unavailable: {exc}"}).encode())
        except Exception as exc:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(exc)}).encode())
        finally:
            try:
                sock.close()
            except Exception:
                pass

    def _snapshot_mjpeg(self) -> None:
        """Read one frame from the MJPEG stream and return it as a single JPEG."""
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3.0)
            sock.connect((MJPEG_PROXY_HOST, MJPEG_PROXY_PORT))

            request = (
                f"GET / HTTP/1.1\r\n"
                f"Host: {MJPEG_PROXY_HOST}:{MJPEG_PROXY_PORT}\r\n"
                f"Connection: close\r\n\r\n"
            )
            sock.sendall(request.encode())

            # Read until we find the first JPEG frame
            data = b""
            jpeg = None
            while jpeg is None:
                chunk = sock.recv(8192)
                if not chunk:
                    break
                data += chunk
                # Look for JPEG magic start and end markers
                soi = data.find(b"\xff\xd8\xff")
                if soi != -1:
                    eoi = data.find(b"\xff\xd9", soi + 3)
                    if eoi != -1:
                        jpeg = data[soi : eoi + 2]

            if jpeg is None:
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "No frame available"}).encode())
                return

            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.send_header("Cache-Control", "no-cache, no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(jpeg)
        except Exception as exc:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(exc)}).encode())
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

    def _send_json(self, data: dict, status: int = 200) -> None:
        payload = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)


# ── Live ticker thread ───────────────────────────────────────────

def run_live_ticker(live: LiveAgent, interval: float = 1.0):
    """Background thread that ticks the live agent."""
    while True:
        live.tick()
        time.sleep(interval)


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Isonome Framework Dashboard")
    parser.add_argument("--port", type=int, default=8420, help="HTTP port (default: 8420)")
    parser.add_argument("--tick-interval", type=float, default=1.0, help="Agent tick interval in seconds")
    args = parser.parse_args()

    print(f"Starting live agent (tick interval: {args.tick_interval}s)")
    live = LiveAgent()
    DashboardHandler.live = live
    t = threading.Thread(target=run_live_ticker, args=(live, args.tick_interval), daemon=True)
    t.start()
    print("Live ticker running in background")

    server = ThreadingHTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print(f"Dashboard running at http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
