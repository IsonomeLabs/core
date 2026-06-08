"""VLA Inspector — real-time logging of what the VLA model sees and outputs.

Intended for debugging and demo visualization.  Captures every inference
step in a ring buffer and formats human-readable summaries.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Any

import numpy as np


class VLAInspector:
    """Ring-buffer logger for VLA I/O.

    Each entry records the observation (proprioception shape, intent,
    camera count) and the resulting action (shape, norm, first values).
    """

    def __init__(self, max_entries: int = 100) -> None:
        self._buffer: deque[dict[str, Any]] = deque(maxlen=max_entries)
        self._step = 0

    def log_step(
        self,
        obs: dict[str, Any],
        action: np.ndarray,
        ee_pos: list[float] | None = None,
    ) -> dict[str, Any]:
        """Log one inference step and return the formatted entry."""
        images = obs.get("image")
        n_cameras = len(images) if isinstance(images, list) else (1 if images is not None else 0)
        entry = {
            "step": self._step,
            "timestamp": time.strftime("%H:%M:%S", time.localtime(obs.get("timestamp", time.time()))),
            "intent": obs.get("intent", ""),
            "proprio_shape": list(obs.get("proprioception", np.array([])).shape),
            "n_cameras": n_cameras,
            "action_shape": list(action.shape),
            "action_norm": round(float(np.linalg.norm(action)), 4),
            "action_head": [round(float(v), 4) for v in action.flat[:4]],
            "ee_pos": [round(float(v), 3) for v in ee_pos] if ee_pos is not None else None,
        }
        self._buffer.append(entry)
        self._step += 1
        return entry

    def latest(self, n: int = 1) -> list[dict[str, Any]]:
        """Return the last *n* entries (most recent first)."""
        return list(self._buffer)[-n:]

    def format_entry(self, entry: dict[str, Any]) -> str:
        """Single-line terminal-friendly representation."""
        parts = [
            f"[{entry['timestamp']}]",
            f"step={entry['step']:04d}",
            f"intent='{entry['intent'][:40]}'",
            f"cameras={entry['n_cameras']}",
            f"proprio={entry['proprio_shape']}",
            f"action={entry['action_shape']} norm={entry['action_norm']:.3f}",
            f"head={entry['action_head']}",
        ]
        if entry.get("ee_pos"):
            parts.append(f"ee=[{entry['ee_pos'][0]:+.2f},{entry['ee_pos'][1]:+.2f},{entry['ee_pos'][2]:+.2f}]")
        return "  ".join(parts)

    def dump(self) -> str:
        """Return all buffered entries as a multi-line string."""
        return "\n".join(self.format_entry(e) for e in self._buffer)

    def clear(self) -> None:
        self._buffer.clear()
        self._step = 0
