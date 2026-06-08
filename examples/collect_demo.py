"""Human tele-operation demo collector for VLA training data.

Uses the MuJoCo bridge WebSocket interface to record
(observation, action, reward, done) tuples while a human
steers the robot via the dashboard or keyboard.

Usage
-----
    # 1. Start the MuJoCo bridge
    python -m isonome.sim.mujoco_bridge

    # 2. In another terminal, run this collector
    python examples/collect_demo.py --output_dir ./vla_demos

    # 3. Drive the robot from the dashboard; episodes auto-save on reset.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import websockets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("collect_demo")


def parse_state(msg: dict) -> dict | None:
    """Extract joint positions / velocities from bridge state message."""
    if not msg.get("ok"):
        return None
    state = msg.get("state")
    if state is None:
        return None
    joints = state.get("joints", [])
    positions = np.array([j["position"] for j in joints], dtype=np.float32)
    velocities = np.array([j["velocity"] for j in joints], dtype=np.float32)
    return {
        "timestamp": state.get("timestamp", 0.0),
        "positions": positions,
        "velocities": velocities,
        "joint_names": [j["name"] for j in joints],
    }


class DemoCollector:
    """Records episodes from the MuJoCo bridge."""

    def __init__(self, output_dir: Path, episode_length: int = 500) -> None:
        self._output_dir = output_dir
        self._episode_length = episode_length
        self._episode: list[dict] = []
        self._episode_idx = 0
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def on_step(self, obs: dict, action: np.ndarray, reward: float, done: bool) -> None:
        """Store one transition."""
        self._episode.append({
            "obs": {
                "image": None,  # No image in state-poll mode; could fetch snapshot
                "proprioception": np.concatenate([obs["positions"], obs["velocities"]]),
                "intent": "",  # Could be enriched from bridge intent field
            },
            "action": action.copy(),
            "reward": reward,
            "done": done,
        })
        if done or len(self._episode) >= self._episode_length:
            self._save_episode()

    def _save_episode(self) -> None:
        path = self._output_dir / f"episode_{self._episode_idx:04d}.pkl"
        with open(path, "wb") as fh:
            pickle.dump(self._episode, fh)
        logger.info("Saved episode %d (%d steps) → %s", self._episode_idx, len(self._episode), path)
        self._episode = []
        self._episode_idx += 1


async def main() -> None:
    parser = argparse.ArgumentParser(description="Collect tele-op demonstrations")
    parser.add_argument("--ws_url", default="ws://localhost:8765")
    parser.add_argument("--output_dir", type=Path, default=Path("./vla_demos"))
    parser.add_argument("--episode_length", type=int, default=500)
    parser.add_argument("--rate_hz", type=float, default=10.0)
    args = parser.parse_args()

    collector = DemoCollector(args.output_dir, args.episode_length)
    logger.info("Connecting to %s …", args.ws_url)

    async with websockets.connect(args.ws_url) as ws:
        logger.info("Connected. Recording will start after first state message.")
        prev_positions: np.ndarray | None = None
        while True:
            await ws.send(json.dumps({"action": "get_state"}))
            msg = json.loads(await ws.recv())
            state = parse_state(msg)
            if state is None:
                await asyncio.sleep(1.0 / args.rate_hz)
                continue

            # Derive action as delta from previous step (human tele-op proxy)
            if prev_positions is not None:
                action = state["positions"] - prev_positions
                # Sparse reward: +1 if any joint moved significantly
                reward = 1.0 if np.max(np.abs(action)) > 0.01 else 0.0
                done = False
                collector.on_step(state, action, reward, done)

            prev_positions = state["positions"].copy()
            await asyncio.sleep(1.0 / args.rate_hz)


if __name__ == "__main__":
    asyncio.run(main())
