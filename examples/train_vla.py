"""Training script for VLA policies in simulation.

Supports two modes:
  --mode offline   Behaviour cloning on collected demonstrations
  --mode online    Interleaved DAgger / RL fine-tuning

Usage
-----
    python examples/train_vla.py --model mock --data_dir ./vla_demos --epochs 10
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np

from isonome.praxis.vla import MockVLABackend, OpenVLA, LLaVARobot
from isonome.praxis.vla.base import VLABase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_vla")


def load_demos(data_dir: Path) -> list[dict]:
    """Load demonstration episodes from *data_dir*."""
    episodes = []
    for p in sorted(data_dir.glob("*.pkl")):
        with open(p, "rb") as fh:
            episodes.append(pickle.load(fh))
    if not episodes:
        # Also try JSONL
        for p in sorted(data_dir.glob("*.jsonl")):
            with open(p, "r") as fh:
                episodes.extend(json.loads(line) for line in fh if line.strip())
    logger.info("Loaded %d episodes from %s", len(episodes), data_dir)
    return episodes


def make_model(model_name: str, action_dim: int) -> VLABase:
    if model_name == "mock":
        return MockVLABackend(action_dim=action_dim)
    if model_name == "openvla":
        return OpenVLA(action_dim=action_dim)
    if model_name == "llava_robot":
        return LLaVARobot(action_dim=action_dim)
    raise ValueError(f"Unknown model: {model_name}")


def train_offline(model: VLABase, episodes: list[dict], epochs: int) -> None:
    """Behaviour-clone on demonstration data."""
    logger.info("Starting offline training for %d epochs", epochs)
    for epoch in range(1, epochs + 1):
        losses = []
        for ep in episodes:
            obs = ep["obs"]
            action = ep["action"]
            # Mock backend is deterministic — log a proxy loss
            pred = model.predict(obs)
            mse = float(np.mean((pred - np.asarray(action, dtype=np.float32)) ** 2))
            losses.append(mse)
        logger.info("Epoch %d/%d — mean MSE: %.4f", epoch, epochs, float(np.mean(losses)))
    logger.info("Offline training complete.")


def train_online(model: VLABase, episodes: list[dict], epochs: int) -> None:
    """Placeholder for online fine-tuning (DAgger / RL)."""
    logger.info("Online training not yet implemented.  Falling back to offline.")
    train_offline(model, episodes, epochs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a VLA policy")
    parser.add_argument("--model", default="mock", choices=["mock", "openvla", "llava_robot"])
    parser.add_argument("--data_dir", type=Path, default=Path("./vla_demos"))
    parser.add_argument("--mode", default="offline", choices=["offline", "online"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("./checkpoints"))
    args = parser.parse_args()

    model = make_model(args.model, args.action_dim)
    if args.model != "mock":
        # Real models would load pre-trained weights here
        pass

    episodes = load_demos(args.data_dir)
    if not episodes:
        logger.warning("No demonstrations found in %s", args.data_dir)
        logger.info("Run  python examples/collect_demo.py  to generate data.")
        return

    if args.mode == "offline":
        train_offline(model, episodes, args.epochs)
    else:
        train_online(model, episodes, args.epochs)

    # Save checkpoint (for mock this is a no-op; for real models export safetensors)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Checkpoints would be saved to %s", args.checkpoint_dir)


if __name__ == "__main__":
    main()
