"""Closed-loop VLA controller for MuJoCo simulation.

Runs the full perceive → predict → act loop in Python so the VLA model
has direct control over the robot.  The bridge's WebSocket / MJPEG servers
can still run in parallel for remote viewing, but physics stepping is
driven synchronously by this controller.

Usage
-----
    bridge = MuJoCoBridge()
    bridge._cmd_load_urdf("examples/robot_arm.xml")

    vla = MockVLABackend(action_dim=len(bridge._joint_names))
    ctrl = VLAController(bridge, vla, intent="reach the target")

    # Run 500 steps at 10 Hz inference / 60 Hz physics
    ctrl.run(n_steps=500, inference_hz=10.0, physics_hz=60.0)
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from isonome.praxis.vla.base import VLABase
from isonome.sim.mujoco_bridge import MuJoCoBridge

try:
    import mujoco
    HAS_MUJOCO = True
except Exception:  # pragma: no cover
    HAS_MUJOCO = False

logger = logging.getLogger("isonome.sim.vla_controller")


class VLAController:
    """Synchronous closed-loop controller: VLA policy → MuJoCo physics.

    Parameters
    ----------
    bridge:
        Loaded :class:`MuJoCoBridge` with a model already initialized.
    vla_model:
        Concrete :class:`VLABase` wrapper (e.g. ``MockVLABackend``).
    intent:
        Task description passed to the VLA at every inference step.
        For the simple arm demo this defaults to ``"reach the target"``.
    action_lpf_alpha:
        Low-pass filter coefficient for action smoothing (0 = all previous,
        1 = no smoothing).
    """

    def __init__(
        self,
        bridge: MuJoCoBridge,
        vla_model: VLABase,
        intent: str = "reach the target",
        action_lpf_alpha: float = 0.3,
    ) -> None:
        self._bridge = bridge
        self._vla = vla_model
        self._intent = intent
        self._action_lpf_alpha = action_lpf_alpha
        self._last_action: np.ndarray | None = None
        self._step_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        n_steps: int = 500,
        inference_hz: float = 10.0,
        physics_hz: float = 60.0,
    ) -> list[dict[str, Any]]:
        """Run the closed-loop for *n_steps* inference cycles.

        Between each VLA inference call the physics are stepped
        ``physics_hz / inference_hz`` times so the sim clock stays
        consistent.

        Returns
        -------
        list[dict]
            Trajectory log — one entry per inference step with keys
            ``step``, ``timestamp``, ``ee_pos``, ``action_norm``,
            ``intent``.
        """
        if not HAS_MUJOCO:
            raise RuntimeError("mujoco is required")
        if self._bridge._model is None or self._bridge._data is None:
            raise RuntimeError("Bridge has no loaded model. Call load_urdf first.")

        physics_substeps = max(1, int(round(physics_hz / inference_hz)))
        dt_inference = 1.0 / inference_hz
        trajectory: list[dict[str, Any]] = []

        logger.info(
            "Starting VLA control loop: %d inference steps, "
            "%d physics sub-steps per inference, intent='%s'",
            n_steps, physics_substeps, self._intent,
        )

        for i in range(n_steps):
            t0 = time.perf_counter()

            # 1. Perceive
            obs = self._bridge.get_observation(self._intent)

            # 2. Predict
            action = self._vla.predict(obs)

            # Handle action chunks: [T, n_joints] → queue, use first
            if action.ndim == 2:
                action = action[0]

            action = np.asarray(action, dtype=np.float32)

            # 3. Smooth
            if self._last_action is not None and action.shape == self._last_action.shape:
                action = (
                    self._action_lpf_alpha * action
                    + (1.0 - self._action_lpf_alpha) * self._last_action
                )
            self._last_action = action.copy()

            # 4. Act + step physics
            self._apply_action(action)
            for _ in range(physics_substeps):
                mujoco.mj_step(self._bridge._model, self._bridge._data)

            # 5. Log
            ee_pos = self._end_effector_pos()
            entry = {
                "step": i,
                "timestamp": obs["timestamp"],
                "ee_pos": ee_pos.tolist() if ee_pos is not None else None,
                "action_norm": float(np.linalg.norm(action)),
                "intent": self._intent,
            }
            trajectory.append(entry)
            self._step_count += 1

            # Throttle to inference_hz
            elapsed = time.perf_counter() - t0
            sleep = dt_inference - elapsed
            if sleep > 0:
                time.sleep(sleep)

        logger.info("VLA control loop finished after %d steps", self._step_count)
        return trajectory

    def reset(self) -> None:
        """Reset the controller state (action filter, step counter)."""
        self._last_action = None
        self._step_count = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_action(self, action: np.ndarray) -> None:
        """Write *action* (delta positions) into the bridge's qpos."""
        model = self._bridge._model
        data = self._bridge._data
        if model is None or data is None:
            return
        for i, name in enumerate(self._bridge._joint_names):
            if i >= len(action):
                break
            idx = self._bridge._joint_map[name]
            qposadr = model.jnt_qposadr[idx]
            jnt_type = model.jnt_type[idx]
            if jnt_type == mujoco.mjtJoint.mjJNT_HINGE or jnt_type == mujoco.mjtJoint.mjJNT_SLIDE:
                data.qpos[qposadr] += float(action[i])
        mujoco.mj_forward(model, data)

    def _end_effector_pos(self) -> np.ndarray | None:
        """Return the 3-D Cartesian position of the end-effector site."""
        model = self._bridge._model
        data = self._bridge._data
        if model is None or data is None:
            return None
        try:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
            return np.array(data.site_xpos[site_id], dtype=np.float32)
        except Exception:
            # No 'ee' site — fall back to last body position
            if model.nbody > 1:
                return np.array(data.xpos[model.nbody - 1], dtype=np.float32)
            return None
