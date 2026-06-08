"""Tests for Iteration 029 — VLA integration pipeline.

Covers:
  - MuJoCoBridge observation packaging
  - VLA wrapper instantiation (mock + placeholders)
  - Action-space mapping
  - Intent round-trip via WebSocket command (simulated)
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from isonome.praxis.vla import MockVLABackend, OpenVLA, LLaVARobot, PiZeroFive
from isonome.praxis.vla.base import VLABase


# ── Helpers ───────────────────────────────────────────────────────

class FakeMuJoCoData:
    """Minimal stand-in for ``mujoco.MjData``."""

    def __init__(self, nq: int, nv: int) -> None:
        self.qpos = np.zeros(nq, dtype=np.float64)
        self.qvel = np.zeros(nv, dtype=np.float64)
        self.ctrl = np.zeros(nq, dtype=np.float64)


class FakeMuJoCoModel:
    """Minimal stand-in for ``mujoco.MjModel``."""

    def __init__(self, n_joints: int = 7) -> None:
        self.njnt = n_joints
        self.nq = n_joints
        self.nv = n_joints
        self.nbody = n_joints + 1
        # Each joint is a hinge with qposadr == idx, dofadr == idx
        self.jnt_qposadr = np.arange(n_joints, dtype=np.int32)
        self.jnt_dofadr = np.arange(n_joints, dtype=np.int32)
        self.jnt_type = np.full(n_joints, 3, dtype=np.int32)  # mjJNT_HINGE == 3
        self.jnt_range = np.tile([-np.pi, np.pi], (n_joints, 1))


# ── VLABase conformance ───────────────────────────────────────────

class TestVLABaseConformance:
    def test_mock_is_instance_of_base(self):
        assert isinstance(MockVLABackend(), VLABase)

    def test_mock_load_is_noop(self):
        m = MockVLABackend()
        m.load("/dev/null")
        assert m._loaded

    def test_mock_predict_shape(self):
        m = MockVLABackend(action_dim=7)
        obs = {"proprioception": np.zeros(7)}
        action = m.predict(obs)
        assert action.shape == (7,)
        assert action.dtype == np.float32

    def test_mock_predict_converges(self):
        """Repeated prediction should drive state toward target."""
        m = MockVLABackend(action_dim=3)
        state = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(50):
            action = m.predict({"proprioception": state})
            state += action
        # Should be close to target [0.0, 0.5, -0.3]
        assert np.allclose(state, np.array([0.0, 0.5, -0.3]), atol=0.05)

    def test_openvla_raises_without_transformers(self):
        vla = OpenVLA(action_dim=7)
        with pytest.raises(RuntimeError, match="transformers"):
            vla.load("openvla/openvla-7b")

    def test_llava_robot_load_raises(self):
        vla = LLaVARobot(action_dim=7)
        with pytest.raises(RuntimeError, match="checkpoint not yet integrated"):
            vla.load("llava-robot/checkpoint")

    def test_pi05_load_raises(self):
        vla = PiZeroFive(action_dim=7)
        with pytest.raises(RuntimeError, match="closed"):
            vla.load("physical-intelligence/pi0.5")


# ── Observation packaging (bridge logic extracted) ────────────────

class TestObservationPackaging:
    """Test the observation helpers that live in MuJoCoBridge."""

    def _make_bridge_state(self, n_joints: int = 7):
        """Return minimal objects mimicking a loaded bridge."""
        model = FakeMuJoCoModel(n_joints)
        data = FakeMuJoCoData(model.nq, model.nv)
        joint_names = [f"j{i}" for i in range(n_joints)]
        joint_map = {name: i for i, name in enumerate(joint_names)}
        # Seed with non-zero positions
        for i in range(n_joints):
            data.qpos[i] = 0.1 * i
            data.qvel[i] = -0.05 * i
        return model, data, joint_names, joint_map

    def _get_proprio(self, model, data, joint_names, joint_map) -> np.ndarray:
        """Mirror of MuJoCoBridge._get_proprio for testing."""
        pos_list = []
        vel_list = []
        for name in joint_names:
            idx = joint_map[name]
            qposadr = model.jnt_qposadr[idx]
            qveladr = model.jnt_dofadr[idx]
            jnt_range = model.jnt_range[idx]
            lo, hi = jnt_range
            span = hi - lo if hi != lo else 1.0
            raw_pos = float(data.qpos[qposadr])
            norm_pos = (raw_pos - lo) / span
            pos_list.append(norm_pos)
            vel_list.append(float(data.qvel[qveladr]))
        return np.array(pos_list + vel_list, dtype=np.float32)

    def _get_observation(self, model, data, joint_names, joint_map, intent: str = "") -> dict:
        """Mirror of MuJoCoBridge.get_observation for testing."""
        return {
            "image": np.zeros((480, 640, 3), dtype=np.uint8),  # placeholder
            "proprioception": self._get_proprio(model, data, joint_names, joint_map),
            "intent": intent,
            "timestamp": 0.0,
        }

    def test_proprio_shape(self):
        model, data, joint_names, joint_map = self._make_bridge_state(7)
        proprio = self._get_proprio(model, data, joint_names, joint_map)
        assert proprio.shape == (14,)  # 7 pos + 7 vel

    def test_proprio_positions_normalized(self):
        model, data, joint_names, joint_map = self._make_bridge_state(7)
        proprio = self._get_proprio(model, data, joint_names, joint_map)
        pos = proprio[:7]
        # All normalized positions should be in [0, 1] because the raw
        # positions (0.1*i) are inside [-π, π]
        assert np.all((pos >= 0.0) & (pos <= 1.0))

    def test_observation_keys(self):
        model, data, joint_names, joint_map = self._make_bridge_state(7)
        obs = self._get_observation(model, data, joint_names, joint_map, intent="reach")
        assert set(obs.keys()) == {"image", "proprioception", "intent", "timestamp"}
        assert obs["intent"] == "reach"

    def test_observation_through_mock_vla(self):
        """End-to-end: observation → mock VLA → action."""
        model, data, joint_names, joint_map = self._make_bridge_state(7)
        obs = self._get_observation(model, data, joint_names, joint_map, intent="reach")
        vla = MockVLABackend(action_dim=7)
        action = vla.predict(obs)
        assert action.shape == (7,)


# ─-- Action-space mapping ─────────────────────────────────────────

class TestActionSpaceMapping:
    def test_delta_position_mapping(self):
        """Action = delta positions → new qpos = old qpos + action."""
        model = FakeMuJoCoModel(7)
        data = FakeMuJoCoData(7, 7)
        data.qpos[:] = np.arange(7) * 0.1
        action = np.ones(7, dtype=np.float32) * 0.05
        # Simulate bridge apply
        data.qpos[:] += action
        expected = np.arange(7) * 0.1 + 0.05
        np.testing.assert_allclose(data.qpos, expected, atol=1e-6)

    def test_action_chunk_queue(self):
        """Action chunks should be consumable one step at a time."""
        chunk = np.ones((8, 7), dtype=np.float32) * 0.01
        queue = list(chunk)
        for _ in range(8):
            action = queue.pop(0)
            assert action.shape == (7,)
        assert len(queue) == 0


# ── Intent round-trip (simulated command handling) ────────────────

class TestIntentRoundTrip:
    def test_set_intent_command(self):
        """Bridge should store intent and return it in observation."""
        current_intent = ""

        def handle_command(cmd: dict):
            nonlocal current_intent
            action = cmd.get("action")
            if action == "set_intent":
                current_intent = cmd.get("text", "")
                return {"ok": True, "intent": current_intent}
            if action == "get_intent":
                return {"ok": True, "intent": current_intent}
            return {"error": "unknown"}

        resp = handle_command({"action": "set_intent", "text": "reach the red cube"})
        assert resp["ok"]
        assert resp["intent"] == "reach the red cube"

        resp2 = handle_command({"action": "get_intent"})
        assert resp2["intent"] == "reach the red cube"


# ── collect_demo script smoke test ────────────────────────────────

class TestCollectDemoHelpers:
    def test_parse_state_valid(self):
        msg = {
            "ok": True,
            "state": {
                "timestamp": 1.23,
                "joints": [
                    {"name": "j1", "position": 0.1, "velocity": 0.01},
                    {"name": "j2", "position": 0.2, "velocity": -0.02},
                ],
            },
        }
        from examples.collect_demo import parse_state
        state = parse_state(msg)
        assert state is not None
        np.testing.assert_allclose(state["positions"], [0.1, 0.2])
        np.testing.assert_allclose(state["velocities"], [0.01, -0.02])

    def test_parse_state_error(self):
        from examples.collect_demo import parse_state
        assert parse_state({"error": "boom"}) is None


# ── VLAController integration ─────────────────────────────────────

try:
    import mujoco
    HAS_MUJOCO_TEST = True
except Exception:
    HAS_MUJOCO_TEST = False


@pytest.mark.skipif(not HAS_MUJOCO_TEST, reason="mujoco not installed")
class TestVLAController:
    """Integration tests for the closed-loop VLA controller."""

    _ARM_XML: str = """
    <mujoco model="test_arm">
      <compiler angle="radian" autolimits="true"/>
      <option timestep="0.002"/>
      <worldbody>
        <body name="base">
          <geom type="box" size="0.1 0.1 0.1"/>
          <body name="l1" pos="0 0 0.2">
            <joint name="j1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
            <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2"/>
            <body name="l2" pos="0 0 0.2">
              <joint name="j2" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
              <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2"/>
              <site name="ee" pos="0 0 0.2" size="0.02"/>
            </body>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """

    def _make_bridge(self) -> "MuJoCoBridge":
        from isonome.sim.mujoco_bridge import MuJoCoBridge
        bridge = MuJoCoBridge()
        # Write temp XML and load
        import tempfile
        path = Path(tempfile.gettempdir()) / "test_vla_arm.xml"
        path.write_text(self._ARM_XML)
        result = bridge._cmd_load_urdf(str(path))
        assert result["ok"], result.get("error")
        return bridge

    def test_controller_rejects_unloaded_bridge(self):
        from isonome.sim.vla_controller import VLAController
        from isonome.sim.mujoco_bridge import MuJoCoBridge
        bridge = MuJoCoBridge()
        vla = MockVLABackend(action_dim=2)
        ctrl = VLAController(bridge, vla)
        with pytest.raises(RuntimeError, match="no loaded model"):
            ctrl.run(n_steps=1)

    def test_controller_run_returns_trajectory(self):
        from isonome.sim.vla_controller import VLAController
        bridge = self._make_bridge()
        n_joints = len(bridge._joint_names)
        target = np.array([0.2, -0.1], dtype=np.float32)
        vla = MockVLABackend(target=target, action_dim=n_joints)
        ctrl = VLAController(bridge, vla, intent="reach the target")
        traj = ctrl.run(n_steps=10, inference_hz=100.0, physics_hz=100.0)
        assert len(traj) == 10
        assert all("step" in t and "ee_pos" in t for t in traj)
        assert traj[0]["intent"] == "reach the target"

    def test_controller_moves_end_effector(self):
        from isonome.sim.vla_controller import VLAController
        bridge = self._make_bridge()
        n_joints = len(bridge._joint_names)
        # Target drives j1 toward 0.5, j2 toward -0.3
        target = np.array([0.5, -0.3], dtype=np.float32)
        vla = MockVLABackend(target=target, action_dim=n_joints)
        ctrl = VLAController(bridge, vla, intent="reach the target")
        traj = ctrl.run(n_steps=50, inference_hz=100.0, physics_hz=100.0)
        start_ee = np.array(traj[0]["ee_pos"])
        end_ee = np.array(traj[-1]["ee_pos"])
        # End-effector should have moved
        assert not np.allclose(start_ee, end_ee, atol=1e-4)

    def test_controller_reset_clears_filter(self):
        from isonome.sim.vla_controller import VLAController
        bridge = self._make_bridge()
        vla = MockVLABackend(action_dim=len(bridge._joint_names))
        ctrl = VLAController(bridge, vla)
        ctrl.run(n_steps=5, inference_hz=100.0, physics_hz=100.0)
        assert ctrl._last_action is not None
        ctrl.reset()
        assert ctrl._last_action is None
        assert ctrl._step_count == 0
