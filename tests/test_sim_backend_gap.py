"""Tests for gap #4: Isaac Lab + MuJoCo MJX simulation backends.

These tests mock the heavy optional dependencies (jax, mujoco.mjx, isaaclab)
so they can run in any environment.  They verify that the bridges expose the
same command protocol as ``MuJoCoBridge`` and that the factory can construct
``BodyBridge`` adapters for the new engines.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
import torch

from isonome.core.config import AppConfig, BridgeConfig, SomaConfig
from isonome.core.state import CorrectedMotorCommand


TEST_URDF = Path(__file__).parent.parent / "examples" / "robot_arm.urdf"


@pytest.fixture
def fake_mujoco_module() -> ModuleType:
    """Return a minimal stand-in for ``mujoco`` with the attributes MJX uses."""
    mod = ModuleType("mujoco")
    mod.MjModel = MagicMock()
    mod.MjData = MagicMock()
    mod.Renderer = MagicMock()
    mod.MjvCamera = MagicMock

    mjtJoint = ModuleType("mjtJoint")
    mjtJoint.mjJNT_HINGE = 3
    mjtJoint.mjJNT_SLIDE = 2
    mjtJoint.mjJNT_BALL = 1
    mjtJoint.mjJNT_FREE = 0
    mod.mjtJoint = mjtJoint

    mjtObj = ModuleType("mjtObj")
    mjtObj.mjOBJ_JOINT = 1
    mod.mjtObj = mjtObj

    mjtCamera = ModuleType("mjtCamera")
    mjtCamera.mjCAMERA_TRACKING = 0
    mod.mjtCamera = mjtCamera

    mod.mj_id2name = MagicMock(return_value="joint_0")
    mod.mj_resetData = MagicMock()
    mod.mj_forward = MagicMock()
    mod.mj_step = MagicMock()

    return mod


@pytest.fixture
def fake_mjx_module(fake_mujoco_module: ModuleType) -> ModuleType:
    """Return a minimal stand-in for ``mujoco.mjx``."""
    mod = ModuleType("mujoco.mjx")
    mod.put_model = MagicMock(return_value=MagicMock())
    mod.put_data = MagicMock(return_value=MagicMock())
    mod.step = MagicMock(return_value=MagicMock())
    mod.get_data_into = MagicMock()
    return mod


@pytest.fixture
def fake_jax_module() -> ModuleType:
    """Return a minimal stand-in for ``jax``."""
    mod = ModuleType("jax")
    mod.__path__ = []  # type: ignore[attr-defined]
    numpy_mod = ModuleType("jax.numpy")
    mod.numpy = numpy_mod
    return mod


@pytest.fixture
def fake_isaaclab_module() -> ModuleType:
    """Return a minimal stand-in for ``isaaclab``."""
    mod = ModuleType("isaaclab")
    mod.__path__ = []  # type: ignore[attr-defined]
    mod.app = ModuleType("isaaclab.app")
    mod.app.__path__ = []  # type: ignore[attr-defined]
    mod.app.AppLauncher = MagicMock()
    mod.envs = ModuleType("isaaclab.envs")
    mod.envs.__path__ = []  # type: ignore[attr-defined]
    mod.envs.ManagerBasedRLEnv = MagicMock()
    mod.envs.ManagerBasedRLEnvCfg = MagicMock()
    mod.managers = ModuleType("isaaclab.managers")
    mod.managers.__path__ = []  # type: ignore[attr-defined]
    mod.managers.ObservationManagerCfg = MagicMock()
    mod.managers.RewardManagerCfg = MagicMock()
    mod.scene = ModuleType("isaaclab.scene")
    mod.scene.__path__ = []  # type: ignore[attr-defined]
    mod.scene.InteractiveSceneCfg = MagicMock()
    mod.assets = ModuleType("isaaclab.assets")
    mod.assets.__path__ = []  # type: ignore[attr-defined]
    mod.assets.ArticulationCfg = MagicMock()
    mod.assets.ArticulationCfg.InitialStateCfg = MagicMock()
    mod.utils = ModuleType("isaaclab.utils")
    mod.utils.__path__ = []  # type: ignore[attr-defined]
    mod.utils.assets = ModuleType("isaaclab.utils.assets")
    mod.utils.assets.__path__ = []  # type: ignore[attr-defined]
    mod.utils.assets.ISAACLAB_NUCLEUS_DIR = ""
    return mod


# ==============================================================================
# MJX Bridge (command protocol)
# ==============================================================================


def test_mjx_bridge_raises_without_jax() -> None:
    """MJXBridge refuses to instantiate when JAX is unavailable."""
    from isonome.sim.mjx_bridge import MJXBridge

    with pytest.raises(RuntimeError, match="jax"):
        MJXBridge()


def test_mjx_bridge_load_urdf_with_mocks(
    fake_jax_module: ModuleType,
    fake_mujoco_module: ModuleType,
    fake_mjx_module: ModuleType,
) -> None:
    """MJXBridge can load a URDF when JAX/MJX are mocked."""
    fake_mujoco_module.mjx = fake_mjx_module
    with (
        patch.dict("sys.modules", {"jax": fake_jax_module}),
        patch.dict("sys.modules", {"jax.numpy": fake_jax_module.numpy}),
        patch.dict("sys.modules", {"mujoco": fake_mujoco_module}),
        patch.dict("sys.modules", {"mujoco.mjx": fake_mjx_module}),
    ):
        # Force a clean import of the bridge module with the fakes in place.
        for key in list(sys.modules):
            if key.startswith("isonome.sim.mjx_bridge"):
                del sys.modules[key]
        from isonome.sim.mjx_bridge import MJXBridge

        bridge = MJXBridge()
        result = bridge._cmd_load_urdf(str(TEST_URDF))
        assert result["ok"] is True
        assert result["dof_count"] == len(result["joints"])
        fake_mjx_module.put_model.assert_called()
        fake_mjx_module.put_data.assert_called()


def test_mjx_bridge_apply_action_with_mocks(
    fake_jax_module: ModuleType,
    fake_mujoco_module: ModuleType,
    fake_mjx_module: ModuleType,
) -> None:
    """MJXBridge apply_action mutates qpos and syncs back to MJX."""
    fake_mujoco_module.mjx = fake_mjx_module
    with (
        patch.dict("sys.modules", {"jax": fake_jax_module}),
        patch.dict("sys.modules", {"jax.numpy": fake_jax_module.numpy}),
        patch.dict("sys.modules", {"mujoco": fake_mujoco_module}),
        patch.dict("sys.modules", {"mujoco.mjx": fake_mjx_module}),
    ):
        for key in list(sys.modules):
            if key.startswith("isonome.sim.mjx_bridge"):
                del sys.modules[key]
        from isonome.sim.mjx_bridge import MJXBridge

        bridge = MJXBridge()
        bridge._cmd_load_urdf(str(TEST_URDF))
        result = bridge._cmd_apply_action([0.1] * bridge.joint_count)
        assert result["ok"] is True
        fake_mujoco_module.mj_forward.assert_called()
        fake_mjx_module.put_data.assert_called()


# ==============================================================================
# Isaac Lab Bridge (command protocol)
# ==============================================================================


def test_isaac_lab_bridge_raises_without_isaaclab() -> None:
    """IsaacLabBridge refuses to instantiate when isaaclab is unavailable."""
    from isonome.sim.isaac_lab_bridge import IsaacLabBridge

    with pytest.raises(RuntimeError, match="Isaac Lab"):
        IsaacLabBridge()


def test_isaac_lab_bridge_load_urdf_with_mocks(
    fake_isaaclab_module: ModuleType,
) -> None:
    """IsaacLabBridge can load a URDF when isaaclab is mocked."""
    with (
        patch.dict("sys.modules", {"isaaclab": fake_isaaclab_module}),
        patch.dict("sys.modules", {"isaaclab.app": fake_isaaclab_module.app}),
        patch.dict("sys.modules", {"isaaclab.envs": fake_isaaclab_module.envs}),
        patch.dict("sys.modules", {"isaaclab.managers": fake_isaaclab_module.managers}),
        patch.dict("sys.modules", {"isaaclab.scene": fake_isaaclab_module.scene}),
        patch.dict("sys.modules", {"isaaclab.assets": fake_isaaclab_module.assets}),
        patch.dict("sys.modules", {"isaaclab.utils": fake_isaaclab_module.utils}),
        patch.dict("sys.modules", {"isaaclab.utils.assets": fake_isaaclab_module.utils.assets}),
    ):
        for key in list(sys.modules):
            if key.startswith("isonome.sim.isaac_lab_bridge"):
                del sys.modules[key]
        from isonome.sim.isaac_lab_bridge import IsaacLabBridge

        # Build a fake articulation with tensor-shaped data
        fake_env = MagicMock()
        fake_robot = MagicMock()
        fake_robot.joint_names = ["joint_0", "joint_1"]
        fake_robot.data.joint_pos = torch.zeros(1, 2)
        fake_robot.data.joint_vel = torch.zeros(1, 2)
        fake_robot.data.default_joint_pos = torch.zeros(1, 2)
        fake_robot.data.default_joint_vel = torch.zeros(1, 2)
        fake_robot.data.root_pos_w = torch.zeros(1, 3)
        fake_robot.data.root_quat_w = torch.zeros(1, 4)
        fake_env.scene = {"robot": fake_robot}
        fake_env.sim = MagicMock()

        fake_isaaclab_module.envs.ManagerBasedRLEnv.return_value = fake_env

        bridge = IsaacLabBridge()
        result = bridge._cmd_load_urdf(str(TEST_URDF))
        assert result["ok"] is True
        assert result["dof_count"] == 2


# ==============================================================================
# BodyBridge adapters + factory
# ==============================================================================


def test_factory_creates_mujoco_mjx_bridge_with_mocked_deps(
    fake_jax_module: ModuleType,
    fake_mujoco_module: ModuleType,
    fake_mjx_module: ModuleType,
) -> None:
    """``build_body_bridge(engine='mujoco_mjx')`` constructs an adapter."""
    fake_mujoco_module.mjx = fake_mjx_module
    with (
        patch.dict("sys.modules", {"jax": fake_jax_module}),
        patch.dict("sys.modules", {"jax.numpy": fake_jax_module.numpy}),
        patch.dict("sys.modules", {"mujoco": fake_mujoco_module}),
        patch.dict("sys.modules", {"mujoco.mjx": fake_mjx_module}),
    ):
        for key in list(sys.modules):
            if key.startswith("isonome.sim.mjx_bridge") or key.startswith(
                "isonome.bridge.adapters.mjx_adapter"
            ):
                del sys.modules[key]
        from isonome.bridge.adapters.mjx_adapter import MJXBridgeAdapter
        from isonome.bridge.factory import build_body_bridge

        cfg = AppConfig(
            soma=SomaConfig(urdf_path=str(TEST_URDF)),
            bridge=BridgeConfig(engine="mujoco_mjx"),
        )
        bridge = build_body_bridge(cfg)
        assert isinstance(bridge, MJXBridgeAdapter)
        assert bridge.name == "mujoco_mjx"


def test_factory_creates_isaac_lab_bridge_with_mocked_deps(
    fake_isaaclab_module: ModuleType,
) -> None:
    """``build_body_bridge(engine='isaac_lab')`` constructs an adapter."""
    with (
        patch.dict("sys.modules", {"isaaclab": fake_isaaclab_module}),
        patch.dict("sys.modules", {"isaaclab.app": fake_isaaclab_module.app}),
        patch.dict("sys.modules", {"isaaclab.envs": fake_isaaclab_module.envs}),
        patch.dict("sys.modules", {"isaaclab.managers": fake_isaaclab_module.managers}),
        patch.dict("sys.modules", {"isaaclab.scene": fake_isaaclab_module.scene}),
        patch.dict("sys.modules", {"isaaclab.assets": fake_isaaclab_module.assets}),
        patch.dict("sys.modules", {"isaaclab.utils": fake_isaaclab_module.utils}),
        patch.dict("sys.modules", {"isaaclab.utils.assets": fake_isaaclab_module.utils.assets}),
    ):
        for key in list(sys.modules):
            if key.startswith("isonome.sim.isaac_lab_bridge") or key.startswith(
                "isonome.bridge.adapters.isaac_lab_adapter"
            ):
                del sys.modules[key]
        from isonome.bridge.adapters.isaac_lab_adapter import IsaacLabBridgeAdapter
        from isonome.bridge.factory import build_body_bridge

        cfg = AppConfig(
            soma=SomaConfig(urdf_path=str(TEST_URDF)),
            bridge=BridgeConfig(engine="isaac_lab"),
        )
        bridge = build_body_bridge(cfg)
        assert isinstance(bridge, IsaacLabBridgeAdapter)
        assert bridge.name == "isaac_lab"


# ==============================================================================
# Adapter lifecycle smoke tests (with mocks)
# ==============================================================================


@pytest.mark.asyncio
async def test_mjx_adapter_boot_shutdown_with_mocks(
    fake_jax_module: ModuleType,
    fake_mujoco_module: ModuleType,
    fake_mjx_module: ModuleType,
) -> None:
    """MJXBridgeAdapter boots and shuts down cleanly with mocked MJX."""
    fake_mujoco_module.mjx = fake_mjx_module
    with (
        patch.dict("sys.modules", {"jax": fake_jax_module}),
        patch.dict("sys.modules", {"jax.numpy": fake_jax_module.numpy}),
        patch.dict("sys.modules", {"mujoco": fake_mujoco_module}),
        patch.dict("sys.modules", {"mujoco.mjx": fake_mjx_module}),
    ):
        for key in list(sys.modules):
            if key.startswith("isonome.sim.mjx_bridge") or key.startswith(
                "isonome.bridge.adapters.mjx_adapter"
            ):
                del sys.modules[key]
        from isonome.bridge.adapters.mjx_adapter import MJXBridgeAdapter

        adapter = MJXBridgeAdapter(TEST_URDF)
        await adapter.boot()
        assert adapter.is_connected
        assert adapter.joint_count > 0

        raw = await adapter.perceive()
        assert raw.proprioception.numel() == adapter.joint_count * 2

        cmd = CorrectedMotorCommand(commands=torch.zeros(1, adapter.joint_count))
        await adapter.act(cmd)

        result = await adapter.observe_result()
        assert result.success

        await adapter.shutdown()
        assert not adapter.is_connected


@pytest.mark.asyncio
async def test_isaac_lab_adapter_boot_shutdown_with_mocks(
    fake_isaaclab_module: ModuleType,
) -> None:
    """IsaacLabBridgeAdapter boots and shuts down cleanly with mocked isaaclab."""
    with (
        patch.dict("sys.modules", {"isaaclab": fake_isaaclab_module}),
        patch.dict("sys.modules", {"isaaclab.app": fake_isaaclab_module.app}),
        patch.dict("sys.modules", {"isaaclab.envs": fake_isaaclab_module.envs}),
        patch.dict("sys.modules", {"isaaclab.managers": fake_isaaclab_module.managers}),
        patch.dict("sys.modules", {"isaaclab.scene": fake_isaaclab_module.scene}),
        patch.dict("sys.modules", {"isaaclab.assets": fake_isaaclab_module.assets}),
        patch.dict("sys.modules", {"isaaclab.utils": fake_isaaclab_module.utils}),
        patch.dict("sys.modules", {"isaaclab.utils.assets": fake_isaaclab_module.utils.assets}),
    ):
        for key in list(sys.modules):
            if key.startswith("isonome.sim.isaac_lab_bridge") or key.startswith(
                "isonome.bridge.adapters.isaac_lab_adapter"
            ):
                del sys.modules[key]
        from isonome.bridge.adapters.isaac_lab_adapter import IsaacLabBridgeAdapter

        fake_env = MagicMock()
        fake_robot = MagicMock()
        fake_robot.joint_names = ["joint_0", "joint_1"]
        fake_robot.data.joint_pos = torch.zeros(1, 2)
        fake_robot.data.joint_vel = torch.zeros(1, 2)
        fake_robot.data.default_joint_pos = torch.zeros(1, 2)
        fake_robot.data.default_joint_vel = torch.zeros(1, 2)
        fake_robot.data.root_pos_w = torch.zeros(1, 3)
        fake_robot.data.root_quat_w = torch.zeros(1, 4)
        fake_env.scene = {"robot": fake_robot}
        fake_env.sim = MagicMock()
        fake_isaaclab_module.envs.ManagerBasedRLEnv.return_value = fake_env

        adapter = IsaacLabBridgeAdapter(TEST_URDF)
        await adapter.boot()
        assert adapter.is_connected
        assert adapter.joint_count == 2

        raw = await adapter.perceive()
        assert raw.proprioception.numel() == 4

        cmd = CorrectedMotorCommand(commands=torch.zeros(1, 2))
        await adapter.act(cmd)

        result = await adapter.observe_result()
        assert result.success

        await adapter.shutdown()
        assert not adapter.is_connected
