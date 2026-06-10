"""Tests for the BodyBridge port and adapters.

These tests verify the core integration point between ``SomaLayer`` and
the various simulation / hardware backends.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import torch

from isonome.bridge.adapters.mock_adapter import MockBridgeAdapter
from isonome.bridge.factory import build_body_bridge
from isonome.core.config import AppConfig, BridgeConfig, SomaConfig
from isonome.core.layers.soma import SomaLayer
from isonome.core.ports.body_bridge import BodyBridge, BridgePortError
from isonome.core.state import CorrectedMotorCommand, RawSensorState


TEST_URDF = Path(__file__).parent.parent / "examples" / "robot_arm.urdf"


class FakeBodyBridge(BodyBridge):
    """In-memory BodyBridge for unit testing."""

    def __init__(self, urdf_path: Path, joint_count: int = 3) -> None:
        super().__init__(urdf_path)
        self._joint_count = joint_count
        self.perceived: list[RawSensorState] = []
        self.commands: list[CorrectedMotorCommand] = []
        self.booted = False
        self.shutdown_ = False

    @property
    def name(self) -> str:
        return "fake"

    @property
    def joint_count(self) -> int:
        return self._joint_count

    async def _do_boot(self) -> None:
        self.booted = True

    async def _do_shutdown(self) -> None:
        self.shutdown_ = True

    async def _do_perceive(self) -> RawSensorState:
        state = RawSensorState(
            proprioception=torch.arange(self._joint_count, dtype=torch.float32),
            camera_frames=[],
        )
        self.perceived.append(state)
        return state

    async def _do_act(self, cmd: CorrectedMotorCommand) -> None:
        self.commands.append(cmd)


# ==============================================================================
# BodyBridge contract
# ==============================================================================


@pytest.mark.asyncio
async def test_body_bridge_lifecycle() -> None:
    bridge = FakeBodyBridge(TEST_URDF, joint_count=3)
    assert not bridge.is_connected

    await bridge.boot()
    assert bridge.is_connected
    assert bridge.booted

    await bridge.boot()  # idempotent
    assert bridge.is_connected

    await bridge.shutdown()
    assert not bridge.is_connected
    assert bridge.shutdown_

    await bridge.shutdown()  # idempotent
    assert not bridge.is_connected


@pytest.mark.asyncio
async def test_body_bridge_perceive_before_boot_raises() -> None:
    bridge = FakeBodyBridge(TEST_URDF, joint_count=3)
    with pytest.raises(BridgePortError):
        await bridge.perceive()


# ==============================================================================
# MockBridgeAdapter
# ==============================================================================


@pytest.mark.asyncio
async def test_mock_bridge_boot_shutdown() -> None:
    bridge = MockBridgeAdapter(TEST_URDF)
    assert bridge.joint_count == 0

    await bridge.boot()
    assert bridge.is_connected
    assert bridge.joint_count == 7  # robot_arm.urdf has 7 revolute joints

    await bridge.shutdown()
    assert not bridge.is_connected


@pytest.mark.asyncio
async def test_mock_bridge_perceive_act_observe() -> None:
    bridge = MockBridgeAdapter(TEST_URDF)
    await bridge.boot()

    raw = await bridge.perceive()
    assert raw.proprioception.numel() == 14  # 7 positions + 7 velocities

    cmd = CorrectedMotorCommand(commands=torch.zeros(1, 7))
    await bridge.act(cmd)

    result = await bridge.observe_result()
    assert result.success
    assert result.final_proprioception.numel() == 14

    await bridge.shutdown()


# ==============================================================================
# SomaLayer integration
# ==============================================================================


@pytest.mark.asyncio
async def test_soma_layer_delegates_to_bridge() -> None:
    fake = FakeBodyBridge(TEST_URDF, joint_count=7)
    soma = SomaLayer(urdf_path=TEST_URDF, body_bridge=fake)

    await soma.boot()
    assert fake.booted

    # Agent.tick() calls _async_perceive which uses the bridge directly,
    # but soma.perceive() should still fall back to stub data.
    raw = soma.perceive()
    assert raw.proprioception.numel() == 7

    await soma.shutdown()
    assert fake.shutdown_


@pytest.mark.asyncio
async def test_soma_layer_no_bridge_is_stub() -> None:
    soma = SomaLayer(urdf_path=TEST_URDF)
    await soma.boot()

    raw = soma.perceive()
    assert torch.all(raw.proprioception == 0)
    assert raw.proprioception.numel() == 7

    await soma.shutdown()


@pytest.mark.asyncio
async def test_soma_layer_joint_count_validation_logs_warning() -> None:
    fake = FakeBodyBridge(TEST_URDF, joint_count=3)  # URDF says 7
    soma = SomaLayer(urdf_path=TEST_URDF, body_bridge=fake)
    # Boot should complete but log a warning about joint count mismatch
    await soma.boot()
    assert soma.body_bridge is fake
    await soma.shutdown()


# ==============================================================================
# Factory
# ==============================================================================


def test_factory_returns_none_for_no_bridge() -> None:
    cfg = AppConfig(bridge=BridgeConfig(engine="none"))
    assert build_body_bridge(cfg) is None


def test_factory_creates_mock_bridge() -> None:
    cfg = AppConfig(
        soma=SomaConfig(urdf_path=str(TEST_URDF)),
        bridge=BridgeConfig(engine="mock"),
    )
    bridge = build_body_bridge(cfg)
    assert isinstance(bridge, MockBridgeAdapter)


def test_factory_legacy_sim_engine_does_not_auto_build_bridge() -> None:
    """Old configs using sim.engine should NOT auto-construct a bridge.

    This preserves backward compatibility for code that set sim.engine but
    never wired up a bridge.
    """
    cfg = AppConfig(
        soma=SomaConfig(urdf_path=str(TEST_URDF)),
        sim=AppConfig().sim,  # default engine=pybullet
        bridge=BridgeConfig(engine="none"),
    )
    bridge = build_body_bridge(cfg)
    assert bridge is None


# ==============================================================================
# Agent integration (smoke test)
# ==============================================================================


@pytest.mark.asyncio
async def test_agent_with_mock_bridge_boots_and_ticks() -> None:
    from isonome.core.agent import Agent

    cfg = AppConfig(
        agent_name="test_bridge_agent",
        soma=SomaConfig(urdf_path=str(TEST_URDF)),
        bridge=BridgeConfig(engine="mock"),
    )
    agent = Agent(cfg)
    await agent.boot()
    assert agent.mode.value == "idle"
    assert agent.soma.body_bridge is not None
    assert agent.soma.body_bridge.is_connected

    # Run a few ticks
    agent.mode = __import__("isonome.core.safety", fromlist=["AgentMode"]).AgentMode.RUNTIME
    for _ in range(3):
        await agent.tick()

    await agent.shutdown()
    assert not agent.soma.body_bridge.is_connected
