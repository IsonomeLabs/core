"""Tests for isonome.core modules: bus, config, safety.

Imports target submodules directly to avoid torch dependency
in isonome.core.__init__.py.
"""
from __future__ import annotations

import asyncio
import pytest

from isonome.core.bus import Channel, MessageBus
from isonome.core.config import (
    AppConfig,
    CortexConfig,
    JEPAConfig,
    PlasticityConfig,
    ReflexConfig,
    SafetyConfig,
    SimConfig,
    SomaConfig,
)
from isonome.core.safety import AgentMode, EmergencyStop, SafetyGovernor


# ======================================================================
# MessageBus tests
# ======================================================================

class TestMessageBus:

    @pytest.mark.asyncio
    async def test_start_stop(self):
        bus = MessageBus()
        await bus.start()
        assert bus._running is True
        await bus.stop()
        assert bus._running is False

    @pytest.mark.asyncio
    async def test_all_channels_initialized(self):
        bus = MessageBus()
        assert len(bus._queues) == len(Channel)
        for ch in Channel:
            assert ch in bus._queues

    @pytest.mark.asyncio
    async def test_publish_and_receive(self):
        bus = MessageBus()
        await bus.start()
        try:
            await bus.publish(Channel.SENSORS, {"temperature": 42.0})
            msg = await bus.receive(Channel.SENSORS, timeout=1.0)
            assert msg == {"temperature": 42.0}
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_publish_when_stopped_is_noop(self):
        bus = MessageBus()
        await bus.publish(Channel.SENSORS, "test")
        success, msg = await bus.try_receive(Channel.SENSORS)
        assert success is False

    @pytest.mark.asyncio
    async def test_try_receive_empty(self):
        bus = MessageBus()
        success, msg = await bus.try_receive(Channel.SENSORS)
        assert success is False
        assert msg is None

    @pytest.mark.asyncio
    async def test_try_receive_with_message(self):
        bus = MessageBus()
        await bus.start()
        try:
            await bus.publish(Channel.ERROR, "test_error")
            success, msg = await bus.try_receive(Channel.ERROR)
            assert success is True
            assert msg == "test_error"
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_subscribe_callback(self):
        bus = MessageBus()
        await bus.start()
        received = []

        async def on_sensor(msg):
            received.append(msg)

        bus.subscribe(Channel.SENSORS, on_sensor)
        await bus.publish(Channel.SENSORS, {"key": "value"})
        await asyncio.sleep(0.1)
        assert len(received) == 1
        assert received[0] == {"key": "value"}
        await bus.stop()

    @pytest.mark.asyncio
    async def test_queue_full_drops_message(self):
        bus = MessageBus(maxsize=2)
        await bus.start()
        try:
            await bus.publish(Channel.SENSORS, "msg1")
            await bus.publish(Channel.SENSORS, "msg2")
            await bus.publish(Channel.SENSORS, "msg3")
            msg1 = await bus.receive(Channel.SENSORS, timeout=1.0)
            msg2 = await bus.receive(Channel.SENSORS, timeout=1.0)
            assert msg1 == "msg1"
            assert msg2 == "msg2"
            success, _ = await bus.try_receive(Channel.SENSORS)
            assert success is False
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_multiple_channels_independent(self):
        bus = MessageBus()
        await bus.start()
        try:
            await bus.publish(Channel.SENSORS, "sensor_data")
            await bus.publish(Channel.ERROR, "error_data")
            s = await bus.receive(Channel.SENSORS, timeout=1.0)
            e = await bus.receive(Channel.ERROR, timeout=1.0)
            assert s == "sensor_data"
            assert e == "error_data"
        finally:
            await bus.stop()


# ======================================================================
# Config tests
# ======================================================================

class TestReflexConfig:
    def test_defaults(self):
        c = ReflexConfig()
        assert c.frequency_hz == 100.0
        assert c.max_latency_ms == 10.0
        assert c.control_freq_hz == 100.0
        assert c.policy_freq_hz == 1.0

    def test_custom(self):
        c = ReflexConfig(frequency_hz=200.0, max_latency_ms=5.0)
        assert c.frequency_hz == 200.0
        assert c.max_latency_ms == 5.0


class TestJEPAConfig:
    def test_defaults(self):
        c = JEPAConfig()
        assert c.frequency_hz == 10.0
        assert c.backend == "openvla"
        assert c.model_id is None

    def test_custom(self):
        c = JEPAConfig(backend="custom", model_id="test-model")
        assert c.backend == "custom"
        assert c.model_id == "test-model"


class TestCortexConfig:
    def test_defaults(self):
        c = CortexConfig()
        assert c.frequency_hz == 0.5


class TestSomaConfig:
    def test_defaults(self):
        c = SomaConfig()
        assert c.urdf_path == ""
        assert c.kernel_path is None


class TestPlasticityConfig:
    def test_defaults(self):
        c = PlasticityConfig()
        assert c.kernel_dir == "~/.isonome/kernels"


class TestSafetyConfig:
    def test_defaults(self):
        c = SafetyConfig()
        assert c.permit_boot_adaptation is False
        assert c.error_window_s == 300.0
        assert c.error_repeat_threshold == 3
        assert c.sim_validation_ticks == 100


class TestSimConfig:
    def test_defaults(self):
        c = SimConfig()
        assert c.engine == "pybullet"
        assert c.timestep == pytest.approx(1.0 / 240.0)
        assert c.gravity == (0.0, 0.0, -9.81)
        assert c.gui is False

    def test_godot_engine(self):
        c = SimConfig(engine="godot")
        assert c.engine == "godot"


class TestAppConfig:
    def test_defaults(self):
        c = AppConfig()
        assert c.agent_name == "isonome_agent"
        assert c.preset is None
        assert isinstance(c.reflex, ReflexConfig)
        assert isinstance(c.jepa, JEPAConfig)
        assert isinstance(c.cortex, CortexConfig)
        assert isinstance(c.soma, SomaConfig)
        assert isinstance(c.plasticity, PlasticityConfig)
        assert isinstance(c.safety, SafetyConfig)
        assert isinstance(c.sim, SimConfig)

    def test_nested_override(self):
        c = AppConfig(
            agent_name="test",
            reflex=ReflexConfig(frequency_hz=500.0),
        )
        assert c.agent_name == "test"
        assert c.reflex.frequency_hz == 500.0

    def test_model_dump_roundtrip(self):
        c = AppConfig(agent_name="roundtrip")
        d = c.model_dump()
        c2 = AppConfig(**d)
        assert c2.agent_name == "roundtrip"
        assert c2.reflex.frequency_hz == c.reflex.frequency_hz


# ======================================================================
# Safety tests
# ======================================================================

class TestAgentMode:
    def test_all_modes_exist(self):
        modes = {m.value for m in AgentMode}
        assert "boot" in modes
        assert "calibrating" in modes
        assert "runtime" in modes
        assert "idle" in modes
        assert "safe_stop" in modes

    def test_string_enum(self):
        assert AgentMode.BOOT == "boot"
        assert AgentMode.RUNTIME == "runtime"


class TestEmergencyStop:
    def test_is_exception(self):
        with pytest.raises(EmergencyStop):
            raise EmergencyStop("test emergency")

    def test_can_be_caught(self):
        try:
            raise EmergencyStop("critical")
        except EmergencyStop as e:
            assert "critical" in str(e)


class TestSafetyGovernor:
    """Test SafetyGovernor with a mock agent."""

    def _make_mock_agent(self, mode=AgentMode.BOOT):
        class MockReflex:
            killed = False
            def kill(self):
                self.killed = True

        class MockAgent:
            def __init__(self, m):
                self.mode = m
                self.reflex = MockReflex()

        return MockAgent(mode)

    def test_can_execute_boot(self):
        agent = self._make_mock_agent(AgentMode.BOOT)
        gov = SafetyGovernor(agent)
        assert gov.can_execute() is False

    def test_can_execute_runtime(self):
        agent = self._make_mock_agent(AgentMode.RUNTIME)
        gov = SafetyGovernor(agent)
        assert gov.can_execute() is True

    def test_can_execute_calibrating(self):
        agent = self._make_mock_agent(AgentMode.CALIBRATING)
        gov = SafetyGovernor(agent)
        assert gov.can_execute() is True

    def test_can_execute_idle(self):
        agent = self._make_mock_agent(AgentMode.IDLE)
        gov = SafetyGovernor(agent)
        assert gov.can_execute() is False

    def test_can_execute_safe_stop(self):
        agent = self._make_mock_agent(AgentMode.SAFE_STOP)
        gov = SafetyGovernor(agent)
        assert gov.can_execute() is False

    def test_can_load_kernel_boot(self):
        agent = self._make_mock_agent(AgentMode.BOOT)
        gov = SafetyGovernor(agent)
        assert gov.can_load_kernel() is True

    def test_can_load_kernel_idle(self):
        agent = self._make_mock_agent(AgentMode.IDLE)
        gov = SafetyGovernor(agent)
        assert gov.can_load_kernel() is True

    def test_can_load_kernel_runtime(self):
        agent = self._make_mock_agent(AgentMode.RUNTIME)
        gov = SafetyGovernor(agent)
        assert gov.can_load_kernel() is False

    def test_emergency_stop(self):
        agent = self._make_mock_agent(AgentMode.RUNTIME)
        gov = SafetyGovernor(agent)
        gov.emergency_stop()
        assert agent.mode == AgentMode.SAFE_STOP
        assert agent.reflex.killed is True


class TestChannel:
    def test_all_channels_exist(self):
        expected = {
            "sensors", "reflex_output", "jepa_adjustment",
            "cortex_advice", "plasticity_patches", "error",
        }
        actual = {ch.value for ch in Channel}
        assert actual == expected

    def test_channel_is_string_enum(self):
        assert Channel.SENSORS == "sensors"
        assert isinstance(Channel.SENSORS, str)
