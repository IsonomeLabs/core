"""Hello Simulation -- Two agents in PyBullet demonstrating JEPA-modulated Reflex.

Agent A: obstacle avoidance robot (Reflex + JEPA modulation)
Agent B: moving obstacle (simple forward motion)

Mock LLM backends for Cortex/Plasticity.
"""
from __future__ import annotations
import asyncio
import logging
from isonome.core.config import AppConfig, SimConfig
from isonome.core.state import (
    SensorState,
    MotorCommand,
    Adjustment,
    JointReading,
    IMUReading,
)
from isonome.core.layers.reflex import ReflexLayer
from isonome.core.layers.jepa import JEPALayer
from isonome.core.agent import Agent
try:
    import pybullet  # noqa: F401

    HAS_PYBULLET = True
except ImportError:
    HAS_PYBULLET = False

from isonome.bridge.sim import SimBridge

logging.basicConfig(level=logging.WARNING)


class ObstacleAvoidanceReflex(ReflexLayer):
    """Reflex that drives forward and steers away from obstacles."""

    def __init__(self) -> None:
        super().__init__(frequency_hz=100.0)
        self._forward_speed = 0.5

    async def react(self, sensors: SensorState) -> MotorCommand:
        # Simple: drive forward, check contacts for avoidance
        left_force = sum(c.force for c in sensors.contacts if "left" in c.link_name)
        right_force = sum(c.force for c in sensors.contacts if "right" in c.link_name)

        steer = 0.0
        if left_force > 0.1:
            steer = -0.3
        elif right_force > 0.1:
            steer = 0.3

        return MotorCommand(
            joint_positions={"drive": self._forward_speed, "steer": steer},
        )


class AvoidanceJEPA(JEPALayer):
    """JEPA that tightens avoidance radius when predicting fast-moving obstacles."""

    def __init__(self) -> None:
        super().__init__(frequency_hz=10.0)

    async def predict_and_adjust(
        self, sensors: SensorState, reflex_cmd: MotorCommand
    ) -> Adjustment:
        # Mock prediction: if base position is far from origin, predict collision
        speed = sensors.extras.get("obstacle_speed", 0.0)

        if speed > 0.5:
            # Tighten avoidance -- adjust steering to be more aggressive
            return Adjustment(
                position_deltas={
                    "steer": reflex_cmd.joint_positions.get("steer", 0.0) * 1.5,
                },
                metadata={"reason": "fast_obstacle_detected", "speed": speed},
            )
        return Adjustment()


class MovingObstacleReflex(ReflexLayer):
    """Simple reflex that drives the obstacle forward."""

    def __init__(self) -> None:
        super().__init__(frequency_hz=100.0)

    async def react(self, sensors: SensorState) -> MotorCommand:
        return MotorCommand(
            joint_positions={"drive": 1.0},
        )


async def main() -> None:
    # --- Agent A: Obstacle Avoidance Robot ---
    config_a = AppConfig(
        agent_name="avoidance_robot",
        sim=SimConfig(gui=False),
    )
    agent_a = Agent(config_a)
    agent_a.reflex = ObstacleAvoidanceReflex()
    agent_a.jepa = AvoidanceJEPA()

    # --- Agent B: Moving Obstacle ---
    config_b = AppConfig(
        agent_name="moving_obstacle",
        sim=SimConfig(gui=False),
    )
    agent_b = Agent(config_b)
    agent_b.reflex = MovingObstacleReflex()

    # --- Sim Bridge (optional — requires pybullet) ---
    sim = None
    if HAS_PYBULLET:
        sim = SimBridge(SimConfig(gui=False))
        sim.connect()
        print("Isonome hello_sim -- 2 agents, PyBullet connected")
    else:
        print("Isonome hello_sim -- 2 agents, mock sensors (install pybullet for sim)")
    print("Agent A: ObstacleAvoidance (Reflex + JEPA modulation)")
    print("Agent B: MovingObstacle (simple forward motion)")
    print()

    # Mock sense/act for Agent A
    tick_count = 0
    for tick in range(10):
        # Simulate sensor data
        sensors_a = SensorState(
            joints=[
                JointReading(name="drive", position=0.1 * tick),
                JointReading(name="steer", position=0.0),
            ],
            imu=IMUReading(),
            extras={
                "base_position": (0.1 * tick, 0.0, 0.5),
                "obstacle_speed": 0.8 if tick > 3 else 0.0,
            },
        )

        # Agent A: Reflex + JEPA
        reflex_cmd = await agent_a.reflex.react(sensors_a)
        jepa_adj = await agent_a.jepa.predict_and_adjust(sensors_a, reflex_cmd)
        merged = agent_a._merge_reflex_jepa(reflex_cmd, jepa_adj)

        tick_count += 1
        print(
            f"Tick {tick_count:2d} "
            f"| Reflex steer: {reflex_cmd.joint_positions.get('steer', 0):6.2f} "
            f"| JEPA delta: {jepa_adj.position_deltas.get('steer', 0):6.2f} "
            f"| Merged steer: {merged.joint_positions.get('steer', 0):6.2f} "
            f"| Reason: {jepa_adj.metadata.get('reason', 'none')}"
        )

    if sim:
        sim.disconnect()
    print(
        f"\nCompleted {tick_count} ticks. "
        "JEPA modulated Reflex after tick 3 (fast obstacle detected)."
    )


if __name__ == "__main__":
    asyncio.run(main())
