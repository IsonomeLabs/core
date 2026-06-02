from __future__ import annotations
import logging
from typing import Any
from isonome.core.state import SensorState, MotorCommand, JointReading, IMUReading
from isonome.core.config import SimConfig


class SimBridge:
    """PyBullet simulation bridge.

    Loads URDF, steps physics, streams SensorState.
    """

    def __init__(self, config: SimConfig | None = None) -> None:
        self._config = config or SimConfig()
        self._physics_client: Any = None
        self._robot_id: int | None = None
        self._logger = logging.getLogger("isonome.bridge.sim")
        self._joint_names: list[str] = []
        self._num_joints: int = 0

    def connect(self) -> None:
        """Connect to the physics server (GUI or headless)."""
        import pybullet
        import pybullet_data

        if self._config.gui:
            self._physics_client = pybullet.connect(pybullet.GUI)
        else:
            self._physics_client = pybullet.connect(pybullet.DIRECT)

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.setGravity(*self._config.gravity)
        pybullet.setTimeStep(self._config.timestep)

        # Ground plane
        pybullet.loadURDF("plane.urdf")
        self._logger.info("sim_connected", extra={"gui": self._config.gui})

    def load_urdf(
        self,
        urdf_path: str,
        start_pos: tuple[float, ...] = (0, 0, 1),
        start_orn: tuple[float, ...] = (0, 0, 0, 1),
    ) -> int:
        """Load a robot URDF into the simulation."""
        import pybullet

        self._robot_id = pybullet.loadURDF(urdf_path, start_pos, start_orn)
        self._num_joints = pybullet.getNumJoints(self._robot_id)
        self._joint_names = []
        for i in range(self._num_joints):
            info = pybullet.getJointInfo(self._robot_id, i)
            self._joint_names.append(info[1].decode("utf-8"))
        self._logger.info(
            "sim_urdf_loaded",
            extra={"path": urdf_path, "joints": self._num_joints},
        )
        return self._robot_id

    def step(self) -> None:
        """Advance the simulation by one timestep."""
        import pybullet

        pybullet.stepSimulation()

    def get_sensor_state(self) -> SensorState:
        """Read the current sensor state from the simulated robot."""
        import pybullet

        if self._robot_id is None:
            return SensorState()

        joints: list[JointReading] = []
        for i in range(self._num_joints):
            state = pybullet.getJointState(self._robot_id, i)
            joints.append(
                JointReading(
                    name=self._joint_names[i] if i < len(self._joint_names) else f"joint_{i}",
                    position=state[0],
                    velocity=state[1],
                )
            )

        pos, orn = pybullet.getBasePositionAndOrientation(self._robot_id)
        linear_vel, angular_vel = pybullet.getBaseVelocity(self._robot_id)

        return SensorState(
            joints=joints,
            imu=IMUReading(
                linear_acceleration=(0.0, 0.0, 0.0),
                angular_velocity=tuple(angular_vel),
                orientation=orn,
            ),
            extras={"base_position": pos, "base_orientation": orn},
        )

    def apply_motor_command(self, cmd: MotorCommand) -> None:
        """Apply a MotorCommand to the simulated robot joints."""
        import pybullet

        if self._robot_id is None:
            return
        for i in range(self._num_joints):
            name = self._joint_names[i] if i < len(self._joint_names) else f"joint_{i}"
            if name in cmd.joint_positions:
                pybullet.setJointMotorControl2(
                    self._robot_id,
                    i,
                    pybullet.POSITION_CONTROL,
                    targetPosition=cmd.joint_positions[name],
                )

    def disconnect(self) -> None:
        """Disconnect from the physics server."""
        import pybullet

        if self._physics_client is not None:
            pybullet.disconnect()
            self._physics_client = None
