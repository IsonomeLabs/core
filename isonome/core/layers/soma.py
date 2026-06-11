"""Soma Layer — the body interface.

Adapts canonical VLA actions to a specific robot body through a learned kernel
or a naive initial guess. Provides RAW sensor data upstream.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from isonome.core.layers.base import LayerBase
from isonome.core.ports.body_bridge import BodyBridge
from isonome.core.state import (
    CanonicalActionChunk,
    CorrectedMotorCommand,
    ExecutionResult,
    RawSensorState,
)
from isonome.utils.logging import get_layer_logger
from isonome.utils.morphology import BaseMorphology, MorphologyAnalyzer, TopologyVector


CANONICAL_DIM = 14


class NaiveMapper:
    """Initial guess mapping from canonical action space to robot DOF."""
    
    def __init__(self, urdf_path: Path) -> None:
        self.urdf_path = urdf_path
        self.joint_count = self._parse_urdf_actuated_joints(urdf_path)
        self.action_space = self._infer_action_space(self.joint_count)
        self._logger = get_layer_logger("soma.naive_mapper")
        self._logger.info(
            "naive_mapper_init",
            extra={"joints": self.joint_count, "canonical_dim": CANONICAL_DIM},
        )
        
    def _parse_urdf_actuated_joints(self, urdf_path: Path) -> int:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        count = 0
        for joint in root.findall("joint"):
            jtype = joint.get("type", "fixed")
            if jtype != "fixed":
                count += 1
        return count
        
    def _infer_action_space(self, joint_count: int) -> torch.Tensor:
        """Build initial guess: map canonical 14-DOF to robot's N-DOF."""
        if joint_count < CANONICAL_DIM:
            # Truncate and pad with zeros
            mapping = torch.zeros(CANONICAL_DIM, joint_count)
            for i in range(joint_count):
                mapping[i, i] = 1.0
            return mapping
        elif joint_count > CANONICAL_DIM:
            self._logger.warning(
                "naive_mapper_joint_count_high",
                extra={"joint_count": joint_count, "canonical_dim": CANONICAL_DIM},
            )
            mapping = torch.zeros(CANONICAL_DIM, joint_count)
            for i in range(CANONICAL_DIM):
                mapping[i, i] = 1.0
            return mapping
        else:
            return torch.eye(CANONICAL_DIM)
            
    def map(self, canonical: torch.Tensor) -> torch.Tensor:
        """Apply initial guess mapping.
        
        canonical: tensor of shape [..., CANONICAL_DIM]
        returns: tensor of shape [..., joint_count]
        """
        device = canonical.device
        matrix = self.action_space.to(device)
        return torch.matmul(canonical, matrix)
        
        
class SomaKernel(nn.Module):
    """Learned residual network that corrects naive mapping for a specific body."""
    
    def __init__(self, canonical_dim: int, robot_state_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(canonical_dim + robot_state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, canonical_dim),
        )
        
    def forward(self, canonical_action: torch.Tensor, raw_state: torch.Tensor) -> torch.Tensor:
        x = torch.cat([canonical_action, raw_state], dim=-1)
        return self.net(x)
        
        
class SomaLayer(LayerBase):
    """Body interface layer.
    
    - perceive() returns RAW sensor state. No correction applied.
    - act() executes corrected motor commands.
    - load_kernel() injects a calibrated kernel for this body.
    
    When ``body_bridge`` is provided, all runtime I/O is delegated to the
    bridge.  Otherwise the layer returns zero-filled stub data, which is
    useful for testing kernel logic without a physics backend.
    """
    
    def __init__(
        self,
        urdf_path: Path,
        frequency_hz: float = 100.0,
        canonical_dim: int = CANONICAL_DIM,
        body_bridge: BodyBridge | None = None,
    ) -> None:
        super().__init__(name="soma", frequency_hz=frequency_hz)
        self.urdf_path = Path(urdf_path)
        self.naive_mapper = NaiveMapper(self.urdf_path)
        self._canonical_dim = canonical_dim
        self._kernel: Optional[SomaKernel] = None
        self._body_bridge = body_bridge
        self._morphology_analyzer = MorphologyAnalyzer(self.urdf_path)
        self._logger = get_layer_logger("soma")
        
    @property
    def body_bridge(self) -> BodyBridge | None:
        return self._body_bridge
        
    @property
    def has_calibrated_kernel(self) -> bool:
        return self._kernel is not None
        
    @property
    def kernel(self) -> Optional[SomaKernel]:
        return self._kernel

    @property
    def morphology(self) -> BaseMorphology:
        """Parsed morphology features from the URDF."""
        return self._morphology_analyzer.base_morphology

    @property
    def topology_vector(self) -> TopologyVector:
        """32-D topology vector and SHA-256 topology hash."""
        return self._morphology_analyzer.topology_vector
        
    def load_kernel(self, weights_path: Path) -> None:
        """Load a pre-trained kernel from disk."""
        self._logger.info("soma_loading_kernel", extra={"path": str(weights_path)})
        if not weights_path.exists():
            raise FileNotFoundError(f"Kernel not found: {weights_path}")
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        # Infer dims from saved state if possible, else use defaults
        canonical_dim = state.get("canonical_dim", self._canonical_dim)
        robot_state_dim = state.get("robot_state_dim", self.naive_mapper.joint_count)
        self._kernel = SomaKernel(canonical_dim, robot_state_dim)
        self._kernel.load_state_dict(state["net"])
        self._kernel.eval()
        for param in self._kernel.parameters():
            param.requires_grad = False
        self._logger.info("soma_kernel_loaded")
        
    def apply_kernel(
        self, canonical_chunk: CanonicalActionChunk, raw_state: RawSensorState
    ) -> CorrectedMotorCommand:
        """Apply the learned kernel to correct canonical actions for this body."""
        if self._kernel is None:
            raise RuntimeError("No calibrated kernel loaded")
        actions = canonical_chunk.actions  # [chunk_size, canonical_dim]
        proprio = raw_state.proprioception
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0).expand(actions.shape[0], -1)
        with torch.no_grad():
            base = self.naive_mapper.map(actions)
            residual = self._kernel(actions, proprio)
            corrected = base + self.naive_mapper.map(residual)
        return CorrectedMotorCommand(
            commands=corrected,
            robot_hash=self._robot_hash(),
        )
        
    def _robot_hash(self) -> str:
        """Topology-aware hash based on morphology features, not raw file bytes.

        This replaces the previous sha256(urdf_bytes)[:16] with a hash derived
        from the 32-D topology vector, ensuring the hash captures semantic
        morphology rather than incidental file formatting.
        """
        return self._morphology_analyzer.topology_vector.topology_hash[:16]
        
    async def on_boot(self) -> None:
        self._logger.info(
            "soma_layer_booting",
            extra={"urdf": str(self.urdf_path), "joints": self.naive_mapper.joint_count},
        )
        if self._body_bridge is not None:
            await self._body_bridge.boot()
            bridge_joints = self._body_bridge.joint_count
            urdf_joints = self.naive_mapper.joint_count
            if bridge_joints != urdf_joints:
                self._logger.warning(
                    "soma_bridge_joint_mismatch",
                    extra={
                        "urdf_joints": urdf_joints,
                        "bridge_joints": bridge_joints,
                    },
                )
                
    async def on_tick(self) -> None:
        pass  # tick logic driven externally by agent.py
        
    async def on_shutdown(self) -> None:
        self._logger.info("soma_layer_shutdown")
        if self._body_bridge is not None:
            await self._body_bridge.shutdown()
            
    def perceive(self) -> RawSensorState:
        """Read RAW sensor state from the body.
        
        Returns uncorrected proprioception and camera frames. This is RAW data.
        Never apply post-processing or kernel correction here.
        
        Note: when a ``BodyBridge`` is connected, ``Agent._async_perceive``
        calls the bridge's async ``perceive()`` directly instead of this
        method.  This sync fallback exists for testing and subclasses.
        """
        joint_count = self.naive_mapper.joint_count
        return RawSensorState(
            proprioception=torch.zeros(joint_count),
            camera_frames=[],
        )
        
    def act(self, cmd: CorrectedMotorCommand) -> None:
        """Send corrected motor commands to the physics bridge or hardware.
        
        Note: when a ``BodyBridge`` is connected, ``Agent._async_act``
        calls the bridge's async ``act()`` directly.
        """
        self._logger.debug("soma_act", extra={"shape": list(cmd.commands.shape)})
        
    def observe_result(self) -> ExecutionResult:
        """Observe the result of the last act() for discrepancy analysis.
        
        Note: when a ``BodyBridge`` is connected, ``Agent._async_observe_result``
        calls the bridge's async ``observe_result()`` directly.
        """
        joint_count = self.naive_mapper.joint_count
        return ExecutionResult(
            final_proprioception=torch.zeros(joint_count),
            success=True,
            error_metric=0.0,
        )
        