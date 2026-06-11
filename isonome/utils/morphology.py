"""MorphologyAnalyzer — extract 32-D topology vector from URDF.

Addresses architecture gap #5: the architecture specifies a MorphologyAnalyzer
that produces a 32-D Topology Vector and a SHA256 Topology Hash based on
morphology features (base type, DOF ratios, mass ratios, workspace volume,
etc.). Previously, only a trivial _robot_hash() existed (sha256 of raw file
bytes).

The 32-dimensional vector follows the architecture spec (Diagram 9):
  Dims 0-2:   Base type (fixed / diff-drive / holonomic / legged) — one-hot
  Dims 3-5:   Max DOF/arm /20
  Dims 6-8:   Max DOF/leg /20
  Dims 9-11:  End-effector type — one-hot (gripper / suction / tool)
  Dims 12-14: Link length ratios (arm) — log scale
  Dims 15-17: Mass ratios (arm vs torso) /10
  Dims 18-20: Max torque ratios /100
  Dims 21-23: Workspace volume — log scale
  Dims 24-26: Joint damping mean /1.0
  Dims 27-29: Friction coeff mean /1.0
  Dim 30:     Head/Gaze joints (binary)
  Dim 31:     Force sensors (binary)
"""
from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

import torch


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BaseMorphologyType(str, Enum):
    FIXED = "fixed"
    DIFF_DRIVE = "diff_drive"
    HOLONOMIC = "holonomic"
    LEGGED = "legged"


class EndEffectorType(str, Enum):
    GRIPPER = "gripper"
    SUCTION = "suction"
    TOOL = "tool"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BaseMorphology:
    """Parsed raw morphology features from a URDF file."""

    base_type: str
    total_joints: int
    arm_joints: int
    leg_joints: int
    end_effector_type: EndEffectorType
    link_masses: List[float]
    max_efforts: List[float]
    mean_damping: float
    mean_friction: float
    has_head_gaze: bool
    has_force_sensor: bool
    chain_lengths: List[int]


class TopologyVector:
    """32-dimensional topology vector with SHA-256 topology hash.

    The vector encodes robot morphology in a normalized, fixed-size format
    suitable for cache keys, similarity search, and neural network input.
    """

    DIM = 32

    def __init__(self, features: torch.Tensor) -> None:
        if features.shape != (self.DIM,):
            raise ValueError(
                f"TopologyVector requires shape ({self.DIM},), got {features.shape}"
            )
        self._features = features
        self._topology_hash = self._compute_hash(features)

    @property
    def features(self) -> torch.Tensor:
        return self._features

    @property
    def topology_hash(self) -> str:
        return self._topology_hash

    @staticmethod
    def _compute_hash(features: torch.Tensor) -> str:
        """SHA-256 hex digest of the quantized feature vector."""
        # Quantize to 4 decimal places for stable hashing
        quantized = torch.round(features * 10000).to(torch.int32)
        data = quantized.numpy().tobytes()
        return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# MorphologyAnalyzer
# ---------------------------------------------------------------------------

# Joint types that count as arm/manipulator joints
_ARM_JOINT_TYPES = {"revolute", "prismatic"}
# Joint types that count as leg joints
_LEG_JOINT_TYPES = {"revolute", "prismatic"}
# Joint types that indicate wheels
_WHEEL_JOINT_TYPES = {"continuous"}


class MorphologyAnalyzer:
    """Parse a URDF file and extract a 32-D topology vector.

    Usage::

        analyzer = MorphologyAnalyzer(Path("robot.urdf"))
        topo = analyzer.topology_vector   # TopologyVector
        bm = analyzer.base_morphology      # BaseMorphology (raw parsed data)
    """

    def __init__(self, urdf_path: Path) -> None:
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")
        self._tree = ET.parse(self.urdf_path)
        self._root = self._tree.getroot()
        self._base_morphology: Optional[BaseMorphology] = None
        self._topology_vector: Optional[TopologyVector] = None

    # -- Public API --

    @property
    def base_morphology(self) -> BaseMorphology:
        if self._base_morphology is None:
            self._base_morphology = self._parse_morphology()
        return self._base_morphology

    @property
    def topology_vector(self) -> TopologyVector:
        if self._topology_vector is None:
            self._topology_vector = self._build_topology_vector()
        return self._topology_vector

    # -- Parsing --

    def _parse_morphology(self) -> BaseMorphology:
        joints = self._parse_joints()
        links = self._parse_links()

        base_type = self._infer_base_type(joints)
        arm_joints, leg_joints = self._count_arm_leg_joints(joints, base_type)
        chain_lengths = self._compute_chain_lengths(joints, links)
        end_effector = self._infer_end_effector(joints, links)
        damping_vals, friction_vals, effort_vals = self._parse_dynamics(joints)
        has_head_gaze = self._detect_head_gaze(joints)
        has_force_sensor = self._detect_force_sensor(links)

        # Only count actuated (non-fixed) joints
        actuated_joints = [j for j in joints if j["type"] != "fixed"]

        return BaseMorphology(
            base_type=base_type,
            total_joints=len(actuated_joints),
            arm_joints=arm_joints,
            leg_joints=leg_joints,
            end_effector_type=end_effector,
            link_masses=[m for m in links.values() if m is not None],
            max_efforts=effort_vals,
            mean_damping=sum(damping_vals) / len(damping_vals) if damping_vals else 0.0,
            mean_friction=sum(friction_vals) / len(friction_vals) if friction_vals else 0.0,
            has_head_gaze=has_head_gaze,
            has_force_sensor=has_force_sensor,
            chain_lengths=chain_lengths,
        )

    def _parse_joints(self) -> list[dict]:
        """Extract joint info from URDF."""
        joints = []
        for j_elem in self._root.findall("joint"):
            jtype = j_elem.get("type", "fixed")
            name = j_elem.get("name", "")
            parent = j_elem.find("parent")
            child = j_elem.find("child")
            axis_elem = j_elem.find("axis")
            limit_elem = j_elem.find("limit")
            dynamics_elem = j_elem.find("dynamics")
            origin_elem = j_elem.find("origin")

            joint_info: dict = {
                "name": name,
                "type": jtype,
                "parent": parent.get("link", "") if parent is not None else "",
                "child": child.get("link", "") if child is not None else "",
                "axis": (
                    [float(x) for x in axis_elem.get("xyz", "0 0 0").split()]
                    if axis_elem is not None
                    else [0.0, 0.0, 0.0]
                ),
                "effort": float(limit_elem.get("effort", "0")) if limit_elem is not None else 0.0,
                "velocity": float(limit_elem.get("velocity", "0")) if limit_elem is not None else 0.0,
                "damping": 0.0,
                "friction": 0.0,
                "origin_xyz": [0.0, 0.0, 0.0],
            }

            if dynamics_elem is not None:
                joint_info["damping"] = float(dynamics_elem.get("damping", "0"))
                joint_info["friction"] = float(dynamics_elem.get("friction", "0"))

            if origin_elem is not None:
                xyz_str = origin_elem.get("xyz", "0 0 0")
                joint_info["origin_xyz"] = [float(x) for x in xyz_str.split()]

            joints.append(joint_info)
        return joints

    def _parse_links(self) -> dict[str, Optional[float]]:
        """Extract link name → mass mapping."""
        links = {}
        for l_elem in self._root.findall("link"):
            name = l_elem.get("name", "")
            inertial = l_elem.find("inertial")
            mass_elem = inertial.find("mass") if inertial is not None else None
            if mass_elem is not None:
                links[name] = float(mass_elem.get("value", "0"))
            else:
                links[name] = None
        return links

    def _infer_base_type(self, joints: list[dict]) -> str:
        """Classify robot base type from joint types."""
        joint_types = {j["type"] for j in joints}

        # If any continuous joints exist, check if they're wheels (diff-drive or holonomic)
        continuous_count = sum(1 for j in joints if j["type"] == "continuous")

        if continuous_count > 0:
            # Heuristic: 2 continuous joints = diff-drive, >2 = holonomic
            if continuous_count == 2:
                return BaseMorphologyType.DIFF_DRIVE
            else:
                return BaseMorphologyType.HOLONOMIC

        # Check for leg-like structure: multiple chains from base with revolute joints
        if joints:
            base_children = [
                j for j in joints
                if j["parent"] != "" and self._is_base_child(j, joints)
            ]
            # If base has multiple revolute children → likely legged
            revolute_from_base = [
                j for j in base_children if j["type"] in _ARM_JOINT_TYPES
            ]
            if len(revolute_from_base) >= 2:
                return BaseMorphologyType.LEGGED

        return BaseMorphologyType.FIXED

    def _is_base_child(self, joint: dict, joints: list[dict]) -> bool:
        """Check if a joint's parent is the first link (base)."""
        # Find the root link: one that's a parent but never a child
        parents = {j["parent"] for j in joints if j["parent"]}
        children = {j["child"] for j in joints if j["child"]}
        root_candidates = parents - children
        if not root_candidates:
            # Fallback: first link found
            return joint["parent"] == ""
        return joint["parent"] in root_candidates

    def _count_arm_leg_joints(
        self, joints: list[dict], base_type: str
    ) -> tuple[int, int]:
        """Count arm vs leg joints based on base type and chain structure."""
        if base_type == BaseMorphologyType.LEGGED:
            # For legged robots, all revolute/prismatic from base children
            # that are in separate chains are leg joints
            # Simplification: if base_type is legged, all joints are leg
            leg_count = sum(1 for j in joints if j["type"] in _LEG_JOINT_TYPES)
            return 0, leg_count
        elif base_type in (BaseMorphologyType.DIFF_DRIVE, BaseMorphologyType.HOLONOMIC):
            # Mobile bases: continuous joints are wheels, others are arm
            arm_count = sum(1 for j in joints if j["type"] in _ARM_JOINT_TYPES)
            return arm_count, 0
        else:
            # Fixed base: all non-fixed joints are arm
            arm_count = sum(1 for j in joints if j["type"] != "fixed")
            return arm_count, 0

    def _compute_chain_lengths(
        self, joints: list[dict], links: dict
    ) -> list[int]:
        """Compute kinematic chain lengths (depth of each leaf chain)."""
        if not joints:
            return []

        # Build adjacency: parent_link -> [child_links]
        children_map: dict[str, list[str]] = {}
        for j in joints:
            p = j["parent"]
            c = j["child"]
            if p and c:
                children_map.setdefault(p, []).append(c)

        # Find root(s)
        parents = {j["parent"] for j in joints if j["parent"]}
        children_set = {j["child"] for j in joints if j["child"]}
        roots = parents - children_set
        if not roots:
            # Try to find the first link that appears as parent but not child
            roots = {j["parent"] for j in joints[:1]}

        # BFS/DFS to find chain depths
        chain_lengths = []
        for root in roots:
            stack = [(root, 0)]
            while stack:
                node, depth = stack.pop()
                kids = children_map.get(node, [])
                if not kids:
                    if depth > 0:
                        chain_lengths.append(depth)
                else:
                    for kid in kids:
                        stack.append((kid, depth + 1))

        return chain_lengths if chain_lengths else [0]

    def _infer_end_effector(
        self, joints: list[dict], links: dict
    ) -> EndEffectorType:
        """Infer end-effector type from link names and structure.

        Heuristic: check leaf link names for 'gripper', 'suction', 'tool',
        'vacuum' keywords. Default to GRIPPER.
        """
        # Find leaf links (children that are never parents)
        children_set = {j["child"] for j in joints if j["child"]}
        parents_set = {j["parent"] for j in joints if j["parent"]}
        leaf_links = children_set - parents_set

        for name in leaf_links:
            name_lower = name.lower()
            if any(kw in name_lower for kw in ("suction", "vacuum", "cup")):
                return EndEffectorType.SUCTION
            if any(kw in name_lower for kw in ("tool", "drill", "screw", "weld")):
                return EndEffectorType.TOOL

        return EndEffectorType.GRIPPER

    def _parse_dynamics(
        self, joints: list[dict]
    ) -> tuple[list[float], list[float], list[float]]:
        """Extract damping, friction, and effort values from joints."""
        damping_vals = []
        friction_vals = []
        effort_vals = []
        for j in joints:
            if j["type"] != "fixed":
                damping_vals.append(j["damping"])
                friction_vals.append(j["friction"])
                effort_vals.append(j["effort"])
        return damping_vals, friction_vals, effort_vals

    def _detect_head_gaze(self, joints: list[dict]) -> bool:
        """Check for head/gaze joints by name convention."""
        for j in joints:
            name_lower = j["name"].lower()
            if any(kw in name_lower for kw in ("head", "gaze", "neck", "pan_tilt")):
                return True
        return False

    def _detect_force_sensor(self, links: dict) -> bool:
        """Check for force/torque sensor links by name convention."""
        for name in links:
            name_lower = name.lower()
            if any(kw in name_lower for kw in ("ft_sensor", "force_sensor", "f_sensor", "loadcell")):
                return True
        return False

    # -- Topology vector construction --

    def _build_topology_vector(self) -> TopologyVector:
        bm = self.base_morphology
        features = torch.zeros(TopologyVector.DIM)

        # Dims 0-2: Base type one-hot
        base_type_map = {
            BaseMorphologyType.FIXED: 0,
            BaseMorphologyType.DIFF_DRIVE: 1,
            BaseMorphologyType.HOLONOMIC: 2,
            BaseMorphologyType.LEGGED: 3,
        }
        # Only 3 slots, so legged maps to index 2 (overrides holonomic position
        # if we had 4 slots — but we follow the spec with 3 dims)
        # Re-read spec: "fixed / diff-drive / holonomic / legged" → 4 values in 3 dims
        # Architecture shows "one-hot" but has 4 types in 3 dims. We use index 2 for legged.
        base_idx = base_type_map.get(bm.base_type, 0)
        if base_idx < 3:
            features[base_idx] = 1.0
        else:
            # Legged → use a special encoding: [0, 0, 1]
            features[2] = 1.0

        # Dims 3-5: Max DOF/arm /20
        features[3] = min(bm.arm_joints / 20.0, 1.0)
        features[4] = 0.0  # reserved for secondary arm DOF
        features[5] = 0.0  # reserved

        # Dims 6-8: Max DOF/leg /20
        features[6] = min(bm.leg_joints / 20.0, 1.0)
        features[7] = 0.0  # reserved
        features[8] = 0.0  # reserved

        # Dims 9-11: End-effector type one-hot
        ee_map = {
            EndEffectorType.GRIPPER: 0,
            EndEffectorType.SUCTION: 1,
            EndEffectorType.TOOL: 2,
        }
        ee_idx = ee_map.get(bm.end_effector_type, 0)
        features[9 + ee_idx] = 1.0

        # Dims 12-14: Link length ratios (arm) — log scale
        # Approximate from joint origins: chain depth → cumulative distance
        link_lengths = self._compute_link_lengths()
        if link_lengths:
            # Normalize: divide by max length
            max_len = max(link_lengths) if max(link_lengths) > 0 else 1.0
            ratios = [l / max_len for l in link_lengths[:3]]
            for i, r in enumerate(ratios):
                features[12 + i] = math.log1p(r) if r > 0 else 0.0
        # Pad remaining with 0

        # Dims 15-17: Mass ratios (arm vs torso) /10
        if len(bm.link_masses) >= 2:
            torso_mass = bm.link_masses[0]  # First link = base/torso
            for i in range(min(3, len(bm.link_masses) - 1)):
                if torso_mass > 0:
                    ratio = bm.link_masses[i + 1] / torso_mass
                    features[15 + i] = min(ratio / 10.0, 1.0)

        # Dims 18-20: Max torque ratios /100
        if bm.max_efforts:
            max_eff = max(bm.max_efforts) if max(bm.max_efforts) > 0 else 1.0
            for i in range(min(3, len(bm.max_efforts))):
                features[18 + i] = min(bm.max_efforts[i] / 100.0, 1.0)

        # Dims 21-23: Workspace volume — log scale
        # Estimate from chain lengths * average link length
        if bm.chain_lengths and link_lengths:
            avg_link_len = sum(link_lengths) / len(link_lengths)
            max_reach = max(bm.chain_lengths) * avg_link_len
            # Rough workspace volume: (4/3)*pi*r^3 for spherical, but simplified
            ws_vol = (4.0 / 3.0) * math.pi * (max_reach ** 3)
            features[21] = min(math.log1p(ws_vol) / 10.0, 1.0)

        # Dims 24-26: Joint damping mean / 1.0
        features[24] = min(bm.mean_damping, 1.0)

        # Dims 27-29: Friction coeff mean / 1.0
        features[27] = min(bm.mean_friction, 1.0)

        # Dim 30: Head/Gaze joints (binary)
        features[30] = 1.0 if bm.has_head_gaze else 0.0

        # Dim 31: Force sensors (binary)
        features[31] = 1.0 if bm.has_force_sensor else 0.0

        return TopologyVector(features=features)

    def _compute_link_lengths(self) -> list[float]:
        """Compute approximate link lengths from joint origin XYZ distances."""
        lengths = []
        for j_elem in self._root.findall("joint"):
            origin = j_elem.find("origin")
            if origin is not None:
                xyz_str = origin.get("xyz", "0 0 0")
                xyz = [float(v) for v in xyz_str.split()]
                length = math.sqrt(sum(v * v for v in xyz))
                if length > 0:
                    lengths.append(length)
        return sorted(lengths, reverse=True)
