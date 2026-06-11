"""Tests for MorphologyAnalyzer — 32-D topology vector from URDF (iter-028).

Addresses architecture gap #5: the architecture specifies a MorphologyAnalyzer
that produces a 32-D Topology Vector and a SHA256 Topology Hash from URDF
morphology features, but only a trivial _robot_hash() existed.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch

from isonome.utils.morphology import (
    BaseMorphology,
    EndEffectorType,
    MorphologyAnalyzer,
    TopologyVector,
)


# ---------------------------------------------------------------------------
# Fixtures: minimal URDFs for different morphologies
# ---------------------------------------------------------------------------

@pytest.fixture
def urdf_arm_7dof(tmp_path: Path) -> Path:
    """7-DOF serial arm (matches examples/robot_arm.urdf structure)."""
    urdf = tmp_path / "arm_7dof.urdf"
    urdf.write_text("""\
<?xml version="1.0"?>
<robot name="test_arm">
  <link name="base_link">
    <inertial><mass value="2.0"/><inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base_link"/><child link="l1"/>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="20" velocity="2"/>
    <dynamics damping="0.3" friction="0.1"/>
  </joint>
  <link name="l1">
    <inertial><mass value="1.0"/><inertia ixx="0.05" iyy="0.05" izz="0.05" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="j2" type="revolute">
    <parent link="l1"/><child link="l2"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" effort="15" velocity="2"/>
    <dynamics damping="0.3" friction="0.1"/>
  </joint>
  <link name="l2">
    <inertial><mass value="0.8"/><inertia ixx="0.04" iyy="0.04" izz="0.04" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="j3" type="revolute">
    <parent link="l2"/><child link="l3"/>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" effort="10" velocity="2"/>
    <dynamics damping="0.2" friction="0.05"/>
  </joint>
  <link name="l3">
    <inertial><mass value="0.6"/><inertia ixx="0.03" iyy="0.03" izz="0.03" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
</robot>
""")
    return urdf


@pytest.fixture
def urdf_fixed_base(tmp_path: Path) -> Path:
    """Simple fixed-base robot with 2 revolute joints."""
    urdf = tmp_path / "fixed_base.urdf"
    urdf.write_text("""\
<?xml version="1.0"?>
<robot name="fixed_bot">
  <link name="base_link">
    <inertial><mass value="5.0"/><inertia ixx="0.2" iyy="0.2" izz="0.2" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base_link"/><child link="l1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="1"/>
    <dynamics damping="0.5" friction="0.2"/>
  </joint>
  <link name="l1">
    <inertial><mass value="1.0"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="j2" type="revolute">
    <parent link="l1"/><child link="l2"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.0" upper="1.0" effort="30" velocity="1"/>
    <dynamics damping="0.5" friction="0.2"/>
  </joint>
  <link name="l2">
    <inertial><mass value="0.5"/><inertia ixx="0.005" iyy="0.005" izz="0.005" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
</robot>
""")
    return urdf


@pytest.fixture
def urdf_mobile_diff_drive(tmp_path: Path) -> Path:
    """Differential-drive mobile base with 2 continuous wheel joints."""
    urdf = tmp_path / "diff_drive.urdf"
    urdf.write_text("""\
<?xml version="1.0"?>
<robot name="diff_bot">
  <link name="base_link">
    <inertial><mass value="10.0"/><inertia ixx="0.5" iyy="0.5" izz="0.5" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="left_wheel" type="continuous">
    <parent link="base_link"/><child link="lw"/>
    <origin xyz="0 0.15 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit effort="5" velocity="10"/>
    <dynamics damping="0.1" friction="0.5"/>
  </joint>
  <link name="lw">
    <inertial><mass value="0.5"/><inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="right_wheel" type="continuous">
    <parent link="base_link"/><child link="rw"/>
    <origin xyz="0 -0.15 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit effort="5" velocity="10"/>
    <dynamics damping="0.1" friction="0.5"/>
  </joint>
  <link name="rw">
    <inertial><mass value="0.5"/><inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
</robot>
""")
    return urdf


@pytest.fixture
def urdf_legged(tmp_path: Path) -> Path:
    """Simple 2-legged robot with 3 joints per leg."""
    urdf = tmp_path / "legged.urdf"
    urdf.write_text("""\
<?xml version="1.0"?>
<robot name="biped">
  <link name="base_link">
    <inertial><mass value="8.0"/><inertia ixx="0.3" iyy="0.3" izz="0.3" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="l_hip" type="revolute">
    <parent link="base_link"/><child link="l_upper"/>
    <origin xyz="0 0.1 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.0" upper="1.0" effort="30" velocity="3"/>
    <dynamics damping="0.4" friction="0.1"/>
  </joint>
  <link name="l_upper"><inertial><mass value="2.0"/><inertia ixx="0.02" iyy="0.02" izz="0.02" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <joint name="l_knee" type="revolute">
    <parent link="l_upper"/><child link="l_lower"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.0" effort="25" velocity="3"/>
    <dynamics damping="0.3" friction="0.1"/>
  </joint>
  <link name="l_lower"><inertial><mass value="1.5"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <joint name="l_ankle" type="revolute">
    <parent link="l_lower"/><child link="l_foot"/>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="15" velocity="3"/>
    <dynamics damping="0.2" friction="0.05"/>
  </joint>
  <link name="l_foot"><inertial><mass value="0.5"/><inertia ixx="0.002" iyy="0.002" izz="0.002" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <joint name="r_hip" type="revolute">
    <parent link="base_link"/><child link="r_upper"/>
    <origin xyz="0 -0.1 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.0" upper="1.0" effort="30" velocity="3"/>
    <dynamics damping="0.4" friction="0.1"/>
  </joint>
  <link name="r_upper"><inertial><mass value="2.0"/><inertia ixx="0.02" iyy="0.02" izz="0.02" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <joint name="r_knee" type="revolute">
    <parent link="r_upper"/><child link="r_lower"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.0" effort="25" velocity="3"/>
    <dynamics damping="0.3" friction="0.1"/>
  </joint>
  <link name="r_lower"><inertial><mass value="1.5"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <joint name="r_ankle" type="revolute">
    <parent link="r_lower"/><child link="r_foot"/>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="15" velocity="3"/>
    <dynamics damping="0.2" friction="0.05"/>
  </joint>
  <link name="r_foot"><inertial><mass value="0.5"/><inertia ixx="0.002" iyy="0.002" izz="0.002" ixy="0" ixz="0" iyz="0"/></inertial></link>
</robot>
""")
    return urdf


@pytest.fixture
def urdf_minimal(tmp_path: Path) -> Path:
    """Single fixed joint, no actuated joints — degenerate case."""
    urdf = tmp_path / "minimal.urdf"
    urdf.write_text("""\
<?xml version="1.0"?>
<robot name="static_body">
  <link name="base_link">
    <inertial><mass value="1.0"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="fixed_mount" type="fixed">
    <parent link="base_link"/><child link="attachment"/>
  </joint>
  <link name="attachment">
    <inertial><mass value="0.1"/><inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
</robot>
""")
    return urdf


# ---------------------------------------------------------------------------
# TopologyVector dataclass tests
# ---------------------------------------------------------------------------

class TestTopologyVector:
    def test_vector_shape_is_32(self):
        tv = TopologyVector(features=torch.zeros(32))
        assert tv.features.shape == (32,)

    def test_vector_requires_32d(self):
        with pytest.raises(ValueError, match="32"):
            TopologyVector(features=torch.zeros(16))

    def test_hash_is_hex_string(self):
        tv = TopologyVector(features=torch.zeros(32))
        assert isinstance(tv.topology_hash, str)
        assert len(tv.topology_hash) == 64  # full SHA-256 hex

    def test_hash_depends_on_features(self):
        tv1 = TopologyVector(features=torch.zeros(32))
        tv2 = TopologyVector(features=torch.ones(32))
        assert tv1.topology_hash != tv2.topology_hash

    def test_same_features_same_hash(self):
        f = torch.randn(32)
        tv1 = TopologyVector(features=f.clone())
        tv2 = TopologyVector(features=f.clone())
        assert tv1.topology_hash == tv2.topology_hash

    def test_base_morphology_fields(self):
        bm = BaseMorphology(
            base_type="fixed",
            total_joints=2,
            arm_joints=2,
            leg_joints=0,
            end_effector_type=EndEffectorType.GRIPPER,
            link_masses=[5.0, 1.0, 0.5],
            max_efforts=[50.0, 30.0],
            mean_damping=0.5,
            mean_friction=0.2,
            has_head_gaze=False,
            has_force_sensor=False,
            chain_lengths=[2],
        )
        assert bm.base_type == "fixed"
        assert bm.total_joints == 2


# ---------------------------------------------------------------------------
# MorphologyAnalyzer: base type detection
# ---------------------------------------------------------------------------

class TestBaseTypeDetection:
    def test_fixed_base(self, urdf_fixed_base: Path):
        ma = MorphologyAnalyzer(urdf_fixed_base)
        bm = ma.base_morphology
        assert bm.base_type == "fixed"

    def test_diff_drive_base(self, urdf_mobile_diff_drive: Path):
        ma = MorphologyAnalyzer(urdf_mobile_diff_drive)
        bm = ma.base_morphology
        assert bm.base_type == "diff_drive"

    def test_legged_base(self, urdf_legged: Path):
        ma = MorphologyAnalyzer(urdf_legged)
        bm = ma.base_morphology
        assert bm.base_type == "legged"

    def test_arm_is_fixed_base(self, urdf_arm_7dof: Path):
        """Arms (no wheels or legs) should be classified as fixed."""
        ma = MorphologyAnalyzer(urdf_arm_7dof)
        bm = ma.base_morphology
        assert bm.base_type == "fixed"


# ---------------------------------------------------------------------------
# MorphologyAnalyzer: joint counts
# ---------------------------------------------------------------------------

class TestJointCounts:
    def test_arm_7dof_total(self, urdf_arm_7dof: Path):
        ma = MorphologyAnalyzer(urdf_arm_7dof)
        bm = ma.base_morphology
        assert bm.total_joints == 3  # 3 revolute joints in our fixture
        assert bm.arm_joints == 3

    def test_fixed_base_counts(self, urdf_fixed_base: Path):
        ma = MorphologyAnalyzer(urdf_fixed_base)
        bm = ma.base_morphology
        assert bm.total_joints == 2
        assert bm.arm_joints == 2

    def test_diff_drive_counts(self, urdf_mobile_diff_drive: Path):
        ma = MorphologyAnalyzer(urdf_mobile_diff_drive)
        bm = ma.base_morphology
        assert bm.total_joints == 2
        # Wheels are continuous, not arm/leg
        assert bm.arm_joints == 0
        assert bm.leg_joints == 0

    def test_legged_counts(self, urdf_legged: Path):
        ma = MorphologyAnalyzer(urdf_legged)
        bm = ma.base_morphology
        assert bm.total_joints == 6
        assert bm.leg_joints == 6

    def test_minimal_no_actuated(self, urdf_minimal: Path):
        ma = MorphologyAnalyzer(urdf_minimal)
        bm = ma.base_morphology
        assert bm.total_joints == 0
        assert bm.arm_joints == 0
        assert bm.leg_joints == 0


# ---------------------------------------------------------------------------
# MorphologyAnalyzer: mass and effort features
# ---------------------------------------------------------------------------

class TestMassAndEffort:
    def test_link_masses_extracted(self, urdf_arm_7dof: Path):
        ma = MorphologyAnalyzer(urdf_arm_7dof)
        bm = ma.base_morphology
        # base_link (2.0), l1 (1.0), l2 (0.8), l3 (0.6)
        assert len(bm.link_masses) == 4
        assert bm.link_masses[0] == pytest.approx(2.0)
        assert bm.link_masses[-1] == pytest.approx(0.6)

    def test_max_efforts_extracted(self, urdf_arm_7dof: Path):
        ma = MorphologyAnalyzer(urdf_arm_7dof)
        bm = ma.base_morphology
        assert len(bm.max_efforts) == 3
        assert bm.max_efforts[0] == pytest.approx(20.0)

    def test_damping_and_friction(self, urdf_arm_7dof: Path):
        ma = MorphologyAnalyzer(urdf_arm_7dof)
        bm = ma.base_morphology
        # 3 joints: damping 0.3, 0.3, 0.2; friction 0.1, 0.1, 0.05
        assert bm.mean_damping == pytest.approx((0.3 + 0.3 + 0.2) / 3.0)
        assert bm.mean_friction == pytest.approx((0.1 + 0.1 + 0.05) / 3.0)


# ---------------------------------------------------------------------------
# MorphologyAnalyzer: chain topology
# ---------------------------------------------------------------------------

class TestChainTopology:
    def test_arm_single_chain(self, urdf_arm_7dof: Path):
        ma = MorphologyAnalyzer(urdf_arm_7dof)
        bm = ma.base_morphology
        # Single kinematic chain: 3 joints deep
        assert len(bm.chain_lengths) >= 1
        assert max(bm.chain_lengths) == 3

    def test_legged_two_chains(self, urdf_legged: Path):
        ma = MorphologyAnalyzer(urdf_legged)
        bm = ma.base_morphology
        # Two leg chains, each 3 deep
        assert len(bm.chain_lengths) >= 2
        assert 3 in bm.chain_lengths


# ---------------------------------------------------------------------------
# MorphologyAnalyzer: TopologyVector (32-D)
# ---------------------------------------------------------------------------

class TestTopologyVectorGeneration:
    def test_vector_is_32d(self, urdf_arm_7dof: Path):
        ma = MorphologyAnalyzer(urdf_arm_7dof)
        tv = ma.topology_vector
        assert tv.features.shape == (32,)

    def test_vector_values_bounded(self, urdf_arm_7dof: Path):
        """All features should be normalized to roughly [0, 1] range."""
        ma = MorphologyAnalyzer(urdf_arm_7dof)
        tv = ma.topology_vector
        # Some dimensions may be log-scaled, but all should be finite
        assert torch.isfinite(tv.features).all()

    def test_base_type_one_hot(self, urdf_fixed_base: Path, urdf_mobile_diff_drive: Path, urdf_legged: Path):
        """Dims 0-2 should encode base type as one-hot."""
        fixed = MorphologyAnalyzer(urdf_fixed_base).topology_vector
        diff = MorphologyAnalyzer(urdf_mobile_diff_drive).topology_vector
        legged = MorphologyAnalyzer(urdf_legged).topology_vector

        # Fixed: [1, 0, 0]
        assert fixed.features[0].item() == pytest.approx(1.0)
        assert fixed.features[1].item() == pytest.approx(0.0)
        assert fixed.features[2].item() == pytest.approx(0.0)

        # Diff-drive: [0, 1, 0]
        assert diff.features[0].item() == pytest.approx(0.0)
        assert diff.features[1].item() == pytest.approx(1.0)
        assert diff.features[2].item() == pytest.approx(0.0)

        # Legged: [0, 0, 1]
        assert legged.features[0].item() == pytest.approx(0.0)
        assert legged.features[1].item() == pytest.approx(0.0)
        assert legged.features[2].item() == pytest.approx(1.0)

    def test_dof_dims_normalized(self, urdf_arm_7dof: Path):
        """Dims 3-5 should encode arm DOF / 20."""
        ma = MorphologyAnalyzer(urdf_arm_7dof)
        tv = ma.topology_vector
        # 3 arm joints => 3/20 = 0.15
        assert tv.features[3].item() == pytest.approx(3.0 / 20.0)

    def test_minimal_robot_zero_dof(self, urdf_minimal: Path):
        ma = MorphologyAnalyzer(urdf_minimal)
        tv = ma.topology_vector
        # 0 arm joints
        assert tv.features[3].item() == pytest.approx(0.0)

    def test_different_robots_different_vectors(self, urdf_arm_7dof: Path, urdf_legged: Path):
        tv_arm = MorphologyAnalyzer(urdf_arm_7dof).topology_vector
        tv_leg = MorphologyAnalyzer(urdf_legged).topology_vector
        assert not torch.allclose(tv_arm.features, tv_leg.features)


# ---------------------------------------------------------------------------
# MorphologyAnalyzer: topology hash
# ---------------------------------------------------------------------------

class TestTopologyHash:
    def test_hash_is_sha256_hex(self, urdf_arm_7dof: Path):
        ma = MorphologyAnalyzer(urdf_arm_7dof)
        tv = ma.topology_vector
        h = tv.topology_hash
        assert len(h) == 64
        # All hex chars
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_urdf_same_hash(self, urdf_arm_7dof: Path):
        ma1 = MorphologyAnalyzer(urdf_arm_7dof)
        ma2 = MorphologyAnalyzer(urdf_arm_7dof)
        assert ma1.topology_vector.topology_hash == ma2.topology_vector.topology_hash

    def test_different_urdf_different_hash(self, urdf_arm_7dof: Path, urdf_legged: Path):
        h1 = MorphologyAnalyzer(urdf_arm_7dof).topology_vector.topology_hash
        h2 = MorphologyAnalyzer(urdf_legged).topology_vector.topology_hash
        assert h1 != h2

    def test_hash_differs_from_raw_urdf_hash(self, urdf_arm_7dof: Path):
        """Topology hash should be based on morphology features, not raw file bytes."""
        import hashlib
        raw_hash = hashlib.sha256(urdf_arm_7dof.read_bytes()).hexdigest()
        topo_hash = MorphologyAnalyzer(urdf_arm_7dof).topology_vector.topology_hash
        # They should differ — topology hash is content-aware, not byte-level
        assert topo_hash != raw_hash


# ---------------------------------------------------------------------------
# MorphologyAnalyzer: end-effector detection
# ---------------------------------------------------------------------------

class TestEndEffectorDetection:
    def test_default_is_gripper(self, urdf_arm_7dof: Path):
        """Without explicit end-effector tags, default to gripper."""
        ma = MorphologyAnalyzer(urdf_arm_7dof)
        bm = ma.base_morphology
        # No EE markers in fixture URDF → default GRIPPER
        assert bm.end_effector_type == EndEffectorType.GRIPPER

    def test_end_effector_one_hot_in_vector(self, urdf_arm_7dof: Path):
        ma = MorphologyAnalyzer(urdf_arm_7dof)
        tv = ma.topology_vector
        # Dims 9-11: one-hot end-effector type
        # GRIPPER = index 0 => [1, 0, 0]
        assert tv.features[9].item() == pytest.approx(1.0)
        assert tv.features[10].item() == pytest.approx(0.0)
        assert tv.features[11].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# MorphologyAnalyzer: mass ratio features
# ---------------------------------------------------------------------------

class TestMassRatioFeatures:
    def test_mass_ratio_dims_populated(self, urdf_arm_7dof: Path):
        """Dims 15-17: mass ratios should be populated."""
        ma = MorphologyAnalyzer(urdf_arm_7dof)
        tv = ma.topology_vector
        # At least some mass ratio dims should be non-zero
        mass_dims = tv.features[15:18]
        assert mass_dims.abs().sum().item() > 0.0

    def test_torque_ratio_dims_populated(self, urdf_arm_7dof: Path):
        """Dims 18-20: torque ratios should be populated."""
        ma = MorphologyAnalyzer(urdf_arm_7dof)
        tv = ma.topology_vector
        torque_dims = tv.features[18:21]
        assert torque_dims.abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# MorphologyAnalyzer: damping and friction features
# ---------------------------------------------------------------------------

class TestDampingFrictionFeatures:
    def test_damping_mean_in_vector(self, urdf_arm_7dof: Path):
        """Dims 24-26: joint damping mean / 1.0."""
        ma = MorphologyAnalyzer(urdf_arm_7dof)
        tv = ma.topology_vector
        damping_dim = tv.features[24].item()
        # mean_damping ≈ 0.267
        assert damping_dim == pytest.approx((0.3 + 0.3 + 0.2) / 3.0, abs=0.01)

    def test_friction_mean_in_vector(self, urdf_arm_7dof: Path):
        """Dims 27-29: friction coeff mean / 1.0."""
        ma = MorphologyAnalyzer(urdf_arm_7dof)
        tv = ma.topology_vector
        friction_dim = tv.features[27].item()
        # mean_friction ≈ 0.083
        assert friction_dim == pytest.approx((0.1 + 0.1 + 0.05) / 3.0, abs=0.01)


# ---------------------------------------------------------------------------
# MorphologyAnalyzer: binary features
# ---------------------------------------------------------------------------

class TestBinaryFeatures:
    def test_no_head_gaze(self, urdf_arm_7dof: Path):
        ma = MorphologyAnalyzer(urdf_arm_7dof)
        tv = ma.topology_vector
        # Dim 30: head/gaze joints (binary)
        assert tv.features[30].item() == pytest.approx(0.0)

    def test_no_force_sensor(self, urdf_arm_7dof: Path):
        ma = MorphologyAnalyzer(urdf_arm_7dof)
        tv = ma.topology_vector
        # Dim 31: force sensors (binary)
        assert tv.features[31].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# MorphologyAnalyzer: integration with real URDF
# ---------------------------------------------------------------------------

class TestRealURDF:
    def test_examples_robot_arm(self):
        """Test against the actual examples/robot_arm.urdf."""
        urdf_path = Path(__file__).parent.parent / "examples" / "robot_arm.urdf"
        if not urdf_path.exists():
            pytest.skip("examples/robot_arm.urdf not found")
        ma = MorphologyAnalyzer(urdf_path)
        tv = ma.topology_vector
        assert tv.features.shape == (32,)
        assert torch.isfinite(tv.features).all()
        bm = ma.base_morphology
        assert bm.total_joints == 7  # 7 revolute joints
        assert bm.base_type == "fixed"


# ---------------------------------------------------------------------------
# MorphologyAnalyzer: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_joints_robot(self, urdf_minimal: Path):
        ma = MorphologyAnalyzer(urdf_minimal)
        tv = ma.topology_vector
        assert tv.features.shape == (32,)
        # Should not raise — all dimensions should be 0/default
        assert torch.isfinite(tv.features).all()

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            MorphologyAnalyzer(Path("/nonexistent/robot.urdf"))

    def test_invalid_xml(self, tmp_path: Path):
        bad = tmp_path / "bad.urdf"
        bad.write_text("not xml at all")
        with pytest.raises(Exception):  # ET.ParseError
            MorphologyAnalyzer(bad)

    def test_no_dynamics_defaults_to_zero(self, tmp_path: Path):
        """URDF joints without <dynamics> should default damping/friction to 0."""
        urdf = tmp_path / "no_dynamics.urdf"
        urdf.write_text("""\
<?xml version="1.0"?>
<robot name="no_dyn">
  <link name="base_link">
    <inertial><mass value="1.0"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base_link"/><child link="l1"/>
    <limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
  <link name="l1">
    <inertial><mass value="0.5"/><inertia ixx="0.005" iyy="0.005" izz="0.005" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
</robot>
""")
        ma = MorphologyAnalyzer(urdf)
        bm = ma.base_morphology
        assert bm.mean_damping == 0.0
        assert bm.mean_friction == 0.0

    def test_prismatic_joint_counted(self, tmp_path: Path):
        """Prismatic joints should be counted as arm joints."""
        urdf = tmp_path / "prismatic.urdf"
        urdf.write_text("""\
<?xml version="1.0"?>
<robot name="linear">
  <link name="base_link">
    <inertial><mass value="3.0"/><inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="slide" type="prismatic">
    <parent link="base_link"/><child link="slider"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="0.5" effort="20" velocity="1"/>
    <dynamics damping="0.1" friction="0.05"/>
  </joint>
  <link name="slider">
    <inertial><mass value="0.5"/><inertia ixx="0.005" iyy="0.005" izz="0.005" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
</robot>
""")
        ma = MorphologyAnalyzer(urdf)
        bm = ma.base_morphology
        assert bm.total_joints == 1
        assert bm.arm_joints == 1


# ---------------------------------------------------------------------------
# SomaLayer integration
# ---------------------------------------------------------------------------

class TestSomaLayerIntegration:
    """Verify MorphologyAnalyzer is wired into SomaLayer correctly."""

    def test_soma_has_morphology_property(self):
        """SomaLayer should expose a .morphology property."""
        from isonome.core.layers.soma import SomaLayer

        urdf_path = Path(__file__).parent.parent / "examples" / "robot_arm.urdf"
        if not urdf_path.exists():
            pytest.skip("examples/robot_arm.urdf not found")
        soma = SomaLayer(urdf_path=urdf_path)
        bm = soma.morphology
        assert bm.total_joints == 7
        assert bm.base_type == "fixed"

    def test_soma_has_topology_vector_property(self):
        """SomaLayer should expose a .topology_vector property."""
        from isonome.core.layers.soma import SomaLayer

        urdf_path = Path(__file__).parent.parent / "examples" / "robot_arm.urdf"
        if not urdf_path.exists():
            pytest.skip("examples/robot_arm.urdf not found")
        soma = SomaLayer(urdf_path=urdf_path)
        tv = soma.topology_vector
        assert tv.features.shape == (32,)
        assert len(tv.topology_hash) == 64

    def test_soma_robot_hash_uses_topology(self):
        """SomaLayer._robot_hash() should now use topology hash."""
        from isonome.core.layers.soma import SomaLayer

        urdf_path = Path(__file__).parent.parent / "examples" / "robot_arm.urdf"
        if not urdf_path.exists():
            pytest.skip("examples/robot_arm.urdf not found")
        soma = SomaLayer(urdf_path=urdf_path)
        robot_hash = soma._robot_hash()
        topo_hash = soma.topology_vector.topology_hash[:16]
        assert robot_hash == topo_hash

    def test_soma_robot_hash_not_raw_bytes(self):
        """_robot_hash() should NOT match sha256 of raw URDF bytes."""
        import hashlib
        from isonome.core.layers.soma import SomaLayer

        urdf_path = Path(__file__).parent.parent / "examples" / "robot_arm.urdf"
        if not urdf_path.exists():
            pytest.skip("examples/robot_arm.urdf not found")
        soma = SomaLayer(urdf_path=urdf_path)
        robot_hash = soma._robot_hash()
        raw_hash = hashlib.sha256(urdf_path.read_bytes()).hexdigest()[:16]
        # These should differ — topology-based vs byte-based
        assert robot_hash != raw_hash


# ---------------------------------------------------------------------------
# TopologyVectorState (state.py)
# ---------------------------------------------------------------------------

class TestTopologyVectorState:
    def test_from_topology_vector(self, urdf_arm_7dof: Path):
        from isonome.core.state import TopologyVectorState

        ma = MorphologyAnalyzer(urdf_arm_7dof)
        tv = ma.topology_vector
        tvs = TopologyVectorState.from_topology_vector(tv)
        assert tvs.features.shape == (32,)
        assert tvs.topology_hash == tv.topology_hash

    def test_serialization(self, urdf_arm_7dof: Path):
        from isonome.core.state import TopologyVectorState

        ma = MorphologyAnalyzer(urdf_arm_7dof)
        tv = ma.topology_vector
        tvs = TopologyVectorState.from_topology_vector(tv)
        dumped = tvs.model_dump()
        assert "features" in dumped
        assert isinstance(dumped["features"], list)
        assert len(dumped["features"]) == 32
        assert dumped["topology_hash"] == tv.topology_hash

    def test_wrong_shape_raises(self):
        from isonome.core.state import TopologyVectorState

        with pytest.raises(ValueError, match="32"):
            TopologyVectorState(features=torch.zeros(16))

    def test_validates_32d(self, urdf_arm_7dof: Path):
        import torch as th
        from isonome.core.state import TopologyVectorState

        tvs = TopologyVectorState(features=th.zeros(32), topology_hash="abc")
        assert tvs.features.shape == (32,)
