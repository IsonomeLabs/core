# Iteration 028: Morphology Analyzer & 32-D Topology Vector

**Date**: 2026-06-10
**Status**: Complete
**Test Coverage**: 52 new tests
**Backward Compatibility**: Full — `SomaLayer._robot_hash()` return type unchanged (16-char hex string)

## Problem

The architecture (Diagram 9) specifies a **Morphology Analyzer** that produces a **32-D Topology Vector** and a **SHA256 Topology Hash** based on morphology features (base type, DOF ratios, mass ratios, workspace volume, etc.). However, only a trivial `_robot_hash()` existed in `SomaLayer` — it was simply `sha256(urdf_file_bytes)[:16]`, which:

1. **Captures file formatting, not morphology**: Two functionally identical URDFs with different whitespace or attribute ordering produce different hashes.
2. **No topology vector**: There is no fixed-size numerical representation of robot morphology for similarity search, cache key composition, or neural network input.
3. **No morphology features**: Base type, joint counts, mass ratios, damping, friction, and end-effector type are not extracted anywhere.
4. **Gap #5 from architecture-gaps.md**: The only "hash" was `SomaLayer._robot_hash()`, which was `sha256(urdf_file_bytes)[:16]`. There was no morphology analyzer, no topology vector, and the cache key logic shown in Diagram 1 didn't exist.

## Solution

### MorphologyAnalyzer (`isonome/utils/morphology.py`, +300 lines)

A standalone class that parses a URDF file and extracts both raw morphology features and a 32-D topology vector.

**BaseMorphology** (frozen dataclass):
- `base_type`: fixed / diff_drive / holonomic / legged (inferred from joint types)
- `total_joints`, `arm_joints`, `leg_joints`: actuated joint counts
- `end_effector_type`: gripper / suction / tool (inferred from leaf link names)
- `link_masses`, `max_efforts`: raw numerical features
- `mean_damping`, `mean_friction`: averaged dynamics properties
- `has_head_gaze`, `has_force_sensor`: binary features from naming conventions
- `chain_lengths`: kinematic chain depth for each leaf chain

**TopologyVector** (class):
- 32-D feature tensor following the architecture spec (Diagram 9):
  - Dims 0-2: Base type one-hot (fixed / diff-drive / legged)
  - Dims 3-5: Max DOF/arm /20
  - Dims 6-8: Max DOF/leg /20
  - Dims 9-11: End-effector type one-hot (gripper / suction / tool)
  - Dims 12-14: Link length ratios (log scale)
  - Dims 15-17: Mass ratios (arm vs torso) /10
  - Dims 18-20: Max torque ratios /100
  - Dims 21-23: Workspace volume (log scale)
  - Dims 24-26: Joint damping mean /1.0
  - Dims 27-29: Friction coeff mean /1.0
  - Dim 30: Head/Gaze joints (binary)
  - Dim 31: Force sensors (binary)
- SHA-256 topology hash: `sha256(quantized_features)` — content-aware, not byte-level

**Base type inference heuristics**:
- `continuous` joints → diff-drive (2) or holonomic (>2)
- Multiple revolute chains from root → legged
- Otherwise → fixed

**End-effector inference heuristics**:
- Leaf link names containing "suction"/"vacuum" → SUCTION
- Leaf link names containing "tool"/"drill"/"weld" → TOOL
- Default → GRIPPER

### SomaLayer Integration

- New `morphology` property: returns `BaseMorphology` from the internal `MorphologyAnalyzer`
- New `topology_vector` property: returns `TopologyVector`
- `_robot_hash()` now returns `topology_vector.topology_hash[:16]` instead of `sha256(urdf_bytes)[:16]`
  - Return type unchanged (16-char hex string) — backward compatible

### TopologyVectorState (`isonome/core/state.py`)

- New Pydantic model `TopologyVectorState` with `features: torch.Tensor[32]` and `topology_hash: str`
- Factory method `from_topology_vector(tv)` for easy conversion
- Full serialization support via `model_dump()`

## Test Coverage

52 new tests across 12 test classes:

| Class | Tests | What it covers |
|---|---|---|
| `TestTopologyVector` | 5 | Shape validation, hash computation, determinism |
| `TestBaseTypeDetection` | 4 | Fixed, diff-drive, legged, arm-as-fixed |
| `TestJointCounts` | 5 | Total, arm, leg counts for various URDFs |
| `TestMassAndEffort` | 3 | Mass extraction, effort extraction, damping/friction |
| `TestChainTopology` | 2 | Kinematic chain length computation |
| `TestTopologyVectorGeneration` | 6 | 32-D shape, boundedness, one-hot encoding, DOF normalization |
| `TestTopologyHash` | 4 | SHA-256 format, determinism, difference from raw bytes hash |
| `TestEndEffectorDetection` | 2 | Default gripper, one-hot in vector |
| `TestMassRatioFeatures` | 2 | Mass and torque ratio dims populated |
| `TestDampingFrictionFeatures` | 2 | Damping and friction mean in vector |
| `TestBinaryFeatures` | 2 | Head/gaze and force sensor dims |
| `TestRealURDF` | 1 | Against examples/robot_arm.urdf |
| `TestEdgeCases` | 5 | Zero joints, file not found, invalid XML, missing dynamics, prismatic |
| `TestSomaLayerIntegration` | 4 | morphology property, topology_vector property, _robot_hash topology-based |
| `TestTopologyVectorState` | 4 | from_topology_vector, serialization, shape validation |

## Files Changed

- `isonome/utils/morphology.py` — **NEW**: MorphologyAnalyzer, TopologyVector, BaseMorphology
- `isonome/utils/__init__.py` — Export MorphologyAnalyzer, TopologyVector
- `isonome/core/layers/soma.py` — Wire MorphologyAnalyzer, add morphology/topology_vector properties, update _robot_hash()
- `isonome/core/state.py` — Add TopologyVectorState model
- `tests/test_morphology_analyzer.py` — **NEW**: 52 tests

## Architectural Impact

This iteration closes **architecture gap #5** (Topology / Morphology Analyzer). The 32-D topology vector enables:

1. **Semantic cache keys**: `SHA256(topology + task_type + vla_version)` as specified in Diagram 1
2. **Morphology-aware similarity**: Compare robots by their feature vectors
3. **Neural network conditioning**: Topology vector as input to SomaKernel or future FSM
4. **Deterministic hashing**: Same morphology → same hash, regardless of URDF formatting
