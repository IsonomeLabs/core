## Product Requirements Document

**Version:** 0.1  
**Date:** 2026-06-01  
**Status:** Pre-Alpha  
**Author:** Isonome Team

---

## 1. Overview

Isonome is an autonomous compiler for embodied AI. It converts natural language task descriptions and robot URDFs into production-ready, sim-certified multi-agent policies without hand-tuned robotics engineering.

Current robot foundation models (Skild Brain, pi0, GR00T) are monolithic: one network controls all joints. Isonome treats robot bodies as composable kinematic chains and foundation models as replaceable inference engines. The system decomposes a user intent into independently optimizable agent tracks, coordinates them via explicit rules, calibrates them in parallel physics simulation, and caches the result so repeat deployments are instantaneous.

**Core Thesis:** Simulation is not a testing tool. It is the production calibration layer.

---

## 2. Goals

1. Accept a natural language task + URDF and output a deployable, calibrated multi-agent policy package in <4 hours for novel morphologies and <30 seconds for cached morphologies.
2. Decompose arbitrary morphologies into kinematic agent partitions automatically without human joint assignment.
3. Coordinate parallel VLA (Vision-Language-Action) agents via explicit, debuggable event/guard rules rather than learned implicit synchronization.
4. Cache calibration results keyed by morphology topology so similar bodies reuse prior work.
5. Guarantee deterministic, real-time execution on edge hardware with hard safety reflexes.

---

## 3. Non-Goals

- Train foundation models. Isonome consumes frozen open-source VLAs (pi0, OpenVLA, etc.) as inference engines only.
- Build low-level motor drivers. Output is ROS2 nodes or standalone torchscript that interfaces with existing hardware stacks.
- Run real-time LLM inference during execution. The LLM generates the Task Manifest once; the runtime executes deterministically.
- Support CPU-only VLA inference. GPU is required for the runtime.
- Handle multi-robot fleet coordination in v0. Single-robot scope only.

---

## 4. Architecture

### 4.1 System Diagram

```
User Prompt + URDF
       |
       v
+------------------------+     +------------------------+
|   LLM Orchestrator     |---->|   Task Manifest (YAML) |
|   (Chamber 1)          |     +------------------------+
+------------------------+                |
                                          v
+------------------------+     +------------------------+
| Morphology Analyzer    |---->|   Agent Partition      |
| (Chamber 2)            |     |   Topology Hash        |
+------------------------+     +------------------------+
                                          |
                                          v
+------------------------+     +------------------------+
| Calibration Cache      |---->| Cache Hit?             |
|                        |     | Yes -> Load Package      |
+------------------------+     | No  -> Run Simulation  |
                               +------------------------+
                                          |
                                          v
+------------------------+     +------------------------+
| Simulation Software    |---->| Certified Policy       |
| (Data Plane)           |     | Package (.zip)         |
+------------------------+     +------------------------+
                                          |
                                          v
+------------------------+     +------------------------+
| VLA Runtime            |---->| Hardware Commands        |
| (Edge Execution)       |     | (ROS2 / Standalone)    |
+------------------------+     +------------------------+
```

### 4.2 Subsystems

| Subsystem           | Responsibility                                                    | Rate                |
| ------------------- | ----------------------------------------------------------------- | ------------------- |
| LLM Orchestrator    | Strategic intent decomposition                                    | Batch (1 Hz replan) |
| Morphology Analyzer | URDF parsing, chain extraction, topology hashing                  | Batch               |
| Calibration Cache   | Exact/near-exact lookup, interpolation                            | <100 ms             |
| Simulation Software | Parallel VLA calibration, composition validation, auto-adjustment | Batch (hours)       |
| Coordination Engine | Event/guard evaluation, action merging, FSM execution             | 200 Hz              |
| Reflex Layer        | Safety overrides, joint limits, emergency stances                 | 1 kHz               |
| VLA Runtime         | Multi-instance inference, sensor routing, shared state            | 50-100 Hz per agent |

---

## 5. Functional Requirements

### 5.1 LLM Orchestrator (Chamber 1: Deliberative)

**FR-1.1:** Accept a natural language task prompt, URDF file path, and optional scene description JSON.

**FR-1.2:** Output a Task Manifest YAML constrained to a strict schema. The LLM must not assign specific joints; it names logical agents only (e.g., "arm", "gripper", "base").

**FR-1.3:** Include strategic constraints (safety rules, object properties, success criteria) in the manifest.

**FR-1.4:** Validate manifest syntax and schema compliance. If invalid, re-prompt the LLM with the validation error. Max 3 retries.

**FR-1.5:** Use few-shot prompting with 5 canonical examples to constrain output format.

### 5.2 Morphology Analyzer (Chamber 2: Tactical)

**FR-2.1:** Parse URDF into a directed kinematic tree and identify end-effectors by collision geometry heuristics.

**FR-2.2:** Trace root-to-end-effector paths and cluster joints into candidate kinematic chains.

**FR-2.3:** Detect shared joints (prefix overlap between chains) and mark them for coordination.

**FR-2.4:** Generate a 32-dimensional topology vector and SHA256 topology hash for cache indexing.

**FR-2.5:** Bind logical agents from the Task Manifest to physical chains. If the LLM names one "arm" and two arm chains exist, auto-instantiate "arm_left" and "arm_right".

**FR-2.6:** Validate that the morphology can physically satisfy the task intent (reachability check via approximate workspace bounding).

### 5.3 Coordination Engine (Chamber 3: Operational)

**FR-3.1:** Convert the manifest coordination block into a deterministic finite state machine (FSM) where states = phases and transitions = events/guards.

**FR-3.2:** Validate the FSM for deadlocks before runtime. Detect cycles in dependency graphs.

**FR-3.3:** At runtime (200 Hz), evaluate guard conditions against shared world state and trigger phase transitions.

**FR-3.4:** Merge partial action vectors from parallel VLA agents into a dense whole-robot command vector.

**FR-3.5:** Support three merge strategies for shared joints: Priority (agent wins), Weighted Average, and Nullspace Projection.

**FR-3.6:** Publish phase transitions and guard evaluations to a debug topic for logging.

### 5.4 Simulation Software (Data Plane)

**FR-4.1:** Spawn isolated Isaac Lab environments for each agent, stripping URDF to agent-assigned joints only. Shared joints are either kinematically held or perturbed as external disturbances.

**FR-4.2:** Run domain randomization per episode: mass scale, friction, damping, surface properties, object geometry, lighting (if vision-based), and camera noise.

**FR-4.3:** Calibrate each agent independently in parallel. Calibration means optimizing an affine action transform `a_calibrated = W * a_vla + b` via CMA-ES or differentiable sim, not retraining the VLA.

**FR-4.4:** Achieve >95% isolated success rate per agent before composition.

**FR-4.5:** Validate composition in full-body simulation with all agents, coordinator, and reflex layer active. Run 1000 episodes with randomized physics.

**FR-4.6:** Log failure modes with precise attribution: agent failure, guard violation, merge conflict, reflex trigger, or physics violation.

**FR-4.7:** Auto-adjust the manifest based on failure patterns (max 5 iterations). Adjustments include guard thresholds, merge strategies, reflex gains, and prompt tuning.

**FR-4.8:** Achieve >99% composition success rate before certification.

**FR-4.9:** Output a certified policy package containing: manifest, per-agent policies, coordinator config, reflex gains, sim metrics, certification video, and deployment launcher.

**FR-4.10:** Support secondary validation in MuJoCo MJX for contact-rich tasks when Isaac Lab reports >10% grasp failures.

### 5.5 Calibration Cache

**FR-5.1:** Cache key = SHA256(topology_hash + task_type + vla_model_version).

**FR-5.2:** Support exact match retrieval (<30 seconds) and near-match interpolation (topology distance < epsilon).

**FR-5.3:** Cache artifacts: manifest, agent configs, coordinator config, reflex gains, and policy package.

**FR-5.4:** Maintain separate public and private cache namespaces. Public cache is open-source contributed; private cache is enterprise-encrypted.

**FR-5.5:** Cache lookup API exposed via SDK and CLI.

### 5.6 VLA Runtime (Edge Execution)

**FR-6.1:** Load a single frozen VLA model once into GPU memory. Spawn N inference contexts (one per agent) with separate prompts, observation masks, and action masks.

**FR-6.2:** Route sensor data to per-agent observation buffers. Maintain ring buffers (size 10) with staleness detection.

**FR-6.3:** Maintain a lock-free Shared World State dictionary (base pose, object poses, phase, agent states) updated by the Coordination Engine at 200 Hz.

**FR-6.4:** Run one inference thread per agent at its specified rate (50-100 Hz). If inference exceeds the tick period, emit the previous action and log a deadline miss.

**FR-6.5:** Support hot-swapping a single agent's policy/config without restarting the runtime.

**FR-6.6:** Graceful degradation: if one agent's VLA crashes, continue execution with safe defaults (hold position) for that agent's joints.

### 5.7 Reflex Layer (Chamber 4: Reactive)

**FR-7.1:** Run at 1 kHz on a dedicated CPU thread with no GPU dependency.

**FR-7.2:** Clamp commands to URDF joint limits.

**FR-7.3:** Enforce velocity limits: if `delta/dt > max_vel`, scale command accordingly.

**FR-7.4:** Emergency stance: if IMU pitch > 15 degrees or COM outside support polygon, freeze non-locomotion agents and execute pre-computed balance recovery.

**FR-7.5:** Collision reflex: if unexpected force/torque spike outside predicted contact window, zero velocity and hold position.

**FR-7.6:** E-stop relay: on physical or remote e-stop signal, initiate software brake and cut motor commands.

**FR-7.7:** Log all overrides with timestamp, trigger condition, and action taken.

---

## 6. Data Models

### 6.1 Task Manifest Schema (YAML)

```yaml
task_id: pick_and_place_001
robot_id: humanoid_v1

intent: "Move the red cup to the shelf without spilling"
priority: safety
constraints:
  - "cup must remain upright"
  - "do not exceed 0.5 m/s in workspace"

agents:
  - id: locomotion
    vla_model: pi0-base
    joints: [hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll]
    prompt: "Maintain upright posture. Walk to coordinates (1.2, 0.5). Stop within 0.1m."
    sensors: [base_imu, base_odom]
    rate: 50

  - id: right_arm
    vla_model: pi0-base
    joints:
      [
        shoulder_pitch,
        shoulder_roll,
        shoulder_yaw,
        elbow,
        wrist_roll,
        wrist_pitch,
      ]
    prompt: "Reach the red cup at pose (1.2, 0.5, 0.8). Keep elbow up."
    sensors: [wrist_camera, arm_encoders]
    rate: 50
    depends_on: [locomotion.arrived]

  - id: gripper
    vla_model: pi0-base
    joints: [finger_l, finger_r]
    prompt: "Close fingers when cup is between them. Maintain 2N force. Do not tilt cup."
    sensors: [finger_force]
    rate: 50
    depends_on: [right_arm.reached]

coordination:
  guards:
    - condition: "base.velocity < 0.05"
      blocks: "right_arm.reached"
      reason: "Do not reach while base is moving"

  events:
    - trigger: "right_arm.reached"
      action: "enable gripper.grasp"
    - trigger: "gripper.grasp_confident"
      action: "enable right_arm.lift"

  merge_strategy:
    shared_joints: [torso_pitch, torso_roll]
    mode: priority

calibration:
  sim_iterations: 10000
  success_rate: 0.99
  parallel_envs: 256
  domain_randomize:
    friction: [0.3, 0.8]
    mass_scale: [0.9, 1.1]
  validate_contact_physics: true
  output_policy: torchscript
```

### 6.2 Topology Vector (32-Dimensional)

| Dim   | Feature                                       | Normalization |
| ----- | --------------------------------------------- | ------------- |
| 0-2   | Base type (fixed/diff-drive/holonomic/legged) | One-hot       |
| 3-5   | Max DOF per arm                               | /20           |
| 6-8   | Max DOF per leg                               | /20           |
| 9-11  | End-effector type                             | One-hot       |
| 12-14 | Link length ratios (arm)                      | Log scale     |
| 15-17 | Mass ratios (arm vs torso)                    | /10           |
| 18-20 | Max torque ratios                             | /100          |
| 21-23 | Workspace volume (approx)                     | Log scale     |
| 24-26 | Joint damping mean                            | /1.0          |
| 27-29 | Friction coefficients mean                    | /1.0          |
| 30    | Presence of head/gaze joints                  | Binary        |
| 31    | Presence of force sensors                     | Binary        |

### 6.3 Runtime Data Structures

**ObservationBuffer**

- `images`: dict[str, np.ndarray]
- `proprioception`: np.ndarray
- `shared_state`: dict
- `timestamp`: float
- `valid`: bool

**PartialAction**

- `agent_id`: str
- `values`: dict[str, float] (joint_name → command)
- `confidence`: float
- `timestamp`: float

**FullAction**

- `joint_commands`: np.ndarray (dense)
- `reflex_override`: bool
- `source_agents`: dict[str, str] (joint → agent_id)

---

## 7. API / SDK

### 7.1 Python SDK

```python
import isonome

# Ingest and decompose
manifest = isonome.ingest(
    task="grab the red cup without spilling",
    urdf_path="robot.urdf",
    scene="kitchen.json"
)

# Calibrate via simulation
package = isonome.calibrate(
    manifest=manifest,
    vla_model="pi0-base",
    sim_backend="isaac_lab",
    use_cache=True
)

# Deploy to hardware
isonome.deploy(
    package=package,
    robot_ip="192.168.1.100",
    protocol="ros2"
)
```

### 7.2 CLI

```bash
# Full calibration pipeline
isonome calibrate \
  --urdf robot.urdf \
  --task "grab the red cup" \
  --vla pi0-base \
  --output policy.zip

# Deploy
isonome deploy policy.zip --robot-ip 192.168.1.100

# Cache operations
isonome cache lookup --urdf robot.urdf --task reach
isonome cache list --topology-hash <hash>
```

### 7.3 Simulation Pipeline API

```python
from isonome.sim import CalibrationPipeline

pipeline = CalibrationPipeline(
    manifest_path="task_manifest.yaml",
    urdf_path="robot.urdf",
    vla_weights="pi0_base.pt",
    sim_backend="isaac_lab",
    gpu_ids=[0, 1, 2, 3]
)

# Step 1: Parallel isolated calibration
agent_policies = pipeline.calibrate_agents(parallel_envs=256, max_hours=4)

# Step 2: Composition validation
report = pipeline.validate_composition(episodes=1000, success_threshold=0.99)

# Step 3: Auto-adjust if needed
if not report.passed:
    report = pipeline.auto_adjust(max_iterations=5)

# Step 4: Export
pipeline.export_package("policy_package.zip")
```

### 7.4 Runtime Launch

```python
from isonome.runtime import RobotRuntime

runtime = RobotRuntime.from_package("policy_package.zip")
runtime.start()
runtime.wait_for_phase("complete")
runtime.stop()
```

### 7.5 ROS2 Integration

```bash
ros2 launch isonome_runtime policy.launch.py package:=policy_package.zip
```

**Topics:**

- `/isonome/agents/{agent_id}/partial_action` (debug)
- `/isonome/coordinator/full_action` (pre-reflex)
- `/isonome/reflex/safe_action` (final hardware command)
- `/isonome/fsm/phase` (current state)

---

## 8. Success Criteria

### 8.1 Framework

- **Decomposition accuracy:** 95% of standard URDFs (UR5, Panda, Unitree Go2, humanoid mockup) partitioned correctly without human correction.
- **Cache hit rate:** >70% after 100 public morphologies contributed.
- **Manifest validation:** LLM generates schema-compliant manifest on first attempt >80% of the time.

### 8.2 Simulation

- **Isolated calibration:** >95% success per agent within 1 hour sim time.
- **Composition validation:** >99% success across 1000 randomized episodes.
- **Auto-adjustment convergence:** >80% of composition failures resolved within 3 iterations.
- **Pipeline throughput:** 3-agent morphology completes in <4 hours on 4x A100.
- **Determinism:** Identical manifest + URDF + seed produces bitwise-identical policy package.

### 8.3 Runtime

- **VLA inference latency:** <10 ms per agent on Jetson AGX Orin.
- **Coordination tick:** <5 ms.
- **Reflex tick:** <1 ms.
- **End-to-end latency:** Sensor to motor command <20 ms.
- **Deadline miss rate:** <0.1%.
- **Hot-swap reload:** <2 seconds.

### 8.4 Sim-to-Real

- **Transfer accuracy:** Real hardware achieves >85% of sim success rate without real-world fine-tuning.
- **Safety:** Zero unreflexed joint limit violations or emergency stops during nominal operation.

---

## 9. Milestones (12-Week Residency)

| Week | Deliverable                                                                   |
| ---- | ----------------------------------------------------------------------------- |
| 1    | URDF parser, Morphology Analyzer, topology vector generator                   |
| 2    | LLM Orchestrator prompt engineering, Task Manifest validator                  |
| 3    | Isaac Lab env spawner, isolated agent stripping                               |
| 4    | Domain randomization config, parallel calibration runner                      |
| 5    | Action bias calibration (CMA-ES / differentiable sim)                         |
| 6    | Sensor router, observation buffers, VLA instance manager                      |
| 7    | Coordination Engine FSM, guard evaluation, action merger (Priority + Average) |
| 8    | Composition validator, failure mode logger, auto-adjustment engine            |
| 9    | Reflex Layer skeleton, joint limit / emergency stance implementation          |
| 10   | Calibration Cache (exact match + lookup API), ROS2 bridge                     |
| 11   | Integration: sim pipeline → policy package → runtime                          |
| 12   | **Validation:** 3 morphologies in sim + 3 morphologies on real hardware       |

---

## 10. Open Questions

1. **Contact Physics:** Isaac Lab/PhysX contact approximations may fail for deformable grippers. Do we mandate MuJoCo MJX for all grasp tasks, or only as fallback?
2. **VLA Licensing:** pi0 is Apache 2.0. If users bring proprietary VLAs (e.g., Skild Brain), how does runtime loading work without exposing weights?
3. **Nullspace Projection:** Requires real-time Jacobian computation. Implement in C++ with Pinocchio, or accept Python overhead at 50 Hz?
4. **Shared Joint Ownership:** If LLM assigns a shared joint to the wrong agent, auto-adjustment must detect and reassign. What is the failure signal?
5. **Edge Compute:** Jetson AGX Orin may only fit 2x pi0 instances. Do we support agent time-sharing (round-robin inference) or require discrete GPUs per agent?
   """
