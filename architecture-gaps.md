# Isonome Architecture → Code Gap Analysis

> Generated: 2026-06-09  
> Comparing `architecture.md` (PRD v0.1 / Code v0.2) against the current `isonome/core/` and `isonome/sim/` implementation.

---

## 1. Core → Sim Bridge Gap (Biggest Disconnect) ✅ CLOSED

**Architecture says:** `SomaLayer` loads URDF via `SimBridge` (PyBullet) **or** `HardwareBridge`, and `perceive()` / `act()` read from / write to that bridge.

**Reality:** ✅ Implemented via `BodyBridge` port + adapters.
- `BodyBridge` abstract port in `isonome/core/ports/body_bridge.py` with async `perceive()` / `act()` / `observe_result()` lifecycle.
- Concrete adapters in `isonome/bridge/adapters/`: `MockBridgeAdapter`, `PyBulletBridgeAdapter`, `MuJoCoBridgeAdapter`, `HardwareBridgeAdapter`.
- `Agent` constructs the bridge via `build_body_bridge(config)` and delegates runtime I/O to it in `_async_perceive()`, `_async_act()`, `_async_observe_result()`.
- `SomaLayer` accepts an optional `body_bridge` and validates joint counts during boot.

**Note:** `VLAController` in `sim/vla_controller.py` still drives MuJoCo directly for closed-loop VLA demos, but the main `Agent` pipeline is now bridged.

---

## 2. Missing FSM Compiler & Action Merger ✅ CLOSED

**Architecture shows:** Chamber 3 has an **FSM Compiler** (Guards / Events / Merge Strategy) and an **Action Merger** (Priority | Weighted Average | Nullspace).

**Reality:** ✅ Implemented in `isonome/core/coordination/`.
- `FSMCompiler` / `FSMExecutor` — builder + runtime engine with guards, events, entry/exit/during actions.
- `ActionMerger` with three strategies: `PriorityMerger`, `WeightedAverageMerger`, `NullspaceMerger`.
- `Coordinator` — multi-agent composition layer that runs sub-agent ticks, collects `PartialAction`s, merges them into `FullAction`, and executes via a body bridge.
- `SomaLayer._last_command` caching so the coordinator can read each agent's output.
- `CoordinationConfig` added to `AppConfig`.
- 33 tests covering FSM, all three merge strategies, and coordinator integration.

---

## 3. Calibration / Training Pipeline (Diagram 7: "PRD Future State")

**Architecture describes:** A full simulation pipeline with:
- URDF Stripper (per-agent joint subsets)
- Isaac Lab Env Spawner (256 parallel envs)
- Domain Randomization (mass, friction, damping, lighting)
- CMA-ES / Differentiable Sim optimization
- Composition Validation (1000 episodes, >99% success target)
- Auto-Adjustment Engine (max 5 iterations)
- Export of a **Certified Policy Package (.zip)**

**Reality:** None of this exists.
- `PlasticityLayer` is **runtime-only** and explicitly comments: *"The open-source runtime only consumes pre-trained kernels. It does NOT implement training loops, loss functions, or cloud API calls."*
- `isaac_bridge.py` is a **remote WebSocket/MJPEG server** for Isaac Sim viewport streaming — not Isaac Lab, not parallel envs, not calibration.
- There is no CMA-ES, no differentiable sim, no composition validation, no auto-adjustment, and no `.zip` export.

---

## 4. Simulation Backends Mismatch

**Architecture shows:** Isaac Lab as primary, MuJoCo **MJX** as fallback for contact-rich tasks.

**Reality:**
- `isaac_bridge.py` uses `omni.isaac.core` (Isaac Sim), not Isaac Lab.
- `mujoco_bridge.py` uses standard **CPU MuJoCo**, not MJX (the GPU-accelerated JAX version).
- `mock_bridge.py` is a software pendulum with no real physics.

---

## 5. Topology / Morphology Analyzer ✅ CLOSED

**Architecture shows:** Chamber 2 produces a **32-D Topology Vector** and a **SHA256 Topology Hash** based on morphology features (base type, DOF ratios, mass ratios, workspace volume, etc.).

**Reality:** ✅ Implemented in `isonome/utils/morphology.py` (iteration-028).
- `MorphologyAnalyzer` parses URDF and extracts `BaseMorphology` features.
- `TopologyVector` produces the 32-D feature vector and a stable SHA-256 topology hash.
- `SomaLayer` exposes `morphology` and `topology_vector` properties and `_robot_hash()` now returns `topology_vector.topology_hash[:16]`.
- `TopologyVectorState` Pydantic model in `isonome/core/state.py` for serialization.
- 52 tests in `tests/test_morphology_analyzer.py`.

---

## 6. Calibration Cache vs. LLM Cache

**Architecture shows:** A `Calibration Cache` keyed by `SHA256(topology + task_type + vla_version)` that stores certified policy packages.

**Reality:** `llm/cache.py` implements a generic `SemanticCache` — a simple dict with TTL eviction for Cortex advice strings. It has nothing to do with topology hashes, task types, VLA versions, or policy packages.

---

## 7. ROS2 Runtime Topology (Diagram 8)

**Architecture shows:** Full ROS2 topic graph with:
- `/isonome/agents/{id}/partial_action`
- `/isonome/coordinator/full_action`
- `/isonome/reflex/safe_action`
- `/isonome/fsm/phase`
- Per-agent VLA instances publishing at 200 Hz

**Reality:** No ROS2 integration anywhere. `Agent.run()` is a single Python `asyncio` loop with no inter-process or network transport.

---

## 8. Reflex Layer — Missing Hard Real-Time Guarantees

**Architecture shows:** Reflex Layer at **1 kHz on a dedicated CPU thread**, with Joint Limit Clamping, Velocity Limits, and Emergency Stance.

**Reality:** `ReflexLayer` is a Python class that does linear interpolation and `torch.clamp`. It runs inside `Agent.tick()` at whatever frequency the asyncio loop manages (typically 100 Hz target). There is no dedicated thread, no 1 kHz guarantee, no velocity limit enforcement (only position clamping), and no emergency stance trajectory.

---

## 9. VLA Inference Contexts & Shared World State

**Architecture shows:** Runtime block with "N Inference Contexts (one per agent)", "Observation Ring Buffers (size 10)", and "Shared World State (lock-free dict @ 200 Hz)".

**Reality:** `JEPALayer` loads one VLA policy (or None) and runs a single `deliberate()` call per tick. There are no inference contexts, no ring buffers in `JEPALayer`, and no shared world state. The `DiscrepancyBuffer` in `CortexLayer` is the only buffer in the system, and it's for discrepancy logging, not observations.

---

## 10. CLI Stubs — PARTIALLY CLOSED

**Architecture shows:** `cli.py` with commands `init | sim | run | deploy`.

**Reality:**
- ✅ `init` implemented — scaffolds a robot project with `main.py`, `config.yaml`, layer stubs, and tests.
- ✅ `sim` implemented — loads config, sets bridge engine, runs `IsonomeApp`.
- ❌ `run` still a stub (hardware mode).
- ❌ `deploy` still a stub.

---

## Summary Table

| Architecture Claim | Actual State |
|---|---|
| `SomaLayer` drives Sim/HW bridge | ✅ `BodyBridge` adapters + Agent integration |
| Isaac Lab + MuJoCo MJX backends | Isaac Sim remote server + CPU MuJoCo |
| FSM Compiler + Action Merger | ✅ Implemented in `isonome/core/coordination/` |
| 32-D Topology Vector + Morphology Hash | ✅ `isonome/utils/morphology.py` |
| Calibration Cache (topology+task+vla) | Generic string TTL cache — **next gap** |
| CMA-ES / 256 envs / Auto-Adjustment | ❌ Missing entirely |
| Certified Policy Package (.zip) export | ❌ Missing entirely |
| ROS2 topic topology | ❌ Missing entirely |
| Reflex @ 1 kHz dedicated thread | Python asyncio ~100 Hz |
| VLA inference contexts + ring buffers | Single policy, no buffers |
| `sim` / `run` / `deploy` CLI | `sim` ✅; `run`/`deploy` stubs |
