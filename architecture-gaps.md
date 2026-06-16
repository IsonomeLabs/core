# Isonome Architecture → Code Gap Analysis

> Generated: 2026-06-09  
> Comparing `architecture.md` (PRD v0.1 / Code v0.2) against the current `isonome/core/` and `isonome/sim/` implementation.

---

## 1. Core → Sim Bridge Gap (Biggest Disconnect) ✅ CLOSED

**Architecture says:** `SomaLayer` loads URDF via `SimBridge` (PyBullet) **or** `HardwareBridge`, and `perceive()` / `act()` read from / write to that bridge.

**Reality:** ✅ Implemented via `BodyBridge` port + adapters.
- `BodyBridge` abstract port in `isonome/core/ports/body_bridge.py` with async `perceive()` / `act()` / `observe_result()` lifecycle.
- Concrete adapters in `isonome/bridge/adapters/`: `MockBridgeAdapter`, `PyBulletBridgeAdapter`, `MuJoCoBridgeAdapter`, `HardwareBridgeAdapter`.
- `Agent` constructs the bridge via `build_body_bridge(config)` and delegates runtime I/O to it.
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

## 3. Calibration / Training Pipeline (Diagram 7: "PRD Future State") ⚠️ PARTIALLY CLOSED

**Architecture describes:** A full simulation pipeline with:
- URDF Stripper (per-agent joint subsets)
- Isaac Lab Env Spawner (256 parallel envs)
- Domain Randomization (mass, friction, damping, lighting)
- CMA-ES / Differentiable Sim optimization
- Composition Validation (1000 episodes, >99% success target)
- Auto-Adjustment Engine (max 5 iterations)
- Export of a **Certified Policy Package (.zip)**

**Reality:** ✅ Foundational pipeline implemented in `isonome/praxis/calibration/` (iteration-031).
- `URDFStripper` extracts per-agent joint subsets and writes stripped URDFs.
- `DomainRandomizer` randomizes URDF mass / friction / damping and produces lighting overrides.
- `CMAESOptimizer` is a lightweight, dependency-free CMA-ES implementation with configurable population, generations, and fitness target.
- `CompositionValidator` runs N episodes, computes success rate, and certifies when threshold is met.
- `PolicyPackageExporter` creates a `.zip` containing manifest, agent/coordinator configs, reflex gains, sim metrics, policy weights, launcher, and optional certification video.
- `CalibrationPipeline` orchestrates the full loop (strip → randomize → optimize → validate → auto-adjust → export → cache).
- CLI `isonome calibrate` added to run the pipeline end-to-end.
- Pipeline integrates with `CalibrationCache` so certified packages are stored under `SHA256(topology + task_type + vla_version)`.

**Gap remaining:** The pipeline is backend-agnostic and defaults to a mock pendulum objective. It does not yet ship an Isaac Lab env spawner or 256 GPU parallel environments. Enterprises can plug those in by providing a custom `BlackBoxObjective` and `episode_runner_factory`. This closes the open-source runtime portion of gap #3.

---

## 4. Simulation Backends Mismatch ✅ CLOSED

**Architecture shows:** Isaac Lab as primary, MuJoCo **MJX** as fallback for contact-rich tasks.

**Reality:** ✅ Implemented.
- `isonome/sim/isaac_lab_bridge.py` — `IsaacLabBridge` that loads URDFs into an Isaac Lab `ManagerBasedRLEnv` and exposes the same command protocol as the other sim bridges.
- `isonome/sim/mjx_bridge.py` — `MJXBridge` that runs MuJoCo physics on the JAX backend via `mujoco.mjx`, copies results back to CPU for rendering/proprioception, and supports `gpu`/`cpu` device selection.
- `isonome/bridge/adapters/isaac_lab_adapter.py` — `BodyBridge` adapter wrapping `IsaacLabBridge` for `SomaLayer` integration.
- `isonome/bridge/adapters/mjx_adapter.py` — `BodyBridge` adapter wrapping `MJXBridge` for `SomaLayer` integration.
- `BridgeConfig.engine` now accepts `isaac_lab` and `mujoco_mjx`; `build_body_bridge()` constructs the corresponding adapter.
- Both backends use guarded imports so the rest of the codebase imports cleanly when the heavy dependencies are not installed; 9 tests mock the optional deps to verify the command protocol and adapter lifecycle.

**Note:** Isaac Lab and JAX/MJX are large optional dependencies. The bridges raise clear runtime errors when the dependencies are missing and the existing Isaac Sim (`isaac_bridge.py`) and CPU MuJoCo (`mujoco_bridge.py`) bridges remain available for backward compatibility.

---

## 5. Topology / Morphology Analyzer ✅ CLOSED

**Architecture shows:** Chamber 2 produces a **32-D Topology Vector** and a **SHA256 Topology Hash** based on morphology features (base type, DOF ratios, mass ratios, workspace volume, etc.).

**Reality:** ✅ Implemented in `isonome/utils/morphology.py` (iteration-028).
- `MorphologyAnalyzer` parses URDF and extracts `BaseMorphology` features.
- `TopologyVector` produces the 32-D feature vector and a stable SHA-256 topology hash.
- `SomaLayer` exposes `morphology` and `topology_vector` properties; `_robot_hash()` returns `topology_vector.topology_hash[:16]`.
- `TopologyVectorState` Pydantic model in `isonome/core/state.py` for serialization.
- 52 tests in `tests/test_morphology_analyzer.py`.

---

## 6. Calibration Cache vs. LLM Cache — ✅ CLOSED

**Architecture shows:** A `Calibration Cache` keyed by `SHA256(topology + task_type + vla_version)` that stores certified policy packages.

**Reality:** Two implementations now exist:
- `isonome/core/calibration_cache.py` — in-memory `CalibrationCache` with `CalibrationCacheKey`, `CalibrationCacheEntry`, `CalibrationCacheStats`, certification filtering, and `to_dict()`/`from_dict()` serialization.
- `isonome/praxis/calibration_cache.py` — on-disk `CalibrationCache` with `CacheKey`, `CertifiedPolicyPackage`, public/private namespaces, topology-vector near-match search (L2), and CLI commands (`cache put`, `cache lookup`, `cache list`).

**Unified implementation:** ✅ `isonome/core/unified_calibration_cache.py` (iteration-030/032).

- `UnifiedCalibrationCache` provides in-memory hot path with hit/miss/put/eviction stats, certification filtering on both exact-match get and near-match search, on-disk persistence so entries survive process restarts, namespaces (public/private) with directory-level isolation, near-match search by L2 topology-vector distance, TTL support with lazy eviction on read, max-size FIFO eviction for bounded caches, and `to_dict()`/`from_dict()` serialization.
- `remove()` and `_evict_oldest()` now correctly clean both in-memory and on-disk entries (fixed iteration-032).
- The original core and praxis implementations remain untouched for backward compatibility. The unified cache is the recommended replacement going forward.

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

## 10. CLI Stubs ✅ CLOSED

**Architecture shows:** `cli.py` with commands `init | sim | run | deploy`.

**Reality:**
- ✅ `init` implemented — scaffolds a robot project with `main.py`, `config.yaml`, layer stubs, and tests.
- ✅ `sim` implemented — loads config, sets bridge engine, runs `IsonomeApp`.
- ✅ `cache` subcommands added — `put`, `lookup`, `list` for calibration cache management.
- ✅ `run` implemented — loads config, validates URDF, forces `bridge.engine = "hardware"`, and runs `IsonomeApp`. Supports an optional certified policy package (`.zip`) whose `policy/policy.pt` is pointed at as the runtime kernel.
- ✅ `deploy` implemented — validates a certified policy package, extracts it to a deployment directory, copies an optional runtime config, and writes a deployment manifest with `robot_ip`, `protocol`, and timestamp. The resulting directory is a self-contained artifact ready for a follow-up ROS2/MQTT/HTTP push.

---

## Summary Table

| Architecture Claim | Actual State |
|---|---|
| `SomaLayer` drives Sim/HW bridge | ✅ `BodyBridge` adapters + Agent integration |
| Isaac Lab + MuJoCo MJX backends | ✅ `IsaacLabBridge` + `MJXBridge` + adapters |
| FSM Compiler + Action Merger | ✅ Implemented in `isonome/core/coordination/` |
| 32-D Topology Vector + Morphology Hash | ✅ `isonome/utils/morphology.py` |
| Calibration Cache (topology+task+vla) | ✅ Unified cache (core + praxis features merged) |
| CMA-ES / 256 envs / Auto-Adjustment | ✅ Pipeline implemented; Isaac Lab backend pending |
| Certified Policy Package (.zip) export | ✅ Implemented in `isonome/praxis/calibration/exporter.py` |
| ROS2 topic topology | ❌ Missing entirely |
| Reflex @ 1 kHz dedicated thread | Python asyncio ~100 Hz |
| VLA inference contexts + ring buffers | Single policy, no buffers |
| `sim` / `run` / `deploy` CLI | ✅ All implemented |
