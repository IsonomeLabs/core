# Isonome Architecture → Code Gap Analysis

> Generated: 2026-06-09  
> Comparing `architecture.md` (PRD v0.1 / Code v0.2) against the current `isonome/core/` and `isonome/sim/` implementation.

---

## 1. Core → Sim Bridge Gap (Biggest Disconnect)

**Architecture says:** `SomaLayer` loads URDF via `SimBridge` (PyBullet) **or** `HardwareBridge`, and `perceive()` / `act()` read from / write to that bridge.

**Reality:** `SomaLayer` **never connects to any bridge**. It parses the URDF file with `xml.etree` to count joints, then `perceive()` returns zero-filled tensors and `act()` is a no-op logger call. The `SimBridge` in `isonome/bridge/sim.py` is a standalone PyBullet wrapper that nothing in `core` imports or uses. Same for `HardwareBridge` — `Agent` and `SomaLayer` don't know it exists.

**Impact:** There is **no integration point** between the four-layer agent pipeline and any physics engine. The `VLAController` in `sim/vla_controller.py` drives MuJoCo directly, bypassing `Agent`, `SomaLayer`, `ReflexLayer`, and `SafetyGovernor` entirely.

---

## 2. Missing FSM Compiler & Action Merger

**Architecture shows:** Chamber 3 has an **FSM Compiler** (Guards / Events / Merge Strategy) and an **Action Merger** (Priority | Weighted Average | Nullspace).

**Reality:** Nothing like this exists in `core/`. `Agent.tick()` runs a single linear pipeline for one body. There is no finite-state machine, no multi-agent composition, no merge strategy, and no coordinator.

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

## 5. Topology / Morphology Analyzer

**Architecture shows:** Chamber 2 produces a **32-D Topology Vector** and a **SHA256 Topology Hash** based on morphology features (base type, DOF ratios, mass ratios, workspace volume, etc.).

**Reality:** The only "hash" is `SomaLayer._robot_hash()`, which is literally `sha256(urdf_file_bytes)[:16]`. There is no morphology analyzer, no topology vector, and the cache key logic shown in Diagram 1 doesn't exist.

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

## 10. CLI Stubs

**Architecture shows:** `cli.py` with commands `init | sim | run | deploy`.

**Reality:** `sim`, `run`, and `deploy` are all stubs that print a string and do nothing:

```python
@app.command()
def sim() -> None:
    typer.echo("Starting simulation... (stub)")
    # TODO: load config, create SimBridge, run Agent
```

---

## Summary Table

| Architecture Claim | Actual State |
|---|---|
| `SomaLayer` drives Sim/HW bridge | No-op; bridges are orphaned |
| Isaac Lab + MuJoCo MJX backends | Isaac Sim remote server + CPU MuJoCo |
| FSM Compiler + Action Merger | ❌ Missing entirely |
| 32-D Topology Vector + Morphology Hash | Raw URDF file SHA256 only |
| Calibration Cache (topology+task+vla) | Generic string TTL cache |
| CMA-ES / 256 envs / Auto-Adjustment | ❌ Missing entirely |
| Certified Policy Package (.zip) export | ❌ Missing entirely |
| ROS2 topic topology | ❌ Missing entirely |
| Reflex @ 1 kHz dedicated thread | Python asyncio ~100 Hz |
| VLA inference contexts + ring buffers | Single policy, no buffers |
| `sim` / `run` / `deploy` CLI | Stubs |
