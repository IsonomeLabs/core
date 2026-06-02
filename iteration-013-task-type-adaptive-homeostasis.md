# isonome-framework — Iteration 013: Task-Type Adaptive Homeostasis

**Date:** 2026-06-02
**Cron Job:** Hourly incremental improvement
**Iteration:** 013
**Change Type:** Cross-pillar — curriculum learning for equilibrium defaults via task-type profiles
**Tests:** 571/571 passing (+39 new)

---

## Summary

The agent now learns **different equilibrium configurations for different task types** — and pre-adapts its tension defaults when it recognizes a known type on new task submission.

**Before:** The outcome-driven homeostatic loop (iter-012) adapted defaults globally. Every task type experienced the same learning signal. A sequence of "analysis" tasks followed by a "coding" task used the same set points, even though analysis and coding might optimally operate at different equilibrium configurations.

**After:** The agent tracks default-position trajectories *per task type*. After enough "analysis" tasks produce a stable default profile, the next "analysis" task submission automatically pre-adapts the engine defaults toward the learned configuration — before the first action executes.

| Mechanism | Before | After |
|-----------|--------|-------|
| Default adaptation scope | Global — one set of set points | Per-task-type — distinct profiles per type |
| Task type awareness | None | Keyword-based type inference from description |
| Pre-adaptation on submit | Never | Auto-applies when profile is converged (≥3 stable observations) |
| Pre-adaptation strength | N/A | Soft (1/3 distance) — gradual, not aggressive |
| Cross-session persistence | No type profiles saved | Full `to_dict()/from_dict()` for all profiles |
| Profile convergence | N/A | RMS-stability check; "is this profile trustworthy?" |
| Profile similarity metric | N/A | Cosine similarity between current and learned defaults |

---

## What Was Built

| File | Action | Change | Key Contribution |
|------|--------|--------|-----------------|
| `isonome/equilibrium/task_type_homeostasis.py` | **Created** | 506 lines | Full module: `TaskTypeHomeostasis`, `TaskTypeProfile`, `infer_task_type()`, 8 built-in types, convergence detection, pre-adaptation, soft pre-adapt, similarity metric, serialization |
| `isonome/agent.py` | **Modified** | +54 / −2 lines | `task_type_homeostasis` property, `current_task_type` property, `submit_task()` type inference + pre-adaptation hook, `_process_execution_outcomes()` recording (M3), `to_dict/from_dict` integration, `stats` inclusion |
| `tests/test_task_type_homeostasis.py` | **Created** | +478 lines | 39 tests across 9 test classes |

---

## Architecture (ASCII)

```


    Task submitted with description
                │
                ▼
    ┌──────────────────────┐
    │   submit_task()      │
    │                      │
    │  1. infer_task_type()│──→ "analysis", "coding", ...
    │  2. set current_type │
    │  3. check profile    │
    │  4. if converged:    │──→ soft_pre_adapt(engine, type)
    │     soft-pre-adapt   │     moves defaults 1/3 of
    │                      │     distance to learned profile
    └──────────────────────┘
                │
                ▼
    ┌──────────────────────┐
    │   tick() loop        │
    │                      │
    │  ... pillars process │
    │                      │
    │  _process_execution_ │
    │   outcomes():        │
    │    M1: calibrator    │
    │    M2: adapt defaults│
    │    M3: record        │──→ TaskTypeHomeostasis.record_defaults()
    │        per-type      │     pushes current defaults into
    │                      │     the profile for current_task_type
    └──────────────────────┘
                │
                ▼
    ┌──────────────────────────────────────┐
    │   TaskTypeHomeostasis                │
    │                                      │
    │   _profiles = {                      │
    │     "analysis": TaskTypeProfile(     │
    │       observations: [[0.15,-0.2,...],│
    │                      [0.16,-0.19,...],│
    │                      [0.14,-0.2,...]] │
    │       is_converged: True             │← stable (RMS<0.05)
    │       get_norm(): {"explore_exploit":│
    │                    0.15, ...}        │← converged profile
    │     ),                               │
    │     "coding": TaskTypeProfile(...),  │
    │   }                                  │
    │                                      │
    │   summary():                         │
    │     profile_count: 2                 │
    │     converged_types: ("analysis",)   │
    │     total_recordings: 6              │
    │     pre_adaptations_applied: 1       │
    └──────────────────────────────────────┘
                │
                ▼
    ┌──────────────────────┐
    │  serialization       │
    │                      │
    │  to_dict():          │
    │    task_type_        │──→ included in agent.to_dict()
    │    homeostasis: {...}│     survives from_dict()
    │  from_dict():        │
    │    restores profiles │
    │    restores current  │
    │    task type         │
    └──────────────────────┘
```

---

## Core Mechanisms (with mathematical formulas)

### Mechanism 1: Task Type Inference

Keyword-based heuristic classifier. Returns one of 8 built-in types or "general". Uses substring matching with specificity ordering:

```
Order: debugging > coding > analysis > research > writing >
       planning > design > data_processing > general
```

Keywords per type:
- **debugging:** debug, bug, fix, error, crash, traceback
- **coding:** code, implement, program, function, class, api
- **analysis:** analy, investigat, evaluat, assess, examine
- **research:** research, literature, paper, survey, study, find
- **writing:** write, draft, compose, essay, report, document
- **planning:** plan, strategy, roadmap, schedule, organize
- **design:** design, architecture, layout, ui, mockup
- **data_processing:** data, transform, extract, load, process, pipeline, etl

### Mechanism 2: Profile Convergence Detection

A profile is "converged" (safe to use for pre-adaptation) when:

1. **Minimum observations:** `n ≥ 3`
2. **Stability check:** The RMS difference between the first-half mean and second-half mean is below 5%:

```
Let:
    obs = [v₀, v₁, ..., v_{n-1}]   # list of default-position vectors
    mid = n // 2
    μ₁ = mean(obs[0:mid])           # first-half mean
    μ₂ = mean(obs[mid:n])           # second-half mean
    Δ = μ₂ - μ₁                     # vector of differences
    convergence_ratio = sqrt(mean(Δ²))
    is_converged = convergence_ratio < 0.05
```

A low ratio means the defaults have stabilized — the system consistently settles to the same configuration for this task type.

### Mechanism 3: Learned Norm (Profile Vector)

The "norm" for a converged profile is the element-wise mean of all observations:

```
norm[axis_id] = mean(obs[i][axis_idx] for i in range(n))
```

This produces a dict like:
```python
{"explore_exploit": 0.12, "shallow_deep": -0.18, "autonomy_safety": -0.45, ...}
```

### Mechanism 4: Full Pre-Adaptation

`apply_task_type_profile()` computes the exact signal needed to move each axis's default to the learned value:

```
For each axis_id, learned_value in norm.items():
    current = axis.default_position
    if |learned_value - current| < 0.001:
        skip (near-identical)
    needed_signal = (learned_value - current) / axis.learning_rate
    engine.adjust_default(axis_id, outcome_signal=needed_signal)
```

This moves the set point to the learned target in a single damped step.

### Mechanism 5: Soft Pre-Adaptation (default)

`soft_pre_adapt()` moves one **third** of the distance, preventing aggressive shifts when the profile may be outdated or the task type loosely matched:

```
target = current + (learned_value - current) / 3.0
needed_signal = (target - current) / axis.learning_rate
```

This is the default in `submit_task()`. The first task of a known type gets a gentle nudge rather than a full reset.

### Mechanism 6: Profile Similarity

Cosine similarity between the current engine default vector and the learned norm vector:

```
similarity = (C · L) / (|C| · |L|)
```

Where C is the vector of current defaults and L is the learned norm. Value range: [0, 1] with 1.0 = identical.

---

## Tension Modulation

The task-type homeostasis system is the **third** layer of learning in the agent, operating orthogonally to existing mechanisms:

| Layer | What It Modifies | When It Acts | Relationship |
|-------|-----------------|-------------|-------------|
| Feedback | Current position | Every tick | Fast, reactive |
| Calibrator → Default (iter-012) | All default positions | After outcomes | Medium, global structural learning |
| **Task-type profiling (iter-013)** | Default positions per type | On task submission + after outcomes | Slow, *discriminative* — different types push defaults in different directions |

Key design principle: **Task-type adaptation does not replace global default adaptation.** The two compose naturally:

1. Global default adaptation (iter-012 M2) shifts set points based on aggregate outcomes
2. Task-type profiling records where those set points *end up* per task type
3. On a new task, soft pre-adaptation nudges toward the stored profile for that type
4. Global adaptation continues from the nudge point

This composability means:
- If a new type has no profile, the agent uses globally adapted defaults
- If a profile exists but is outdated, the agent adapts from the nudge point
- Multiple types can converge to different attractors simultaneously

### Interaction Examples

| Scenario | Global Default | Analysis Profile | Coding Profile | Behavior |
|----------|---------------|-----------------|---------------|----------|
| Fresh agent, first task | Stock (-0.4 safe) | — | — | No pre-adaptation |
| After 5 analysis cycles | -0.5 safe, +0.2 exploit | Same as global | — | Pre-adaptation activates for analysis tasks |
| After 5 coding cycles | Now -0.3 safe | -0.5 safe | -0.3 safe | Each type pre-adapts to its own profile |
| Restored from serialization | Saved state | Saved state | Saved state | Profiles survive across sessions |

---

## Key Design Decisions

1. **Keyword-based inference, not LLM** — The type classifier is rule-based and deterministic. This ensures reproducible behavior: the same task description always produces the same type. An LLM-based classifier would introduce variance that makes homeostatic learning hard to debug.

2. **Convergence gating** — Profiles require ≥3 observations AND stability (convergence_ratio < 0.05) before they're used for pre-adaptation. This prevents premature adjustment from noisy early data. The 0.05 threshold is conservative — only profiles that are genuinely stable activate.

3. **Soft pre-adaptation as default** — `submit_task()` uses `soft_pre_adapt()` (1/3 distance) rather than `apply_task_type_profile()` (full distance). This is the homeostatic analogue of gradual integration: the profile represents a long-term attractor, but the agent shouldn't snap to it on a single task submission. When the agent encounters multiple tasks of the same type, the nudge compounds.

4. **Recording after outcome processing, not after submit** — Profiles are built from `_process_execution_outcomes()`, not from `submit_task()`. This means defaults are recorded *after* they've been adapted by outcome signals — the recorded values represent the *settled* configuration, not the initial guess. This gives cleaner profiles.

5. **Mechanism 3 as a sep from iter-012** — The recording step (M3) lives in `_process_execution_outcomes()` right after M2 (default adaptation). This makes the recording depend on the same guard conditions (actions_total > 2, reports exist) and ensures it only fires when meaningful outcome processing has happened.

6. **8 built-in types with 'general' fallback** — Enough coverage for diverse tasks without over-specialization. The `BUILTIN_TASK_TYPES` tuple is a public constant that consumers can extend.

7. **Cosine similarity as a diagnostic** — `get_profile_similarity()` lets callers ask "how different is the current configuration from this type's profile?" This is useful for drift detection: if similarity drops over time, the task environment may be changing.

8. **Serialization is full, not partial** — `TaskTypeHomeostasis.to_dict()` serializes all profiles, axis order, and counters. Restored agents resume with complete task-type knowledge — no warmup needed. This is the foundation for meta-learning across sessions.

9. **apply_task_type_profile is idempotent** — Calling it twice in a row has no effect because the first call moves defaults to the learned values, and the second finds no difference > 0.001. This matters when multiple code paths might trigger pre-adaptation.

---

## Mathematical Foundation Summary

| Concept | Formula | Purpose |
|---------|---------|---------|
| Convergence ratio | `RMS(mean(recent) - mean(initial))` | Profile stability check |
| Learned norm | `element-wise mean(observations)` | Canonical profile vector |
| Full adjustment | `signal = (target - current) / lr` | One-shot to learned config |
| Soft adjustment | `target' = current + (target - current)/3` | Gradual pre-adaptation |
| Similarity | `(C·L) / (|C|·|L|)` | Cosine distance diagnostic |

---

## Test Coverage

### New Tests (39 in `test_task_type_homeostasis.py`)

| Test Class | Tests | What It Covers |
|-----------|-------|----------------|
| `TestTaskTypeInference` | 5 | Type detection for analysis, coding, debugging, general, all builtins |
| `TestTaskTypeProfile` | 6 | Empty profile, convergence, norms, serialization, drift detection |
| `TestTaskTypeHomeostasisCore` | 8 | Empty state, recorded defaults, separate types, get_profile, get_or_create, convergence across types, summary |
| `TestPreAdaptation` | 4 | Unknown type skip, no-adapt for unconverged, identity (learned = current), learned ≠ current |
| `TestSoftPreAdaptation` | 3 | Unknown skip, unconverged skip, partial-distance movement |
| `TestProfileSimilarity` | 2 | Zero for unknown, near-1.0 when identical |
| `TestHomeostasisSerialization` | 3 | Empty roundtrip, populated roundtrip, converged survives |
| `TestAgentHomeostasisIntegration` | 5 | Agent creation, submit_task type inference, multi-type tracking, stats inclusion, serialization preservation |
| `TestFullHomeostaticCycle` | 3 | Converged profile activates pre-adaptation, different types produce distinct profiles, full cycle survives serialization |

### Full Suite (571 total)

```
tests/test_agent.py                                25 tests
tests/test_attention.py                            23 tests
tests/test_calibration.py                          63 tests
tests/test_calibration_attention.py                51 tests
tests/test_calibration_mneme.py                    26 tests
tests/test_calibration_rehearsal_pattern.py        29 tests
tests/test_equilibrium.py                          18 tests
tests/test_mneme.py                                50 tests
tests/test_outcome_learning_loop.py                25 tests
tests/test_praxis.py                               68 tests
tests/test_reasoning.py                            63 tests
tests/test_serialization.py                        41 tests
tests/test_task_type_homeostasis.py                39 tests  ← NEW
tests/test_uncertainty_planning.py                 35 tests
tests/test_confidence_gating.py                    24 tests
tests/test_confidence_verify.py                    24 tests
---
Total: 571 tests — 571/571 passing ✅
```

---

## Why This Creates Impact

### Short-term (immediate value)

- **Discriminative homeostatic learning** — The agent no longer treats all tasks as the same. After 5 "analysis" tasks, the defaults settle at one attractor; after 5 "debugging" tasks, a different one. This is the framework's first step toward *task-type-adaptive behavior*.

- **Pre-adaptation before the first action** — When a known task type arrives, defaults shift before any action executes. This means the first few actions in a known domain are already operating at the right equilibrium — no warmup time wasted.

- **Transparent reporting** — `agent.stats["task_type_homeostasis"]` shows all known types, converged types, observation counts, and pre-adaptations applied. The agent's curriculum learning is fully observable.

- **No performance regression** — The pre-adaptation is a no-op for unknown types, so existing agents see zero behavior change. New agents gradually accumulate profiles as they work.

### Long-term (strategic value)

- **Foundation for cross-session meta-learning** — Profiles survive serialization. An agent that's been running for 1000 tasks across 50 sessions has learned 8 distinct equilibrium attractors. This is the infrastructure for "waking up ready for the task at hand" after any number of sleep/wake cycles.

- **Task-type drift detection** — By comparing similarity over time, the system can detect when the same task type starts producing a different equilibrium profile. This is a signal for concept drift — the environment or agent capabilities have changed.

- **Enables curriculum analytics** — A supervisor can inspect profiles per type and ask "why does 'analysis' converge at -0.5 safe while 'coding' converges at -0.3 safe?" The profile trajectory itself becomes a diagnostic tool.

- **Natural delegation trigger** — When a task type has a new observation that deviates strongly from the learned norm, it could trigger a "type drift" signal. Combined with calibrator ECE (iter-005) and outcome-driven defaults (iter-012), this creates a multi-signal stress indicator for each task type.

- **Extensible taxonomy** — The `BUILTIN_TASK_TYPES` constant is public. Future iterations can add domain-specific types (e.g., "security_audit", "code_review") or use embedding-based classifiers for finer-grained distinctions.

- **Meta-learning on type:type relationships** — Are "analysis" and "research" profiles more similar to each other than "analysis" and "coding"? The similarity metric enables type-type correlation matrices. Over many sessions, the agent could learn that "analysis" and "research" share an attractor while "coding" and "debugging" share another — discovering *type families*.

---

## Architecture: Full Framework State

```
╔════════════════════════════════════════════════════════╗
║                    IsonomeAgent                       ║
║                                                        ║
║  ╔═══════════════════════════════════════════════╗     ║
║  ║           EquilibriumEngine                  ║     ║
║  ║  ┌─────────┐ ┌─────────┐ ... ┌────────────┐ ║     ║
║  ║  │explore  │ │shallow  │     │consolidate │ ║     ║
║  ║  │_exploit │ │_deep    │     │_prune      │ ║     ║
║  ║  │pos:0.15 │ │pos:-0.2 │     │pos:-0.1    │ ║     ║
║  ║  │def:-0.1 │ │def:-0.25│     │def:-0.15   │ ║     ║
║  ║  └─────────┘ └─────────┘     └────────────┘ ║     ║
║  ║  + adjust_default() ← set-point adaptation  ║     ║
║  ╚═══════════════════════════════════════════════╝     ║
║                                                        ║
║  ╔═══════════════════════════════════════════════╗     ║
║  ║        TaskTypeHomeostasis ← NEW              ║     ║
║  ║                                               ║     ║
║  ║  profiles = {                               ║     ║
║  ║    "analysis": TaskTypeProfile               ║     ║
║  ║      obs: [(0.15,-0.2,...)*3]               ║     ║
║  ║      converged: True                         ║     ║
║  ║      norm: {explore:0.15,...}                ║     ║
║  ║    "coding": TaskTypeProfile                 ║     ║
║  ║      obs: [(0.12,-0.18,...)*5]               ║     ║
║  ║      converged: True                         ║     ║
║  ║      norm: {explore:0.12,...}                ║     ║
║  ║  }                                           ║     ║
║  ║  record_counts: 8                           ║     ║
║  ║  pre_adaptations: 2                         ║     ║
║  ╚═══════════════════════════════════════════════╝     ║
║                                                        ║
║  ╔═══════════════════════════════════════════════╗     ║
║  ║  tick() → _process_execution_outcomes()      ║     ║
║  ║    M1: calibrator.record()                   ║     ║
║  ║    M2: engine.adjust_default(global)          ║     ║
║  ║    M3: homeostasis.record_defaults(per-type)  ║     ║
║  ╚═══════════════════════════════════════════════╝     ║
║                                                        ║
║  ╔═══════════════════════════════════════════════╗     ║
║  ║  submit_task(Task)                           ║     ║
║  ║    1. infer_task_type() → "coding"           ║     ║
║  ║    2. if profile.converged:                  ║     ║
║  ║       soft_pre_adapt(engine, "coding")       ║     ║
║  ╚═══════════════════════════════════════════════╝     ║
║                                                        ║
║  ╔═══════════════════════════════════════════════╗     ║
║  ║  Pillars                                     ║     ║
║  ║  ┌──────────┐  ┌──────────┐  ┌──────────┐   ║     ║
║  ║  │Cognition │  │ Praxis   │  │ Mneme    │   ║     ║
║  ║  │(νοῦς)    │◄─┤(πρᾶξις) │──┤(μνήμη)   │   ║     ║
║  ║  │          │  │ Execution│  │          │   ║     ║
║  ║  │Calibrator│  │ Report   │  │          │   ║     ║
║  ║  └──────────┘  └──────────┘  └──────────┘   ║     ║
║  ╚═══════════════════════════════════════════════╝     ║
╚════════════════════════════════════════════════════════╝
```

---

## Next Iteration Candidates

1. **Calibration-stress delegation trigger** — When all three conditions are met (ECE > 0.20, success rate < 50%, defaults shifting toward safe/verify across multiple task types), emit a structured `calibration_stress` signal. Could be gated per task type using the new profile infrastructure.

2. **Per-axis learning rate adaptation** — Tension axes that have drifted further from initial defaults during outcome-driven learning should increase their learning_rate. Now that we have per-type profiles, this could also be type-specific: an axis that drifts +0.3 for "analysis" but stays at 0.0 for "coding" should increase learning_rate only for analysis tasks.

3. **task_type_homeostasis in Mneme rehearsal** — Use task-type profiles to bias rehearsal scheduling: if the current task type is "coding," boost entries tagged with coding-related events during spaced repetition. Mneme's `rehearse_by_tags()` (iter-010) already supports calibration-gated boost; extend to type-gated boost.

4. **Task-type drift detection** — Compare the similarity of observed default positions against the stored norm. If similarity drops below 0.7 over 3 consecutive recordings, emit a `type_drift` signal — the task type label may no longer fit.

5. **Type-type correlation matrix** — Over many sessions, compute pairwise cosine similarity between all learned profiles. This reveals "type families" (analysis↔research, coding↔debugging) and could dynamically merge or split profiles.

6. **Rehearsal scheduling from outcome patterns** — Use the execution log (stored via πρᾶξις → μνήμη) combined with task-type to identify which action types fail most often per task type, scheduling more rehearsal cycles accordingly.

---

## Files Changed (tree)

```
isonome/
├── agent.py                                  +54/−2 lines — homeostasis integration
└── equilibrium/
    └── task_type_homeostasis.py              +506 lines — New module: profiles, inference, pre-adaptation, serialization
tests/
└── test_task_type_homeostasis.py             +478 lines — 39 tests across 9 classes
```

---

## Commit Stack

```bash
# 1. Feature: TaskTypeHomeostasis module — profiles, inference, pre-adaptation
# 2. Feat: Integrate into agent.py — submit_task hook, outcome recording, serialization
# 3. Test: 39 tests for task-type adaptive homeostasis
# 4. Docs: iteration-013 — task-type adaptive homeostasis
```
