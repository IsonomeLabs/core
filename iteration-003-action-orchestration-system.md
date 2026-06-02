# isonome-framework — Iteration 003: Action Orchestration System (Praxis)

**Date:** 2026-06-01
**Cron Job:** Hourly incremental improvement
**Iteration:** 003 — Third Iteration (First Praxis Iteration)
**Change Type:** Pillar 2 (Praxis / πρᾶξις) — Full Action Orchestration System
**Tests:** 169/169 passing (68 new)

---

## Summary

Built the **Action Orchestration System** — the πρᾶξις (praxis) pillar of isonome-framework. This is a DAG-based action scheduler that converts cognition plans into executable work, modulates execution behavior through three equilibrium tension axes (autonomy_safety, sequential_parallel, verify_execute), and provides execution results to Mneme for learning. It closes the largest gap in the framework: the Praxis pillar was a stub with 3 orphaned tension axes and no execution engine — now it's a fully-realized system with risk-gated execution, variable parallelism, configurable verification depth, exponential backoff retry, and cross-pillar pipelines.

This is the highest-impact next step because:
1. Without Praxis, the agent can think (Cognition) and remember (Mneme) but cannot DO anything — it's a brain without hands
2. Three tension axes (`autonomy_safety`, `sequential_parallel`, `verify_execute`) had no consumer — they now actively modulate execution behavior
3. The νοῦς → πρᾶξις pipeline completes the thought-to-action flow
4. The πρᾶξις → μνήμη pipeline enables learning from execution outcomes

## What Was Built

### Files Created

| File | LOC | Description |
|------|-----|-------------|
| `isonome/praxis/orchestrator.py` | 902 | Core DAG-based action scheduler with tension modulation |
| `isonome/praxis/pillar.py` | 352 | BasePillar wrapper for agent lifecycle integration |
| `isonome/praxis/__init__.py` | 38 | Updated with full public API (was 1-line stub) |
| `tests/test_praxis.py` | 973 | 68 comprehensive tests |

### Architecture: DAG Execution Engine

```
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │  Cognition   │────▶│   Praxis    │────▶│    Mneme    │
   │  (νοῦς)      │     │  (πρᾶξις)   │     │  (μνήμη)    │
   │             │     │             │     │             │
   │  Plans →    │     │  ┌───────┐  │     │  Memories ← │
   │  import_    │     │  │Action │  │     │  export_    │
   │  from_      │     │  │  DAG  │  │     │  to_mneme() │
   │  cognition()│     │  │       │  │     │             │
   └─────────────┘     │  └───┬───┘  │     └─────────────┘
                       │      │      │
           tensions ───│──────┼──────│─── feedback
           modulate    │  ┌───▼───┐  │
                       │  │Safety │  │
                       │  │ Gate  │  │  τ = 0.5 + autonomy × 0.5
                       │  └───┬───┘  │
                       │  ┌───▼───┐  │
                       │  │Parallel│  │
                       │  │Sched. │  │  C = max(1, ⌊4·(p+1)/2 + 1⌋)
                       │  └───┬───┘  │
                       │  ┌───▼───┐  │
                       │  │Verif- │  │
                       │  │ication│  │  ω = 0.5·(1 − p_verify)
                       │  └───┬───┘  │
                       │  ┌───▼───┐  │
                       │  │ Retry │  │  delay = base × 2^attempt
                       │  │  w/   │  │
                       │  │ backoff│  │
                       │  └───────┘  │
                       └─────────────┘

Action Lifecycle:
  PENDING → QUEUED → EXECUTING → VERIFYING → COMPLETED
                                       └→ FAILED → RETRYING → EXECUTING
  BLOCKED (safety gate)
  CANCELLED (external signal)
```

### The Three Pillars — Now All Active

```
    νοῦς (Cognition)     πρᾶξις (Praxis)      μνήμη (Mneme)
    ─────────────────     ────────────────      ─────────────
    · Reasoning           · ★ DAG Execution     · 3-tier memory
    · Planning            · ★ Safety gating     · Ebbinghaus decay
    · Context management  · ★ Parallel scheduling· Spaced rehearsal
    · ★ Attention EQ      · ★ Verification depth· Consolidation
    ·                      · ★ Retry w/ backoff · ★ Cross-session
    ·                      · ★ Cross-pillar     ·
                          ·   pipelines        ·
```

## Core Mechanisms (with mathematical formulas)

### 1. Safety Gate (autonomy_safety tension)

Every action has a risk level (TRIVIAL=0 to CRITICAL=4). The gate threshold τ is:

```
τ = 0.5 + p_autonomy × 0.5    where p_autonomy ∈ [-1, 1]
```

Risk is normalized: `risk_q = action.risk.value / 4.0`. An action is blocked if `risk_q > τ`.

| autonomy_safety | τ (gate) | TRIVIAL (q=0) | LOW (q=0.25) | HIGH (q=0.75) | CRITICAL (q=1.0) |
|-----------------|----------|---------------|--------------|----------------|-------------------|
| −1.0 (safe)     | 0.00     | ✓ pass        | ✗ block      | ✗ block        | ✗ block          |
| −0.4 (default)  | 0.30     | ✓ pass        | ✓ pass       | ✗ block        | ✗ block          |
|  0.0 (neutral)  | 0.50     | ✓ pass        | ✓ pass       | ✗ block        | ✗ block          |
| +0.5            | 0.75     | ✓ pass        | ✓ pass       | ✓ pass         | ✗ block          |
| +1.0 (auto)     | 1.00     | ✓ pass        | ✓ pass       | ✓ pass         | ✓ pass           |

External approval can override any block via `approve_fn`.

### 2. Parallelism Scheduling (sequential_parallel tension)

Actions are organized by topological level ℓ(v) in the DAG:

```
ℓ(v) = 0 if no dependencies, else max(ℓ(u) for (u→v) ∈ E) + 1
```

Maximum concurrency is:
```
C = max(1, ⌊4·(p_parallel + 1) / 2 + 1⌋)
```

| p_parallel | Max concurrency |
|-----------|-----------------|
| −1.0 (sequential) | 1 |
| −0.5 | 2 |
| 0.0 | 3 |
| 0.5 | 4 |
| +1.0 (parallel) | 5 |

Actions at the same topological level with no inter-dependencies execute concurrently up to C.

### 3. Verification Depth (verify_execute tension)

Verification depth ω controls post-execution validation:

```
ω = 0.5 × (1 − p_verify)
```

| p_verify | ω | Behavior |
|-----------|-----|----------|
| −1.0 (verify_heavy) | 1.0 | Full validation required, threshold=1.0 |
| −0.5 | 0.75 | Full validation, threshold=0.825 |
| 0.0 | 0.50 | Light validation, threshold=0.65 |
| +0.5 | 0.25 | Light validation, threshold=0.475 |
| +1.0 (execute_fast) | 0.0 | No validation at all |

Validation threshold: `val_threshold = 0.3 + ω × 0.7`. Validator returns (passed: bool, score: float).

### 4. Exponential Backoff Retry

```
delay(attempt) = min(base_delay × backoff_factor^attempt, max_delay)
```

Default: base_delay=1.0s, backoff_factor=2.0, max_delay=300s, max_retries=3.

| Attempt | Delay |
|---------|-------|
| 0 | 1.0s |
| 1 | 2.0s |
| 2 | 4.0s |
| 3 | 8.0s |

### 5. Cross-Pillar Pipelines

**νοῦς → πρᾶξις:** `import_from_cognition(tasks)` converts a list of task dicts into executable Action nodes with resolved dependency DAG:

```python
tasks = [
    {"description": "step A", "tool_name": "a", "ref": "a"},
    {"description": "step B", "tool_name": "b", "ref": "b", "dependencies": ("a",)},
]
action_ids = orchestrator.import_from_cognition(tasks)
```

**πρᾶξις → μνήμη:** `export_to_mneme()` returns execution log entries as dicts suitable for HierarchicalMneme storage:

```python
memories = orchestrator.export_to_mneme()
# Each: {action_id, description, tool_name, success, error, attempt, duration_ms, ...}
mneme.store(m["description"], significance=m["success"] ? 0.8 : 0.3, tags=("execution",))
```

## Tension Modulation

All three Praxis tension axes are now actively consumed:

| Axis | Consumer | Modulation |
|------|----------|------------|
| `autonomy_safety` | Safety gate | τ = 0.5 + p × 0.5 — higher autonomy = higher gate (more actions pass) |
| `sequential_parallel` | Parallel scheduler | C = floor(4·(p+1)/2 + 1) — higher parallel = more concurrency |
| `verify_execute` | Validation depth | ω = 0.5·(1−p) — higher execute = less validation |

Feedback is emitted after each execution batch:

- **autonomy_safety**: Success rate ≥ 0.95 → push autonomous (+0.15); rate < 0.50 → push safe (−0.20)
- **sequential_parallel**: Parallelism > 1 and success ≥ 0.80 → push parallel (+0.10)
- **verify_execute**: Low validation scores → push verify_heavy (−0.12); high scores → push execute_fast (+0.08)

## Test Coverage

| Test Class | Tests | Description |
|-----------|-------|-------------|
| TestRetryPolicy | 7 | Exponential backoff delay calculation, capping, edge cases |
| TestAction | 8 | Creation, dependency checking, ready state, risk levels, tags, metadata |
| TestExecutionResult | 2 | Success and failure result data structures |
| TestExecutionReport | 1 | Report field completeness |
| TestOrchestratorRegistration | 5 | Single/batch registration, DAG depth computation (linear, diamond) |
| TestOrchestratorExecution | 6 | Single action, multiple independent, dependency ordering, failure, retry success, retry exhaust |
| TestOrchestratorSafetyGating | 5 | TRIVIAL passes in safe mode, CRITICAL blocked, approval override, HIGH at autonomy, max autonomy CRITICAL |
| TestOrchestratorValidation | 3 | verify_heavy passes/fails, execute_fast skips validation |
| TestOrchestratorParallelism | 2 | Sequential mode (C=1), parallel mode (C>1) |
| TestOrchestratorProperties | 7 | pending/completed/blocked actions, states, totals, stats, execution log |
| TestOrchestratorImportExport | 5 | Single import, risk import, dependency import, empty export, populated export |
| TestOrchestratorSerialization | 6 | Empty round-trip, with actions, risk preservation, completed state, keys, level recomputation |
| TestPraxisPillar | 11 | Init, import_plan signal, execute_plan signal, execute_pending signal, feedback emission (3 axes), memory export, serialize/restore, cancel, tension profile update, no-executor warning, last_report |
| **Total** | **68** | |

All 101 existing tests continue to pass, bringing the framework from 101 to 169 total tests.

## Design Decisions

1. **Separate Safety Gate from Execution**: The gate runs before scheduling, so blocked actions never enter the execution queue. This avoids race conditions between approval checks and actual execution.

2. **Topological Levels for Parallelism**: Rather than computing all pairs of independent actions (O(n²)), we use topological levels — actions at the same level are independent by construction (they share no dependency chain). This is O(n+e) and matches the natural structure of plan DAGs.

3. **Verification as a Continuum, Not Binary**: Instead of "verify yes/no", verification depth ω moves smoothly from 0 (skip everything) to 1 (exhaustive validation with high threshold). The threshold formula `0.3 + ω × 0.7` ensures even light verification has a minimum bar.

4. **Retry Counting from Attempt Index**: Rather than tracking retries in the orchestrator loop, we count `result.attempt > 0` as a retry. This correctly captures both the "retried then succeeded" case and the "retried and failed" case.

5. **Cross-Level Re-Enqueue Only**: When an action completes, we only re-enqueue dependents at strictly higher topological levels. Same-level actions are already in the processing queue — re-adding them would cause duplicate execution.

6. **from_dict Rebuilds Stats from Data**: Following the pattern from HierarchicalMneme, `from_dict()` rebuilds mutable statistics from the deserialized actions rather than trusting saved stats (which may be stale). Topological levels are also recomputed from the reconstructed DAG.

7. **frozendict for Hashability**: A small immutable dictionary implementation enables action parameter maps to be used in frozensets and as dict keys.

8. **Three Separate Feedback Emissions**: After each execution batch, the PraxisPillar emits three Feedback signals — one per tension axis — each with its own confidence score and descriptive reason. This allows the equilibrium engine to independently adjust each axis based on execution outcomes.

## Why This Creates Impact

### Short-Term
- **Immediate functional agent**: With all three pillars now implemented, the isonome agent has a complete think→act→learn loop
- **3 orphaned tension axes now active**: `autonomy_safety`, `sequential_parallel`, and `verify_execute` go from dormant defaults to continuously modulated positions
- **νοῦς → πρᾶξις pipeline**: Plans from cognition can directly become executable actions — no manual bridging required
- **Testable execution engine**: 68 tests covering every subsystem — DAG scheduling, safety gating, validation, retry, serialization, and pillar integration

### Long-Term
- **Foundation for tool integration**: The ActionOrchestrator's `executor_fn` is a pluggable callback — any tool system (shell, HTTP, browser, etc.) can be wired in
- **Learning from execution**: The πρᾶξις → μνήμη pipeline feeds execution outcomes into the memory system, enabling the agent to learn which actions work and which don't
- **Safety as a spectrum**: Risk-gated execution with tension modulation means the agent can gradually earn autonomy as it proves reliable
- **Scalable parallelism**: The DAG-based scheduler naturally supports complex multi-step plans with many parallel branches

## Architecture: Full Framework State

```
                         ┌──────────────────────────────────┐
                         │     EquilibriumEngine             │
                         │     (8 tension axes)              │
                         │                                  │
                         │  Cognition  Praxis     Mneme     │
                         │  ─────────  ──────     ─────     │
                         │  explore_   autonomy_  consol-   │
                         │  exploit    safety     idate_    │
                         │  shallow_   sequen-    prune     │
                         │  deep       tial_      specif-   │
                         │  diverge-   parallel   ic_gen-   │
                         │  nt_conve-  verify_    eral      │
                         │  rgent      execute              │
                         └──────┬───────┬──────────┬───────┘
                                │       │          │
                  tensions ─────┤       │          ├── tensions
                  modulate      │       │          │   modulate
                  ┌─────────────▼──┐ ┌──▼───────────▼──┐ ┌─────────────┐
                  │   νοῦς         │ │   πρᾶξις        │ │   μνήμη      │
                  │  (Cognition)   │ │  (Praxis)       │ │  (Mneme)     │
                  │                │ │                 │ │              │
                  │ Attention EQ   │ │ Action Orch.    │ │ Hierarchical │
                  │ System         │ │  ┌─────────┐   │ │ Mneme        │
                  │  · Scoring     │ │  │Safety   │   │ │  · Working   │
                  │  · GC          │ │  │Gate     │   │ │  · Episodic  │
                  │  · Decay       │ │  ├─────────┤   │ │  · Semantic  │
                  │  · Surprisal   │ │  │DAG      │   │ │  · Forgetting│
                  │                │ │  │Scheduler│   │ │  · Rehearsal │
                  │                │ │  ├─────────┤   │ │  · Patterns  │
                  │                │ │  │Verify   │   │ │              │
                  │                │ │  │Pipeline │   │ │              │
                  │                │ │  ├─────────┤   │ │              │
                  │                │ │  │Retry w/ │   │ │              │
                  │                │ │  │Backoff  │   │ │              │
                  │                │ │  └─────────┘   │ │              │
                  └────────┬───────┘ └───────┬────────┘ └──────┬───────┘
                           │                 │                  │
                           └──import_from────┘                  │
                           └──attention─────────────────────────┘
                           └──export_to_mneme───────────────────┘
```

## Next Iteration Candidates

1. **Cognition pillar wrapper**: Wrap the AttentionEquilibriumSystem in a BasePillar (like MnemePillar and PraxisPillar) for full agent integration
2. **Cognition ↔ Praxis bridge**: Implement the concrete signal protocol for "plan ready → import into execution"
3. **Full integration test**: Wire all three pillars into a single IsonomeAgent with a real task flow (plan → execute → remember)
4. **Mneme ↔ Cognition bridge**: Recall relevant past executions when planning similar tasks
5. **Tool registry**: A concrete tool system (shell, HTTP, browser) that plugs into the ActionOrchestrator's `executor_fn`

## Files Changed

```
isonome/praxis/
├── __init__.py          (updated: 1 line → 38 lines — full public API)
├── orchestrator.py      (new: 902 lines)
└── pillar.py            (new: 352 lines)
tests/
└── test_praxis.py       (new: 973 lines)
```

## Commit Stack

```
git add -A
git commit -m "feat: Action Orchestration System — DAG-based Praxis pillar with
safety gating, parallel scheduling, verification depth, and cross-pillar pipelines"
```
