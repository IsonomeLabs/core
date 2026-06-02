# isonome-framework — Iteration 004: CognitionPillar — Completing the Three-Pillar Architecture

**Date:** 2026-06-01
**Cron Job:** Hourly incremental improvement
**Iteration:** 004
**Change Type:** feat: CognitionPillar wrapper + comprehensive tests — completes three-pillar agent loop
**Tests:** 232/232 passing (169 existing + 63 new)

---

## Summary

The Cognition pillar previously had only a reasoning engine (`reasoning.py`) with no BasePillar wrapper — making it the only pillar that couldn't participate in the agent lifecycle, receive signals from other pillars, or emit feedback to the equilibrium engine. This iteration builds the **CognitionPillar** — a full `BasePillar` subclass that wraps both the `RecursiveReasoningEngine` and `AttentionEquilibriumSystem`, completing the three-pillar architecture. The framework now supports the full end-to-end agent pipeline: Cognition (reasons, plans) → Praxis (executes) → Mneme (learns) → Cognition (informed by results).

Additionally, 63 new tests cover every aspect of the Cognition pillar: ReasoningNode data structures, RecursiveReasoningEngine behavior under all tension profiles, CognitionPillar lifecycle and signal handling, attention-reasoning integration, and edge cases.

## What Was Built

| File | Action | Lines | Description |
|------|--------|-------|-------------|
| `isonome/cognition/pillar.py` | **Created** | 413 | CognitionPillar — BasePillar wrapper for Reasoning + Attention |
| `isonome/cognition/__init__.py` | **Updated** | 44 | Exports all Cognition systems (Attention, Reasoning, Pillar) |
| `tests/test_reasoning.py` | **Created** | 521 | 63 tests: nodes, engine, tensions, pillar, attention, edge cases |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         IsonomeAgent.tick()                             │
│                                                                         │
│  1. drain_feedback() from all pillars → apply to EquilibriumEngine      │
│  2. drain_signals() from each pillar → route to targets                 │
│  3. process_queued() on each pillar → handle signals, emit feedback     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│   CognitionPillar    │ │    PraxisPillar      │ │    MnemePillar       │
│   (NEW — νοῦς)       │ │    (πρᾶξις)          │ │    (μνήμη)           │
├──────────────────────┤ ├──────────────────────┤ ├──────────────────────┤
│                      │ │                      │ │                      │
│  ReasoningEngine ────┼─┼──► plan_ready ──────►│ │                      │
│                      │ │                      │ │                      │
│                      │ │  ◄── evaluate_result │ │                      │
│                      │ │                      │ │                      │
│  AttentionSystem ────┼─┼─── pruned_chunks ────┼─┼──► import_from_attn │
│                      │ │                      │ │                      │
│  ◄── add_context ────┼─┼──────────────────────┼─┼── store              │
│                      │ │                      │ │                      │
└──────────────────────┘ └──────────────────────┘ └──────────────────────┘
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   ▼
                    ┌──────────────────────────┐
                    │   EquilibriumEngine      │
                    │   (8-axis tension system) │
                    └──────────────────────────┘
```

## CognitionPillar Design

### Signal Handling (6 signal kinds)

| Signal Kind | Direction | Behavior |
|-------------|-----------|----------|
| `reason` | From any pillar | Decomposes `{task}` into a ReasoningPlan, auto-emits `plan_ready` feedback |
| `add_context` | From Mneme/Praxis | Adds content to the attention system |
| `collect_garbage` | From any pillar | Forces GC cycle on the attention system |
| `set_priority` | From Mneme | Boosts importance tags on matching chunks |
| `evaluate_result` | From Praxis | Feeds execution outcomes back as context, emits explore/exploit feedback |

### Tick Operations (via `update_tension_profile()`)

```
Each tick:
  1. Propagate tension profile to reasoning engine
  2. Apply recency decay (3% per tick) to all attention chunks
  3. Auto-GC: if budget utilization ≥ threshold, run GC cycle
  4. Emit feedback: utilization → shallow/deep tension
  5. Emit feedback: pruned count → consolidate/prune tension
```

### Auto-GC

When `auto_gc=True` (default) and `budget.utilization ≥ gc_utilization_threshold` (default 0.80), the pillar automatically runs a garbage collection cycle. This prevents context window overflow without requiring explicit external signals.

## Core Mechanisms

### Confidence Propagation

```
C(node) = evidence_ratio(node) × 0.7 + mean(children_confidences) × 0.3

Where:
  - evidence_ratio = Σ weight_for / (Σ weight_for + Σ weight_against)
  - Terminal nodes rely purely on evidence
  - Internal nodes blend evidence with subtree quality
```

### Plan Collapse

```
Convergent mode:   return [(best_path, best_conf)]
Divergent mode:    return [(path₁, conf₁), ..., (pathₖ, confₖ)]

Path ∈ {walks from root → terminal leaves, collecting action steps}
Best path = argmax confidence × action_count
```

### Evidence Sources (3-tier)

```
Tier 1: Custom evidence_fn (if provided) — highest priority
Tier 2: Attention system context (top N chunks) — runtime context
Tier 3: Inherited evidence from parent node — structural knowledge
         ↓ discounted by 0.6× weight per inheritance hop
```

## Tension Modulation

The CognitionPillar reads all 5 Cognition-relevant axes and modulates both attention and reasoning:

| Tension Axis | Consumer | Modulation |
|-------------|----------|------------|
| `shallow_deep` | Attention GC thresholds, Reasoning max depth | Shallow → higher prune thresholds, depth 2-3; Deep → lower thresholds, depth 5-8 |
| `explore_exploit` | Reasoning branching factor | Explore → B=6, consider alternatives; Exploit → B=1, commit early |
| `divergent_convergent` | Reasoning plan output | Divergent → return multiple plans; Convergent → return single best |
| `consolidate_prune` | Attention GC feedback | Budget pressure → emit consolidate signal for Mneme |
| `specific_general` | (Future: recall precision) | Specific → exact match; General → semantic overlap |

## Test Coverage (63 new tests)

| Test Class | Count | Focus |
|-----------|-------|-------|
| `TestReasoningNode` | 10 | Data structure: evidence, confidence, tree properties |
| `TestRecursiveReasoningEngine` | 10 | Plan generation, action inference, stats accumulation |
| `TestReasoningTensionModulation` | 8 | Depth, branching, divergence under all tension extremes |
| `TestReasoningWithAttention` | 4 | Attention system integration for evidence gathering |
| `TestCognitionPillar` | 16 | Lifecycle, all signal kinds, stats, serialization |
| `TestCognitionPillarSignaling` | 3 | Plan-to-Praxis format, priority boosting, auto-GC |
| `TestReasoningEdgeCases` | 12 | Empty tasks, custom decomposers, long tasks, tool inference |

### Key Test Cases

- **test_shallow_mode_limits_depth**: At `shallow_deep = -1.0`, max depth ≤ 3
- **test_deep_mode_allows_deeper**: At `shallow_deep = 1.0`, decomposition goes deeper
- **test_auto_gc_triggers_on_high_utilization**: Auto-GC fires when utilization exceeds threshold
- **test_plan_ready_signal_contains_actions**: Output actions are Praxis-compatible (description, tool_name, risk, dependencies)
- **test_set_priority_boosts_tags**: Priority signal adds `"priority"` tag to matching chunks
- **test_custom_decomposer/evidence_fn**: Custom functions override defaults

## Design Decisions

1. **Pillar owns both systems**: The CognitionPillar creates and manages both the `RecursiveReasoningEngine` and `AttentionEquilibriumSystem`. This is the same pattern as MnemePillar (owns HierarchicalMneme) and PraxisPillar (owns ActionOrchestrator). Single-owner lifecycle management.

2. **Shared equilibrium engine**: When an `EquilibriumEngine` is passed to the pillar constructor, both subsystems share it — ensuring consistent tension reads across attention and reasoning. When not provided (standalone), a minimal engine is created internally.

3. **Auto-emit `plan_ready`**: When `reason()` completes, the pillar automatically emits `_emit_plan_ready()` feedback to the equilibrium engine. This ensures that plan quality immediately affects future tension positions without requiring the caller to manually route feedback.

4. **Auto-GC default on**: Context window management is critical — defaulting `auto_gc=True` with an 80% utilization threshold prevents silent overflow. This is conservative but safe.

5. **Signal naming follows existing conventions**: `reason`, `add_context`, `collect_garbage`, `set_priority`, `evaluate_result` match the naming patterns in MnemePillar (`store`, `recall:<query>`, `consolidate_now`) and PraxisPillar (`import_plan`, `execute_plan`, `cancel_action`).

6. **Recency decay per tick**: Each `update_tension_profile()` call applies 3% recency decay to all attention chunks. This models the natural decay of attention over time without requiring explicit GC calls.

## Why This Creates Impact

### Short-term impact
- **Completes the three-pillar architecture**: All three pillars now have full `BasePillar` wrappers that participate in the agent lifecycle. The agent loop (`tick()`) can now route signals through all three pillars.
- **End-to-end testable**: The full pipeline (Cognition plans → Praxis executes → Mneme learns) is now wire-able from a single test fixture.
- **Immediate value**: The `reason()` method produces action plans consumable by `Praxis.import_from_cognition()` — the pipeline that was designed but non-functional is now operational.

### Long-term impact
- **Foundation for metacognition**: The CognitionPillar's `evaluate_result` signal handler creates a feedback loop where execution outcomes inform future reasoning. This is the seed of self-improving agents.
- **Extensible reasoning architecture**: The `decomposer_fn` and `evidence_fn` hooks allow plugging in LLM-based or heuristic decomposition without changing the core engine.
- **Cross-pillar coherence**: The pillar emits feedback to all 5 Cognition-relevant tension axes, ensuring the entire framework responds coherently to cognitive load.

## Architecture: Full Framework State

```
isonome-framework/
├── pyproject.toml
├── isonome/
│   ├── __init__.py
│   ├── agent.py              # IsonomeAgent — tick() loop, pillar wiring
│   ├── base.py               # BasePillar — signal/feedback queue, lifecycle
│   ├── types/__init__.py     # 21 Pydantic models, enums, protocols
│   ├── equilibrium/__init__.py # EquilibriumEngine — 8-axis PID regulator
│   ├── cognition/             # νοῦς — REASON + PLAN
│   │   ├── __init__.py        # ✅ Updated — exports all 3 systems
│   │   ├── attention.py       # ✅ AttentionEquilibriumSystem — context mgmt
│   │   ├── reasoning.py       # ✅ RecursiveReasoningEngine — task → plan
│   │   └── pillar.py          # ✅ NEW — CognitionPillar wrapper
│   ├── praxis/                # πρᾶξις — EXECUTE
│   │   ├── __init__.py
│   │   ├── orchestrator.py    # ✅ ActionOrchestrator — DAG scheduler
│   │   └── pillar.py          # ✅ PraxisPillar wrapper
│   └── mneme/                 # μνήμη — REMEMBER
│       ├── __init__.py
│       ├── hierarchical.py    # ✅ HierarchicalMneme — 3-tier memory
│       └── pillar.py          # ✅ MnemePillar wrapper
└── tests/
    ├── test_agent.py          # 10 tests — agent lifecycle
    ├── test_attention.py      # 23 tests — attention system
    ├── test_equilibrium.py    # 18 tests — engine
    ├── test_mneme.py          # 50 tests — memory system
    ├── test_praxis.py         # 68 tests — execution system
    └── test_reasoning.py      # 63 tests — NEW — reasoning + pillar
```

**Total: 232 tests, 8 source modules, 4 test files**

## Cross-Pillar Pipeline Status

| Pipeline | Status | Details |
|----------|--------|---------|
| νοῦς → πρᾶξις | ✅ Complete | CognitionPillar.reason() → plan_ready → PraxisPillar.import_plan() |
| πρᾶξις → μνήμη | ✅ Complete | PraxisPillar.export_to_mneme() → MnemePillar.store() |
| πρᾶξις → νοῦς | ✅ Complete | PraxisPillar → evaluate_result signal → CognitionPillar context |
| νοῦς → μνήμη | ✅ Complete | Attention pruned chunks → import_from_attention() in Mneme |
| μνήμη → νοῦς | ✅ Complete | MnemePillar → add_context signal → CognitionPillar.attention |

## Next Iteration Candidates

1. **Metacognition Loop**: Wire `evaluate_result` feedback into a self-improving reasoning cycle where low-confidence plans trigger automatic re-reasoning with adjusted parameters.
2. **LLM-Backed Decomposition**: Implement a default `decomposer_fn` that calls an LLM for hypothesis decomposition, making the reasoning engine truly intelligent rather than heuristic-based.
3. **Attention Stewardship**: Add explicit "chunk aging" — chunks that haven't been accessed in N ticks get automatically pruned during GC.
4. **Reasoning Caching**: Cache decomposition results for similar tasks to avoid redundant reasoning — store reasoning traces in Mneme for reuse.
5. **Plan Validation**: Add a validator that checks action plans for internal consistency (no circular dependencies, all refs resolve, risk levels match tool capabilities) before emitting to Praxis.
6. **Confidence Calibration**: Track actual vs. predicted confidence over time and adjust the `_evaluate_confidence()` formula if calibration drifts.

## Files Changed

```
isonome/
├── cognition/
│   ├── __init__.py            (modified — exports)
│   └── pillar.py              (NEW — 413 lines)
tests/
└── test_reasoning.py          (NEW — 521 lines, 63 tests)
```

## Commit Stack

```
feat: CognitionPillar — BasePillar wrapper completing the three-pillar architecture
  - isonome/cognition/pillar.py: CognitionPillar with 6 signal kinds, auto-GC, tension modulation
  - isonome/cognition/__init__.py: Exports all Cognition systems (Attention, Reasoning, Pillar)
  - tests/test_reasoning.py: 63 tests covering nodes, engine, tensions, pillar, attention, edge cases
  - 232/232 tests passing (169 existing + 63 new)
```
