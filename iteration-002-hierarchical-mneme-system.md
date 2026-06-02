# isonome-framework — Iteration 002: Hierarchical Mneme System

**Date:** 2026-06-01
**Cron Job:** Hourly incremental improvement
**Iteration:** 002 — Second Iteration
**Change Type:** Pillar 3 (Mneme / μνήμη) — Full Hierarchical Memory System
**Tests:** 101/101 passing (50 new)

---

## Summary

Built the **Hierarchical Mneme System** — the μνήμη (memory) pillar of isonome-framework. This is a three-tier memory architecture (Working → Episodic → Semantic) governed by Ebbinghaus forgetting curves, spaced-repetition rehearsal, significance-gated consolidation, and tension-modulated thresholds. It activates the previously dormant `consolidate_prune` and `specific_general` equilibrium axes, and establishes the **νοῦς → μνήμη cross-pillar pipeline** via `import_from_attention()`.

This is the highest-impact next step because:
1. The attention system already prunes context — now those pruned chunks are evaluated for long-term storage instead of being discarded
2. Memory is the foundation for agents that **learn across sessions** — without it, every session starts from scratch
3. The equilibrium engine had Mneme tension axes with no consumer — this closes that gap

## What Was Built

### Files Created

| File | LOC | Description |
|------|-----|-------------|
| `isonome/mneme/hierarchical.py` | 1,086 | Core three-tier memory system |
| `isonome/mneme/pillar.py` | 212 | BasePillar wrapper for agent integration |
| `isonome/mneme/__init__.py` | — | Updated with full public API |
| `tests/test_mneme.py` | 443 | 50 comprehensive tests |

### Files Modified

| File | Change |
|------|--------|
| `isonome/types/__init__.py` | No changes needed — type system was already complete |

### Architecture: Three-Tier Memory

```
  Working Memory       Episodic Memory       Semantic Memory
  ─────────────        ───────────────       ───────────────
  Capacity: 7          Capacity: 1,000       Capacity: 10,000
  Lifetime: minutes    Lifetime: hours–days   Lifetime: persistent
  Detail: verbatim     Detail: timestamped    Detail: abstracted
  Decay: fast (1h HL)  Decay: medium (24h HL) Decay: near-permanent

      │                      │                       │
      ▼                      ▼                       ▼
  ┌─────────┐         ┌───────────┐          ┌────────────┐
  │  LRU    │  sig.   │ timestamp │ pattern  │  abstract   │
  │ bounded │ ──────► │  records  │ ───────► │  knowledge  │
  │  store  │  gate   │           │  gate    │             │
  └─────────┘         └───────────┘          └────────────┘
       │                    │                       │
       └────────────────────┴───────────────────────┘
                     Ebbinghaus Forgetting
                     R(t) = e^(-t·ln(2)/S)
```

### Core Mechanisms

#### 1. Ebbinghaus Forgetting Curves

Every `MemoryEntry` has a `base_half_life` and `rehearsal_count`. The effective half-life is:

```
S_effective = base_half_life × 1.5^rehearsal_count
```

The forgetting function:

```
R(t) = e^(-t·ln(2) / S_effective)
```

After one half-life, strength drops to 50%. After two, to 25%. Each rehearsal extends the half-life by 1.5× (the spacing effect), making well-rehearsed memories dramatically more durable.

| Rehearsals | Effective HL multiplier |
|-----------|------------------------|
| 0         | 1.0×                   |
| 1         | 1.5×                   |
| 2         | 2.25×                  |
| 3         | 3.375×                 |
| 5         | 7.59×                  |

#### 2. Tier Promotion

| Tier promotion   | Gate                          | Half-life multiplier |
|------------------|-------------------------------|---------------------|
| WM → Episodic    | significance ≥ cons_threshold  | 24×                 |
| Episodic → Semantic | significance ≥ prom_threshold + pattern support | 720× |

Pattern support is computed via n-gram frequency overlap. An episodic entry has "pattern support" if ≥30% of its bigrams and trigrams appear ≥3 times across the memory system.

#### 3. Tension Modulation

The `consolidate_prune` and `specific_general` tension axes modulate thresholds:

```
consolidate < 0 (Consolidate mode):
    cons_threshold ↓  →  more WM → Episodic promotions
    prom_threshold ↓  →  more Ep → Semantic promotions

consolidate > 0 (Prune mode):
    cons_threshold ↑  →  fewer promotions, more eviction
    prom_threshold ↑  →  conservative semantic consolidation

specific_general < 0 (Specific):
    recall() boosts exact token matches (Jaccard × 1.5)

specific_general > 0 (General):
    recall() weights semantic overlap more heavily
    consolidation thresholds lower slightly (more abstraction)
```

#### 4. νοῦς → μνήμη Cross-Pillar Pipeline

```python
# When the Attention Equilibrium System prunes a chunk:
mneme.import_from_attention(
    content="<pruned chunk>",
    attention_score=0.65,       # from GarbageCollectionReport
    tags=("system", "critical"),
)
# → significance = sigmoid(attention_score)
# → if significance ≥ 0.15: store in WM
# → otherwise: discard as noise
```

The sigmoid mapping from attention score to significance:
```
sig(score) = 1 / (1 + e^(-8 × (score - 0.35)))
```
- Score 0.05 → significance ~0.08 → rejected (noise)
- Score 0.50 → significance ~0.77 → stored
- Score 0.90 → significance ~0.99 → strongly stored

#### 5. Relevance-Based Recall

`recall(query)` uses multi-strategy scoring:

```
relevance(entry, query) = (
    0.6 × Jaccard(entry_tokens, query_tokens) +
    0.4 × tag_overlap
) × entry.strength
```

Tension-modulated:
- `specific_general < 0` → Jaccard weight boosted by 1.5× (exact matching)
- `specific_general > 0` → standard weighting (semantic flexibility)

Results are sorted by relevance and the top-N are returned, each receiving an `access()` boost (retrieval practice effect, +0.03 strength).

### MemoryEntry Lifecycle

```
Creation:        store("fact", significance=0.8)
                       │
                       ▼
               Working Memory (strength=1.0, half_life=1h)
                       │
         ┌─────────────┼─────────────┐
         │             │             │
    rehearse()    consolidate()   forget()
         │             │             │
    strength ↑    significance    strength ↓
    HL extended   gate check      eventual eviction
                       │
         ┌─────────────┴─────────────┐
         │                           │
    sig ≥ threshold             sig < threshold
         │                           │
    Episodic (24h HL)           stays in WM
         │                    (decays naturally)
    pattern support?
         │
    ┌────┴────┐
    │         │
   Yes       No
    │         │
 Semantic   stays in
 (720h HL)  Episodic
```

### MnemePillar Integration

The `MnemePillar` wraps `HierarchicalMneme` as a `BasePillar`:

- **initialize**: Creates `HierarchicalMneme`, reads initial tension profile
- **on_signal**: Handles `store`, `consolidate_now`, `import_from_attention`, `rehearse`, `rehearse_by_tags`
- **tick processing**: `update_tension_profile()` + light `consolidate()` for gradual decay
- **shutdown**: Serializes memory state for cross-session persistence
- **feedback**: Emits `Feedback` on `consolidate_prune` axis based on memory pressure

Memory pressure feedback:
```
total > 500  →  signal +0.3 (push toward prune)
total > 200  →  signal +0.1
total < 200  →  signal −0.1 (push toward consolidate)
active consolidations  →  additional −0.1 (reinforce if productive)
```

### Serialization

Full `to_dict()` / `from_dict()` round-trip support for cross-session persistence. Preserves:
- All three tier contents with entry metadata
- N-gram pattern frequencies (for semantic promotion continuity)
- Stats reconstruction on restore

## Test Coverage

| Test Class | Tests | What It Covers |
|-----------|-------|----------------|
| TestMemoryEntry | 12 | Forgetting curves, rehearsal half-life, access boost, promotion tier multipliers, strength bounds |
| TestHierarchicalMnemeStore | 5 | Basic store, batch store, tags, LRU eviction at capacity, pattern frequency updates |
| TestHierarchicalMnemeRecall | 5 | Token-based recall, no-match, tag recall, match-all, access count increment |
| TestHierarchicalMnemeConsolidation | 6 | WM→Ep promotion, low-sig retention, forgetting+pruning, Ep→Sem with patterns, tension modulation, report format |
| TestHierarchicalMnemeAttentionImport | 3 | High score → significance mapping, low score rejection, mid score |
| TestHierarchicalMnemeRehearsal | 3 | By entry ID, by tags, nonexistent entry |
| TestHierarchicalMnemeSerialization | 3 | Round-trip, empty state, content preservation |
| TestHierarchicalMnemeProperties | 4 | Stats dict, tier accessors, total count, consolidation log |
| TestConsolidationReport | 1 | Summary string format |
| TestMnemePillar | 8 | Init, store signal, consolidate signal, attention import signal, shutdown serialization, serialize/restore round-trip, tension profile update, double-init guard |
| **Total new** | **50** | |
| **Total all** | **101** | **All passing** ✅ |

## Design Decisions

1. **Single-pass consolidation can skip tiers**: WM entries with high significance AND pattern support promote directly to Semantic in one cycle. This models "aha moments" — when a new fact immediately connects to existing knowledge.

2. **Base half-life stays fixed; rehearsal_count extends decay curve**: `base_half_life` is never mutated by `rehearse()` — only `rehearsal_count` increments. The effective half-life `base × 1.5^count` is computed in `_effective_half_life()`. This prevents double-multiplication bugs and keeps the base value interpretable.

3. **Sigmoid attention→significance mapping** (not linear): The steep sigmoid centered at 0.35 means low-attention items are aggressively rejected as noise, mid-tier items get moderate significance, and high-attention items saturate near 1.0. This prevents the memory from filling with marginal content.

4. **LRU eviction with promotion check**: When WM evicts (at capacity 7), the weakest entry is evaluated for episodic promotion before being discarded. This means even "evicted" items can survive if they're significant enough.

5. **Pattern support uses n-gram overlap**: Bigrams and trigrams from stored content are tracked as frequency counts. An episodic entry needs ≥30% of its n-grams to appear at least `pattern_count_threshold` (default 3) times across the system before promoting to semantic. This prevents single-occurrence trivia from becoming "knowledge."

6. **Consolidation threshold direction**: `consolidate_prune < 0` (Consolidate mode) → thresholds go DOWN, allowing more promotion. `consolidate_prune > 0` (Prune mode) → thresholds go UP, restricting promotion. This matches the intuitive semantics: "consolidate" means "be more open to storing," "prune" means "be more selective."

## Why This Creates Impact

### Short-Term Impact
- **νοῦς → μνήμη pipeline**: Pruned attention chunks are no longer wasted — they're evaluated for long-term storage, immediately making the attention system more valuable
- **Cross-tick learning**: Memories survive across consolidation cycles, allowing agents to reference past context
- **Working memory is bounded**: Miller's Law (7±2 items) prevents context bloat while keeping the most relevant items accessible
- **Observable memory lifecycle**: `ConsolidationReport` and `consolidation_log` provide full visibility into memory operations

### Long-Term Impact
- **Cross-session persistence**: `to_dict()`/`from_dict()` enable agents that remember across restarts
- **Self-improving consolidation**: The tension feedback loop adjusts consolidation aggressiveness based on memory pressure, creating a self-tuning system
- **Pattern extraction**: N-gram statistics build a learned model of recurring patterns, enabling semantic abstraction over time
- **Foundation for agent specialization**: Different agents can share serialized memory state (inheritance of experience)

## Architecture: Full Framework State After Iteration 002

```
                    ┌──────────────────────────┐
                    │    Equilibrium Engine     │
                    │  ┌────────────────────┐   │
                    │  │ explore_exploit    │   │
                    │  │ shallow_deep       │   │
                    │  │ divergent_convergent│  │
                    │  │ autonomy_safety    │   │
                    │  │ sequential_parallel│   │
                    │  │ verify_execute     │   │
                    │  │ consolidate_prune  │◄──┼─── Feedback from Mneme
                    │  │ specific_general   │   │
                    │  └────────┬───────────┘   │
                    └───────────┼───────────────┘
                                │
                    tension profile (dict)
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
    ┌─────▼─────┐        ┌──────▼──────┐       ┌─────▼─────┐
    │ Cognition  │        │   Praxis    │       │   Mneme   │
    │  (νοῦς)   │        │  (πρᾶξις)  │       │  (μνήμη)  │
    │            │        │             │       │           │
    │ ┌───────┐  │        │  [STUB]     │       │ ┌───────┐  │
    │ │Attention│ │        │             │       │ │ 3-Tier│  │
    │ │  EQ    │──┼────────┼─────────────┼──────►│ │Memory │  │
    │ │ System │  │ pruned │             │import │ │System │  │
    │ └───────┘  │ chunks │             │       │ └───────┘  │
    └────────────┘        └─────────────┘       └───────────┘

    ✅ = built     ⬜ = stub
```

## Next Iteration Candidates

1. **Praxis: Tool Orchestration with Backpressure** — Finally activate the πρᾶξις pillar with `sequential_parallel` and `verify_execute` tensions to batch/pipeline tool calls. This is the only pillar still a stub.

2. **Cross-Pillar: Attention→Memory→Attention Feedback Loop** — When memories are recalled during attention scoring, increase the mutual_info of related chunks. Creates a self-reinforcing loop where the agent "notices" things it remembers.

3. **Cognition: Delegation Planning with Complexity Estimation** — Use `divergent_convergent` tension to decide when to decompose tasks vs. execute directly.

4. **Mneme: Semantic Abstraction Engine** — Beyond n-gram pattern detection, implement actual concept extraction from episodic clusters (TF-IDF topic modeling or embedding-based clustering).

## Files Changed

```
isonome-framework/
├── isonome/
│   └── mneme/
│       ├── __init__.py          (updated — full public API)
│       ├── hierarchical.py      (NEW — 1,086 lines)
│       └── pillar.py            (NEW — 212 lines)
└── tests/
    └── test_mneme.py            (NEW — 443 lines)
```

**Total: +1,741 lines of Python across 3 new files, 1 updated.**

## Commit Stack

```
feat: implement HierarchicalMneme — 3-tier memory with Ebbinghaus decay
feat: add MnemePillar — BasePillar wrapper with tension integration
test: 50 tests for Mneme — forgetting, consolidation, serialization, pillar
```
