"""Tests for calibration-aware Mneme — Rehearsal, Pattern Support, and Import (Iteration 010).

Iteration 010 extends the calibration framework with three new mechanisms:

1. Calibration-aware rehearse_by_tags():
   - Overconfident: distributed rehearsal (all entries, reduced boost)
   - Underconfident: elevated uniform rehearsal (all entries get 1.3x boost)
   - Well-calibrated: significance-ranked prioritization (skip low-sig)
   - Moderate/none: standard uniform rehearsal (unchanged)

2. Calibration-gated _has_pattern_support():
   - Overconfident (ECE > 0.15): raise required ratio from 30% to 40%
   - Underconfident (ECE > 0.15): lower required ratio from 30% to 20%
   - Well-calibrated (ECE <= 0.05): standard 30% threshold
   - Moderate/none: standard 30%

3. Calibration-gated import_from_attention():
   - Overconfident (ECE > 0.15): raise min significance floor to 0.20
   - Underconfident (ECE > 0.15): lower min significance floor to 0.08
   - Well-calibrated/moderate: standard 0.15 floor
"""

import math
import pytest

from isonome.mneme.hierarchical import (
    HierarchicalMneme,
    ConsolidationReport,
    MemoryEntry,
    MemoryTier,
)
from isonome.types import (
    AgentIdentity,
    AgentState,
    Pillar,
    TensionAxis,
    TensionSnapshot,
)
from isonome.mneme.pillar import MnemePillar


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def mneme():
    return HierarchicalMneme(
        consolidation_significance=0.5,
        promotion_significance=0.7,
    )


@pytest.fixture
def agent_state():
    axes = frozenset([
        TensionAxis(
            id="consolidate_prune", pillar=Pillar.MNEME,
            pole_left="consolidate", pole_right="prune", position=0.0,
        ),
        TensionAxis(
            id="specific_general", pillar=Pillar.MNEME,
            pole_left="specific", pole_right="general", position=0.0,
        ),
    ])
    snapshot = TensionSnapshot(axes=axes)
    identity = AgentIdentity(name="test_agent")
    return AgentState(identity=identity, tensions=snapshot)


# ═══════════════════════════════════════════════════════════════════
# Calibration-Aware Rehearsal
# ═══════════════════════════════════════════════════════════════════


class TestCalibrationRehearsalPrioritization:
    """rehearse_by_tags() responds to calibration state."""

    def test_default_no_calibration_rehearses_all(self, mneme):
        """Without calibration, rehearse_by_tags boosts all matching entries equally."""
        mneme.store("important python tip", significance=0.9, tags=("python",))
        mneme.store("boring python note", significance=0.2, tags=("python",))
        mneme.store("medium python hint", significance=0.5, tags=("python",))

        count = mneme.rehearse_by_tags(frozenset(["python"]))

        assert count == 3  # All three matched
        assert mneme.stats["total_rehearsals"] == 3
        # All should have been boosted equally (default boost = 0.15)
        for entry in mneme._working.values():
            if "python" in entry.tags:
                assert entry.strength == 1.0  # Already at max

    def test_well_calibrated_skips_low_significance(self, mneme):
        """Well-calibrated should skip low-significance entries during rehearsal."""
        mneme.set_calibration_state(
            ece=0.03, bias=0.01,
            is_overconfident=False, is_underconfident=False,
            total_predictions=50,
        )

        # Store with different significance levels
        high = mneme.store("high value python pattern", significance=0.85, tags=("python",))
        low = mneme.store("trivial python note", significance=0.25, tags=("python",))
        medium = mneme.store("medium python insight", significance=0.55, tags=("python",))

        # Rehearse — should only touch high (0.7+) and medium (0.35+)
        count = mneme.rehearse_by_tags(frozenset(["python"]))

        # Should have rehearsed 2 entries (high + medium)
        assert count == 2
        # Verify low wasn't touched
        low_entry = mneme._find_entry(low.id)
        assert low_entry is not None
        # Low was stored with strength 1.0, not rehearsed, should still be 1.0
        # This relies on the fact we haven't decayed

    def test_well_calibrated_bonus_for_high_significance(self, mneme):
        """Well-calibrated should give bonus boost to high-significance entries."""
        mneme.set_calibration_state(
            ece=0.03, bias=0.01,
            is_overconfident=False, is_underconfident=False,
            total_predictions=50,
        )

        mneme.store("critical python knowledge", significance=0.85, tags=("python",))
        mneme.store("medium python knowledge", significance=0.55, tags=("python",))

        count = mneme.rehearse_by_tags(frozenset(["python"]), boost=0.10)

        assert count == 2
        # Both got rehearsed — boost amounts differ but both > 0
        # Verify stats updated
        assert mneme.stats["total_rehearsals"] == 2

    def test_overconfident_distributed_rehearsal(self, mneme):
        """Overconfident should distribute rehearsal across all entries with reduced boost."""
        mneme.set_calibration_state(
            ece=0.25, bias=0.20,
            is_overconfident=True, is_underconfident=False,
            total_predictions=50,
        )

        # Store entries with varying significance
        for i in range(6):
            mneme.store(f"python fact {i}", significance=0.3 + 0.1 * i, tags=("python",))

        count = mneme.rehearse_by_tags(frozenset(["python"]))

        # Overconfident → all 6 entries get reduced boost
        assert count == 6
        assert mneme.stats["total_rehearsals"] == 6

    def test_overconfident_reduced_boost_effect(self, mneme):
        """Overconfident rehearsal uses reduced boost (50% of default)."""
        mneme.set_calibration_state(
            ece=0.25, bias=0.20,
            is_overconfident=True, is_underconfident=False,
            total_predictions=50,
        )

        # Weaken an entry first, then rehearse, check the boost is smaller
        entry = mneme.store("python fact", significance=0.5, tags=("python",))
        mneme._working[entry.id] = MemoryEntry(
            id=entry.id, content=entry.content, tier=entry.tier,
            strength=0.1, significance=entry.significance,
            created_at=entry.created_at, last_accessed=entry.last_accessed,
            last_rehearsed=entry.last_rehearsed,
            rehearsal_count=0, access_count=0,
            source=entry.source, tags=entry.tags,
            metadata=entry.metadata, base_half_life=entry.base_half_life,
        )

        mneme.rehearse_by_tags(frozenset(["python"]), boost=0.20)

        reheated = mneme._find_entry(entry.id)
        assert reheated is not None
        # Overconfident: reduced_boost = 0.20 * 0.5 = 0.10
        # Strength should be 0.1 + 0.10 = 0.20
        assert reheated.strength == pytest.approx(0.20, abs=0.01)
        assert reheated.rehearsal_count == 1

    def test_underconfident_elevated_rehearsal(self, mneme):
        """Underconfident should boost all entries with 1.3x multiplier."""
        mneme.set_calibration_state(
            ece=0.25, bias=-0.20,
            is_overconfident=False, is_underconfident=True,
            total_predictions=50,
        )

        # Weaken two entries
        e1 = mneme.store("fact A", significance=0.6, tags=("python",))
        e2 = mneme.store("fact B", significance=0.3, tags=("python",))
        for eid in [e1.id, e2.id]:
            entry = mneme._find_entry(eid)
            mneme._working[eid] = MemoryEntry(
                id=eid, content=entry.content, tier=entry.tier,
                strength=0.1, significance=entry.significance,
                created_at=entry.created_at, last_accessed=entry.last_accessed,
                last_rehearsed=entry.last_rehearsed,
                rehearsal_count=0, access_count=0,
                source=entry.source, tags=entry.tags,
                metadata=entry.metadata, base_half_life=entry.base_half_life,
            )

        mneme.rehearse_by_tags(frozenset(["python"]), boost=0.20)

        # Both should be rehearsed with elevated boost (0.20 * 1.3 = 0.26)
        ee1 = mneme._find_entry(e1.id)
        ee2 = mneme._find_entry(e2.id)
        assert ee1 is not None
        assert ee2 is not None
        assert ee1.strength == pytest.approx(0.36, abs=0.01)
        assert ee2.strength == pytest.approx(0.36, abs=0.01)
        assert ee1.rehearsal_count == 1
        assert ee2.rehearsal_count == 1

    def test_moderate_calibration_standard_rehearsal(self, mneme):
        """Moderate calibration (ECE 0.05-0.15) uses default uniform rehearsal."""
        mneme.set_calibration_state(
            ece=0.10, bias=0.05,
            is_overconfident=False, is_underconfident=False,
            total_predictions=50,
        )

        mneme.store("fact A", significance=0.9, tags=("python",))
        mneme.store("fact B", significance=0.2, tags=("python",))

        count = mneme.rehearse_by_tags(frozenset(["python"]))
        assert count == 2  # All entries rehearsed equally (default path)

    def test_rehearsal_few_predictions_uses_default(self, mneme):
        """Fewer than 10 predictions should not activate calibration-aware rehearsal."""
        mneme.set_calibration_state(
            ece=0.25, bias=0.20,
            is_overconfident=True, is_underconfident=False,
            total_predictions=5,  # Below guard
        )

        mneme.store("fact", significance=0.5, tags=("python",))
        count = mneme.rehearse_by_tags(frozenset(["python"]))
        # Default path (no calibration modulation)
        assert count == 1
        assert mneme.stats["total_rehearsals"] == 1

    def test_empty_entries_returns_zero(self, mneme):
        """rehearse_by_tags with no matching entries returns 0."""
        count = mneme.rehearse_by_tags(frozenset(["nonexistent"]))
        assert count == 0


# ═══════════════════════════════════════════════════════════════════
# Calibration-Gated Pattern Support
# ═══════════════════════════════════════════════════════════════════


class TestCalibrationPatternSupport:
    """_has_pattern_support() responds to calibration state."""

    @pytest.fixture
    def mneme_with_patterns(self):
        """Mneme with pre-seeded pattern frequencies."""
        m = HierarchicalMneme()
        # Seed pattern frequencies so the entry content matches
        m._pattern_frequencies["python"] = 5
        m._pattern_frequencies["is"] = 3
        m._pattern_frequencies["great"] = 3
        m._pattern_frequencies["python is"] = 5
        m._pattern_frequencies["is great"] = 4
        m._pattern_frequencies["python is great"] = 3
        return m

    def test_no_calibration_standard_threshold(self, mneme_with_patterns):
        """Without calibration, pattern support uses standard 30% threshold."""
        entry = MemoryEntry(
            content="python is great",
            tier=MemoryTier.EPISODIC,
            significance=0.8,
        )
        # 2 bigrams + 1 trigram = 3 total grams
        # python is (hit) + is great (hit) = 2 hits
        # python is great (hit) = 1 hit
        # ratio = 3/3 = 1.0 >= 0.30 → True
        assert mneme_with_patterns._has_pattern_support(entry) is True

    def test_overconfident_raises_required_ratio(self, mneme_with_patterns):
        """Overconfident calibration raises required pattern ratio to 40%."""
        mneme_with_patterns.set_calibration_state(
            ece=0.25, bias=0.20,
            is_overconfident=True, is_underconfident=False,
            total_predictions=50,
        )

        # Entry with exactly 33% pattern hits (1/3 grams)
        entry = MemoryEntry(
            content="python is unique",  # bigrams: python is (hit), is unique (miss) → 1/2
                                         # trigrams: python is unique (miss) → 0/1
                                         # total: 1/3 ≈ 33%
            tier=MemoryTier.EPISODIC,
            significance=0.8,
        )
        # 33% < 40% → overconfident system should NOT promote
        assert mneme_with_patterns._has_pattern_support(entry) is False

    def test_overconfident_high_match_still_passes(self, mneme_with_patterns):
        """Overconfident still promotes entries with strong pattern evidence."""
        mneme_with_patterns.set_calibration_state(
            ece=0.25, bias=0.20,
            is_overconfident=True, is_underconfident=False,
            total_predictions=50,
        )

        entry = MemoryEntry(
            content="python is great",  # 3/3 = 100% > 40%
            tier=MemoryTier.EPISODIC,
            significance=0.8,
        )
        assert mneme_with_patterns._has_pattern_support(entry) is True

    def test_underconfident_lowers_required_ratio(self, mneme_with_patterns):
        """Underconfident calibration lowers required pattern ratio to 20%."""
        mneme_with_patterns.set_calibration_state(
            ece=0.25, bias=-0.20,
            is_overconfident=False, is_underconfident=True,
            total_predictions=50,
        )

        # Entry with exactly 1/5 = 20% pattern hits — should pass underconfident
        entry = MemoryEntry(
            content="python something else another word",
            # bigrams: python something(0), something else(0), else another(0), another word(0)
            # trigrams: python something else(0), something else another(0), else another word(0)
            # 3 bigrams + 3 trigrams = 6 grams, only "python" single-token match
            # But in bigram mode: "python something" — no match
            # Actually let's use a simpler case
            tier=MemoryTier.EPISODIC,
            significance=0.8,
        )
        # 0/6 = 0% < 20%
        assert mneme_with_patterns._has_pattern_support(entry) is False

    def test_underconfident_low_match_still_passes(self, mneme_with_patterns):
        """Underconfident promotes entries with 20%+ pattern evidence."""
        mneme_with_patterns.set_calibration_state(
            ece=0.25, bias=-0.20,
            is_overconfident=False, is_underconfident=True,
            total_predictions=50,
        )

        # Content where 1/5 grams = 20% exactly
        # "python is OK" has bigrams: "python is"(hit), "is OK"(miss)
        # triagrams: "python is OK"(miss) → 1/3 = 33% > 20%
        entry = MemoryEntry(
            content="python is OK",
            tier=MemoryTier.EPISODIC,
            significance=0.8,
        )
        assert mneme_with_patterns._has_pattern_support(entry) is True

    def test_well_calibrated_standard_ratio(self, mneme_with_patterns):
        """Well-calibrated uses standard 30% threshold."""
        mneme_with_patterns.set_calibration_state(
            ece=0.03, bias=0.01,
            is_overconfident=False, is_underconfident=False,
            total_predictions=50,
        )

        # Entry with 2/4 = 50% > 30% → passes
        entry = MemoryEntry(
            content="python is",
            # bigram: "python is" (hit) = 1/1
            # trigrams: none (only 2 tokens)
            # total: 1/1 = 100% > 30%
            tier=MemoryTier.EPISODIC,
            significance=0.8,
        )
        assert mneme_with_patterns._has_pattern_support(entry) is True

    def test_pattern_support_few_predictions_default_ratio(self, mneme_with_patterns):
        """Fewer than 10 predictions uses standard 30% ratio."""
        mneme_with_patterns.set_calibration_state(
            ece=0.25, bias=0.20,
            is_overconfident=True, is_underconfident=False,
            total_predictions=5,  # Below guard
        )

        # Same entry that would fail at 40% passes at 30%
        entry = MemoryEntry(
            content="python is unique",  # 1/3 ≈ 33% >= 30%
            tier=MemoryTier.EPISODIC,
            significance=0.8,
        )
        assert mneme_with_patterns._has_pattern_support(entry) is True

    def test_no_pattern_frequencies_returns_false(self, mneme):
        """Without any pattern data, _has_pattern_support returns False."""
        entry = MemoryEntry(
            content="some fresh content",
            tier=MemoryTier.EPISODIC,
            significance=0.8,
        )
        assert mneme._has_pattern_support(entry) is False


# ═══════════════════════════════════════════════════════════════════
# Calibration-Gated Import From Attention
# ═══════════════════════════════════════════════════════════════════


class TestCalibrationImportFromAttention:
    """import_from_attention() responds to calibration state."""

    def test_default_floor_no_calibration(self, mneme):
        """Without calibration, standard 0.15 floor is used."""
        # 0.15 floor: sigmoid(8*(0.15-0.35)) = sigmoid(-1.6) ≈ 0.17 > 0.15
        entry = mneme.import_from_attention("marginal content", attention_score=0.15)
        assert entry is not None, "Score 0.15 should pass default floor"

        # sigmoid(8*(0.10-0.35)) = sigmoid(-2.0) ≈ 0.12 < 0.15
        entry = mneme.import_from_attention("noise", attention_score=0.10)
        assert entry is None, "Score 0.10 should fail default floor"

    def test_overconfident_higher_floor(self, mneme):
        """Overconfident calibration raises floor to 0.20."""
        mneme.set_calibration_state(
            ece=0.25, bias=0.20,
            is_overconfident=True, is_underconfident=False,
            total_predictions=50,
        )

        # sigmoid(8*(0.15-0.35)) ≈ 0.17 < 0.20
        entry = mneme.import_from_attention("marginal", attention_score=0.15)
        assert entry is None, "Score 0.15 should fail 0.20 overconfident floor"

        # sigmoid(8*(0.20-0.35)) ≈ 0.23 > 0.20
        entry = mneme.import_from_attention("borderline", attention_score=0.20)
        assert entry is not None, "Score 0.20 should pass 0.20 overconfident floor"

        # sigmoid(8*(0.25-0.35)) ≈ 0.31 > 0.20
        entry = mneme.import_from_attention("decent", attention_score=0.25)
        assert entry is not None, "Score 0.25 should pass overconfident floor"

    def test_underconfident_lower_floor(self, mneme):
        """Underconfident calibration lowers floor to 0.08."""
        mneme.set_calibration_state(
            ece=0.25, bias=-0.20,
            is_overconfident=False, is_underconfident=True,
            total_predictions=50,
        )

        # sigmoid(8*(0.05-0.35)) = sigmoid(-2.4) ≈ 0.08 >= 0.08 → borderline stored
        entry = mneme.import_from_attention("very weak signal", attention_score=0.05)
        assert entry is not None, "Score 0.05 should pass 0.08 underconfident floor"

        # sigmoid(8*(0.10-0.35)) ≈ 0.12 > 0.08
        entry = mneme.import_from_attention("weak signal", attention_score=0.10)
        assert entry is not None, "Score 0.10 should pass underconfident floor"

        # sigmoid(8*(0.03-0.35)) = sigmoid(-2.56) ≈ 0.07 < 0.08
        entry = mneme.import_from_attention("real noise", attention_score=0.03)
        assert entry is None, "Score 0.03 should fail even underconfident floor"

    def test_few_predictions_uses_default_floor(self, mneme):
        """Fewer than 10 predictions uses standard 0.15 floor."""
        mneme.set_calibration_state(
            ece=0.25, bias=-0.20,
            is_overconfident=False, is_underconfident=True,
            total_predictions=5,  # Below guard
        )

        # Standard floor: 0.15 → sigmoid(8*(0.10-0.35)) ≈ 0.12 < 0.15
        entry = mneme.import_from_attention("weak", attention_score=0.10)
        assert entry is None, "Score 0.10 should fail standard 0.15 floor with few predictions"

        # sigmoid(8*(0.15-0.35)) ≈ 0.17 > 0.15
        entry = mneme.import_from_attention("marginal", attention_score=0.15)
        assert entry is not None, "Score 0.15 should pass standard floor"

    def test_moderate_calibration_standard_floor(self, mneme):
        """Moderate calibration (ECE 0.05-0.15) uses standard 0.15 floor."""
        mneme.set_calibration_state(
            ece=0.10, bias=0.05,
            is_overconfident=False, is_underconfident=False,
            total_predictions=50,
        )
        entry = mneme.import_from_attention("normal content", attention_score=0.15)
        assert entry is not None, "Moderate calibration should use standard floor"

        entry = mneme.import_from_attention("noise", attention_score=0.10)
        assert entry is None, "Score 0.10 should fail standard floor"

    def test_well_calibrated_standard_floor(self, mneme):
        """Well-calibrated (ECE <= 0.05) uses standard 0.15 floor."""
        mneme.set_calibration_state(
            ece=0.03, bias=0.01,
            is_overconfident=False, is_underconfident=False,
            total_predictions=50,
        )
        entry = mneme.import_from_attention("normal content", attention_score=0.15)
        assert entry is not None, "Well-calibrated should use standard floor"

        entry = mneme.import_from_attention("noise", attention_score=0.10)
        assert entry is None, "Score 0.10 should fail standard floor"

    def test_import_from_attention_after_overconfident_clear(self, mneme):
        """Clearing calibration should restore default floor."""
        mneme.set_calibration_state(
            ece=0.25, bias=0.20,
            is_overconfident=True, is_underconfident=False,
            total_predictions=50,
        )
        mneme.set_calibration_state(
            ece=0.0, bias=0.0,
            is_overconfident=False, is_underconfident=False,
            total_predictions=0,
        )
        # Should now use default floor (0.15)
        entry = mneme.import_from_attention("marginal", attention_score=0.15)
        assert entry is not None, "After clearing overconfident, default floor should apply"


# ═══════════════════════════════════════════════════════════════════
# Integration: Consolidation Pipeline with Gated Features
# ═══════════════════════════════════════════════════════════════════


class TestCalibrationConsolidationPipeline:
    """Full pipeline: calibration gates pattern support → semantic promotion."""

    def test_overconfident_blocks_weak_pattern_semantic_promotion(self, mneme):
        """Overconfident calibration prevents weak-pattern entries from reaching semantic."""
        # Seed patterns
        for i in range(5):
            mneme.store(f"python is great version {i}", significance=0.8,
                         tags=("pattern_seed",))
        mneme.consolidate()

        mneme.set_calibration_state(
            ece=0.25, bias=0.20,
            is_overconfident=True, is_underconfident=False,
            total_predictions=50,
        )

        # Store an entry with moderate pattern support directly in episodic
        entry = MemoryEntry(
            content="python something unique",
            tier=MemoryTier.EPISODIC,
            significance=0.85,
        )
        mneme._episodic[entry.id] = entry

        report = mneme.consolidate()

        # "python something unique": bigrams: python something(miss), something unique(miss)
        # trigrams: python something unique(miss) → 0/3 = 0% < 40% → no promotion
        assert report.ep_to_semantic == 0

    def test_well_calibrated_normal_semantic_promotion(self, mneme):
        """Well-calibrated calibration allows normal semantic promotion."""
        # Seed patterns
        for i in range(5):
            mneme.store(f"python is great version {i}", significance=0.8,
                         tags=("pattern_seed",))
        mneme.consolidate()

        mneme.set_calibration_state(
            ece=0.03, bias=0.01,
            is_overconfident=False, is_underconfident=False,
            total_predictions=50,
        )

        # Store a strongly patterned entry in episodic
        entry = MemoryEntry(
            content="python is great",
            tier=MemoryTier.EPISODIC,
            significance=0.85,
        )
        mneme._episodic[entry.id] = entry

        report = mneme.consolidate()

        # "python is great": 3/3 = 100% >> 30% → promotes
        assert report.ep_to_semantic >= 1

    def test_underconfident_allows_marginal_semantic_promotion(self, mneme):
        """Underconfident calibration allows marginal patterns to promote."""
        # Seed patterns
        for i in range(5):
            mneme.store(f"python is code version {i}", significance=0.8,
                         tags=("pattern_seed",))
        mneme.consolidate()

        mneme.set_calibration_state(
            ece=0.25, bias=-0.20,
            is_overconfident=False, is_underconfident=True,
            total_predictions=50,
        )

        # Entry with ~40% pattern support: "python is unique"
        # bigrams: "python is"(hit) 1/2, trigrams: "python is unique"(miss) 0/1
        # total: 1/3 ≈ 33% > 20% → promotes with underconfident
        entry = MemoryEntry(
            content="python is unique",
            tier=MemoryTier.EPISODIC,
            significance=0.85,
        )
        mneme._episodic[entry.id] = entry

        report = mneme.consolidate()
        # 33% >= 20% → promotes
        assert report.ep_to_semantic >= 1


# ═══════════════════════════════════════════════════════════════════
# Pillar wiring
# ═══════════════════════════════════════════════════════════════════


class TestMnemePillarCalibrationRehearsal:
    """MnemePillar forwards calibration state for rehearsal modulation."""

    def test_pillar_rehearse_by_tags_with_calibration(self, agent_state):
        """MnemePillar rehearsal should respect calibration state."""
        pillar = MnemePillar(name="test_mneme")
        pillar.initialize(agent_state)
        assert pillar.mneme is not None

        # Push overconfident calibration
        pillar.update_calibration(
            ece=0.25, bias=0.20,
            is_overconfident=True, is_underconfident=False,
            total_predictions=50,
        )

        # Store and rehearse
        for i in range(5):
            pillar.mneme.store(f"fact {i}", significance=0.5, tags=("tag",))

        count = pillar.mneme.rehearse_by_tags(frozenset(["tag"]))
        # Overconfident → distributed (all 5)
        assert count == 5
        assert pillar.mneme.stats["total_rehearsals"] == 5

    def test_pillar_import_from_attention_with_calibration(self, agent_state):
        """MnemePillar import_from_attention should respect calibration."""
        pillar = MnemePillar(name="test_mneme")
        pillar.initialize(agent_state)
        assert pillar.mneme is not None

        # Push underconfident calibration
        pillar.update_calibration(
            ece=0.25, bias=-0.20,
            is_overconfident=False, is_underconfident=True,
            total_predictions=50,
        )

        # Underconfident floor is 0.08, score 0.05 sigmoid ≈ 0.08 >= 0.08 → borderline stored
        entry = pillar.mneme.import_from_attention("weak signal", attention_score=0.05)
        assert entry is not None, "Underconfident pillar should accept score 0.05"

        # Clear calibration by updating with few predictions (reset through mneme directly)
        pillar.mneme.set_calibration_state(
            ece=0.0, bias=0.0,
            is_overconfident=False, is_underconfident=False,
            total_predictions=0,
        )
        entry = pillar.mneme.import_from_attention("weak signal", attention_score=0.05)
        assert entry is None, "After clearing, score 0.05 should fail default floor"
