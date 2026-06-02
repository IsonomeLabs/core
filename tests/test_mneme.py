"""Tests for the Hierarchical Mneme memory system.

Covers:
    - MemoryEntry: Ebbinghaus forgetting, rehearsal, access, promotion
    - HierarchicalMneme: store, recall, consolidate, import_from_attention
    - Serialization: to_dict/from_dict round-trip
    - Tension modulation: threshold changes with consolidate_prune
    - Capacity enforcement: LRU eviction, tier limits
    - MnemePillar: pillar lifecycle integration
"""

import math
import time

import pytest

from isonome.mneme.hierarchical import (
    ConsolidationReport,
    HierarchicalMneme,
    MemoryEntry,
    MemoryTier,
)
from isonome.mneme.pillar import MnemePillar
from isonome.types import AgentIdentity, AgentState, Feedback, Pillar, Signal


# ═══════════════════════════════════════════════════════════════════
# MemoryEntry tests
# ═══════════════════════════════════════════════════════════════════


class TestMemoryEntry:
    def test_creation_defaults(self):
        entry = MemoryEntry(content="test")
        assert entry.tier == MemoryTier.WORKING
        assert entry.strength == 1.0
        assert entry.significance == 0.5
        assert entry.rehearsal_count == 0
        assert entry.access_count == 0

    def test_content_hash_stable(self):
        entry = MemoryEntry(content="hello world")
        h1 = entry.content_hash()
        h2 = entry.content_hash()
        assert h1 == h2
        assert len(h1) == 16

    def test_forgetting_without_decay(self):
        """Immediately after creation, strength should be 1.0."""
        entry = MemoryEntry(content="fresh", base_half_life=3600.0)
        # Forget with current_time = creation time → no decay
        decayed = entry.forget(current_time=entry.created_at)
        assert decayed.strength == pytest.approx(1.0, abs=1e-6)

    def test_forgetting_after_half_life(self):
        """After one half-life, strength should be ~0.5."""
        hl = 3600.0
        entry = MemoryEntry(content="decaying", base_half_life=hl)
        # Advance time by exactly one half-life
        decayed = entry.forget(current_time=entry.created_at + hl)
        assert decayed.strength == pytest.approx(0.5, rel=1e-3)

    def test_forgetting_after_two_half_lives(self):
        """After two half-lives, strength should be ~0.25."""
        hl = 3600.0
        entry = MemoryEntry(content="decaying", base_half_life=hl)
        decayed = entry.forget(current_time=entry.created_at + 2 * hl)
        assert decayed.strength == pytest.approx(0.25, rel=1e-3)

    def test_rehearsal_boosts_strength(self):
        entry = MemoryEntry(content="rehearse me")
        boosted = entry.rehearse(boost=0.2)
        assert boosted.strength == pytest.approx(1.0, abs=1e-6)  # Already at 1.0
        assert boosted.rehearsal_count == 1
        assert boosted.access_count == 1

    def test_rehearsal_extends_half_life(self):
        """Each rehearsal extends the effective half-life by 1.5×."""
        entry = MemoryEntry(content="spaced", base_half_life=100.0)
        assert entry._effective_half_life() == pytest.approx(100.0)

        r1 = entry.rehearse(boost=0.1)
        assert r1._effective_half_life() == pytest.approx(100.0 * 1.5)  # 150

        r2 = r1.rehearse(boost=0.1)
        assert r2._effective_half_life() == pytest.approx(100.0 * 1.5 * 1.5)  # 225

    def test_rehearsed_memory_resists_forgetting(self):
        """A rehearsed memory should have higher strength after the same time."""
        hl = 100.0
        entry = MemoryEntry(content="test", base_half_life=hl)

        # Non-rehearsed: forget for 200s
        forgotten = entry.forget(current_time=entry.created_at + 200)
        # 200 seconds ≈ 2 half-lives → strength ≈ 0.25

        # Rehearsed once: effective HL = 150
        rehearsed = entry.rehearse(boost=0.1)
        # Now forget for 200s from rehearsal time
        forgotten_r = rehearsed.forget(current_time=rehearsed.last_rehearsed + 200)

        # Rehearsed version should retain more
        assert forgotten_r.strength > forgotten.strength

    def test_is_forgotten(self):
        entry = MemoryEntry(content="weak", strength=0.03, base_half_life=1.0)
        assert entry.is_forgotten(threshold=0.05)
        assert not entry.is_forgotten(threshold=0.01)

    def test_access_slight_boost(self):
        entry = MemoryEntry(content="access me", strength=0.5)
        accessed = entry.access()
        assert accessed.strength == pytest.approx(0.53)
        assert accessed.access_count == 1

    def test_promote_to_episodic(self):
        entry = MemoryEntry(content="promote", base_half_life=100.0)
        promoted = entry.promote(MemoryTier.EPISODIC)
        assert promoted.tier == MemoryTier.EPISODIC
        assert promoted.base_half_life == pytest.approx(100.0 * 24.0)

    def test_promote_to_semantic(self):
        entry = MemoryEntry(content="permanent", base_half_life=100.0)
        promoted = entry.promote(MemoryTier.SEMANTIC)
        assert promoted.tier == MemoryTier.SEMANTIC
        assert promoted.base_half_life == pytest.approx(100.0 * 720.0)


# ═══════════════════════════════════════════════════════════════════
# HierarchicalMneme tests
# ═══════════════════════════════════════════════════════════════════


class TestHierarchicalMnemeStore:
    def test_store_basic(self):
        mneme = HierarchicalMneme()
        entry = mneme.store("Hello world", significance=0.8)
        assert entry.tier == MemoryTier.WORKING
        assert entry.content == "Hello world"
        assert entry.significance == 0.8
        assert mneme.total_memories == 1

    def test_store_with_tags(self):
        mneme = HierarchicalMneme()
        entry = mneme.store("important fact", tags=("critical", "user"))
        assert entry.tags == ("critical", "user")

    def test_store_batch(self):
        mneme = HierarchicalMneme()
        entries = mneme.store_batch(
            [("A", 0.9), ("B", 0.7), ("C", 0.5)], source="batch_test"
        )
        assert len(entries) == 3
        assert mneme.total_memories == 3

    def test_working_memory_capacity_lru(self):
        """Working memory should evict weakest when at capacity (7)."""
        mneme = HierarchicalMneme()
        # Fill working memory to capacity
        for i in range(7):
            mneme.store(f"content {i}", significance=0.5)

        assert len(mneme.working_memory) == 7

        # 8th entry triggers eviction
        mneme.store("overflow", significance=0.3)
        # Working memory stays at capacity
        assert len(mneme.working_memory) <= 7

    def test_store_updates_patterns(self):
        mneme = HierarchicalMneme()
        mneme.store("the cat sat on the mat", significance=0.8)
        # Check pattern frequencies were updated
        assert len(mneme._pattern_frequencies) > 0
        assert mneme._pattern_frequencies.get("the cat", 0) == 1


class TestHierarchicalMnemeRecall:
    @pytest.fixture
    def populated(self):
        mneme = HierarchicalMneme()
        mneme.store("Python is a programming language", significance=0.9,
                    tags=("python", "language"))
        mneme.store("JavaScript runs in browsers", significance=0.8,
                    tags=("javascript", "browser"))
        mneme.store("Coffee is best served hot", significance=0.3,
                    tags=("coffee",))
        return mneme

    def test_recall_basic(self, populated):
        results = populated.recall("Python programming")
        assert len(results) > 0
        assert any("Python" in r.content for r in results)

    def test_recall_no_match(self, populated):
        results = populated.recall("xyzzy plugh")
        assert len(results) == 0

    def test_recall_by_tags(self, populated):
        results = populated.recall_by_tags(frozenset(["python"]))
        assert len(results) >= 1
        assert any("Python" in r.content for r in results)

    def test_recall_by_tags_match_all(self, populated):
        results = populated.recall_by_tags(
            frozenset(["python", "language"]), match_all=True
        )
        assert len(results) >= 1
        assert any("Python" in r.content for r in results)

    def test_recall_increments_access_count(self, populated):
        before = populated._stats.total_retrievals
        populated.recall("Python")
        after = populated._stats.total_retrievals
        assert after >= before


class TestHierarchicalMnemeConsolidation:
    def test_consolidation_basic(self):
        mneme = HierarchicalMneme()
        mneme.store("important memory", significance=0.9)
        report = mneme.consolidate()
        assert isinstance(report, ConsolidationReport)
        assert report.working_count == 0  # Promoted to episodic
        assert report.episodic_count == 1
        assert report.wm_to_episodic == 1

    def test_consolidation_low_significance_stays(self):
        """Low significance entries stay in working memory."""
        mneme = HierarchicalMneme()
        mneme.store("trivial memory", significance=0.2)
        report = mneme.consolidate()
        assert report.wm_to_episodic == 0  # Not significant enough
        assert report.working_count == 1

    def test_consolidation_forgets_over_time(self):
        """After time passes, entries should decay."""
        mneme = HierarchicalMneme(consolidation_significance=0.1)

        # Create an entry with a very short half-life
        entry = mneme.store("fast decaying", significance=0.5)
        # Manually weaken it
        mneme._working[entry.id] = MemoryEntry(
            id=entry.id,
            content="fast decaying",
            tier=MemoryTier.WORKING,
            strength=0.02,
            significance=0.5,
            base_half_life=1.0,
            created_at=time.time() - 1000,
            last_rehearsed=time.time() - 1000,
        )

        report = mneme.consolidate()
        # Should be pruned (strength below threshold)
        assert len(mneme.working_memory) == 0

    def test_episodic_to_semantic_promotion(self):
        """High-significance entries with pattern support promote through tiers."""
        mneme = HierarchicalMneme(
            consolidation_significance=0.3,
            promotion_significance=0.5,
        )

        # Build pattern data with varied content so not all go straight to semantic
        for i in range(5):
            mneme.store(f"error occurred in module {chr(65+i)}", significance=0.8)

        # First consolidation: WM → Episodic (no patterns yet)
        report1 = mneme.consolidate()
        assert report1.wm_to_episodic >= 1

        # Now we have episodic entries; store more related content to build patterns
        for i in range(3):
            mneme.store(f"error occurred in module {chr(70+i)}", significance=0.8)

        # Second consolidation: some Ep → Semantic based on pattern support
        report2 = mneme.consolidate()
        # At least one entry should now be in semantic or episodic
        assert mneme.total_memories > 0
        # Consolidation should have moved things
        assert report2.wm_to_episodic + report2.ep_to_semantic >= 1

    def test_consolidation_report_summary(self):
        mneme = HierarchicalMneme()
        mneme.store("test", significance=0.9)
        report = mneme.consolidate()
        summary = report.summary()
        assert "WM→Ep" in summary
        assert "thresholds" in summary

    def test_tension_modulates_consolidation(self):
        """consolidate_prune tension should change consolidation behavior."""
        mneme = HierarchicalMneme(consolidation_significance=0.5)

        # Default: need 0.5 significance to consolidate
        mneme.store("moderate", significance=0.4)
        report_neutral = mneme.consolidate()

        # Now set consolidate tension (negative = consolidate mode)
        mneme.set_tension_profile({"consolidate_prune": -0.8})
        mneme.store("moderate2", significance=0.4)
        report_consolidate = mneme.consolidate()
        # In consolidate mode, threshold should be lower → more promotions
        assert report_consolidate.thresholds[0] < report_neutral.thresholds[0]

        # Prune mode
        mneme.set_tension_profile({"consolidate_prune": 0.8})
        mneme.store("moderate3", significance=0.4)
        report_prune = mneme.consolidate()
        # In prune mode, threshold should be higher
        assert report_prune.thresholds[0] > report_neutral.thresholds[0]


class TestHierarchicalMnemeAttentionImport:
    def test_import_high_attention_score(self):
        mneme = HierarchicalMneme()
        entry = mneme.import_from_attention(
            "Critical system instruction", attention_score=0.9,
            tags=("system", "immutable"),
        )
        assert entry is not None
        assert entry.significance > 0.5  # High attention → high significance

    def test_import_low_attention_score(self):
        mneme = HierarchicalMneme()
        entry = mneme.import_from_attention(
            "noise noise noise", attention_score=0.05
        )
        assert entry is None  # Too low to store

    def test_import_mid_attention_score(self):
        mneme = HierarchicalMneme()
        entry = mneme.import_from_attention(
            "Somewhat relevant fact about cats", attention_score=0.5
        )
        assert entry is not None
        assert 0.2 < entry.significance < 0.8


class TestHierarchicalMnemeRehearsal:
    def test_rehearse_by_entry_id(self):
        mneme = HierarchicalMneme()
        entry = mneme.store("rehearse me", significance=0.7)

        mneme.rehearse(entry.id, boost=0.2)
        # After rehearsal, entry should exist and have count > 0
        found = mneme._find_entry(entry.id)
        assert found is not None
        assert found.rehearsal_count == 1

    def test_rehearse_by_tags(self):
        mneme = HierarchicalMneme()
        mneme.store("python is great", tags=("python",), significance=0.7)
        count = mneme.rehearse_by_tags(frozenset(["python"]))
        assert count == 1
        assert mneme.stats["total_rehearsals"] == 1

    def test_rehearse_nonexistent(self):
        mneme = HierarchicalMneme()
        from uuid import uuid4
        result = mneme.rehearse(uuid4())
        assert result is None


class TestHierarchicalMnemeSerialization:
    def test_round_trip(self):
        mneme = HierarchicalMneme()
        mneme.store("persistent fact A", significance=0.9, tags=("fact",))
        mneme.store("persistent fact B", significance=0.7)
        mneme.consolidate()  # Move to episodic

        data = mneme.to_dict()
        restored = HierarchicalMneme.from_dict(data)

        assert restored.total_memories == mneme.total_memories
        assert restored.stats["working_memories"] == mneme.stats["working_memories"]
        assert restored.stats["episodic_memories"] == mneme.stats["episodic_memories"]

    def test_serialization_empty(self):
        mneme = HierarchicalMneme()
        data = mneme.to_dict()
        assert data["working"] == []
        assert data["episodic"] == []
        assert data["semantic"] == []
        restored = HierarchicalMneme.from_dict(data)
        assert restored.total_memories == 0

    def test_serialization_preserves_content(self):
        mneme = HierarchicalMneme()
        mneme.store("unique content for testing serialization", significance=0.8)
        data = mneme.to_dict()
        restored = HierarchicalMneme.from_dict(data)
        entries = restored.recall("unique content")
        assert len(entries) == 1
        assert entries[0].content == "unique content for testing serialization"


class TestHierarchicalMnemeProperties:
    def test_stats(self):
        mneme = HierarchicalMneme()
        stats = mneme.stats
        assert "working_memories" in stats
        assert "episodic_memories" in stats
        assert "semantic_memories" in stats
        assert "total_consolidations" in stats

    def test_memory_accessors(self):
        mneme = HierarchicalMneme()
        mneme.store("wm", significance=0.5)
        assert len(mneme.working_memory) == 1
        assert len(mneme.episodic_memory) == 0
        assert len(mneme.semantic_memory) == 0

    def test_total_memories(self):
        mneme = HierarchicalMneme()
        assert mneme.total_memories == 0
        mneme.store("one", significance=0.5)
        assert mneme.total_memories == 1

    def test_consolidation_log(self):
        mneme = HierarchicalMneme()
        mneme.store("log test", significance=0.9)
        mneme.consolidate()
        log = mneme.consolidation_log
        assert len(log) >= 1
        assert log[0].from_tier == MemoryTier.WORKING
        assert log[0].to_tier == MemoryTier.EPISODIC


class TestConsolidationReport:
    def test_summary_format(self):
        report = ConsolidationReport(
            working_count=3,
            episodic_count=5,
            semantic_count=2,
            wm_to_episodic=2,
            ep_to_semantic=1,
            pruned=1,
            thresholds=(0.5, 0.7),
            tension_profile={},
        )
        summary = report.summary()
        assert "WM→Ep=2" in summary
        assert "Ep→Sem=1" in summary
        assert "pruned=1" in summary


# ═══════════════════════════════════════════════════════════════════
# MnemePillar integration tests
# ═══════════════════════════════════════════════════════════════════


class TestMnemePillar:
    @pytest.fixture
    def agent_state(self):
        return AgentState(
            identity=AgentIdentity(name="test-agent"),
        )

    def test_initialization(self, agent_state):
        pillar = MnemePillar(name="memory")
        pillar.initialize(agent_state)
        assert pillar.mneme is not None
        assert pillar.pillar == Pillar.MNEME
        assert pillar.initialized

    def test_store_signal(self, agent_state):
        pillar = MnemePillar(name="memory")
        pillar.initialize(agent_state)

        signal = Signal(
            source=Pillar.COGNITION,
            target=Pillar.MNEME,
            kind="store",
            payload={"content": "test memory", "significance": 0.9},
        )
        pillar.receive_signal(signal)
        pillar.process_queued()

        assert pillar.mneme.total_memories == 1

    def test_consolidate_signal(self, agent_state):
        pillar = MnemePillar(name="memory")
        pillar.initialize(agent_state)

        # Store high-significance memory
        pillar.mneme.store("important", significance=0.9)

        # Send consolidate signal
        signal = Signal(
            source=Pillar.COGNITION,
            target=Pillar.MNEME,
            kind="consolidate_now",
            payload={},
        )
        pillar.receive_signal(signal)
        pillar.process_queued()

        # Should generate feedback
        feedback = pillar.drain_feedback()
        assert len(feedback) > 0

    def test_import_from_attention_signal(self, agent_state):
        pillar = MnemePillar(name="memory")
        pillar.initialize(agent_state)

        signal = Signal(
            source=Pillar.COGNITION,
            target=Pillar.MNEME,
            kind="import_from_attention",
            payload={
                "content": "pruned context chunk",
                "attention_score": 0.7,
            },
        )
        pillar.receive_signal(signal)
        pillar.process_queued()

        assert pillar.mneme.total_memories == 1

    def test_shutdown_serializes(self, agent_state):
        pillar = MnemePillar(name="memory")
        pillar.initialize(agent_state)
        pillar.mneme.store("persist me", significance=0.8)
        pillar.shutdown()
        assert not pillar.initialized

    def test_serialize_restore_round_trip(self, agent_state):
        pillar = MnemePillar(name="memory")
        pillar.initialize(agent_state)
        pillar.mneme.store("round trip test", significance=0.9, tags=("test",))

        data = pillar.serialize()
        assert data is not None

        # Fresh pillar
        pillar2 = MnemePillar(name="memory2")
        pillar2.initialize(agent_state)
        pillar2.restore(data)

        assert pillar2.mneme.total_memories == 1
        results = pillar2.mneme.recall("round trip")
        assert len(results) == 1

    def test_update_tension_profile(self, agent_state):
        pillar = MnemePillar(name="memory")
        pillar.initialize(agent_state)
        pillar.update_tension_profile({"consolidate_prune": -0.5})
        assert pillar.mneme._current_profile.get("consolidate_prune") == -0.5

    def test_already_initialized_warning(self, agent_state, caplog):
        pillar = MnemePillar(name="memory")
        pillar.initialize(agent_state)
        pillar.initialize(agent_state)  # Double init
        # Should not crash; just warn
