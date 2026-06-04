"""
Tests for iter-018: Calibration-Based Rehearsal Scheduling.

Covers:
- RehearsalScheduler unit tests (interval computation, calibration modes, tension)
- HierarchicalMneme integration (get_rehearsal_candidates, rehearse_due_candidates)
- Serialization round-trip
- Edge cases (empty mneme, clamping, no calibration data)
"""

import time
import dataclasses
import pytest
from isonome.mneme.hierarchical import (
    HierarchicalMneme,
    MemoryEntry,
    MemoryTier,
    RehearsalScheduler,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def scheduler():
    """Fresh RehearsalScheduler with no calibration state."""
    return RehearsalScheduler()


@pytest.fixture
def entry():
    """A standard memory entry for testing (unfrozen for mutation)."""
    return MemoryEntry(
        id="test-1",
        content="test content",
        significance=0.5,
        tier=MemoryTier.WORKING,
        base_half_life=3600.0,
    )


def _make_entry(**kwargs):
    """Create a MemoryEntry with sensible defaults, allowing overrides."""
    defaults = dict(
        content="test",
        significance=0.5,
        tier=MemoryTier.WORKING,
        base_half_life=3600.0,
    )
    defaults.update(kwargs)
    return MemoryEntry(**defaults)


def _set_last_rehearsed(entry, t):
    """Helper to mutate last_rehearsed on a frozen dataclass."""
    return dataclasses.replace(entry, last_rehearsed=t)


def _set_rehearsal_count(entry, count):
    """Helper to mutate rehearsal_count on a frozen dataclass."""
    return dataclasses.replace(entry, rehearsal_count=count)


# =====================================================================
# RehearsalScheduler Unit Tests
# =====================================================================

class TestRehearsalSchedulerBasic:
    """Basic interval computation and clamping."""

    def test_min_interval_clamp(self, scheduler):
        """Very short half-life entries should still respect MIN_INTERVAL."""
        short_entry = _make_entry(significance=0.01, base_half_life=1.0)
        interval = scheduler.compute_next_rehearsal_interval(short_entry)
        assert interval >= RehearsalScheduler.MIN_INTERVAL

    def test_max_interval_clamp(self, scheduler):
        """Very long half-life entries should still respect MAX_INTERVAL."""
        long_entry = _make_entry(
            significance=0.99,
            tier=MemoryTier.SEMANTIC,
            base_half_life=1_000_000.0,
            rehearsal_count=50,
        )
        interval = scheduler.compute_next_rehearsal_interval(long_entry)
        assert interval <= RehearsalScheduler.MAX_INTERVAL

    def test_base_interval_proportional_to_half_life(self, scheduler):
        """Entries with longer half-lives should have longer intervals."""
        e_short = _make_entry(base_half_life=600.0)
        e_long = _make_entry(base_half_life=7200.0, tier=MemoryTier.EPISODIC)
        i_short = scheduler.compute_next_rehearsal_interval(e_short)
        i_long = scheduler.compute_next_rehearsal_interval(e_long)
        assert i_long > i_short

    def test_significance_factor(self, scheduler):
        """Higher significance -> longer interval (more stable, rehearse less)."""
        e_low = _make_entry(significance=0.1)
        e_high = _make_entry(significance=0.9)
        i_low = scheduler.compute_next_rehearsal_interval(e_low)
        i_high = scheduler.compute_next_rehearsal_interval(e_high)
        assert i_high > i_low

    def test_rehearsal_expansion(self, scheduler):
        """More rehearsals -> exponentially longer interval (spacing effect)."""
        e0 = _make_entry(rehearsal_count=0)
        e3 = _make_entry(rehearsal_count=3)
        i0 = scheduler.compute_next_rehearsal_interval(e0)
        i3 = scheduler.compute_next_rehearsal_interval(e3)
        assert i3 > i0
        ratio = i3 / i0
        assert 3.0 < ratio < 4.0  # ~1.5**3 = 3.375 (effective_hl spacing)

    def test_compute_next_rehearsal_at(self, scheduler, entry):
        """next_rehearsal_at = current_time + interval."""
        t = 1000.0
        next_at = scheduler.compute_next_rehearsal_at(entry, current_time=t)
        interval = scheduler.compute_next_rehearsal_interval(entry)
        assert abs(next_at - (t + interval)) < 0.01

    def test_is_due_for_rehearsal_old_entry(self, scheduler):
        """Old entry should be due for rehearsal."""
        old_entry = _set_last_rehearsed(_make_entry(), 0.0)
        assert scheduler.is_due_for_rehearsal(old_entry, current_time=10000.0)

    def test_is_due_for_rehearsal_not_yet(self, scheduler):
        """Recently rehearsed entry should not be due."""
        recent_entry = _set_last_rehearsed(
            _make_entry(), time.time() - 1.0
        )
        assert not scheduler.is_due_for_rehearsal(recent_entry)


class TestRehearsalSchedulerCalibration:
    """Calibration mode effects on interval computation."""

    def test_overconfident_shortens_interval(self, entry):
        """Overconfident mode should produce shorter intervals than no calibration."""
        s_none = RehearsalScheduler()
        s_over = RehearsalScheduler()
        s_over.set_calibration_state(
            ece=0.25, bias=0.3,
            is_overconfident=True, is_underconfident=False,
            total_predictions=20,
        )
        i_none = s_none.compute_next_rehearsal_interval(entry)
        i_over = s_over.compute_next_rehearsal_interval(entry)
        assert i_over < i_none

    def test_underconfident_extends_interval(self, entry):
        """Underconfident mode should produce longer intervals than no calibration."""
        s_none = RehearsalScheduler()
        s_under = RehearsalScheduler()
        s_under.set_calibration_state(
            ece=0.20, bias=-0.25,
            is_overconfident=False, is_underconfident=True,
            total_predictions=15,
        )
        i_none = s_none.compute_next_rehearsal_interval(entry)
        i_under = s_under.compute_next_rehearsal_interval(entry)
        assert i_under > i_none

    def test_well_calibrated_slightly_extends(self, entry):
        """Well-calibrated mode should slightly extend intervals."""
        s_none = RehearsalScheduler()
        s_well = RehearsalScheduler()
        s_well.set_calibration_state(
            ece=0.03, bias=0.01,
            is_overconfident=False, is_underconfident=False,
            total_predictions=30,
        )
        i_none = s_none.compute_next_rehearsal_interval(entry)
        i_well = s_well.compute_next_rehearsal_interval(entry)
        assert i_well > i_none

    def test_calibration_modifier_values(self):
        """Verify get_calibration_modifier returns expected values."""
        s = RehearsalScheduler()
        assert s.get_calibration_modifier() == 0.0

        s.set_calibration_state(ece=0.03, bias=0.0, is_overconfident=False, is_underconfident=False, total_predictions=20)
        assert s.get_calibration_modifier() == pytest.approx(0.10)

        s.set_calibration_state(ece=0.25, bias=0.3, is_overconfident=True, is_underconfident=False, total_predictions=20)
        assert s.get_calibration_modifier() == pytest.approx(-0.25)

        s.set_calibration_state(ece=0.20, bias=-0.2, is_overconfident=False, is_underconfident=True, total_predictions=20)
        assert s.get_calibration_modifier() == pytest.approx(0.20)

    def test_insufficient_predictions_no_modifier(self):
        """With < 10 predictions, no calibration modifier should apply."""
        s = RehearsalScheduler()
        s.set_calibration_state(
            ece=0.50, bias=0.5,
            is_overconfident=True, is_underconfident=False,
            total_predictions=5,
        )
        assert s.get_calibration_modifier() == 0.0
        assert not s.calibration_active


class TestRehearsalSchedulerTension:
    """Tension axis effects on interval computation."""

    def test_consolidate_shortens_interval(self, scheduler, entry):
        """Consolidate position (negative) should shorten intervals."""
        i_neutral = scheduler.compute_next_rehearsal_interval(
            entry, consolidate_prune_position=0.0,
        )
        i_consolidate = scheduler.compute_next_rehearsal_interval(
            entry, consolidate_prune_position=-0.8,
        )
        assert i_consolidate < i_neutral

    def test_prune_extends_interval(self, scheduler, entry):
        """Prune position (positive) should extend intervals."""
        i_neutral = scheduler.compute_next_rehearsal_interval(
            entry, consolidate_prune_position=0.0,
        )
        i_prune = scheduler.compute_next_rehearsal_interval(
            entry, consolidate_prune_position=0.8,
        )
        assert i_prune > i_neutral

    def test_combined_calib_and_tension(self, entry):
        """Calibration and tension modifiers compose multiplicatively."""
        s_base = RehearsalScheduler()
        s_over = RehearsalScheduler()
        s_over.set_calibration_state(
            ece=0.25, bias=0.3,
            is_overconfident=True, is_underconfident=False,
            total_predictions=20,
        )
        i = s_over.compute_next_rehearsal_interval(
            entry, consolidate_prune_position=-0.8,
        )
        i_base = s_base.compute_next_rehearsal_interval(
            entry, consolidate_prune_position=0.0,
        )
        assert i < i_base * 0.8


# =====================================================================
# HierarchicalMneme Integration Tests
# =====================================================================

class TestMnemeGetRehearsalCandidates:
    """Integration: get_rehearsal_candidates on HierarchicalMneme."""

    def test_empty_mneme_returns_empty(self):
        """No entries -> no candidates."""
        m = HierarchicalMneme()
        candidates = m.get_rehearsal_candidates(current_time=time.time() + 1e6)
        assert candidates == []

    def test_stale_entries_are_due(self):
        """Entries with very old last_rehearsed should be due."""
        m = HierarchicalMneme()
        m.store("alpha", significance=0.7)
        m.store("beta", significance=0.3)
        # Force entries to have very old last_rehearsed using dataclasses.replace
        for tier in [m._working, m._episodic, m._semantic]:
            keys = list(tier.keys())
            for k in keys:
                old = tier[k]
                tier[k] = dataclasses.replace(old, last_rehearsed=0.0)

        candidates = m.get_rehearsal_candidates(current_time=time.time())
        assert len(candidates) > 0

    def test_recent_entries_not_due(self):
        """Entries just stored should not be immediately due."""
        m = HierarchicalMneme()
        m.store("alpha", significance=0.7)
        candidates = m.get_rehearsal_candidates()
        assert candidates == []

    def test_tier_filter(self):
        """tier_filter should restrict candidates to specified tier."""
        m = HierarchicalMneme()
        m.store("alpha", significance=0.7)
        # Force working entries to be stale
        for k in list(m._working.keys()):
            old = m._working[k]
            m._working[k] = dataclasses.replace(old, last_rehearsed=0.0)

        candidates = m.get_rehearsal_candidates(
            current_time=time.time(),
            tier_filter=MemoryTier.WORKING,
        )
        assert all(e.tier == MemoryTier.WORKING for e in candidates)

    def test_max_candidates_limit(self):
        """max_candidates should limit the number of results."""
        m = HierarchicalMneme()
        for i in range(5):
            m.store(f"entry-{i}", significance=0.5)
        for tier in [m._working, m._episodic, m._semantic]:
            for k in list(tier.keys()):
                old = tier[k]
                tier[k] = dataclasses.replace(old, last_rehearsed=0.0)

        candidates = m.get_rehearsal_candidates(
            current_time=time.time(),
            max_candidates=2,
        )
        assert len(candidates) <= 2

    def test_urgency_sorting(self):
        """Most overdue entries should come first."""
        m = HierarchicalMneme()
        m.store("old_entry", significance=0.5)
        m.store("newer_entry", significance=0.5)
        entries = list(m._working.values())
        if len(entries) >= 2:
            k0 = list(m._working.keys())[0]
            k1 = list(m._working.keys())[1]
            m._working[k0] = dataclasses.replace(entries[0], last_rehearsed=0.0)
            m._working[k1] = dataclasses.replace(entries[1], last_rehearsed=time.time() - 10.0)

            candidates = m.get_rehearsal_candidates(
                current_time=time.time() + 1e6,
            )
            if len(candidates) >= 2:
                assert candidates[0].last_rehearsed <= candidates[1].last_rehearsed


class TestMnemeRehearseDueCandidates:
    """Integration: rehearse_due_candidates on HierarchicalMneme."""

    def test_rehearse_due_count(self):
        """Should rehearse all due entries and return count."""
        m = HierarchicalMneme()
        m.store("alpha", significance=0.7)
        for k in list(m._working.keys()):
            old = m._working[k]
            m._working[k] = dataclasses.replace(old, last_rehearsed=0.0)

        count = m.rehearse_due_candidates(current_time=time.time())
        assert count > 0

    def test_rehearsed_entries_no_longer_due(self):
        """After rehearsing, entries should not be immediately due again."""
        m = HierarchicalMneme()
        m.store("alpha", significance=0.7)
        for k in list(m._working.keys()):
            old = m._working[k]
            m._working[k] = dataclasses.replace(old, last_rehearsed=0.0)

        m.rehearse_due_candidates(current_time=time.time())
        candidates = m.get_rehearsal_candidates(current_time=time.time())
        assert candidates == []


class TestRehearsalSchedulerSerialization:
    """Round-trip serialization of RehearsalScheduler."""

    def test_round_trip(self):
        """to_dict -> from_dict should preserve all state."""
        s = RehearsalScheduler(min_interval=60.0, max_interval=7200.0)
        s.set_calibration_state(
            ece=0.15, bias=0.1,
            is_overconfident=True, is_underconfident=False,
            total_predictions=25,
        )
        data = s.to_dict()
        s2 = RehearsalScheduler.from_dict(data)
        assert s2._min_interval == 60.0
        assert s2._max_interval == 7200.0
        assert s2._calibration_ece == pytest.approx(0.15)
        assert s2._calibration_bias == pytest.approx(0.1)
        assert s2._calibration_overconfident is True
        assert s2._calibration_underconfident is False
        assert s2._calibration_total_predictions == 25

    def test_mneme_round_trip(self):
        """Full mneme serialization should include rehearsal_scheduler."""
        m = HierarchicalMneme()
        m.store("test", significance=0.5)
        m.set_calibration_state(
            ece=0.10, bias=-0.05,
            is_overconfident=False, is_underconfident=True,
            total_predictions=15,
        )
        data = m.to_dict()
        assert "rehearsal_scheduler" in data
        assert data["rehearsal_scheduler"]["calibration_ece"] == pytest.approx(0.10)

        m2 = HierarchicalMneme.from_dict(data)
        assert m2._rehearsal_scheduler._calibration_ece == pytest.approx(0.10)
        assert m2._rehearsal_scheduler._calibration_underconfident is True

    def test_mneme_round_trip_no_scheduler(self):
        """Deserializing data without rehearsal_scheduler should work."""
        m = HierarchicalMneme()
        m.store("test", significance=0.5)
        data = m.to_dict()
        del data["rehearsal_scheduler"]
        m2 = HierarchicalMneme.from_dict(data)
        assert isinstance(m2._rehearsal_scheduler, RehearsalScheduler)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
