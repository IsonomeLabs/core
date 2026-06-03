"""MnemePillar — BasePillar wrapper for the HierarchicalMneme system.

Integrates the memory system into the agent lifecycle:
    - initialize: Creates the HierarchicalMneme instance
    - tick: Runs consolidation cycle, emits feedback about memory pressure
    - signals: Handles 'store', 'recall:<query>', 'consolidate_now', 'import_from_attention'
    - shutdown: Serializes memory state (can be saved for cross-session persistence)
"""

from __future__ import annotations

import logging
from typing import Any

from isonome.base import BasePillar
from isonome.mneme.hierarchical import HierarchicalMneme
from isonome.types import (
    AgentState,
    Feedback,
    Pillar,
    Signal,
)

logger = logging.getLogger(__name__)


class MnemePillar(BasePillar):
    """The Mneme pillar — wraps HierarchicalMneme for agent integration.

    On each tick (via process_queued → _on_signal), the pillar:
    1. Updates the tension profile from the agent's equilibrium engine
       (done externally via set_tension_profile())
    2. Processes any incoming signals (store, recall, etc.)
    3. During consolidate cycles, emits Feedback to the equilibrium engine
       about memory pressure levels.

    Usage:
        mneme = MnemePillar(name="memory")
        # Set tension profile each tick:
        mneme.mneme.set_tension_profile(agent.get_tension_profile())
        # Access memory directly:
        mneme.mneme.store("important fact", significance=0.9)
        results = mneme.mneme.recall("fact")
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        consolidation_significance: float | None = None,
        promotion_significance: float | None = None,
    ):
        super().__init__(name=name)
        self.mneme: HierarchicalMneme | None = None
        self._cons_sig = consolidation_significance
        self._prom_sig = promotion_significance

    # ── Abstract interface ──────────────────────────────────────

    @property
    def pillar(self) -> Pillar:
        return Pillar.MNEME

    def _on_initialize(self, state: AgentState) -> None:
        """Create the HierarchicalMneme system."""
        self.mneme = HierarchicalMneme(
            consolidation_significance=self._cons_sig,
            promotion_significance=self._prom_sig,
        )
        # Set initial tension profile from agent state
        if state.tensions is not None:
            profile = {}
            for axis in state.tensions.axes:
                profile[axis.id] = axis.position
            self.mneme.set_tension_profile(profile)
        logger.info(f"{self.name}: HierarchicalMneme initialized")

    def _on_signal(self, signal: Signal) -> None:
        """Handle incoming signals from other pillars.

        Supported signal kinds:
            - 'store': Store content. payload: {content, significance?, tags?}
            - 'recall:<query>': Recall memories matching query.
            - 'consolidate_now': Force a consolidation cycle.
            - 'import_from_attention': Import pruned attention chunk.
              payload: {content, attention_score, tags?}
            - 'rehearse': Rehearse a specific memory. payload: {entry_id}
            - 'rehearse_by_tags': Rehearse all with matching tags. payload: {tags}
        """
        if self.mneme is None:
            logger.warning(f"{self.name}: not initialized, ignoring signal")
            return

        kind = signal.kind
        payload = signal.payload

        try:
            if kind == "store":
                content = payload.get("content", "")
                significance = float(payload.get("significance", 0.5))
                tags = tuple(payload.get("tags", ()))
                self.mneme.store(
                    content,
                    significance=significance,
                    tags=tags,
                    source=f"signal:{signal.source.value}",
                )

            elif kind == "consolidate_now":
                report = self.mneme.consolidate()
                # Emit feedback about memory pressure
                self._emit_memory_pressure_feedback(report)

            elif kind == "import_from_attention":
                content = payload.get("content", "")
                attention_score = float(payload.get("attention_score", 0.5))
                tags = tuple(payload.get("tags", ()))
                self.mneme.import_from_attention(
                    content, attention_score, tags=tags
                )

            elif kind == "rehearse":
                from uuid import UUID
                entry_id = UUID(payload.get("entry_id", ""))
                self.mneme.rehearse(entry_id)

            elif kind == "rehearse_by_tags":
                tags = frozenset(payload.get("tags", []))
                self.mneme.rehearse_by_tags(tags)

            else:
                logger.debug(f"{self.name}: unknown signal kind '{kind}'")

        except Exception:
            logger.exception(f"{self.name}: error handling signal {kind}")

    def _on_shutdown(self) -> None:
        """Serialize memory state for potential cross-session persistence."""
        if self.mneme is not None:
            try:
                state = self.mneme.to_dict()
                logger.info(
                    f"{self.name}: shutting down — "
                    f"{self.mneme.total_memories} memories persisted "
                    f"(WM={len(state.get('working',[]))}, "
                    f"Ep={len(state.get('episodic',[]))}, "
                    f"Sem={len(state.get('semantic',[]))})"
                )
            except Exception:
                logger.exception(f"{self.name}: error serializing state")

    # ── Equilibrium pull integration ──────────────────────────────

    def _on_equilibrium_sync(self, view) -> None:
        """Auto-sync tension state from the equilibrium view.

        When bound to an engine, this is called automatically at the
        start of each process_queued() tick. It replaces the need
        for external update_tension_profile() calls.

        Applies the view's all_positions to the mneme system and
        runs a light consolidation cycle. Also reads cross-pillar
        influence: when Praxis is failing, reduce memory pruning
        to preserve potentially-useful context.
        """
        if self.mneme is not None:
            self.mneme.set_tension_profile(view.all_positions)
            # Run a light consolidation each tick for gradual decay
            self.mneme.consolidate()

        # Cross-pillar modulation: if Praxis is in safe mode
        # (low autonomy_safety), reduce pruning aggressiveness
        # to preserve context that might help understand failures
        autonomy_safety = view.cross_axes.get("autonomy_safety", 0.0)
        if autonomy_safety < -0.5 and self.mneme is not None:
            # Reduce pruning: lower the consolidation significance
            # threshold so more memories survive
            try:
                self.mneme.set_calibration_state(
                    ece=0.3,  # Moderate miscalibration signal
                    bias=0.0,
                    is_overconfident=False,
                    is_underconfident=False,
                    total_predictions=20,  # Enough to activate
                )
            except Exception:
                logger.debug(f"{self.name}: could not apply cross-pillar modulation")

    # ── Feedback ──────────────────────────────────────────────────

    def _emit_memory_pressure_feedback(self, report) -> None:
        """Emit feedback about memory system pressure to equilibrium engine.

        High memory pressure → push toward pruning
        Consolidations happening → push toward consolidation (keep going)
        """
        total = report.working_count + report.episodic_count + report.semantic_count

        # Memory pressure: high total → need to prune
        if total > 500:
            pressure_signal = 0.3  # Push toward prune
        elif total > 200:
            pressure_signal = 0.1
        else:
            pressure_signal = -0.1  # Slight consolidate push

        if report.wm_to_episodic > 0 and report.ep_to_semantic > 0:
            # Consolidation is active and productive — reinforce it
            pressure_signal -= 0.1

        self.emit_feedback(
            Feedback(
                source=self.pillar,
                tension_axis_id="consolidate_prune",
                signal=max(-1.0, min(1.0, pressure_signal)),
                confidence=0.7,
                reason=f"memory pressure: {total} total, "
                       f"{report.wm_to_episodic} wm→ep, {report.ep_to_semantic} ep→sem",
            )
        )

    # ── Convenience methods ────────────────────────────────────────

    def update_tension_profile(self, profile: dict) -> None:
        """Update the mneme's tension profile (call each tick)."""
        if self.mneme is not None:
            self.mneme.set_tension_profile(profile)
            # Run a light consolidation each tick for gradual decay
            self.mneme.consolidate()

    def update_calibration(
        self,
        ece: float,
        bias: float,
        is_overconfident: bool,
        is_underconfident: bool,
        total_predictions: int,
    ) -> None:
        """Push calibration metrics from the reasoning engine to Mneme.

        Called by CognitionPillar.update_tension_profile() each tick.
        When calibration is poor, Mneme consolidates more cautiously
        and prunes less aggressively — the memory system trusts its
        own relevance judgments less.
        """
        if self.mneme is not None and total_predictions >= 10:
            self.mneme.set_calibration_state(
                ece=ece,
                bias=bias,
                is_overconfident=is_overconfident,
                is_underconfident=is_underconfident,
                total_predictions=total_predictions,
            )

    def serialize(self) -> dict | None:
        """Get the full serializable memory state."""
        if self.mneme is None:
            return None
        return self.mneme.to_dict()

    def restore(self, data: dict) -> None:
        """Restore memory from serialized state."""
        self.mneme = HierarchicalMneme.from_dict(data)
        logger.info(f"{self.name}: restored {self.mneme.total_memories} memories")
