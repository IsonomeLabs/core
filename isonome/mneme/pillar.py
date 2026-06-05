"""MnemePillar — BasePillar wrapper for the HierarchicalMneme system.

Integrates the memory system into the agent lifecycle:
    - initialize: Creates the HierarchicalMneme instance
    - tick: Runs consolidation cycle, emits feedback about memory pressure
    - signals: Handles 'store', 'recall:<query>', 'consolidate_now', 'import_from_attention'
    - shutdown: Serializes memory state (can be saved for cross-session persistence)
"""

from __future__ import annotations

import logging

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
        self._last_recall_results: list | None = None

    @property
    def last_recall_results(self) -> list | None:
        """Results from the most recent 'recall:<query>' signal."""
        return self._last_recall_results

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

            elif kind.startswith("recall:"):
                # Handle 'recall:<query>' signal - extract query from kind suffix
                query = kind[len("recall:"):]
                max_results = int(payload.get("max_results", 10))
                tier_filter = payload.get("tier_filter", None)
                results = self.mneme.recall(
                    query,
                    max_results=max_results,
                    tier_filter=tier_filter,
                )
                logger.info(
                    f"{self.name}: recalled {len(results)} memories "
                    f"for '{query}'"
                )
                # Store results for caller inspection via last_recall_results
                self._last_recall_results = results

            elif kind == "execution_results":
                # Handle Praxis → Mneme pipeline: store execution outcomes
                # as memories for cross-session learning.
                entries = payload.get("entries", [])
                stored_count = 0
                for entry in entries:
                    # Compute significance from execution outcome:
                    # - Success → high significance (confirmed patterns)
                    # - Failure → moderate significance (learning signal)
                    # - Validation score further modulates significance
                    is_success = entry.get("success", False)
                    base_sig = 0.7 if is_success else 0.4
                    val_score = entry.get("validation_score")
                    if val_score is not None and isinstance(
                        val_score, (int, float)
                    ):
                        significance = min(1.0, base_sig + val_score * 0.2)
                    else:
                        significance = base_sig

                    # Build content string from execution entry
                    desc = entry.get("description", "unknown")
                    tool = entry.get("tool_name", "unknown")
                    status = "succeeded" if is_success else "failed"
                    content = f"Action '{desc}' ({tool}) {status}"
                    error = entry.get("error")
                    if error:
                        content += f": {error}"

                    # Tags capture action metadata for recall
                    tags = (
                        "execution",
                        tool,
                        "success" if is_success else "failure",
                        f"batch-{entry.get('batch', 0)}",
                    )

                    self.mneme.store(
                        content,
                        significance=significance,
                        tags=tags,
                        source="praxis:execution_results",
                    )
                    stored_count += 1

                logger.info(
                    f"{self.name}: stored {stored_count} execution "
                    f"memories from Praxis"
                )

            else:
                logger.debug(
                    f"{self.name}: unknown signal kind '{kind}'"
                )

        except Exception:
            logger.exception(
                f"{self.name}: error handling signal {kind}"
            )

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
        """Update the mneme's tension profile (call each tick).

        When the pillar is bound to an engine, consolidation is handled
        automatically by _on_equilibrium_sync() during process_queued().
        This method only syncs the profile without re-consolidating to
        avoid double-consolidation per tick.

        For standalone use (no engine bound), consolidation is still
        triggered here since there is no auto-sync path.
        """
        if self.mneme is not None:
            self.mneme.set_tension_profile(profile)
            # Only consolidate if not bound to an engine (which already
            # handles consolidation in _on_equilibrium_sync)
            if self._engine is None:
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
        """Get the full serializable memory state.

        Includes both the mneme system state and the pillar configuration
        (consolidation_significance, promotion_significance, pillar name)
        so that restore() can faithfully reconstruct the full pillar.
        """
        if self.mneme is None:
            return None

        from isonome import SERIALIZATION_SCHEMA_VERSION

        result = self.mneme.to_dict()
        # Layer pillar config on top of mneme state
        result["_pillar_config"] = {
            "name": self.name,
            "consolidation_significance": self._cons_sig,
            "promotion_significance": self._prom_sig,
        }
        result["_schema_version"] = SERIALIZATION_SCHEMA_VERSION
        return result

    def restore(self, data: dict) -> None:
        """Restore memory from serialized data.

        Reconstructs both the HierarchicalMneme and the pillar's config
        parameters (consolidation/promotion significance thresholds).
        """
        from isonome import SERIALIZATION_SCHEMA_VERSION

        # Validate schema version for forward-compat detection
        saved_version = data.get("_schema_version", 0)
        if saved_version > SERIALIZATION_SCHEMA_VERSION:
            logger.warning(
                f"{self.name}: serialized with schema v{saved_version}, "
                f"current is v{SERIALIZATION_SCHEMA_VERSION} — "
                f"some fields may be ignored"
            )

        self.mneme = HierarchicalMneme.from_dict(data)

        # Restore pillar config if present (schema v1+)
        config = data.get("_pillar_config", {})
        if config:
            self.name = config.get("name", self.name)
            cons_sig = config.get("consolidation_significance")
            prom_sig = config.get("promotion_significance")
            # Apply restored thresholds to the reconstructed mneme
            if cons_sig is not None and self.mneme is not None:
                self._cons_sig = float(cons_sig)
                self.mneme._consolidation_significance = self._cons_sig
            if prom_sig is not None and self.mneme is not None:
                self._prom_sig = float(prom_sig)
                self.mneme._promotion_significance = self._prom_sig

        logger.info(f"{self.name}: restored {self.mneme.total_memories} memories")
