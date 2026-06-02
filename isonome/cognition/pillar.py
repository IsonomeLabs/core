"""CognitionPillar — BasePillar wrapper for the Cognition systems.

Integrates the Recursive Reasoning Engine and the Attention Equilibrium
System into the agent lifecycle:
    - initialize: Creates both Reasoning and Attention systems
    - tick: Applies recency decay, runs GC, handles signals, emits feedback
    - signals: Handles 'reason', 'add_context', 'collect_garbage',
               'set_priority', 'import_from_praxis_execution'
    - shutdown: Serializes attention state and exports pruned content to Mneme

Cross-pillar pipelines:
    - νοῦς → πρᾶξις: reason() produces plans that Praxis imports via Signal
    - πρᾶξις → νοῦς: execution results flow back as context/priorities
    - νοῦς → μνήμη: pruned attention chunks can be stored as episodic memories

Design: The pillar is self-contained — it owns both systems and manages
their lifecycle. The agent loop calls process_queued() which drains and
handles incoming signals, then drains feedback for the equilibrium engine.
"""

from __future__ import annotations

import logging
from typing import Any

from isonome.base import BasePillar
from isonome.cognition.attention import AttentionEquilibriumSystem
from isonome.cognition.reasoning import RecursiveReasoningEngine
from isonome.equilibrium import EquilibriumEngine
from isonome.types import (
    AgentState,
    Feedback,
    Pillar,
    Signal,
    TensionID,
)

logger = logging.getLogger(__name__)


class CognitionPillar(BasePillar):
    """The Cognition pillar — wraps Reasoning + Attention for agent integration.

    On each tick (via process_queued → _on_signal), the pillar:
    1. Updates the tension profile from the agent's equilibrium engine
    2. Applies recency decay to attention chunks
    3. Processes incoming signals (reason, add_context, etc.)
    4. Optionally runs a lightweight GC if budget utilization is high
    5. Emits Feedback to the equilibrium engine about context pressure
       and plan confidence

    This is the direct counterpart to MnemePillar and PraxisPillar,
    completing the three-pillar architecture.

    Usage:
        cognition = CognitionPillar(
            name="thinker",
            engine=agent.engine,  # Shared equilibrium engine
            token_capacity=128_000,
        )
        # Set tension profile each tick:
        cognition.update_tension_profile(agent.get_tension_profile())
        # Reason about a task:
        cognition.reason("analyze database and produce report")
        # Output plan is available via:
        plan = cognition.latest_plan
        # Or sent as a Signal to Praxis automatically
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        engine: EquilibriumEngine | None = None,
        token_capacity: int = 128_000,
        compress_ratio: float = 0.20,
        decomposer_fn: Any = None,
        evidence_fn: Any = None,
        auto_gc: bool = True,
        gc_utilization_threshold: float = 0.80,
        mneme_pillar: Any | None = None,
    ):
        """Initialize the Cognition pillar.

        Args:
            name: Pillar display name.
            engine: Shared EquilibriumEngine for tension-driven modulation.
            token_capacity: Max tokens for the attention budget.
            compress_ratio: Target compression ratio for compressed chunks.
            decomposer_fn: Optional custom hypothesis decomposer.
            evidence_fn: Optional custom evidence gatherer.
            auto_gc: Whether to auto-run GC on high utilization.
            gc_utilization_threshold: Auto-GC trigger threshold [0, 1].
            mneme_pillar: Optional MnemePillar reference for pushing
                calibration state on each tick.
        """
        super().__init__(name=name)
        self._engine = engine
        self._token_capacity = token_capacity
        self._compress_ratio = compress_ratio
        self._decomposer_fn = decomposer_fn
        self._evidence_fn = evidence_fn
        self._auto_gc = auto_gc
        self._gc_util_threshold = gc_utilization_threshold
        self._mneme_pillar = mneme_pillar

        # Systems — created during _on_initialize
        self.attention: AttentionEquilibriumSystem | None = None
        self.reasoning: RecursiveReasoningEngine | None = None

        # State
        self._last_plan: Any = None  # ReasoningPlan | None
        self._last_gc_report: Any = None  # GarbageCollectionReport | None
        self._context_added: int = 0
        self._tasks_reasoned: int = 0
        self._ticks_without_gc: int = 0

    # ── Abstract interface ──────────────────────────────────────

    @property
    def pillar(self) -> Pillar:
        return Pillar.COGNITION

    def _on_initialize(self, state: AgentState) -> None:
        """Create the Attention and Reasoning systems."""
        # Use the shared engine if provided, otherwise standalone
        attn_engine = self._engine
        if attn_engine is None:
            # Create a minimal engine for standalone attention
            from isonome.equilibrium import EquilibriumEngine
            attn_engine = EquilibriumEngine()

        self.attention = AttentionEquilibriumSystem(
            engine=attn_engine,
            token_capacity=self._token_capacity,
            compress_ratio=self._compress_ratio,
        )

        self.reasoning = RecursiveReasoningEngine(
            attention_system=self.attention,
            decomposer_fn=self._decomposer_fn,
            evidence_fn=self._evidence_fn,
        )

        # Set initial tension profile from agent state
        if state.tensions is not None:
            profile = {}
            for axis in state.tensions.axes:
                profile[axis.id] = axis.position
            self.reasoning.set_tension_profile(profile)

        logger.info(
            f"{self.name}: CognitionPillar initialized — "
            f"attention capacity={self._token_capacity:,} tokens"
        )

    def _on_signal(self, signal: Signal) -> None:
        """Handle incoming signals from other pillars.

        Supported signal kinds:
            - 'reason': Decompose a task into an action plan.
              payload: {task: str, context?: [str]}
              Automatically fires a 'plan_ready' Signal back to Praxis.
            - 'add_context': Add content to the attention system.
              payload: {content: str, mutual_info?: float, task_relevance?: float,
                        importance_tags?: [str]}
            - 'collect_garbage': Force a GC cycle.
              payload: {} (no extra data needed)
            - 'set_priority': Mark specific chunks as important.
              payload: {chunk_ids?: [str], content_filter?: str}
            - 'evaluate_result': Feed execution result back to adjust reasoning priors.
              payload: {success: bool, description: str, confidence?: float}
        """
        if self.reasoning is None or self.attention is None:
            logger.warning(f"{self.name}: not initialized, ignoring signal")
            return

        kind = signal.kind
        payload = signal.payload

        try:
            if kind == "reason":
                task = payload.get("task", "")
                context = payload.get("context", None)
                if task:
                    plan = self.reasoning.reason(task, initial_context=context)
                    self._last_plan = plan
                    self._tasks_reasoned += 1
                    logger.info(
                        f"{self.name}: reasoned about task — "
                        f"{plan.summary()}"
                    )
                    # Automatically emit plan_ready signal to Praxis
                    self._emit_plan_ready(plan)

            elif kind == "add_context":
                content = payload.get("content", "")
                if content:
                    chunk = self.attention.add_chunk(
                        content,
                        mutual_info=float(payload.get("mutual_info", 0.0)),
                        task_relevance=float(payload.get("task_relevance", 0.5)),
                        importance_tags=tuple(payload.get("importance_tags", ())),
                    )
                    self._context_added += 1
                    logger.debug(
                        f"{self.name}: added context chunk (id={chunk.id}, "
                        f"tokens={chunk.token_count})"
                    )

            elif kind == "collect_garbage":
                self._last_gc_report = self.attention.collect_garbage()
                self._emit_gc_feedback(self._last_gc_report)
                self._ticks_without_gc = 0
                logger.info(f"{self.name}: GC — {self._last_gc_report.summary()}")

            elif kind == "set_priority":
                # Boost specific chunks or content patterns
                # This adjusts importance tags on matching chunks
                chunk_ids = payload.get("chunk_ids", [])
                content_filter = payload.get("content_filter", "")
                boosted = 0
                for chunk in self.attention.equilibrium_chunks:
                    matches = False
                    if chunk_ids and str(chunk.id) in chunk_ids:
                        matches = True
                    elif content_filter and content_filter.lower() in chunk.content.lower():
                        matches = True
                    if matches:
                        # Re-create with boosted importance
                        from isonome.cognition.attention import AttentionChunk
                        boosted_chunk = AttentionChunk(
                            id=chunk.id,
                            content=chunk.content,
                            token_count=chunk.token_count,
                            surprisal=chunk.surprisal,
                            mutual_info=chunk.mutual_info,
                            recency=chunk.recency,
                            task_relevance=chunk.task_relevance,
                            importance_tags=chunk.importance_tags + ("priority",),
                        )
                        self.attention._chunks[chunk.id] = boosted_chunk
                        boosted += 1
                if boosted:
                    logger.info(f"{self.name}: boosted priority of {boosted} chunks")

            elif kind == "evaluate_result":
                success = payload.get("success", False)
                description = payload.get("description", "")
                confidence = float(payload.get("confidence", 0.5))
                # Feed execution outcome back as context
                outcome = (
                    f"Execution result: {'SUCCESS' if success else 'FAILURE'} — "
                    f"{description} (confidence: {confidence:.2f})"
                )
                self.attention.add_chunk(
                    outcome,
                    mutual_info=0.3 if success else 0.7,
                    task_relevance=0.8,
                    importance_tags=("execution_result",),
                )

                # ── METACOGNITIVE CALIBRATION ──
                # Record the predicted confidence vs actual outcome to
                # calibrate the reasoning engine's confidence estimates.
                # This is the learning step — the calibrator adjusts
                # evidence/child weights based on observed accuracy.
                if self.reasoning is not None:
                    try:
                        cal_result = self.reasoning.calibrate(
                            predicted_confidence=confidence,
                            actual_success=success,
                        )
                        logger.debug(
                            f"{self.name}: calibration — ECE={cal_result['ece']:.4f}, "
                            f"w_ev={cal_result['evidence_weight']:.2f}, "
                            f"w_ch={cal_result['child_weight']:.2f}, "
                            f"adjusted={cal_result['adjusted']}"
                        )
                    except Exception:
                        logger.exception(f"{self.name}: calibration error")

                # Emit feedback about plan quality
                signal_val = 0.08 if success else -0.12
                # Modulate signal strength by calibration quality
                if self.reasoning is not None and self.reasoning.calibrator.is_overconfident:
                    signal_val *= 1.3  # Stronger push toward explore when overconfident
                self.emit_feedback(
                    Feedback(
                        source=self.pillar,
                        tension_axis_id="explore_exploit",
                        signal=max(-1.0, min(1.0, signal_val)),
                        confidence=0.65,
                        reason=f"execution outcome: {'success' if success else 'failure'}",
                    )
                )

            else:
                logger.debug(f"{self.name}: unknown signal kind '{kind}'")

        except Exception:
            logger.exception(f"{self.name}: error handling signal {kind}")

    def _on_shutdown(self) -> None:
        """Export pruned chunks to Mneme and serialize state."""
        if self.attention is not None:
            try:
                # Final GC to clean up
                self._last_gc_report = self.attention.collect_garbage()
                stats = self.attention.stats
                logger.info(
                    f"{self.name}: shutting down — "
                    f"{stats['chunks_active']} active chunks, "
                    f"{stats['gc_cycles']} GC cycles, "
                    f"{stats['total_pruned']} pruned over lifetime"
                )
            except Exception:
                logger.exception(f"{self.name}: error during final GC")

        if self.reasoning is not None:
            try:
                rstats = self.reasoning.stats
                logger.info(
                    f"{self.name}: reasoning summary — "
                    f"{rstats['sessions']} sessions, "
                    f"{rstats['total_nodes']} nodes, "
                    f"{rstats['total_actions']} actions produced, "
                    f"avg confidence={rstats['avg_confidence']:.3f}"
                )
            except Exception:
                logger.exception(f"{self.name}: error reading reasoning stats")

    # ── Pillar tick operations ──────────────────────────────────

    def update_tension_profile(self, profile: dict[TensionID, float]) -> None:
        """Update tension profiles on both systems (call each tick).

        This is the key metacognitive integration point: calibration metrics
        from the reasoning engine are pushed to the attention system so that
        poor calibration triggers wider retention, slower decay, and higher
        GC thresholds — closing the attention/calibration feedback loop.
        """
        if self.reasoning is not None:
            self.reasoning.set_tension_profile(profile)

        if self.attention is not None:
            # ── Push calibration state to attention system ──
            # When the reasoning engine is poorly calibrated, the attention
            # system should retain MORE context (wider window, slower decay).
            if self.reasoning is not None:
                cal = self.reasoning.calibrator
                if cal.total_predictions >= 10:
                    self.attention.set_calibration_state(
                        ece=cal.compute_ece(),
                        bias=cal.compute_bias(),
                        is_overconfident=cal.is_overconfident,
                        is_underconfident=cal.is_underconfident,
                        total_predictions=cal.total_predictions,
                    )

            # ── Calibration-sensitive auto-GC threshold ──
            # When poorly calibrated, raise the GC trigger threshold so we
            # don't garbage collect until the budget is genuinely bursting.
            # Nominal: 0.80, miscalibrated: up to 0.92
            effective_gc_threshold = self._gc_util_threshold
            if self.reasoning is not None and self.reasoning.calibrator.total_predictions >= 10:
                ece = self.reasoning.calibrator.compute_ece()
                bias = abs(self.reasoning.calibrator.compute_bias())
                cal_gc_boost = min(0.12, 0.4 * ece * (1.0 + bias))
                effective_gc_threshold = min(0.92, self._gc_util_threshold + cal_gc_boost)

            self.attention.apply_recency_decay(decay_rate=0.03)
            self._ticks_without_gc += 1

            # ── Push calibration state to Mneme pillar ──
            # When the reasoning engine is poorly calibrated, the memory
            # system should consolidate more cautiously and prune less
            # aggressively — the agent's relevance judgments are suspect.
            if self._mneme_pillar is not None and self.reasoning is not None:
                cal = self.reasoning.calibrator
                if cal.total_predictions >= 10:
                    self._mneme_pillar.update_calibration(
                        ece=cal.compute_ece(),
                        bias=cal.compute_bias(),
                        is_overconfident=cal.is_overconfident,
                        is_underconfident=cal.is_underconfident,
                        total_predictions=cal.total_predictions,
                    )

            # Auto-GC: run if budget utilization exceeds threshold
            if self._auto_gc:
                util = self.attention.budget.utilization
                if util >= effective_gc_threshold:
                    self._last_gc_report = self.attention.collect_garbage()
                    self._emit_gc_feedback(self._last_gc_report)
                    self._ticks_without_gc = 0
                    logger.debug(
                        f"{self.name}: auto-GC triggered — "
                        f"utilization={util:.1%} (threshold={effective_gc_threshold:.2f})"
                    )

    # ── Convenience methods ────────────────────────────────────────

    def reason(self, task: str, *, context: list[str] | None = None) -> Any:
        """Reason about a task and return a ReasoningPlan.

        This is the primary public method — it produces action plans
        that can be forwarded to Praxis for execution.
        """
        if self.reasoning is None:
            return None
        plan = self.reasoning.reason(task, initial_context=context)
        self._last_plan = plan
        self._tasks_reasoned += 1
        # Emit plan_ready signal
        self._emit_plan_ready(plan)
        return plan

    def add_context(
        self,
        content: str,
        *,
        mutual_info: float = 0.0,
        task_relevance: float = 0.5,
        importance_tags: tuple[str, ...] = (),
    ) -> Any:
        """Add content to the attention system."""
        if self.attention is None:
            return None
        chunk = self.attention.add_chunk(
            content,
            mutual_info=mutual_info,
            task_relevance=task_relevance,
            importance_tags=importance_tags,
        )
        self._context_added += 1
        return chunk

    def collect_garbage(self) -> Any:
        """Force a garbage collection cycle on the attention system."""
        if self.attention is None:
            return None
        report = self.attention.collect_garbage()
        self._last_gc_report = report
        self._emit_gc_feedback(report)
        self._ticks_without_gc = 0
        return report

    def serialize(self) -> dict | None:
        """Get the full serializable cognition state."""
        if self.attention is None:
            return None
        return {
            "attention": {
                "token_capacity": self._token_capacity,
                "stats": self.attention.stats,
                "chunk_count": self.attention.chunk_count,
                "budget_utilization": self.attention.budget.utilization,
            },
            "reasoning": self.reasoning.stats if self.reasoning else {},
            "context_added": self._context_added,
            "tasks_reasoned": self._tasks_reasoned,
        }

    # ── Properties ──────────────────────────────────────────────────

    @property
    def latest_plan(self) -> Any:
        """Most recent reasoning plan, or None."""
        return self._last_plan

    @property
    def latest_gc_report(self) -> Any:
        """Most recent GC report, or None."""
        return self._last_gc_report

    @property
    def stats(self) -> dict[str, Any]:
        """Aggregate pillar statistics."""
        base = {
            "context_added": self._context_added,
            "tasks_reasoned": self._tasks_reasoned,
            "ticks_without_gc": self._ticks_without_gc,
        }
        if self.attention is not None:
            base["attention"] = self.attention.stats
        if self.reasoning is not None:
            base["reasoning"] = self.reasoning.stats
        return base

    # ── Internal ────────────────────────────────────────────────────

    def _emit_plan_ready(self, plan: Any) -> None:
        """Emit feedback about plan quality to the equilibrium engine.

        High-confidence plans → push toward exploit (commit)
        Low-confidence plans → push toward explore (reconsider)
        Deep plans → reinforce deep; shallow plans → reinforce shallow
        """
        if plan is None:
            return

        # Plan confidence → explore/exploit modulation
        try:
            best_conf = plan.best_confidence
            # High confidence: exploit more, converge
            if best_conf > 0.8:
                # Don't push exploit signal here — let Praxis feedback drive it
                pass
            elif best_conf < 0.4:
                # Low confidence: push toward explore (reconsider approach)
                self.emit_feedback(
                    Feedback(
                        source=self.pillar,
                        tension_axis_id="explore_exploit",
                        signal=-0.15,
                        confidence=min(0.7, 1.0 - best_conf),
                        reason=f"low plan confidence ({best_conf:.2f})",
                    )
                )

            # Depth feedback: deep plans → push toward deep
            if plan.max_depth_reached >= 5:
                self.emit_feedback(
                    Feedback(
                        source=self.pillar,
                        tension_axis_id="shallow_deep",
                        signal=0.10,
                        confidence=0.5,
                        reason=f"deep reasoning beneficial (depth={plan.max_depth_reached})",
                    )
                )

            # Divergence feedback: many branches → push convergent
            if plan.branches_explored > 5:
                self.emit_feedback(
                    Feedback(
                        source=self.pillar,
                        tension_axis_id="divergent_convergent",
                        signal=0.08,
                        confidence=0.5,
                        reason=f"many branches explored ({plan.branches_explored})",
                    )
                )
        except Exception:
            pass

    def _emit_gc_feedback(self, report: Any) -> None:
        """Emit feedback about attention system pressure.

        High utilization → push toward shallow (faster processing)
        Many pruned → push toward consolidate (save to Mneme)
        """
        if report is None:
            return

        try:
            # Budget pressure → shallow/deep
            util = report.budget_utilization_after
            if util > 0.85:
                self.emit_feedback(
                    Feedback(
                        source=self.pillar,
                        tension_axis_id="shallow_deep",
                        signal=-0.12,  # Push toward shallow
                        confidence=0.6,
                        reason=f"high budget utilization ({util:.1%})",
                    )
                )

            # Pruned chunks → consolidate_prune (save to Mneme)
            if report.pruned_count > 0:
                self.emit_feedback(
                    Feedback(
                        source=self.pillar,
                        tension_axis_id="consolidate_prune",
                        signal=-0.08,  # Push toward consolidate
                        confidence=0.5,
                        reason=f"pruned {report.pruned_count} chunks — consider saving to Mneme",
                    )
                )
        except Exception:
            pass
