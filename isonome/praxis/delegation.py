"""CalibrationGatedDelegation — metacognitive delegation gate for Praxis.

When the confidence calibrator reports poor calibration (high ECE),
the system should not trust its own execution judgments for high-risk
actions. Instead, it delegates those actions to a subagent or external
executor, reducing the risk of poorly-calibrated autonomous execution.

This is a **calibration-only** decision mechanism (iter-010 pattern):
delegation is a qualitative behavior change, not a continuous tension
modulation. The gate is either open (execute directly) or closed
(delegate instead).

Mathematical foundation:
  ECE > θ_delegate → delegate actions at risk level ≥ R_threshold
  ECE ≤ θ_delegate → execute all actions normally (gate open)

Where:
  - θ_delegate: ECE threshold for activation (default: 0.15)
  - R_threshold: minimum risk level for delegation (default: HIGH)
  - Overconfidence bonus: overconfident agents delegate at lower
    risk thresholds (HIGH→MODERATE), because their confidence
    estimates are known to be inflated

Three operating modes:
  1. **WELL_CALIBRATED** (ECE ≤ 0.05): Gate open. No delegation.
     The system's confidence judgments are reliable.
  2. **MODERATE** (0.05 < ECE ≤ 0.15): Gate open. Some miscalibration
     but not severe enough to warrant delegation overhead.
  3. **POORLY_CALIBRATED** (ECE > 0.15):
     - Overconfident: Delegate MODERATE+ risk actions. The system
       over-estimates its own capability, so we don't trust its
       risk assessments for anything above LOW.
     - Underconfident/neutral: Delegate HIGH+ risk actions. The
       system may underestimate, but we still trust its risk
       classifications — just not its confidence in executing
       high-risk operations itself.

Delegation Outcome Tracking (iter-020):
  When a delegated action's outcome is observed, it is recorded via
  record_outcome(). This closes the metacognitive feedback loop:

  1. Delegated actions that SUCCEED indicate the system was too
     cautious — the calibrator records (predicted_confidence, True),
     which gradually lowers ECE and can shift delegation mode.
  2. Delegated actions that FAIL justify the delegation decision —
     the calibrator records (predicted_confidence, False), confirming
     the system's self-assessment of poor calibration.

  Dynamic threshold adaptation:
  θ_delegate adapts based on delegation accuracy over a rolling window:

    delegation_accuracy = successful_delegations / total_delegated_outcomes

    If delegation_accuracy > 0.8:  θ_delegate *= (1 − α)  (tighten —
      system is delegating well but maybe too much; lower threshold
      means fewer actions delegated)
    If delegation_accuracy < 0.5:  θ_delegate *= (1 + α)  (loosen —
      system needs to delegate more; higher threshold means more
      actions delegated)
    α = 0.02 per outcome (slow, conservative adaptation)

Cross-pillar integration:
  Cognition → Praxis: Calibrator ECE flows through the shared
    confidence_calibrator reference. When Praxis reads the calibrator
  during execute_batch(), it checks whether delegation is warranted.

  Praxis → Cognition: Delegation decisions feed back as calibration
    signals — record_outcome() feeds (predicted_confidence,
  actual_success) pairs directly to the calibrator, closing
  the metacognitive loop.

  Praxis → Mneme: Delegated action outcomes are stored with a
    "delegated" tag for future pattern matching and learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
from uuid import UUID


class DelegationMode(Enum):
    """Delegation operating mode based on calibration quality."""
    WELL_CALIBRATED = auto()    # ECE ≤ 0.05 — no delegation
    MODERATE = auto()           # 0.05 < ECE ≤ 0.15 — no delegation
    OVERCONFIDENT = auto()      # ECE > 0.15, overconfident — delegate MODERATE+
    UNDERCONFIDENT = auto()     # ECE > 0.15, underconfident — delegate HIGH+
    UNCALIBRATED = auto()       # < 10 predictions — no delegation (insufficient data)


class DelegationDecision(Enum):
    """Result of a delegation gate check for a single action."""
    EXECUTE = auto()    # Execute directly — calibration is adequate
    DELEGATE = auto()   # Delegate to subagent — calibration too poor for autonomous execution
    SKIP = auto()       # Skip delegation check — action too low-risk to matter


@dataclass(frozen=True, slots=True)
class DelegationRecord:
    """Record of a delegation decision for audit/logging."""
    action_id: UUID
    action_description: str
    action_risk: int          # ActionRisk.value (0-4)
    mode: DelegationMode
    decision: DelegationDecision
    ece: float
    bias: float
    reason: str


@dataclass(frozen=True, slots=True)
class DelegationOutcome:
    """Record of a delegated action's outcome for feedback tracking.

    This is the return path of the delegation loop: when a delegated
    action completes (or fails), the outcome is recorded so the
    calibrator can learn from the delegation decision.

    The predicted_confidence field captures what the system's
    confidence WAS at delegation time — it's the self-assessment
    that the delegation gate used to decide. If the action succeeds
    despite low confidence, the system was underconfident. If it
    fails, the delegation was justified.
    """
    action_id: UUID
    action_description: str
    action_risk: int              # ActionRisk.value (0-4)
    predicted_confidence: float   # System's confidence at delegation time
    actual_success: bool          # Whether the delegated action succeeded
    delegated_mode: DelegationMode  # Mode that triggered delegation
    ece_at_delegation: float      # ECE when delegation was decided
    feedback_to_calibrator: bool = True  # Whether to feed back to calibrator


class DelegationGate:
    """Calibration-gated delegation controller.

    Determines whether actions should be executed directly or delegated
    to a subagent based on the confidence calibrator's ECE and bias.

    Usage:
        gate = DelegationGate(calibrator=my_calibrator)
        for action in pending_actions:
            decision = gate.check(action)
            if decision == DelegationDecision.DELEGATE:
                delegate_to_subagent(action)
            else:
                execute_directly(action)

    After delegated actions complete:
        gate.record_outcome(DelegationOutcome(
            action_id=action.id,
            action_description=action.description,
            action_risk=action.risk.value,
            predicted_confidence=0.6,
            actual_success=True,
            delegated_mode=DelegationMode.OVERCONFIDENT,
            ece_at_delegation=0.18,
        ))

    Attributes:
        ece_threshold: ECE above which delegation activates (default 0.15).
        min_predictions: Minimum predictions before delegation is considered.
        overconfident_risk_threshold: Risk level at which overconfident
            agents delegate (default MODERATE = 2).
        underconfident_risk_threshold: Risk level at which underconfident
            agents delegate (default HIGH = 3).
        adaptation_rate: How fast the ECE threshold adapts (default 0.02).
        outcome_window: How many recent outcomes to track (default 50).
    """

    def __init__(
        self,
        *,
        calibrator: Any = None,
        ece_threshold: float = 0.15,
        min_predictions: int = 10,
        overconfident_risk_threshold: int = 2,  # ActionRisk.MODERATE.value
        underconfident_risk_threshold: int = 3,  # ActionRisk.HIGH.value
        adaptation_rate: float = 0.02,
        outcome_window: int = 50,
    ):
        self._calibrator = calibrator
        self._ece_threshold = ece_threshold
        self._ece_threshold_initial = ece_threshold  # For reset
        self._min_predictions = min_predictions
        self._overconfident_risk_threshold = overconfident_risk_threshold
        self._underconfident_risk_threshold = underconfident_risk_threshold
        self._adaptation_rate = adaptation_rate
        self._outcome_window = outcome_window

        # Statistics
        self._total_checks: int = 0
        self._total_delegated: int = 0
        self._total_executed: int = 0
        self._total_skipped: int = 0
        self._records: list[DelegationRecord] = []

        # Outcome tracking (iter-020)
        self._outcomes: list[DelegationOutcome] = []
        self._total_outcomes_recorded: int = 0
        self._total_successful_delegations: int = 0
        self._total_failed_delegations: int = 0
        self._total_calibrator_feedbacks: int = 0
        self._threshold_adaptations: int = 0

    # ── Configuration ──────────────────────────────────────────

    def set_calibrator(self, calibrator: Any) -> None:
        """Set or replace the confidence calibrator."""
        self._calibrator = calibrator

    @property
    def ece_threshold(self) -> float:
        return self._ece_threshold

    @ece_threshold.setter
    def ece_threshold(self, value: float) -> None:
        self._ece_threshold = max(0.0, min(1.0, value))

    @property
    def min_predictions(self) -> int:
        return self._min_predictions

    @min_predictions.setter
    def min_predictions(self, value: int) -> None:
        self._min_predictions = max(1, value)

    @property
    def adaptation_rate(self) -> float:
        return self._adaptation_rate

    @property
    def outcome_window(self) -> int:
        return self._outcome_window

    # ── Core Logic ─────────────────────────────────────────────

    def compute_mode(self) -> DelegationMode:
        """Determine the current delegation mode from calibration state.

        Returns:
            DelegationMode indicating how delegation should behave.
        """
        if self._calibrator is None:
            return DelegationMode.UNCALIBRATED

        if self._calibrator.total_predictions < self._min_predictions:
            return DelegationMode.UNCALIBRATED

        ece = self._calibrator.compute_ece()

        if ece <= 0.05:
            return DelegationMode.WELL_CALIBRATED
        elif ece <= self._ece_threshold:
            return DelegationMode.MODERATE
        else:
            # Poorly calibrated — determine direction
            if self._calibrator.is_overconfident:
                return DelegationMode.OVERCONFIDENT
            else:
                return DelegationMode.UNDERCONFIDENT

    def check(self, action: Action, *, risk_value: int | None = None) -> DelegationDecision:  # type: ignore[name-defined]
        """Check whether an action should be delegated.

        Decision logic:
          1. UNCALIBRATED / WELL_CALIBRATED / MODERATE → EXECUTE
          2. OVERCONFIDENT:
             - risk < overconfident_risk_threshold → EXECUTE
             - risk >= overconfident_risk_threshold → DELEGATE
          3. UNDERCONFIDENT:
             - risk < underconfident_risk_threshold → EXECUTE
             - risk >= underconfident_risk_threshold → DELEGATE

        TRIVIAL risk (0) is always EXECUTE — no point delegating
        a no-side-effect action.

        Args:
            action: The action to evaluate.
            risk_value: Override risk value (0-4). If None, uses action.risk.value.

        Returns:
            DelegationDecision indicating what to do with this action.
        """
        self._total_checks += 1
        mode = self.compute_mode()

        if risk_value is None:
            risk_value = action.risk.value

        # TRIVIAL risk never delegates
        if risk_value == 0:
            decision = DelegationDecision.EXECUTE
            reason = "trivial risk — always execute directly"
        elif mode == DelegationMode.UNCALIBRATED:
            decision = DelegationDecision.EXECUTE
            reason = f"insufficient calibration data ({self._calibrator.total_predictions if self._calibrator else 0} < {self._min_predictions})"
        elif mode == DelegationMode.WELL_CALIBRATED:
            decision = DelegationDecision.EXECUTE
            reason = f"well-calibrated (ECE ≤ 0.05) — gate open"
        elif mode == DelegationMode.MODERATE:
            decision = DelegationDecision.EXECUTE
            reason = f"moderate miscalibration — gate open"
        elif mode == DelegationMode.OVERCONFIDENT:
            if risk_value >= self._overconfident_risk_threshold:
                decision = DelegationDecision.DELEGATE
                reason = f"overconfident + risk_value={risk_value} ≥ {self._overconfident_risk_threshold} — delegate"
            else:
                decision = DelegationDecision.EXECUTE
                reason = f"overconfident but risk_value={risk_value} < {self._overconfident_risk_threshold} — execute"
        elif mode == DelegationMode.UNDERCONFIDENT:
            if risk_value >= self._underconfident_risk_threshold:
                decision = DelegationDecision.DELEGATE
                reason = f"underconfident + risk_value={risk_value} ≥ {self._underconfident_risk_threshold} — delegate"
            else:
                decision = DelegationDecision.EXECUTE
                reason = f"underconfident but risk_value={risk_value} < {self._underconfident_risk_threshold} — execute"
        else:
            decision = DelegationDecision.EXECUTE
            reason = "unknown mode — default to execute"

        # Update statistics
        if decision == DelegationDecision.DELEGATE:
            self._total_delegated += 1
        elif decision == DelegationDecision.EXECUTE:
            self._total_executed += 1
        elif decision == DelegationDecision.SKIP:
            self._total_skipped += 1

        # Record for audit
        ece = 0.0
        bias = 0.0
        if self._calibrator is not None and self._calibrator.total_predictions >= self._min_predictions:
            ece = self._calibrator.compute_ece()
            bias = self._calibrator.bias

        action_id = action.id
        action_description = action.description

        record = DelegationRecord(
            action_id=action_id,
            action_description=action_description,
            action_risk=risk_value,
            mode=mode,
            decision=decision,
            ece=ece,
            bias=bias,
            reason=reason,
        )
        self._records.append(record)

        return decision

    def check_batch(self, actions: list[Action]) -> dict[UUID, DelegationDecision]:  # type: ignore[name-defined]
        """Check delegation for a batch of actions.

        Returns:
            Dict mapping action UUID to DelegationDecision.
        """
        results = {}
        for action in actions:
            results[action.id] = self.check(action)
        return results

    # ── Outcome Tracking (iter-020) ────────────────────────────

    def record_outcome(self, outcome: DelegationOutcome) -> None:
        """Record the outcome of a delegated action and feed back to calibrator.

        This closes the metacognitive feedback loop: when a delegated
        action's outcome is observed, the gate:
          1. Records the outcome for audit/tracking
          2. Feeds (predicted_confidence, actual_success) to the calibrator
             if feedback_to_calibrator is True
          3. Adapts the ECE threshold based on delegation accuracy

        Threshold adaptation logic:
          - delegation_accuracy > 0.8: tighten threshold (decrease by α),
            because the system is delegating well but perhaps too cautiously
          - delegation_accuracy < 0.5: loosen threshold (increase by α),
            because the system needs to delegate more aggressively
          - 0.5 ≤ delegation_accuracy ≤ 0.8: no adaptation needed

        The adaptation rate α is conservative (default 0.02) to prevent
        oscillation in the threshold. Threshold bounds: [0.05, 0.5].

        Args:
            outcome: The DelegationOutcome to record.
        """
        self._outcomes.append(outcome)
        self._total_outcomes_recorded += 1

        if outcome.actual_success:
            self._total_successful_delegations += 1
        else:
            self._total_failed_delegations += 1

        # Feed to calibrator
        if outcome.feedback_to_calibrator and self._calibrator is not None:
            self._calibrator.record(
                outcome.predicted_confidence,
                outcome.actual_success,
            )
            self._total_calibrator_feedbacks += 1

        # Adapt ECE threshold based on delegation accuracy
        self._adapt_threshold()

    def _adapt_threshold(self) -> None:
        """Adapt ECE threshold based on recent delegation accuracy.

        Uses a rolling window of recent outcomes. The adaptation is
        intentionally conservative (α = 0.02) to prevent oscillation.

        Bounds: θ_delegate ∈ [0.05, 0.5]
            - Floor 0.05: never drop below well-calibrated boundary
            - Ceiling 0.5: never exceed half the ECE range (extreme
              delegation would be counterproductive)
        """
        recent = self._outcomes[-self._outcome_window:]
        if len(recent) < 3:
            return  # Not enough data for meaningful adaptation

        successful = sum(1 for o in recent if o.actual_success)
        accuracy = successful / len(recent)

        if accuracy > 0.8:
            # High accuracy: system is delegating well but possibly
            # too much. Tighten threshold → fewer delegations.
            self._ece_threshold = max(
                0.05,
                self._ece_threshold * (1.0 - self._adaptation_rate),
            )
            self._threshold_adaptations += 1
        elif accuracy < 0.5:
            # Low accuracy: system should delegate more. Loosen
            # threshold → more delegations.
            self._ece_threshold = min(
                0.5,
                self._ece_threshold * (1.0 + self._adaptation_rate),
            )
            self._threshold_adaptations += 1
        # 0.5 ≤ accuracy ≤ 0.8: no adaptation (sweet spot)

    @property
    def delegation_accuracy(self) -> float:
        """Fraction of recent delegated outcomes that succeeded.

        Returns 0.0 if no outcomes have been recorded.
        Uses the rolling outcome window for computation.
        """
        recent = self._outcomes[-self._outcome_window:]
        if not recent:
            return 0.0
        successful = sum(1 for o in recent if o.actual_success)
        return successful / len(recent)

    @property
    def outcomes(self) -> tuple[DelegationOutcome, ...]:
        """Immutable view of all recorded delegation outcomes."""
        return tuple(self._outcomes)

    def reset_threshold(self) -> None:
        """Reset ECE threshold to its initial value.

        Useful for testing or when the adaptation has drifted
        significantly from the starting point.
        """
        self._ece_threshold = self._ece_threshold_initial

    # ── Statistics ─────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        """Aggregate delegation statistics."""
        total = max(1, self._total_checks)
        mode = self.compute_mode()
        ece = 0.0
        bias = 0.0
        n_preds = 0
        if self._calibrator is not None:
            ece = self._calibrator.compute_ece()
            bias = self._calibrator.bias
            n_preds = self._calibrator.total_predictions

        return {
            "total_checks": self._total_checks,
            "total_delegated": self._total_delegated,
            "total_executed": self._total_executed,
            "total_skipped": self._total_skipped,
            "delegation_rate": self._total_delegated / total,
            "mode": mode.name,
            "ece": round(ece, 4),
            "bias": round(bias, 4),
            "calibrator_predictions": n_preds,
            "ece_threshold": self._ece_threshold,
            "ece_threshold_initial": self._ece_threshold_initial,
            "min_predictions": self._min_predictions,
            "overconfident_risk_threshold": self._overconfident_risk_threshold,
            "underconfident_risk_threshold": self._underconfident_risk_threshold,
            # Outcome tracking stats (iter-020)
            "total_outcomes_recorded": self._total_outcomes_recorded,
            "total_successful_delegations": self._total_successful_delegations,
            "total_failed_delegations": self._total_failed_delegations,
            "total_calibrator_feedbacks": self._total_calibrator_feedbacks,
            "delegation_accuracy": round(self.delegation_accuracy, 4),
            "threshold_adaptations": self._threshold_adaptations,
            "adaptation_rate": self._adaptation_rate,
            "outcome_window": self._outcome_window,
        }

    @property
    def records(self) -> tuple[DelegationRecord, ...]:
        """Immutable view of all delegation records."""
        return tuple(self._records)

    @property
    def delegation_rate(self) -> float:
        """Fraction of checked actions that were delegated."""
        total = max(1, self._total_checks)
        return self._total_delegated / total

    # ── Serialization ──────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize delegation gate state for cross-session persistence."""
        data = {
            "ece_threshold": self._ece_threshold,
            "ece_threshold_initial": self._ece_threshold_initial,
            "min_predictions": self._min_predictions,
            "overconfident_risk_threshold": self._overconfident_risk_threshold,
            "underconfident_risk_threshold": self._underconfident_risk_threshold,
            "adaptation_rate": self._adaptation_rate,
            "outcome_window": self._outcome_window,
            "total_checks": self._total_checks,
            "total_delegated": self._total_delegated,
            "total_executed": self._total_executed,
            "total_skipped": self._total_skipped,
            "total_outcomes_recorded": self._total_outcomes_recorded,
            "total_successful_delegations": self._total_successful_delegations,
            "total_failed_delegations": self._total_failed_delegations,
            "total_calibrator_feedbacks": self._total_calibrator_feedbacks,
            "threshold_adaptations": self._threshold_adaptations,
            "records": [
                {
                    "action_id": str(r.action_id),
                    "action_description": r.action_description,
                    "action_risk": r.action_risk,
                    "mode": r.mode.name,
                    "decision": r.decision.name,
                    "ece": r.ece,
                    "bias": r.bias,
                    "reason": r.reason,
                }
                for r in self._records
            ],
            "outcomes": [
                {
                    "action_id": str(o.action_id),
                    "action_description": o.action_description,
                    "action_risk": o.action_risk,
                    "predicted_confidence": o.predicted_confidence,
                    "actual_success": o.actual_success,
                    "delegated_mode": o.delegated_mode.name,
                    "ece_at_delegation": o.ece_at_delegation,
                    "feedback_to_calibrator": o.feedback_to_calibrator,
                }
                for o in self._outcomes
            ],
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, calibrator: Any = None) -> DelegationGate:
        """Deserialize delegation gate from saved state.

        Args:
            data: Dict produced by to_dict().
            calibrator: Confidence calibrator to attach (cannot be serialized).

        Returns:
            Restored DelegationGate with saved configuration and stats.
        """
        gate = cls(
            calibrator=calibrator,
            ece_threshold=data.get("ece_threshold", 0.15),
            min_predictions=data.get("min_predictions", 10),
            overconfident_risk_threshold=data.get("overconfident_risk_threshold", 2),
            underconfident_risk_threshold=data.get("underconfident_risk_threshold", 3),
            adaptation_rate=data.get("adaptation_rate", 0.02),
            outcome_window=data.get("outcome_window", 50),
        )

        # Restore initial threshold (for reset)
        gate._ece_threshold_initial = data.get("ece_threshold_initial", gate._ece_threshold)

        # Restore cumulative stats
        gate._total_checks = int(data.get("total_checks", 0))
        gate._total_delegated = int(data.get("total_delegated", 0))
        gate._total_executed = int(data.get("total_executed", 0))
        gate._total_skipped = int(data.get("total_skipped", 0))
        gate._total_outcomes_recorded = int(data.get("total_outcomes_recorded", 0))
        gate._total_successful_delegations = int(data.get("total_successful_delegations", 0))
        gate._total_failed_delegations = int(data.get("total_failed_delegations", 0))
        gate._total_calibrator_feedbacks = int(data.get("total_calibrator_feedbacks", 0))
        gate._threshold_adaptations = int(data.get("threshold_adaptations", 0))

        # Restore records
        for r_data in data.get("records", []):
            try:
                record = DelegationRecord(
                    action_id=UUID(r_data["action_id"]),
                    action_description=r_data.get("action_description", ""),
                    action_risk=int(r_data.get("action_risk", 0)),
                    mode=DelegationMode[r_data["mode"]],
                    decision=DelegationDecision[r_data["decision"]],
                    ece=float(r_data.get("ece", 0.0)),
                    bias=float(r_data.get("bias", 0.0)),
                    reason=r_data.get("reason", ""),
                )
                gate._records.append(record)
            except (KeyError, ValueError):
                pass  # Skip malformed records

        # Restore outcomes
        for o_data in data.get("outcomes", []):
            try:
                outcome = DelegationOutcome(
                    action_id=UUID(o_data["action_id"]),
                    action_description=o_data.get("action_description", ""),
                    action_risk=int(o_data.get("action_risk", 0)),
                    predicted_confidence=float(o_data.get("predicted_confidence", 0.5)),
                    actual_success=bool(o_data.get("actual_success", False)),
                    delegated_mode=DelegationMode[o_data["delegated_mode"]],
                    ece_at_delegation=float(o_data.get("ece_at_delegation", 0.0)),
                    feedback_to_calibrator=bool(o_data.get("feedback_to_calibrator", True)),
                )
                gate._outcomes.append(outcome)
            except (KeyError, ValueError):
                pass  # Skip malformed outcomes

        return gate

    def __repr__(self) -> str:
        mode = self.compute_mode()
        return (
            f"DelegationGate(mode={mode.name}, "
            f"delegated={self._total_delegated}/{self._total_checks}, "
            f"outcomes={self._total_outcomes_recorded}, "
            f"accuracy={self.delegation_accuracy:.2f}, "
            f"threshold={self._ece_threshold:.4f})"
        )
