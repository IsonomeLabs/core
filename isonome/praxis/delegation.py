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

Cross-pillar integration:
  Cognition → Praxis: Calibrator ECE flows through the shared
  confidence_calibrator reference. When Praxis reads the calibrator
  during execute_batch(), it checks whether delegation is warranted.
  
  Praxis → Cognition: Delegation decisions feed back as calibration
  signals — delegated actions generate (predicted_confidence, False)
  pairs indicating the system chose not to execute autonomously.

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
    WELL_CALIBRATED = auto()  # ECE ≤ 0.05 — no delegation
    MODERATE = auto()          # 0.05 < ECE ≤ 0.15 — no delegation
    OVERCONFIDENT = auto()     # ECE > 0.15, overconfident — delegate MODERATE+
    UNDERCONFIDENT = auto()    # ECE > 0.15, underconfident — delegate HIGH+
    UNCALIBRATED = auto()      # < 10 predictions — no delegation (insufficient data)


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
    action_risk: int       # ActionRisk.value (0-4)
    mode: DelegationMode
    decision: DelegationDecision
    ece: float
    bias: float
    reason: str


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

    Attributes:
        ece_threshold: ECE above which delegation activates (default 0.15).
        min_predictions: Minimum predictions before delegation is considered.
        overconfident_risk_threshold: Risk level at which overconfident
            agents delegate (default MODERATE = 2).
        underconfident_risk_threshold: Risk level at which underconfident
            agents delegate (default HIGH = 3).
    """

    def __init__(
        self,
        *,
        calibrator: Any = None,
        ece_threshold: float = 0.15,
        min_predictions: int = 10,
        overconfident_risk_threshold: int = 2,  # ActionRisk.MODERATE.value
        underconfident_risk_threshold: int = 3,  # ActionRisk.HIGH.value
    ):
        self._calibrator = calibrator
        self._ece_threshold = ece_threshold
        self._min_predictions = min_predictions
        self._overconfident_risk_threshold = overconfident_risk_threshold
        self._underconfident_risk_threshold = underconfident_risk_threshold

        # Statistics
        self._total_checks: int = 0
        self._total_delegated: int = 0
        self._total_executed: int = 0
        self._total_skipped: int = 0
        self._records: list[DelegationRecord] = []

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

    def check(self, action: Action, *, risk_value: int | None = None) -> DelegationDecision:
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

    def check_batch(self, actions: list[Action]) -> dict[UUID, DelegationDecision]:
        """Check delegation for a batch of actions.

        Returns:
            Dict mapping action UUID to DelegationDecision.
        """
        results = {}
        for action in actions:
            results[action.id] = self.check(action)
        return results

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
            "min_predictions": self._min_predictions,
            "overconfident_risk_threshold": self._overconfident_risk_threshold,
            "underconfident_risk_threshold": self._underconfident_risk_threshold,
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
        return {
            "ece_threshold": self._ece_threshold,
            "min_predictions": self._min_predictions,
            "overconfident_risk_threshold": self._overconfident_risk_threshold,
            "underconfident_risk_threshold": self._underconfident_risk_threshold,
            "total_checks": self._total_checks,
            "total_delegated": self._total_delegated,
            "total_executed": self._total_executed,
            "total_skipped": self._total_skipped,
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
        }

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
        )

        # Restore cumulative stats
        gate._total_checks = int(data.get("total_checks", 0))
        gate._total_delegated = int(data.get("total_delegated", 0))
        gate._total_executed = int(data.get("total_executed", 0))
        gate._total_skipped = int(data.get("total_skipped", 0))

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

        return gate

    def __repr__(self) -> str:
        mode = self.compute_mode()
        return (
            f"DelegationGate(mode={mode.name}, "
            f"delegated={self._total_delegated}/{self._total_checks}, "
            f"threshold={self._ece_threshold})"
        )
