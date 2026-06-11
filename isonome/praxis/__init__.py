"""πρᾶξις (Praxis) — tool use, execution, and action orchestration.

The Praxis pillar is the execution engine of the isonome framework.
It converts plans from Cognition (νοῦς) into executable action DAGs,
schedules execution modulated by three equilibrium tension axes, and
exports results to Mneme (μνήμη) for learning.

Core system: ActionOrchestrator — DAG-based scheduler with:
 - Risk-gated execution (autonomy_safety tension)
 - Variable parallelism (sequential_parallel tension)
 - Configurable verification depth (verify_execute tension)
 - Exponential backoff retry
 - Serialization for cross-session persistence

Delegation: DelegationGate — calibration-gated delegation controller:
 - When ECE exceeds threshold, high-risk actions are delegated
 - Overconfident: delegate MODERATE+ risk (system over-trusts itself)
 - Underconfident: delegate HIGH+ risk (system undervalues capability)
 - Well-calibrated: gate open, no delegation overhead

Integration: PraxisPillar — BasePillar wrapper for agent lifecycle.
"""

from isonome.praxis.delegation import (
    DelegationDecision,
    DelegationGate,
    DelegationMode,
    DelegationOutcome,
    DelegationRecord,
)
from isonome.praxis.orchestrator import (
    Action,
    ActionOrchestrator,
    ActionRisk,
    ActionState,
    ExecutionReport,
    ExecutionResult,
    RetryPolicy,
)
from isonome.praxis.calibration_cache import (
    CacheKey,
    CalibrationCache,
    CertifiedPolicyPackage,
)
from isonome.praxis.pillar import PraxisPillar

__all__ = [
    "Action",
    "ActionOrchestrator",
    "ActionRisk",
    "ActionState",
    "CacheKey",
    "CalibrationCache",
    "CertifiedPolicyPackage",
    "DelegationDecision",
    "DelegationOutcome",
    "DelegationGate",
    "DelegationMode",
    "DelegationRecord",
    "ExecutionReport",
    "ExecutionResult",
    "PraxisPillar",
    "RetryPolicy",
    "VLABase",
    "MockVLABackend",
    "OpenVLA",
    "LLaVARobot",
    "PiZeroFive",
]

try:
    from isonome.praxis.vla import (
        VLABase,
        MockVLABackend,
        OpenVLA,
        LLaVARobot,
        PiZeroFive,
    )
except Exception:
    pass
