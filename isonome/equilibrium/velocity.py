"""Tension Velocity Tracker — tracks per-axis velocity and momentum.

The velocity tracker computes the first derivative of each tension axis's
position over time, enabling three key capabilities:

1. Oscillation prediction: Velocity sign changes (reversals) are an
   early warning of oscillation, detectable before position stddev
   exceeds the threshold.
2. Momentum awareness: Axes moving toward their default can be allowed
   to coast, while axes moving away need stronger restoring force.
3. Smarter damping: The AdaptiveDampingController can use velocity
   reversal counts to anticipate oscillation and boost damping
   preemptively.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from isonome.types import TensionID


class TensionVelocityTracker:
    """Tracks per-axis velocity (rate of position change) and oscillation momentum.

    Mathematical model:
    - velocity[i] = position[i] - position[i-1]  (simple finite difference)
    - reversal = sign(velocity[i]) != sign(velocity[i-1])
                 AND |velocity[i]| > min_reversal_magnitude
    - reversal_rate = reversals_in_window / window_size
    - momentum_score = velocity * (default - position)
        positive = heading home, negative = drifting away

    The tracker is optional -- engines without it operate exactly as before.
    When enabled, the engine feeds position updates after each feedback tick.
    """

    __slots__ = (
        "_velocity",
        "_prev_velocity",
        "_reversal_counts",
        "_position_history",
        "_momentum_scores",
        "_window_size",
        "_min_reversal_magnitude",
        "_total_reversals",
        "_total_updates",
    )

    def __init__(
        self,
        *,
        window_size: int = 10,
        min_reversal_magnitude: float = 0.005,
    ):
        """Initialize the velocity tracker.

        Args:
            window_size: Number of recent position samples for reversal
                counting and rate computation.
            min_reversal_magnitude: Minimum |velocity| for a reversal to
                count. Filters out noise from near-zero velocity
                fluctuations.
        """
        if window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {window_size}")
        if min_reversal_magnitude < 0:
            raise ValueError(
                f"min_reversal_magnitude must be >= 0, got {min_reversal_magnitude}"
            )

        self._velocity: dict[TensionID, float] = {}
        self._prev_velocity: dict[TensionID, float] = {}
        self._reversal_counts: dict[TensionID, int] = {}
        self._position_history: dict[TensionID, deque[float]] = {}
        self._momentum_scores: dict[TensionID, float] = {}
        self._window_size = window_size
        self._min_reversal_magnitude = min_reversal_magnitude
        self._total_reversals: int = 0
        self._total_updates: int = 0

    def register_axis(self, axis_id: TensionID) -> None:
        """Register an axis for velocity tracking.

        Args:
            axis_id: The tension axis identifier.
        """
        self._velocity[axis_id] = 0.0
        self._prev_velocity[axis_id] = 0.0
        self._reversal_counts[axis_id] = 0
        self._position_history[axis_id] = deque(maxlen=self._window_size)
        self._momentum_scores[axis_id] = 0.0

    def unregister_axis(self, axis_id: TensionID) -> None:
        """Remove an axis from velocity tracking."""
        self._velocity.pop(axis_id, None)
        self._prev_velocity.pop(axis_id, None)
        self._reversal_counts.pop(axis_id, None)
        self._position_history.pop(axis_id, None)
        self._momentum_scores.pop(axis_id, None)

    def on_position_update(
        self,
        axis_id: TensionID,
        new_position: float,
        default_position: float,
    ) -> None:
        """Record a position update and compute velocity, momentum, reversals.

        Called by the engine after each apply_feedback() or
        apply_feedback_batch().

        Args:
            axis_id: The tension axis identifier.
            new_position: The axis's new position after feedback.
            default_position: The axis's default (homeostasis target).
        """
        # Auto-register unknown axes
        if axis_id not in self._velocity:
            self.register_axis(axis_id)

        hist = self._position_history[axis_id]
        if len(hist) > 0:
            prev_pos = hist[-1]
            velocity = new_position - prev_pos
        else:
            velocity = 0.0

        # Detect reversal: velocity sign change with sufficient magnitude
        # Compare against the CURRENT stored velocity (from the previous update),
        # not against _prev_velocity (which is from two updates ago).
        current_vel = self._velocity.get(axis_id, 0.0)
        if (
            abs(velocity) >= self._min_reversal_magnitude
            and abs(current_vel) >= self._min_reversal_magnitude
            and math.copysign(1, velocity) != math.copysign(1, current_vel)
        ):
            self._reversal_counts[axis_id] = (
                self._reversal_counts.get(axis_id, 0) + 1
            )
            self._total_reversals += 1

        # Update state: current_vel becomes previous, new velocity becomes current
        self._prev_velocity[axis_id] = current_vel
        self._velocity[axis_id] = velocity
        self._position_history[axis_id].append(new_position)
        self._total_updates += 1

        # Compute momentum score: positive = heading toward default
        drift_direction = default_position - new_position
        self._momentum_scores[axis_id] = velocity * drift_direction

    def get_velocity(self, axis_id: TensionID) -> float:
        """Get the current velocity of an axis.

        Returns:
            The velocity (position change per tick). 0.0 if not tracked.
        """
        return self._velocity.get(axis_id, 0.0)

    def get_momentum_score(self, axis_id: TensionID) -> float:
        """Get the momentum score for an axis.

        Positive = axis is moving toward its default (good momentum).
        Negative = axis is drifting away from default (bad momentum).
        Zero = no movement or axis at default.

        Returns:
            The momentum score. 0.0 if not tracked.
        """
        return self._momentum_scores.get(axis_id, 0.0)

    def get_reversal_count(self, axis_id: TensionID) -> int:
        """Get the total velocity reversal count for an axis.

        Returns:
            Cumulative reversal count. 0 if not tracked.
        """
        return self._reversal_counts.get(axis_id, 0)

    def get_reversal_rate(self, axis_id: TensionID) -> float:
        """Get the recent velocity reversal rate for an axis.

        Computed as the number of reversals in the position history window
        divided by the number of possible reversal points. High rates
        indicate oscillatory behavior.

        Returns:
            Reversal rate in [0.0, 1.0]. 0.0 if insufficient data.
        """
        hist = self._position_history.get(axis_id)
        if hist is None or len(hist) < 3:
            return 0.0

        positions = list(hist)
        reversals = 0
        prev_delta = None
        for i in range(1, len(positions)):
            delta = positions[i] - positions[i - 1]
            if (
                prev_delta is not None
                and abs(delta) >= self._min_reversal_magnitude
                and abs(prev_delta) >= self._min_reversal_magnitude
            ):
                if math.copysign(1, delta) != math.copysign(1, prev_delta):
                    reversals += 1
            prev_delta = delta

        max_possible = len(positions) - 2
        return reversals / max_possible if max_possible > 0 else 0.0

    def is_approaching_default(self, axis_id: TensionID) -> bool:
        """Whether an axis is moving toward its default position.

        Returns:
            True if momentum_score > 0 (velocity and drift have same sign).
        """
        return self._momentum_scores.get(axis_id, 0.0) > 0

    def is_drifting_from_default(self, axis_id: TensionID) -> bool:
        """Whether an axis is moving away from its default position.

        Returns:
            True if momentum_score < 0 (velocity and drift have opposite
            sign).
        """
        return self._momentum_scores.get(axis_id, 0.0) < 0

    def all_velocities(self) -> dict[TensionID, float]:
        """Snapshot of all current velocity values."""
        return dict(self._velocity)

    def all_momentum_scores(self) -> dict[TensionID, float]:
        """Snapshot of all current momentum scores."""
        return dict(self._momentum_scores)

    def is_oscillation_imminent(
        self, axis_id: TensionID, threshold: float = 0.4
    ) -> bool:
        """Predict oscillation from velocity reversal rate.

        Unlike the engine's _check_oscillation which requires position
        stddev to exceed a threshold (post-hoc), this detects oscillation
        tendency from velocity sign changes (predictive).

        Args:
            axis_id: The tension axis identifier.
            threshold: Reversal rate above which oscillation is predicted.
                Default 0.4 = 40% of consecutive deltas are reversing.

        Returns:
            True if the axis shows oscillatory velocity patterns.
        """
        return self.get_reversal_rate(axis_id) > threshold

    @property
    def window_size(self) -> int:
        """The position history window size."""
        return self._window_size

    @property
    def min_reversal_magnitude(self) -> float:
        """The minimum velocity magnitude for a reversal to count."""
        return self._min_reversal_magnitude

    @property
    def total_reversals(self) -> int:
        """Total velocity reversals across all axes."""
        return self._total_reversals

    @property
    def total_updates(self) -> int:
        """Total position updates processed."""
        return self._total_updates

    def reset(self) -> None:
        """Reset all velocity tracking state."""
        self._velocity.clear()
        self._prev_velocity.clear()
        self._reversal_counts.clear()
        self._position_history.clear()
        self._momentum_scores.clear()
        self._total_reversals = 0
        self._total_updates = 0

    # -- Serialization --

    def to_dict(self) -> dict[str, Any]:
        """Serialize the tracker state for cross-session persistence."""
        return {
            "velocity": dict(self._velocity),
            "prev_velocity": dict(self._prev_velocity),
            "reversal_counts": dict(self._reversal_counts),
            "position_history": {
                k: list(v) for k, v in self._position_history.items()
            },
            "momentum_scores": dict(self._momentum_scores),
            "window_size": self._window_size,
            "min_reversal_magnitude": self._min_reversal_magnitude,
            "total_reversals": self._total_reversals,
            "total_updates": self._total_updates,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TensionVelocityTracker:
        """Deserialize tracker state from a dict produced by to_dict()."""
        tracker = cls(
            window_size=data.get("window_size", 10),
            min_reversal_magnitude=data.get("min_reversal_magnitude", 0.005),
        )
        tracker._velocity = data.get("velocity", {})
        tracker._prev_velocity = data.get("prev_velocity", {})
        tracker._reversal_counts = {
            k: int(v) for k, v in data.get("reversal_counts", {}).items()
        }
        for k, v_list in data.get("position_history", {}).items():
            tracker._position_history[k] = deque(
                v_list, maxlen=tracker._window_size
            )
        tracker._momentum_scores = data.get("momentum_scores", {})
        tracker._total_reversals = int(data.get("total_reversals", 0))
        tracker._total_updates = int(data.get("total_updates", 0))
        return tracker

    def __repr__(self) -> str:
        n_axes = len(self._velocity)
        return (
            f"TensionVelocityTracker("
            f"axes={n_axes}, "
            f"reversals={self._total_reversals}, "
            f"updates={self._total_updates})"
        )
