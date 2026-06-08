"""Feedback Cooldown Manager — per-axis feedback rate dampening.

When an axis receives high-frequency feedback from the same source
(e.g., Cognition repeatedly pushing explore_exploit), the engine
becomes susceptible to oscillation even with adaptive damping. The
cooldown system adds a per-(pillar, axis) cooldown that dampens
repeated feedback from the same source on the same axis within a
configurable tick window.

Design principles:

1. **First feedback always passes** — no cooldown on initial contact
2. **Decay is multiplicative** — each repeat within the window compounds
   the decay_factor: multiplier = decay_factor^(hit_count - 1)
3. **Cooldown resets after window** — once enough ticks pass with no
   feedback from a (pillar, axis) pair, the next feedback is fresh
4. **Floor prevents total suppression** — multiplier never drops below
   MIN_MULTIPLIER (0.01), ensuring feedback always has some effect
5. **Transparent when disabled** — no overhead, no behavioral change

Integration:

- EquilibriumEngine applies cooldown in apply_feedback() and
  apply_feedback_batch(), multiplying the effective_delta by the
  cooldown multiplier
- PillarEquilibriumView exposes cooldown state for pillar introspection
- Serialization preserves cooldown state for cross-session persistence
"""

from __future__ import annotations

from typing import Any

from isonome.types import Pillar, TensionID

# Minimum multiplier floor — feedback always has at least 1% effect
MIN_MULTIPLIER: float = 0.01

# Type alias for the cooldown key: (pillar, axis_id)
_CooldownKey = tuple[str, str]


class FeedbackCooldownManager:
    """Tracks per-(pillar, axis) feedback cooldown state.

    Mathematical model:
    - On first feedback from (pillar, axis): multiplier = 1.0
    - On repeat within cooldown_window: multiplier *= decay_factor
      Equivalently: multiplier = decay_factor^(hit_count - 1)
    - After cooldown_window ticks with no feedback: reset to 1.0
    - Multiplier is floored at MIN_MULTIPLIER (0.01)

    The cooldown key is (source_pillar, axis_id), so:
    - Cognition → explore_exploit and Praxis → explore_exploit
      are tracked independently
    - Cognition → explore_exploit and Cognition → shallow_deep
      are tracked independently
    """

    __slots__ = (
        "_cooldown_window",
        "_decay_factor",
        "_state",
        "_total_cooldowns",
        "_total_suppressions",
    )

    def __init__(
        self,
        *,
        cooldown_window: int = 5,
        decay_factor: float = 0.5,
    ) -> None:
        """Initialize the feedback cooldown manager.

        Args:
            cooldown_window: Number of ticks after the last feedback
                before the cooldown resets. Must be >= 1.
            decay_factor: Multiplier applied for each repeat feedback
                within the cooldown window. Must be in (0.0, 1.0].
                A value of 1.0 means no dampening (tracking only).
                Values < 1.0 progressively dampen repeated feedback.

        Raises:
            ValueError: If cooldown_window < 1 or decay_factor is
                outside (0.0, 1.0].
        """
        if cooldown_window < 1:
            raise ValueError(
                f"cooldown_window must be >= 1, got {cooldown_window}"
            )
        if not (0.0 < decay_factor <= 1.0):
            raise ValueError(
                f"decay_factor must be in (0.0, 1.0], got {decay_factor}"
            )

        self._cooldown_window = cooldown_window
        self._decay_factor = decay_factor
        self._state: dict[_CooldownKey, dict[str, Any]] = {}
        self._total_cooldowns: int = 0
        self._total_suppressions: float = 0.0

    # ── Core API ────────────────────────────────────────────────

    def check_and_apply(
        self,
        axis_id: TensionID,
        source_pillar: Pillar,
        tick: int,
    ) -> float:
        """Check cooldown state and return the multiplier for this feedback.

        This method both returns the multiplier AND updates the internal
        cooldown state. It must be called exactly once per feedback
        application.

        Args:
            axis_id: The tension axis identifier.
            source_pillar: Which pillar is sending the feedback.
            tick: The current engine tick number.

        Returns:
            A multiplier in [MIN_MULTIPLIER, 1.0] to apply to the
            feedback signal. 1.0 means no cooldown (first feedback
            or window expired).
        """
        key = (source_pillar.value, axis_id)
        existing = self._state.get(key)

        if existing is None:
            # First feedback from this (pillar, axis) — no cooldown
            self._state[key] = {
                "last_tick": tick,
                "hit_count": 1,
                "current_multiplier": 1.0,
            }
            return 1.0

        # Check if cooldown window has expired
        ticks_since_last = tick - existing["last_tick"]
        if ticks_since_last > self._cooldown_window:
            # Window expired — treat as fresh feedback
            self._state[key] = {
                "last_tick": tick,
                "hit_count": 1,
                "current_multiplier": 1.0,
            }
            return 1.0

        # Within cooldown window — apply decay
        new_hit_count = existing["hit_count"] + 1
        # multiplier = decay_factor^(hit_count - 1)
        # e.g., hit_count=2 → decay_factor^1, hit_count=3 → decay_factor^2
        raw_multiplier = self._decay_factor ** (new_hit_count - 1)
        multiplier = max(raw_multiplier, MIN_MULTIPLIER)

        self._state[key] = {
            "last_tick": tick,
            "hit_count": new_hit_count,
            "current_multiplier": multiplier,
        }

        # Track statistics
        self._total_cooldowns += 1
        suppression = 1.0 - multiplier
        self._total_suppressions += suppression

        return multiplier

    # ── Properties ──────────────────────────────────────────────

    @property
    def cooldown_window(self) -> int:
        """The cooldown window in ticks."""
        return self._cooldown_window

    @property
    def decay_factor(self) -> float:
        """The decay factor applied per repeat feedback."""
        return self._decay_factor

    @property
    def total_cooldowns(self) -> int:
        """Total number of times cooldown was applied (multiplier < 1.0)."""
        return self._total_cooldowns

    @property
    def total_suppressions(self) -> float:
        """Total suppression amount (sum of 1.0 - multiplier for cooled feedbacks)."""
        return self._total_suppressions

    @property
    def active_keys(self) -> list[_CooldownKey]:
        """Currently tracked (pillar_value, axis_id) pairs."""
        return list(self._state.keys())

    # ── Query methods ───────────────────────────────────────────

    def cooldown_state_for(
        self,
        axis_id: TensionID,
        source_pillar: Pillar,
    ) -> dict[str, Any] | None:
        """Return the cooldown state for a specific (pillar, axis) pair.

        Args:
            axis_id: The tension axis identifier.
            source_pillar: Which pillar to check.

        Returns:
            A dict with 'last_tick', 'hit_count', 'current_multiplier',
            or None if the pair has no cooldown state.
        """
        key = (source_pillar.value, axis_id)
        entry = self._state.get(key)
        if entry is None:
            return None
        return dict(entry)

    def get_multiplier_for(
        self,
        axis_id: TensionID,
        source_pillar: Pillar,
    ) -> float:
        """Get the current multiplier for a (pillar, axis) pair without updating.

        Returns:
            The current multiplier (from the last check_and_apply call),
            or 1.0 if the pair has no cooldown state.
        """
        key = (source_pillar.value, axis_id)
        entry = self._state.get(key)
        if entry is None:
            return 1.0
        return entry["current_multiplier"]

    # ── Reset ───────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all cooldown tracking state and counters."""
        self._state.clear()
        self._total_cooldowns = 0
        self._total_suppressions = 0.0

    # ── Serialization ───────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize the cooldown manager state for cross-session persistence."""
        return {
            "cooldown_window": self._cooldown_window,
            "decay_factor": self._decay_factor,
            "total_cooldowns": self._total_cooldowns,
            "total_suppressions": self._total_suppressions,
            "state": {
                f"{pillar_val}:{axis_id}": {
                    "last_tick": entry["last_tick"],
                    "hit_count": entry["hit_count"],
                    "current_multiplier": entry["current_multiplier"],
                }
                for (pillar_val, axis_id), entry in self._state.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeedbackCooldownManager:
        """Deserialize cooldown manager state from a dict produced by to_dict().

        Missing fields use sensible defaults, allowing forward-compatible
        deserialization when new fields are added in future versions.
        """
        mgr = cls(
            cooldown_window=data.get("cooldown_window", 5),
            decay_factor=data.get("decay_factor", 0.5),
        )
        mgr._total_cooldowns = int(data.get("total_cooldowns", 0))
        mgr._total_suppressions = float(data.get("total_suppressions", 0.0))

        # Restore per-key state
        raw_state = data.get("state", {})
        for composite_key, entry in raw_state.items():
            # Parse "pillar:axis" format
            if ":" in composite_key:
                pillar_val, axis_id = composite_key.split(":", 1)
            else:
                # Fallback for malformed keys
                continue
            mgr._state[(pillar_val, axis_id)] = {
                "last_tick": int(entry.get("last_tick", 0)),
                "hit_count": int(entry.get("hit_count", 1)),
                "current_multiplier": float(entry.get("current_multiplier", 1.0)),
            }

        return mgr

    def __repr__(self) -> str:
        n_active = len(self._state)
        return (
            f"FeedbackCooldownManager("
            f"window={self._cooldown_window}, "
            f"decay={self._decay_factor:.2f}, "
            f"active={n_active}, "
            f"cooldowns={self._total_cooldowns})"
        )
