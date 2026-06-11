"""FSM Compiler & Executor — Chamber 3 finite-state machine.

Architecture:
  FSMCompiler  →  builder that accumulates states / transitions / guards
  FSMExecutor  →  runtime engine that ticks at control frequency

Guards are callables ``guard(ctx: FSMContext) -> bool``.
Events are strings that trigger transitions when a matching guard is satisfied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from isonome.core.state import FSMStateSnapshot
from isonome.utils.logging import get_layer_logger


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class FSMStateDef:
    """Static definition of a single FSM state."""

    name: str
    entry: Callable[[FSMContext], None] | None = None
    exit: Callable[[FSMContext], None] | None = None
    during: Callable[[FSMContext], None] | None = None


@dataclass
class FSMTransitionDef:
    """Static definition of a single transition."""

    source: str
    target: str
    guard: Callable[[FSMContext], bool] | None = None
    event: str | None = None


@dataclass
class FSMContext:
    """Mutable runtime context passed to guards and actions.

    The coordinator (or a single agent) owns one ``FSMContext`` and
    updates it each tick before calling ``executor.tick()``.
    """

    current_state: str = ""
    previous_state: str | None = None
    event: str | None = None
    tick_count: int = 0
    data: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> FSMStateSnapshot:
        return FSMStateSnapshot(
            current_state=self.current_state,
            previous_state=self.previous_state,
            tick_count=self.tick_count,
            event=self.event,
            data=dict(self.data),
        )


# ---------------------------------------------------------------------------
# FSM Compiler (builder)
# ---------------------------------------------------------------------------

class FSMCompiler:
    """Builder for finite-state machines.

    Usage:
        fsm = (
            FSMCompiler()
            .add_state("idle", entry=on_idle_entry)
            .add_state("walk", during=on_walk_during)
            .add_transition("idle", "walk", event="start", guard=can_walk)
            .add_transition("walk", "idle", event="stop")
            .compile()
        )
    """

    def __init__(self) -> None:
        self._states: dict[str, FSMStateDef] = {}
        self._transitions: list[FSMTransitionDef] = []
        self._initial: str | None = None

    # -- states --------------------------------------------------------------

    def add_state(
        self,
        name: str,
        *,
        entry: Callable[[FSMContext], None] | None = None,
        exit: Callable[[FSMContext], None] | None = None,
        during: Callable[[FSMContext], None] | None = None,
        initial: bool = False,
    ) -> FSMCompiler:
        """Register a state.  Only one state may be marked ``initial``."""
        if name in self._states:
            raise ValueError(f"State '{name}' already defined")
        self._states[name] = FSMStateDef(
            name=name, entry=entry, exit=exit, during=during
        )
        if initial:
            if self._initial is not None:
                raise ValueError(
                    f"Initial state already set to '{self._initial}'"
                )
            self._initial = name
        return self

    # -- transitions ---------------------------------------------------------

    def add_transition(
        self,
        source: str,
        target: str,
        *,
        guard: Callable[[FSMContext], bool] | None = None,
        event: str | None = None,
    ) -> FSMCompiler:
        """Register a transition.

        If ``event`` is provided, the transition only fires when that exact
        event is present in the context.  If ``guard`` is provided, it must
        also return ``True``.
        """
        self._transitions.append(
            FSMTransitionDef(source=source, target=target, guard=guard, event=event)
        )
        return self

    # -- compile -------------------------------------------------------------

    def compile(self) -> "FSMDefinition":
        """Validate and freeze the FSM definition."""
        if not self._states:
            raise ValueError("FSM has no states")
        if self._initial is None:
            raise ValueError("No initial state set (use initial=True on one state)")
        if self._initial not in self._states:
            raise ValueError(f"Initial state '{self._initial}' not defined")

        # Validate transition endpoints
        for t in self._transitions:
            if t.source not in self._states:
                raise ValueError(f"Transition source '{t.source}' not defined")
            if t.target not in self._states:
                raise ValueError(f"Transition target '{t.target}' not defined")

        return FSMDefinition(
            states=dict(self._states),
            transitions=list(self._transitions),
            initial=self._initial,
        )


# ---------------------------------------------------------------------------
# FSM Definition (immutable result of compilation)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FSMDefinition:
    """Validated, immutable FSM ready for execution."""

    states: Dict[str, FSMStateDef]
    transitions: List[FSMTransitionDef]
    initial: str


# ---------------------------------------------------------------------------
# FSM Executor (runtime)
# ---------------------------------------------------------------------------

class FSMExecutor:
    """Runtime engine that ticks an FSM definition.

    The executor maintains its own ``FSMContext`` and updates it on every
    ``tick()``.  Callers can inject events and data into the context before
    each tick.
    """

    def __init__(self, definition: FSMDefinition) -> None:
        self._definition = definition
        self._logger = get_layer_logger("coordination.fsm")
        self._ctx = FSMContext(current_state=definition.initial)
        self._entered = False  # lazy entry on first tick

    # -- read-only properties ------------------------------------------------

    @property
    def current_state(self) -> str:
        return self._ctx.current_state

    @property
    def context(self) -> FSMContext:
        """Return the mutable runtime context (caller may inject data/events)."""
        return self._ctx

    def snapshot(self) -> FSMStateSnapshot:
        return self._ctx.snapshot()

    # -- runtime ---------------------------------------------------------------

    def tick(self) -> str:
        """Execute one FSM step.

        1. Run entry action if we just entered this state.
        2. Evaluate transitions; if one fires, run exit → entry.
        3. Run during action.
        4. Increment tick count.

        Returns the state name *after* this tick.
        """
        ctx = self._ctx
        state_def = self._definition.states[ctx.current_state]

        # Lazy entry for the initial state (first tick)
        if not self._entered:
            self._entered = True
            if state_def.entry:
                state_def.entry(ctx)
            self._logger.debug(
                "fsm_state_enter",
                extra={"state": ctx.current_state, "event": ctx.event},
            )

        # Evaluate transitions
        fired = self._evaluate_transitions(ctx)
        if fired:
            state_def = self._definition.states[ctx.current_state]

        # During action
        if state_def.during:
            state_def.during(ctx)

        ctx.tick_count += 1
        ctx.event = None  # consume event after tick
        return ctx.current_state

    def inject_event(self, event: str) -> None:
        """Queue an event for the next tick."""
        self._ctx.event = event

    def reset(self) -> None:
        """Return to the initial state (runs exit if currently in a state)."""
        ctx = self._ctx
        if ctx.current_state != self._definition.initial:
            old_def = self._definition.states[ctx.current_state]
            if old_def.exit:
                old_def.exit(ctx)
            ctx.previous_state = ctx.current_state
            ctx.current_state = self._definition.initial
            ctx.tick_count = 0
            self._entered = False
            self._logger.info(
                "fsm_reset",
                extra={
                    "from": ctx.previous_state,
                    "to": self._definition.initial,
                },
            )

    # -- internals -----------------------------------------------------------

    def _evaluate_transitions(self, ctx: FSMContext) -> bool:
        """Check all transitions from the current state; fire the first match."""
        for t in self._definition.transitions:
            if t.source != ctx.current_state:
                continue
            if t.event is not None and t.event != ctx.event:
                continue
            if t.guard is not None and not t.guard(ctx):
                continue

            # Fire transition
            old_def = self._definition.states[t.source]
            new_def = self._definition.states[t.target]

            if old_def.exit:
                old_def.exit(ctx)
            ctx.previous_state = ctx.current_state
            ctx.current_state = t.target
            if new_def.entry:
                new_def.entry(ctx)

            self._logger.info(
                "fsm_transition",
                extra={
                    "from": t.source,
                    "to": t.target,
                    "event": t.event,
                    "tick": ctx.tick_count,
                },
            )
            return True

        return False
