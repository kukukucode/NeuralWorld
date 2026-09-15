"""Pure deterministic movement and collision rules."""

from __future__ import annotations

from dataclasses import dataclass

from game.objects import ACTION_DELTAS, Action, GameState


@dataclass(frozen=True, slots=True)
class MovementResult:
    """Outcome of applying one action to the ground-truth state."""

    state: GameState
    moved: bool
    pushed: bool
    collided: bool


def _add(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    return left[0] + right[0], left[1] + right[1]


def apply_action(state: GameState, action: Action) -> MovementResult:
    """Apply an action without randomness or side effects."""

    if action is Action.NOOP:
        return MovementResult(
            state=GameState(state.player, state.box, state.goal, state.walls, state.step_count + 1),
            moved=False,
            pushed=False,
            collided=False,
        )

    destination = _add(state.player, ACTION_DELTAS[action])
    if destination in state.walls:
        return MovementResult(
            state=GameState(state.player, state.box, state.goal, state.walls, state.step_count + 1),
            moved=False,
            pushed=False,
            collided=True,
        )

    next_box = state.box
    pushed = destination == state.box
    if pushed:
        box_destination = _add(state.box, ACTION_DELTAS[action])
        if box_destination in state.walls:
            return MovementResult(
                state=GameState(state.player, state.box, state.goal, state.walls, state.step_count + 1),
                moved=False,
                pushed=False,
                collided=True,
            )
        next_box = box_destination

    return MovementResult(
        state=GameState(destination, next_box, state.goal, state.walls, state.step_count + 1),
        moved=True,
        pushed=pushed,
        collided=False,
    )

