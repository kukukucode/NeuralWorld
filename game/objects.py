"""Value objects shared by the game engine, renderer, and future datasets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

Coordinate = tuple[int, int]


class Action(IntEnum):
    """The discrete v0.1 action space."""

    NOOP = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


ACTION_DELTAS: dict[Action, Coordinate] = {
    Action.NOOP: (0, 0),
    Action.UP: (0, -1),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0),
}


@dataclass(frozen=True, slots=True)
class GameState:
    """Complete, immutable ground-truth state for a single simulation step."""

    player: Coordinate
    box: Coordinate
    goal: Coordinate
    walls: frozenset[Coordinate]
    step_count: int = 0

    def observation(self, *, width: int, height: int) -> dict[str, object]:
        """Convert this state to values accepted by the Gymnasium space."""

        wall_grid = np.zeros((height, width), dtype=np.int8)
        for x, y in self.walls:
            wall_grid[y, x] = 1
        return {
            "player": np.asarray(self.player, dtype=np.int32),
            "box": np.asarray(self.box, dtype=np.int32),
            "goal": np.asarray(self.goal, dtype=np.int32),
            "walls": wall_grid,
            "step_count": self.step_count,
        }

