"""Sample future action sequences and execute their best first action."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from data.dataset import StateCodec
from game import Action
from models import StateMLP, rollout


def _goal_distances(walls: np.ndarray, goal: tuple[int, int]) -> np.ndarray:
    """Distance around the observed static walls, ignoring movable objects."""
    height, width = walls.shape
    x, y = goal
    distances = np.full(walls.shape, height * width, dtype=np.int32)
    distances[y, x] = 0
    frontier = deque([goal])
    while frontier:
        x, y = frontier.popleft()
        distance = distances[y, x] + 1
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if (
                0 <= nx < width and 0 <= ny < height
                and not walls[ny, nx] and distance < distances[ny, nx]
            ):
                distances[ny, nx] = distance
                frontier.append((nx, ny))
    return distances


class RandomShootingPlanner:
    """Replan from the latest observation, with a fixed inference budget."""

    def __init__(
        self,
        model: StateMLP,
        codec: StateCodec,
        *,
        horizon: int = 5,
        candidates: int = 64,
    ) -> None:
        if horizon < 1 or candidates < len(Action):
            raise ValueError(f"horizon must be positive and candidates must be at least {len(Action)}")
        if (model.state_size, model.target_size, model.action_count) != (
            codec.state_size, codec.target_size, len(Action),
        ):
            raise ValueError("checkpoint model dimensions or action space do not match the game codec")
        self.model = model
        self.codec = codec
        self.horizon = horizon
        self.candidates = candidates

    def select_action(self, observation: dict[str, object], rng: np.random.Generator) -> Action:
        """Use predicted progress toward the goal to choose one real action."""
        walls = np.asarray(observation["walls"], dtype=bool)
        if walls.shape != (self.codec.height, self.codec.width):
            raise ValueError("observation dimensions do not match the checkpoint codec")
        goal_x, goal_y = map(int, np.asarray(observation["goal"]))
        distances = _goal_distances(walls, (goal_x, goal_y))

        actions = rng.integers(len(Action), size=(self.horizon, self.candidates))
        # Every possible first action gets a chance, even with a small budget.
        actions[0] = rng.permutation(np.arange(self.candidates) % len(Action))
        parameter = next(self.model.parameters())
        state = torch.as_tensor(
            self.codec.encode(observation), dtype=parameter.dtype, device=parameter.device,
        )
        states = state.expand(self.candidates, -1)
        trajectory = rollout(self.model, self.codec, states, torch.from_numpy(actions))
        positions = trajectory[1:, :, : self.codec.target_size]
        predicted = positions.to(dtype=torch.float32).cpu().numpy()
        finite = np.isfinite(predicted).all(axis=-1)
        grid = self.codec.target_to_grid(np.where(finite[..., None], predicted, 0.0))
        px, py, bx, by = np.moveaxis(grid, -1, 0)
        invalid = ~finite | walls[py, px] | walls[by, bx] | ((px == bx) & (py == by))

        # Only score valid prefixes and prefer reaching a given distance sooner.
        valid_prefix = np.logical_and.accumulate(~invalid, axis=0)
        delay = np.arange(self.horizon)[:, None] / self.horizon
        scores = np.where(valid_prefix, distances[py, px] + delay, np.inf).min(axis=0)
        if not np.isfinite(scores).any():
            return Action(int(rng.integers(len(Action))))
        return Action(int(actions[0, scores.argmin()]))
