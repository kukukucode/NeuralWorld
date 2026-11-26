"""Extract bounded soundtrack features from short world-model forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from data.dataset import StateCodec
from game import Action
from models import StateMLP, rollout
from models.state_mlp import load_checkpoint


@dataclass(frozen=True, slots=True)
class MusicState:
    danger: float
    uncertainty: float
    progress: float
    movement_intensity: float
    isolation: float
    predicted_collision_probability: float


GOAL_STATE = MusicState(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

# Explore every two-action prefix and continue the second action for one step.
_ACTION_IDS = torch.arange(len(Action))
_FIRST = _ACTION_IDS.repeat_interleave(len(Action))
_SECOND = _ACTION_IDS.repeat(len(Action))
_FUTURES = torch.stack((_FIRST, _SECOND, _SECOND))


class MusicForecaster:
    """Map one observation to music features with one batched, three-step rollout."""

    def __init__(self, model: StateMLP, codec: StateCodec) -> None:
        if (model.state_size, model.target_size, model.action_count) != (
            codec.state_size, codec.target_size, len(Action),
        ):
            raise ValueError("checkpoint model dimensions or action space do not match the game codec")
        self.model = model
        self.codec = codec

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> MusicForecaster:
        model, codec, _ = load_checkpoint(path)
        return cls(model, codec)

    def __call__(self, observation: dict[str, object]) -> MusicState:
        walls = np.asarray(observation["walls"], dtype=bool)
        if walls.shape != (self.codec.height, self.codec.width):
            raise ValueError("observation dimensions do not match the checkpoint codec")
        player = np.asarray(observation["player"])
        goal = np.asarray(observation["goal"])
        if np.array_equal(player, goal):
            return GOAL_STATE

        parameter = next(self.model.parameters())
        start = torch.as_tensor(
            self.codec.encode(observation), dtype=parameter.dtype, device=parameter.device,
        )
        trajectory = rollout(self.model, self.codec, start.expand(_FUTURES.shape[1], -1), _FUTURES)
        positions = trajectory[:, :, : self.codec.target_size].float().cpu().numpy()
        finite = np.isfinite(positions).all(axis=-1)
        grid = self.codec.target_to_grid(np.where(finite[..., None], positions, 0.0))
        players, boxes = grid[1:, :, :2], grid[1:, :, 2:]
        px, py = np.moveaxis(players, -1, 0)
        bx, by = np.moveaxis(boxes, -1, 0)
        invalid = ~finite[1:] | walls[py, px] | walls[by, bx] | np.all(players == boxes, axis=-1)
        legal_prefix = np.logical_and.accumulate(~invalid, axis=0)

        distances = np.abs(players - goal).sum(axis=-1)
        reached = legal_prefix & (distances == 0)
        # The real game stops at the goal, so exclude continuation after predicted success.
        finished = np.logical_or.accumulate(reached, axis=0)
        active = np.concatenate((np.ones_like(reached[:1]), ~finished[:-1]))
        valid_active = legal_prefix & active
        displacement = np.abs(np.diff(grid[..., :2], axis=0)).sum(axis=-1)
        blocked = (displacement == 0) & (_FUTURES.numpy() != int(Action.NOOP))
        risk = (invalid | blocked) & active
        collision_probability = float(risk.any(axis=0).mean())
        danger = 0.65 * collision_probability + 0.35 * float(risk[0].mean())

        initial_distance = int(np.abs(player - goal).sum())
        closest = int(np.where(valid_active, distances, initial_distance).min())
        max_distance = (self.codec.width - 3) + (self.codec.height - 3)
        progress = float(np.clip(1 - min(initial_distance, closest) / max(1, max_distance), 0, 1))

        # Spread under different possible actions plus invalid predictions is a heuristic.
        final_steps = valid_active.sum(axis=0)
        final_positions = grid[final_steps, np.arange(len(final_steps)), :2]
        spread = np.linalg.norm(final_positions.std(axis=0)) / len(_FUTURES)
        uncertainty = float(np.clip(spread + invalid[active].mean(), 0.0, 1.0))
        movement = float(np.mean(np.minimum(displacement, 1) * valid_active))
        x, y = player
        isolation = float(np.mean([walls[y - 1, x], walls[y + 1, x], walls[y, x - 1], walls[y, x + 1]]))
        return MusicState(danger, uncertainty, progress, movement, isolation, collision_probability)
