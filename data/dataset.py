"""Tensor-ready transition datasets for state dynamics learning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class Split(IntEnum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2


@dataclass(frozen=True, slots=True)
class StateCodec:
    """Encode ground-truth observations and prediction targets as normalized vectors."""

    width: int
    height: int

    @property
    def state_size(self) -> int:
        return 6 + self.width * self.height

    @property
    def target_size(self) -> int:
        return 4

    def encode(self, observation: dict[str, object]) -> np.ndarray:
        scale = np.asarray((self.width - 1, self.height - 1), dtype=np.float32)
        positions = [
            np.asarray(observation[name], dtype=np.float32) / scale
            for name in ("player", "box", "goal")
        ]
        walls = np.asarray(observation["walls"], dtype=np.float32).reshape(-1)
        return np.concatenate((*positions, walls), dtype=np.float32)

    def encode_target(self, observation: dict[str, object]) -> np.ndarray:
        scale = np.asarray((self.width - 1, self.height - 1), dtype=np.float32)
        player = np.asarray(observation["player"], dtype=np.float32) / scale
        box = np.asarray(observation["box"], dtype=np.float32) / scale
        return np.concatenate((player, box), dtype=np.float32)

    def with_predicted_positions(
        self, state: torch.Tensor, predicted_positions: torch.Tensor
    ) -> torch.Tensor:
        """Build the next model input while carrying Goal and walls forward."""

        if not isinstance(state, torch.Tensor) or not isinstance(
            predicted_positions, torch.Tensor
        ):
            raise TypeError("state and predicted_positions must be torch tensors")
        if state.ndim < 1 or state.shape[-1] != self.state_size:
            raise ValueError(f"state must end with {self.state_size} features")
        expected_shape = (*state.shape[:-1], self.target_size)
        if tuple(predicted_positions.shape) != expected_shape:
            raise ValueError(f"predicted_positions must have shape {expected_shape}")
        if predicted_positions.dtype != state.dtype or predicted_positions.device != state.device:
            raise ValueError("state and predicted_positions must share dtype and device")

        return torch.cat(
            (predicted_positions.clamp(0.0, 1.0), state[..., self.target_size :]),
            dim=-1,
        )

    def target_to_grid(self, targets: np.ndarray) -> np.ndarray:
        scale = np.asarray(
            (self.width - 1, self.height - 1, self.width - 1, self.height - 1),
            dtype=np.float32,
        )
        return np.rint(np.clip(targets, 0.0, 1.0) * scale).astype(np.int64)


class TransitionDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """A selected split of state/action/next-state transitions."""

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
    ) -> None:
        if not (len(states) == len(actions) == len(next_states)):
            raise ValueError("Transition arrays must have equal lengths")
        self.states = torch.as_tensor(states, dtype=torch.float32)
        self.actions = torch.as_tensor(actions, dtype=torch.long)
        self.next_states = torch.as_tensor(next_states, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.states[index], self.actions[index], self.next_states[index]


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    train: TransitionDataset
    validation: TransitionDataset
    test: TransitionDataset
    codec: StateCodec
    metadata: dict[str, Any]


def load_dataset(path: str | Path) -> DatasetBundle:
    """Load a generated NPZ file and expose disjoint episode-level splits."""

    with np.load(Path(path), allow_pickle=False) as archive:
        states = archive["states"]
        actions = archive["actions"]
        next_states = archive["next_states"]
        splits = archive["splits"]
        width = int(archive["width"])
        height = int(archive["height"])
        metadata = {
            "episodes": int(archive["episodes"]),
            "seed": int(archive["seed"]),
            "transitions": len(states),
        }

    def selected(split: Split) -> TransitionDataset:
        mask = splits == int(split)
        if not np.any(mask):
            raise ValueError(f"Dataset contains no {split.name.lower()} transitions")
        return TransitionDataset(states[mask], actions[mask], next_states[mask])

    return DatasetBundle(
        train=selected(Split.TRAIN),
        validation=selected(Split.VALIDATION),
        test=selected(Split.TEST),
        codec=StateCodec(width=width, height=height),
        metadata=metadata,
    )
