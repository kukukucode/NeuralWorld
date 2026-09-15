"""MLP baseline for one-step ground-truth state prediction."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional
from torch.utils.data import DataLoader

from data.dataset import StateCodec, TransitionDataset


class StateMLP(nn.Module):
    """Predict normalized next Agent/Box coordinates from state and action."""

    def __init__(
        self,
        state_size: int,
        *,
        action_count: int = 5,
        target_size: int = 4,
        hidden_sizes: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        if not hidden_sizes or any(size < 1 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive layer sizes")
        self.state_size = state_size
        self.action_count = action_count
        self.target_size = target_size
        self.hidden_sizes = hidden_sizes

        sizes = (state_size + action_count, *hidden_sizes, target_size)
        layers: list[nn.Module] = []
        for layer_index, (input_size, output_size) in enumerate(
            zip(sizes, sizes[1:])
        ):
            layers.append(nn.Linear(input_size, output_size))
            if layer_index < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        one_hot_actions = functional.one_hot(
            actions.long(), num_classes=self.action_count
        ).to(dtype=states.dtype)
        return self.network(torch.cat((states, one_hot_actions), dim=-1))


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 42

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")


@dataclass(frozen=True, slots=True)
class TrainingResult:
    train_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]
    best_epoch: int


def _mean_loss(
    model: StateMLP,
    dataset: TransitionDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_examples = 0
    model.eval()
    with torch.no_grad():
        for states, actions, targets in loader:
            states, actions, targets = (
                states.to(device),
                actions.to(device),
                targets.to(device),
            )
            predictions = model(states, actions)
            total_loss += functional.mse_loss(
                predictions, targets, reduction="sum"
            ).item()
            total_examples += targets.numel()
    return total_loss / total_examples


def train_model(
    model: StateMLP,
    train: TransitionDataset,
    validation: TransitionDataset,
    config: TrainingConfig,
    *,
    device: str | torch.device = "cpu",
) -> TrainingResult:
    """Train deterministically and restore the best validation checkpoint."""

    config.validate()
    device = torch.device(device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model.to(device)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        train,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_history: list[float] = []
    validation_history: list[float] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_values = 0
        for states, actions, targets in loader:
            states, actions, targets = (
                states.to(device),
                actions.to(device),
                targets.to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            predictions = model(states, actions)
            loss = functional.mse_loss(predictions, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.numel()
            total_values += targets.numel()

        train_history.append(total_loss / total_values)
        validation_loss = _mean_loss(
            model,
            validation,
            batch_size=config.batch_size,
            device=device,
        )
        validation_history.append(validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainingResult(tuple(train_history), tuple(validation_history), best_epoch)


def evaluate_model(
    model: StateMLP,
    dataset: TransitionDataset,
    codec: StateCodec,
    *,
    batch_size: int = 1024,
    device: str | torch.device = "cpu",
) -> dict[str, float | int]:
    """Evaluate one-step predictions on a held-out transition split."""

    device = torch.device(device)
    model.to(device).eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for states, actions, next_states in loader:
            predicted = model(states.to(device), actions.to(device))
            predictions.append(predicted.cpu().numpy())
            targets.append(next_states.numpy())

    prediction_array = np.concatenate(predictions)
    target_array = np.concatenate(targets)
    predicted_grid = codec.target_to_grid(prediction_array)
    target_grid = codec.target_to_grid(target_array)
    exact_rows = np.all(predicted_grid == target_grid, axis=1)
    player_exact = np.all(predicted_grid[:, :2] == target_grid[:, :2], axis=1)
    box_exact = np.all(predicted_grid[:, 2:] == target_grid[:, 2:], axis=1)
    coordinate_accuracy = np.mean(predicted_grid == target_grid)
    grid_mae = np.mean(np.abs(predicted_grid - target_grid))
    return {
        "samples": len(dataset),
        "mse": float(np.mean(np.square(prediction_array - target_array))),
        "coordinate_accuracy": float(coordinate_accuracy),
        "player_position_accuracy": float(np.mean(player_exact)),
        "box_position_accuracy": float(np.mean(box_exact)),
        "exact_state_accuracy": float(np.mean(exact_rows)),
        "grid_mae": float(grid_mae),
    }


def save_checkpoint(
    path: str | Path,
    model: StateMLP,
    *,
    codec: StateCodec,
    training_config: TrainingConfig,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model": {
                "state_size": model.state_size,
                "action_count": model.action_count,
                "target_size": model.target_size,
                "hidden_sizes": model.hidden_sizes,
            },
            "codec": {"width": codec.width, "height": codec.height},
            "training": asdict(training_config),
        },
        path,
    )
    return path


def load_checkpoint(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> tuple[StateMLP, StateCodec, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model_config = checkpoint["model"]
    model = StateMLP(
        model_config["state_size"],
        action_count=model_config["action_count"],
        target_size=model_config["target_size"],
        hidden_sizes=tuple(model_config["hidden_sizes"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    codec = StateCodec(**checkpoint["codec"])
    return model, codec, checkpoint["training"]
