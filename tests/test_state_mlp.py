from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from data.dataset import StateCodec, TransitionDataset
from models.state_mlp import (
    StateMLP,
    TrainingConfig,
    evaluate_model,
    load_checkpoint,
    save_checkpoint,
    train_model,
)


def make_identity_dataset(samples: int = 96) -> TransitionDataset:
    rng = np.random.default_rng(3)
    states = rng.uniform(0.0, 1.0, size=(samples, 10)).astype(np.float32)
    actions = rng.integers(0, 5, size=samples, dtype=np.int64)
    targets = states[:, :4].copy()
    return TransitionDataset(states, actions, targets)


def test_model_output_shape() -> None:
    model = StateMLP(87, hidden_sizes=(16, 16))
    predictions = model(torch.zeros((7, 87)), torch.arange(7) % 5)
    assert predictions.shape == (7, 4)


def test_training_reduces_validation_loss() -> None:
    torch.manual_seed(11)
    dataset = make_identity_dataset()
    model = StateMLP(10, hidden_sizes=(32, 32))
    result = train_model(
        model,
        dataset,
        dataset,
        TrainingConfig(epochs=30, batch_size=32, learning_rate=0.01, seed=11),
    )
    assert min(result.validation_loss) < result.validation_loss[0] * 0.25


def test_evaluation_reports_one_step_metrics() -> None:
    codec = StateCodec(width=3, height=3)
    states = np.zeros((4, codec.state_size), dtype=np.float32)
    actions = np.zeros(4, dtype=np.int64)
    targets = np.zeros((4, codec.target_size), dtype=np.float32)
    dataset = TransitionDataset(states, actions, targets)
    model = StateMLP(codec.state_size, hidden_sizes=(8,))
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)

    metrics = evaluate_model(model, dataset, codec)
    assert metrics["samples"] == 4
    assert metrics["mse"] == 0.0
    assert metrics["coordinate_accuracy"] == 1.0
    assert metrics["player_position_accuracy"] == 1.0
    assert metrics["box_position_accuracy"] == 1.0
    assert metrics["exact_state_accuracy"] == 1.0


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    codec = StateCodec(width=9, height=9)
    model = StateMLP(codec.state_size, hidden_sizes=(12,))
    config = TrainingConfig(epochs=2)
    path = save_checkpoint(tmp_path / "model.pt", model, codec=codec, training_config=config)
    restored, restored_codec, restored_training = load_checkpoint(path)
    assert restored_codec == codec
    assert restored_training["epochs"] == 2
    assert restored.hidden_sizes == (12,)
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, actual)
