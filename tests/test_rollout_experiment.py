"""Check held-out windows, horizon alignment, and plot generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data.dataset import StateCodec
from experiments.rollout import HORIZONS, evaluate_rollouts, run_experiment
from models.state_mlp import StateMLP, TrainingConfig, save_checkpoint


def identity_model(codec: StateCodec) -> StateMLP:
    size = codec.state_size + 5
    model = StateMLP(codec.state_size, hidden_sizes=(size,))
    with torch.no_grad():
        model.network[0].weight.copy_(torch.eye(size))
        model.network[0].bias.zero_()
        model.network[-1].weight.zero_()
        model.network[-1].weight[:, :4].copy_(torch.eye(4))
        model.network[-1].bias.zero_()
    return model


def example_dataset(path: Path, *, steps: int = 50, perturb_last_target: bool = False) -> StateCodec:
    codec = StateCodec(width=4, height=4)
    state = np.zeros(codec.state_size, dtype=np.float32)
    state[:6] = [1 / 3, 1 / 3, 2 / 3, 1 / 3, 2 / 3, 2 / 3]
    states = np.tile(state, (steps * 3, 1))
    targets = states[:, :4].copy()
    if perturb_last_target:
        targets[steps * 2 - 1, 0] = 2 / 3
    np.savez_compressed(
        path,
        states=states,
        actions=np.zeros(steps * 3, dtype=np.int64),
        next_states=targets,
        episode_ids=np.repeat(np.arange(3), steps),
        step_indices=np.tile(np.arange(steps), 3),
        splits=np.repeat([0, 2, 2], steps),
        dones=np.tile([False] * (steps - 1) + [True], 3),
        width=codec.width,
        height=codec.height,
    )
    return codec


def test_five_horizons_use_complete_held_out_trajectories(tmp_path: Path) -> None:
    dataset = tmp_path / "transitions.npz"
    codec = example_dataset(dataset, perturb_last_target=True)
    result = evaluate_rollouts(identity_model(codec), codec, dataset, batch_size=1)

    assert result["test_episodes"] == 2
    assert result["eligible_windows"] == result["evaluated_windows"] == 2
    assert list(map(int, result["horizons"])) == list(HORIZONS)
    for horizon in HORIZONS[:-1]:
        assert result["horizons"][str(horizon)]["mse"] == pytest.approx(0.0)
        assert result["horizons"][str(horizon)]["exact_state_accuracy"] == 1.0
    assert result["horizons"]["50"]["mse"] == pytest.approx(1 / 72)
    assert result["horizons"]["50"]["exact_state_accuracy"] == 0.5
    assert all(item["invalid_state_rate"] == 0 for item in result["horizons"].values())


def test_cli_experiment_writes_machine_readable_metrics_and_png(tmp_path: Path) -> None:
    dataset = tmp_path / "transitions.npz"
    codec = example_dataset(dataset)
    checkpoint = save_checkpoint(
        tmp_path / "model.pt", identity_model(codec),
        codec=codec, training_config=TrainingConfig(epochs=1),
    )
    output = tmp_path / "results"
    result = run_experiment(
        checkpoint_path=checkpoint, dataset_path=dataset, output_dir=output,
        batch_size=1, max_windows=1,
    )
    assert result["evaluated_windows"] == 1
    assert result["window_selection"] == "evenly_spaced"
    assert json.loads((output / "metrics.json").read_text(encoding="utf-8")) == result
    assert (output / "rollout_errors.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (output / "rollout_errors.png").stat().st_size > 1000


def test_evaluation_rejects_short_or_incompatible_data(tmp_path: Path) -> None:
    dataset = tmp_path / "short.npz"
    codec = example_dataset(dataset, steps=49)
    model = identity_model(codec)
    with pytest.raises(ValueError, match="50-step windows"):
        evaluate_rollouts(model, codec, dataset)
    with pytest.raises(ValueError, match="batch_size"):
        evaluate_rollouts(model, codec, dataset, batch_size=0)
    with pytest.raises(ValueError, match="dimensions"):
        evaluate_rollouts(model, StateCodec(5, 5), dataset)


def test_evaluation_rejects_episode_boundary_corruption(tmp_path: Path) -> None:
    dataset = tmp_path / "corrupt.npz"
    codec = example_dataset(dataset)
    with np.load(dataset) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["dones"][51] = True
    np.savez_compressed(dataset, **arrays)
    with pytest.raises(ValueError, match="terminal transition"):
        evaluate_rollouts(identity_model(codec), codec, dataset)
