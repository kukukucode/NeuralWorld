"""Evaluate open-loop world-model predictions against held-out trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.dataset import Split, StateCodec
from models import rollout
from models.state_mlp import StateMLP, load_checkpoint

HORIZONS = (1, 5, 10, 20, 50)


def _test_windows(
    dataset_path: Path, codec: StateCodec, max_windows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Return identical, complete test starts for every requested horizon."""

    with np.load(dataset_path, allow_pickle=False) as archive:
        if int(archive["width"]) != codec.width or int(archive["height"]) != codec.height:
            raise ValueError("dataset dimensions do not match the checkpoint codec")
        states = archive["states"]
        actions = archive["actions"]
        targets = archive["next_states"]
        episode_ids = archive["episode_ids"]
        step_indices = archive["step_indices"]
        splits = archive["splits"]
        dones = archive["dones"]

    count = len(states)
    if not all(len(array) == count for array in (actions, targets, episode_ids, step_indices, splits, dones)):
        raise ValueError("dataset transition arrays must have equal lengths")
    if states.shape != (count, codec.state_size) or targets.shape != (count, codec.target_size):
        raise ValueError("dataset state dimensions do not match the checkpoint codec")

    test_rows = np.flatnonzero(splits == int(Split.TEST))
    start_rows: list[int] = []
    action_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    episode_count = 0
    for episode_id in np.unique(episode_ids[test_rows]):
        if np.any(splits[episode_ids == episode_id] != int(Split.TEST)):
            raise ValueError("an episode spans multiple dataset splits")
        rows = test_rows[episode_ids[test_rows] == episode_id]
        episode_count += 1
        if not np.array_equal(step_indices[rows], np.arange(len(rows))):
            raise ValueError("test episode steps must be contiguous and start at zero")
        if np.any(dones[rows[:-1]]):
            raise ValueError("test episode contains a terminal transition before its end")
        for start in range(len(rows) - max(HORIZONS) + 1):
            window = rows[start : start + max(HORIZONS)]
            start_rows.append(int(window[0]))
            action_rows.append(actions[window])
            target_rows.append(targets[window[np.asarray(HORIZONS) - 1]])

    eligible = len(start_rows)
    if eligible == 0:
        raise ValueError("test split has no complete 50-step windows; generate longer episodes")
    if max_windows and eligible > max_windows:
        selected = np.linspace(0, eligible - 1, max_windows, dtype=np.int64)
    else:
        selected = np.arange(eligible)
    starts = states[np.asarray(start_rows)[selected]].astype(np.float32, copy=False)
    action_windows = np.asarray(action_rows, dtype=np.int64)[selected]
    target_windows = np.asarray(target_rows, dtype=np.float32)[selected]
    return starts, action_windows, target_windows, eligible, episode_count


def _horizon_metrics(
    predicted: np.ndarray, target: np.ndarray, starts: np.ndarray, codec: StateCodec,
) -> dict[str, float]:
    predicted_grid = codec.target_to_grid(predicted)
    target_grid = codec.target_to_grid(target)
    walls = starts[:, 6:].reshape(-1, codec.height, codec.width) > 0.5
    rows = np.arange(len(starts))
    player_x, player_y, box_x, box_y = predicted_grid.T
    invalid = (
        walls[rows, player_y, player_x]
        | walls[rows, box_y, box_x]
        | ((player_x == box_x) & (player_y == box_y))
    )
    return {
        "mse": float(np.mean(np.square(predicted - target))),
        "grid_mae": float(np.mean(np.abs(predicted_grid - target_grid))),
        "exact_state_accuracy": float(np.mean(np.all(predicted_grid == target_grid, axis=1))),
        "invalid_state_rate": float(np.mean(invalid)),
    }


def evaluate_rollouts(
    model: StateMLP,
    codec: StateCodec,
    dataset_path: str | Path,
    *,
    batch_size: int = 128,
    max_windows: int = 512,
) -> dict[str, Any]:
    """Measure free-running errors at 1/5/10/20/50 steps on the same test starts.

    Windows never cross terminal transitions or split boundaries. A zero
    max_windows evaluates every eligible window; otherwise evenly spaced
    windows are selected deterministically across the held-out episodes.
    """

    if batch_size < 1 or max_windows < 0:
        raise ValueError("batch_size must be positive and max_windows non-negative")
    if model.state_size != codec.state_size or model.target_size != codec.target_size:
        raise ValueError("model dimensions must match the state codec")
    starts, actions, targets, eligible, episodes = _test_windows(
        Path(dataset_path), codec, max_windows
    )
    parameter = next(model.parameters())
    predicted = np.empty_like(targets)
    for start in range(0, len(starts), batch_size):
        end = min(start + batch_size, len(starts))
        state_batch = torch.as_tensor(starts[start:end], device=parameter.device, dtype=parameter.dtype)
        action_batch = torch.as_tensor(actions[start:end].T.copy(), device=parameter.device)
        trajectory = rollout(model, codec, state_batch, action_batch)
        predicted[start:end] = trajectory[list(HORIZONS), :, : codec.target_size].permute(1, 0, 2).cpu().numpy()

    metrics = {
        str(horizon): _horizon_metrics(predicted[:, index], targets[:, index], starts, codec)
        for index, horizon in enumerate(HORIZONS)
    }
    return {
        "version": "0.3.0",
        "dataset": str(dataset_path),
        "test_episodes": episodes,
        "eligible_windows": eligible,
        "evaluated_windows": len(starts),
        "window_selection": "all" if max_windows == 0 or eligible <= max_windows else "evenly_spaced",
        "metric_notes": {
            "mse": "mean squared error over four normalized Player/Box coordinates",
            "grid_mae": "mean absolute error after rounding to grid cells",
            "exact_state_accuracy": "fraction with both Player and Box at correct grid cells",
            "invalid_state_rate": "fraction with Player/Box on a wall or in the same cell",
        },
        "horizons": metrics,
    }


def save_error_plot(result: dict[str, Any], path: str | Path) -> Path:
    """Save a headless PNG comparing error at the five fixed horizons."""

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    horizons = list(HORIZONS)
    mse = [result["horizons"][str(h)]["mse"] for h in horizons]
    grid_mae = [result["horizons"][str(h)]["grid_mae"] for h in horizons]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for axis, values, title, ylabel in (
        (axes[0], mse, "Normalized position error", "MSE"),
        (axes[1], grid_mae, "Grid position error", "Mean absolute error (cells)"),
    ):
        axis.plot(horizons, values, marker="o")
        axis.set(xlabel="Rollout steps", ylabel=ylabel, title=title, xticks=horizons)
        axis.grid(alpha=0.3)
    figure.suptitle(f"Held-out test rollouts · {result['evaluated_windows']} common starts")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def run_experiment(
    *,
    checkpoint_path: str | Path = "artifacts/v0.2/state_mlp.pt",
    dataset_path: str | Path = "artifacts/v0.2/transitions.npz",
    output_dir: str | Path = "artifacts/v0.3",
    batch_size: int = 128,
    max_windows: int = 512,
) -> dict[str, Any]:
    model, codec, _ = load_checkpoint(checkpoint_path)
    result = evaluate_rollouts(
        model, codec, dataset_path, batch_size=batch_size, max_windows=max_windows
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result["checkpoint"] = str(checkpoint_path)
    result["plot"] = str(save_error_plot(result, output_dir / "rollout_errors.png"))
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NeuralWorld v0.3 long-horizon rollouts")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/v0.2/state_mlp.pt"))
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/v0.2/transitions.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/v0.3"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-windows", type=int, default=512, help="0 evaluates all eligible windows")
    args = parser.parse_args()
    result = run_experiment(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        output_dir=args.output,
        batch_size=args.batch_size,
        max_windows=args.max_windows,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
