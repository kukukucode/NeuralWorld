from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data.dataset import StateCodec
from experiments.planning import run_experiment
from game import Action, NeuralWorldEnv
from models.state_mlp import StateMLP, TrainingConfig, save_checkpoint
from planning import RandomShootingPlanner


def test_planner_uses_predictions_and_rejects_invalid_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = NeuralWorldEnv(layout=("#####", "#P#G#", "#...#", "#B..#", "#####"))
    observation, _ = env.reset(seed=1)
    codec = StateCodec(env.width, env.height)
    model = StateMLP(codec.state_size, hidden_sizes=(8,))
    planner = RandomShootingPlanner(model, codec, horizon=2, candidates=len(Action))

    def predict(_model: StateMLP, _codec: StateCodec, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        assert set(actions[0].tolist()) == set(Action)
        assert actions.shape == (2, len(Action))
        prediction = states.expand(3, -1, -1).clone()
        moves_down = actions[0] == Action.DOWN
        moves_right = actions[0] == Action.RIGHT
        prediction[1, moves_down, :2] = torch.tensor([0.25, 0.50])
        prediction[2, moves_down, :2] = torch.tensor([0.50, 0.50])
        # This candidate predicts the goal only after placing the player in a wall.
        prediction[1, moves_right, :2] = torch.tensor([0.50, 0.25])
        prediction[2, moves_right, :2] = torch.tensor([0.75, 0.25])
        return prediction

    monkeypatch.setattr("planning.random_shooting.rollout", predict)
    action = planner.select_action(observation, np.random.default_rng(4))
    assert action is Action.DOWN
    assert env.state.player == (1, 1)
    env.close()


def test_planning_experiment_records_matched_cases_and_metrics(tmp_path: Path) -> None:
    codec = StateCodec(9, 9)
    model = StateMLP(codec.state_size, hidden_sizes=(8,))
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    checkpoint = save_checkpoint(
        tmp_path / "model.pt", model, codec=codec, training_config=TrainingConfig(epochs=1),
    )
    # Zero weights predict the wall at (0, 0), exercising the random fallback.
    report = run_experiment(
        checkpoint_path=checkpoint, output_dir=tmp_path, horizons=(1, 2),
        candidates=len(Action), episodes=3, max_steps=2, seed=88,
    )
    assert report["cases"] == [
        {"layout_index": index, "seed": 88 + index} for index in range(3)
    ]
    assert set(report["policies"]) == {"random", "planning_1", "planning_2"}
    for metrics in report["policies"].values():
        assert metrics["episodes"] == 3
        assert metrics["success_rate"] == 0.0
        assert metrics["mean_steps_to_goal"] is None
        assert 0 <= metrics["collision_rate"] <= 1
    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8")) == report
