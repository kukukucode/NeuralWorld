from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from data.dataset import Split, StateCodec, load_dataset
from data.generate import generate_dataset
from game import NeuralWorldEnv


def test_generation_is_reproducible(tmp_path: Path) -> None:
    first = generate_dataset(tmp_path / "first.npz", episodes=9, steps_per_episode=8, seed=7)
    second = generate_dataset(tmp_path / "second.npz", episodes=9, steps_per_episode=8, seed=7)

    with np.load(first) as left, np.load(second) as right:
        assert left.files == right.files
        for name in left.files:
            np.testing.assert_array_equal(left[name], right[name])


def test_splits_are_disjoint_at_episode_level(tmp_path: Path) -> None:
    path = generate_dataset(tmp_path / "data.npz", episodes=12, steps_per_episode=6)
    with np.load(path) as archive:
        episode_ids = archive["episode_ids"]
        splits = archive["splits"]
        episode_sets = [
            set(episode_ids[splits == int(split)].tolist()) for split in Split
        ]
    assert all(episode_sets)
    assert episode_sets[0].isdisjoint(episode_sets[1])
    assert episode_sets[0].isdisjoint(episode_sets[2])
    assert episode_sets[1].isdisjoint(episode_sets[2])


def test_loaded_dataset_has_expected_shapes(tmp_path: Path) -> None:
    path = generate_dataset(tmp_path / "data.npz", episodes=9, steps_per_episode=5)
    bundle = load_dataset(path)
    state, action, target = bundle.train[0]
    assert state.shape == (bundle.codec.state_size,)
    assert action.shape == ()
    assert target.shape == (bundle.codec.target_size,)
    assert sum(map(len, (bundle.train, bundle.validation, bundle.test))) <= 45


def test_predicted_positions_replace_only_player_and_box() -> None:
    env = NeuralWorldEnv()
    observation, _ = env.reset(seed=7, options={"layout_index": 0})
    codec = StateCodec(width=env.width, height=env.height)
    state = torch.from_numpy(codec.encode(observation))
    original = state.clone()
    prediction = torch.tensor((0.375, 0.5, 0.625, 0.75))

    next_state = codec.with_predicted_positions(state, prediction)

    torch.testing.assert_close(next_state[:4], prediction)
    torch.testing.assert_close(next_state[4:], state[4:])
    torch.testing.assert_close(state, original)
    assert next_state.shape == state.shape


def test_predicted_positions_support_batches_and_clip_to_world_bounds() -> None:
    env = NeuralWorldEnv()
    codec = StateCodec(width=env.width, height=env.height)
    observations = [
        env.reset(seed=7, options={"layout_index": index})[0] for index in (0, 1)
    ]
    states = torch.stack(
        [torch.from_numpy(codec.encode(observation)) for observation in observations]
    )
    predictions = torch.tensor(
        ((-0.2, 0.375, 1.2, 0.625), (0.25, 1.1, 0.75, -0.1))
    )

    next_states = codec.with_predicted_positions(states, predictions)

    torch.testing.assert_close(next_states[:, :4], predictions.clamp(0.0, 1.0))
    torch.testing.assert_close(next_states[:, 4:], states[:, 4:])
    assert next_states.shape == states.shape


def test_predicted_positions_reject_incompatible_shapes() -> None:
    codec = StateCodec(width=9, height=9)
    state = torch.zeros(codec.state_size)

    with pytest.raises(ValueError, match="state must end"):
        codec.with_predicted_positions(state[:-1], torch.zeros(codec.target_size))
    with pytest.raises(ValueError, match="must have shape"):
        codec.with_predicted_positions(state, torch.zeros(2, codec.target_size))
