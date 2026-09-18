from __future__ import annotations

from pathlib import Path

import numpy as np

from data.dataset import Split, load_dataset
from data.generate import generate_dataset


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
