"""Generate reproducible gameplay transitions from the ground-truth engine."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data.dataset import Split, StateCodec
from game import NeuralWorldEnv


def episode_splits(
    episodes: int,
    *,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> np.ndarray:
    """Assign entire episodes to disjoint splits with deterministic shuffling."""

    if episodes < 3:
        raise ValueError("At least three episodes are required for train/validation/test")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must leave room for test data")

    train_count = max(1, int(episodes * train_fraction))
    validation_count = max(1, int(episodes * validation_fraction))
    if train_count + validation_count >= episodes:
        validation_count = 1
        train_count = episodes - 2

    assignments = np.full(episodes, int(Split.TEST), dtype=np.int8)
    permutation = np.random.default_rng(seed).permutation(episodes)
    assignments[permutation[:train_count]] = int(Split.TRAIN)
    assignments[permutation[train_count : train_count + validation_count]] = int(
        Split.VALIDATION
    )
    return assignments


def generate_dataset(
    output: str | Path,
    *,
    episodes: int = 300,
    steps_per_episode: int = 100,
    seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> Path:
    """Generate and save transitions, including episode-level split labels."""

    if steps_per_episode < 1:
        raise ValueError("steps_per_episode must be at least 1")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    assignments = episode_splits(
        episodes,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    env = NeuralWorldEnv(max_steps=steps_per_episode)
    codec = StateCodec(width=env.width, height=env.height)

    states: list[np.ndarray] = []
    actions: list[int] = []
    next_states: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[bool] = []
    episode_ids: list[int] = []
    step_indices: list[int] = []
    splits: list[int] = []
    layout_indices: list[int] = []

    try:
        for episode_id in range(episodes):
            observation, reset_info = env.reset(seed=seed + episode_id)
            layout_index = int(reset_info["layout_index"])
            for step_index in range(steps_per_episode):
                action = int(rng.integers(env.action_space.n))
                next_observation, reward, terminated, truncated, _ = env.step(action)

                states.append(codec.encode(observation))
                actions.append(action)
                next_states.append(codec.encode_target(next_observation))
                rewards.append(reward)
                dones.append(terminated or truncated)
                episode_ids.append(episode_id)
                step_indices.append(step_index)
                splits.append(int(assignments[episode_id]))
                layout_indices.append(layout_index)
                observation = next_observation
                if terminated or truncated:
                    break
    finally:
        env.close()

    np.savez_compressed(
        output,
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        next_states=np.asarray(next_states, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.bool_),
        episode_ids=np.asarray(episode_ids, dtype=np.int32),
        step_indices=np.asarray(step_indices, dtype=np.int32),
        splits=np.asarray(splits, dtype=np.int8),
        layout_indices=np.asarray(layout_indices, dtype=np.int8),
        width=np.asarray(env.width, dtype=np.int32),
        height=np.asarray(env.height, dtype=np.int32),
        episodes=np.asarray(episodes, dtype=np.int32),
        seed=np.asarray(seed, dtype=np.int64),
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NeuralWorld transitions")
    parser.add_argument("--output", type=Path, default=Path("artifacts/transitions.npz"))
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path = generate_dataset(
        args.output,
        episodes=args.episodes,
        steps_per_episode=args.steps,
        seed=args.seed,
    )
    print(f"Saved dataset to {path}")


if __name__ == "__main__":
    main()
