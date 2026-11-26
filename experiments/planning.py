"""Benchmark receding-horizon planning against uniform random actions."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from game import Action, NeuralWorldEnv
from game._version import __version__
from game.layouts import LAYOUTS
from models.state_mlp import load_checkpoint
from planning import RandomShootingPlanner

DEFAULT_HORIZONS = (1, 5, 10)
Policy = Callable[[dict[str, object], np.random.Generator], Action]


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    success: bool
    steps: int
    total_reward: float
    collisions: int
    decision_seconds: float


def _random_action(_observation: dict[str, object], rng: np.random.Generator) -> Action:
    return Action(int(rng.integers(len(Action))))


def run_episode(
    env: NeuralWorldEnv, policy: Policy, *, layout_index: int, seed: int,
) -> EpisodeResult:
    observation, _ = env.reset(seed=seed, options={"layout_index": layout_index})
    rng = np.random.default_rng(seed)
    total_reward = 0.0
    collisions = 0
    decision_seconds = 0.0
    for step in range(1, env.max_steps + 1):
        started = perf_counter()
        action = policy(observation, rng)
        decision_seconds += perf_counter() - started
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        collisions += int(info["collided"])
        if terminated or truncated:
            return EpisodeResult(bool(info["success"]), step, total_reward, collisions, decision_seconds)
    raise RuntimeError("the environment did not stop at its step limit")


def _summarize(episodes: list[EpisodeResult]) -> dict[str, float | int | None]:
    successes = [episode.steps for episode in episodes if episode.success]
    total_steps = sum(episode.steps for episode in episodes)
    return {
        "episodes": len(episodes),
        "successes": len(successes),
        "success_rate": len(successes) / len(episodes),
        "mean_steps_to_goal": sum(successes) / len(successes) if successes else None,
        "average_return": sum(episode.total_reward for episode in episodes) / len(episodes),
        "collision_rate": sum(episode.collisions for episode in episodes) / total_steps,
        "mean_decision_ms": 1000 * sum(episode.decision_seconds for episode in episodes) / total_steps,
    }


def run_experiment(
    *,
    checkpoint_path: str | Path = "artifacts/v0.2/state_mlp.pt",
    output_dir: str | Path = "artifacts/v0.4",
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    candidates: int = 64,
    episodes: int = 30,
    max_steps: int = 100,
    seed: int = 2027,
) -> dict[str, Any]:
    """Compare every policy on the same sequence of layouts and reset seeds."""
    if episodes < 1 or seed < 0:
        raise ValueError("episodes must be positive and seed must be non-negative")
    if not horizons or len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be non-empty and unique")
    model, codec, _ = load_checkpoint(checkpoint_path)
    policies: dict[str, Policy] = {"random": _random_action}
    for horizon in horizons:
        planner = RandomShootingPlanner(model, codec, horizon=horizon, candidates=candidates)
        policies[f"planning_{horizon}"] = planner.select_action

    cases = [(index % len(LAYOUTS), seed + index) for index in range(episodes)]
    env = NeuralWorldEnv(max_steps=max_steps)
    try:
        if (env.width, env.height) != (codec.width, codec.height):
            raise ValueError("environment dimensions do not match the checkpoint codec")
        results = {
            name: _summarize([
                run_episode(env, policy, layout_index=layout, seed=episode_seed)
                for layout, episode_seed in cases
            ])
            for name, policy in policies.items()
        }
    finally:
        env.close()

    result = {
        "version": __version__,
        "checkpoint": str(checkpoint_path),
        "seed": seed,
        "max_steps": max_steps,
        "candidates": candidates,
        "horizons": list(horizons),
        "cases": [{"layout_index": layout, "seed": episode_seed} for layout, episode_seed in cases],
        "policies": results,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Compare NeuralWorld v{__version__} planning policies")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/v0.2/state_mlp.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/v0.4"))
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    result = run_experiment(
        checkpoint_path=args.checkpoint, output_dir=args.output, horizons=args.horizons,
        candidates=args.candidates, episodes=args.episodes, max_steps=args.max_steps, seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
