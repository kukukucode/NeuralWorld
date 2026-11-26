"""Export an adaptive soundtrack for a reproducible game trajectory."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from game import Action, NeuralWorldEnv
from game.layouts import LAYOUTS
from music.midi import save_midi
from music.score import Bar, MusicRules, choose_style, compose_bar
from music.state import GOAL_STATE, MusicForecaster

DEMO_ACTIONS = (Action.RIGHT,) * 6 + (Action.DOWN,) * 5


def soundtrack_bars(
    forecaster: MusicForecaster,
    rules: MusicRules,
    actions: Iterable[Action],
    *,
    seed: int,
    layout_index: int,
) -> Iterable[Bar]:
    env = NeuralWorldEnv()
    try:
        observation, _ = env.reset(seed=seed, options={"layout_index": layout_index})
        yield compose_bar(choose_style(forecaster(observation), rules), 0)
        for index, action in enumerate(actions, start=1):
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated:
                yield compose_bar(choose_style(GOAL_STATE, rules), 1)
                break
            yield compose_bar(choose_style(forecaster(observation), rules), index)
            if truncated:
                break
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the NeuralWorld adaptive soundtrack as MIDI")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/v0.2/state_mlp.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/v0.5/soundtrack.mid"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layout", type=int, choices=range(len(LAYOUTS)), default=0)
    parser.add_argument("--music-rules", type=Path, help="JSON overrides for soundtrack rules")
    parser.add_argument("--actions", choices=[action.name for action in Action], nargs="+")
    args = parser.parse_args()
    actions = DEMO_ACTIONS if args.actions is None else tuple(Action[name] for name in args.actions)
    try:
        rules = MusicRules.from_json(args.music_rules)
        forecaster = MusicForecaster.from_checkpoint(args.checkpoint)
        bars = soundtrack_bars(forecaster, rules, actions, seed=args.seed, layout_index=args.layout)
        path = save_midi(bars, args.output)
    except FileNotFoundError as exc:
        parser.error(f"{exc}. Train the model with python -m experiments.one_step to export music.")
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.error(f"could not export adaptive music: {exc}")
    print(f"Saved soundtrack to {path}")


if __name__ == "__main__":
    main()
