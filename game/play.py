"""Keyboard-controlled entry point for the game environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from game._version import APP_NAME
from game.env import NeuralWorldEnv
from game.objects import Action


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Play {APP_NAME}")
    parser.add_argument("--seed", type=int, default=0, help="deterministic reset seed")
    parser.add_argument("--music", action="store_true", help="play the adaptive soundtrack")
    parser.add_argument(
        "--music-checkpoint", type=Path, default=Path("artifacts/v0.2/state_mlp.pt"),
        help="trained model for the adaptive soundtrack",
    )
    parser.add_argument("--music-rules", type=Path, help="JSON overrides for soundtrack rules")
    args = parser.parse_args()

    try:
        import pygame
    except ImportError as exc:
        raise SystemExit('Pygame is required: pip install "pygame>=2.5,<3"') from exc

    if args.music:
        from music.playback import SoundtrackPlayer
        from music.score import MusicRules
        from music.state import MusicForecaster

        try:
            rules = MusicRules.from_json(args.music_rules)
            forecaster = MusicForecaster.from_checkpoint(args.music_checkpoint)
        except FileNotFoundError as exc:
            parser.error(f"{exc}. Train the model with python -m experiments.one_step for --music.")
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            parser.error(f"could not load adaptive music: {exc}")

    key_actions = {
        pygame.K_UP: Action.UP,
        pygame.K_w: Action.UP,
        pygame.K_DOWN: Action.DOWN,
        pygame.K_s: Action.DOWN,
        pygame.K_LEFT: Action.LEFT,
        pygame.K_a: Action.LEFT,
        pygame.K_RIGHT: Action.RIGHT,
        pygame.K_d: Action.RIGHT,
        pygame.K_SPACE: Action.NOOP,
    }

    env = NeuralWorldEnv(render_mode="human")
    soundtrack = None
    try:
        if args.music and (env.width, env.height) != (forecaster.codec.width, forecaster.codec.height):
            parser.error("music checkpoint dimensions do not match the game")
        observation, _ = env.reset(seed=args.seed)
        if args.music:
            try:
                soundtrack = SoundtrackPlayer(rules, forecaster(observation))
                soundtrack.update()
            except pygame.error as exc:
                if soundtrack is not None:
                    soundtrack.close()
                soundtrack = None
                print(f"Adaptive music unavailable: {exc}", file=sys.stderr)

        running = True
        while running:
            music_dirty = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break
                    elif event.key == pygame.K_r:
                        observation, _ = env.reset(seed=args.seed)
                        music_dirty = True
                    elif event.key in key_actions:
                        observation, _, terminated, truncated, info = env.step(key_actions[event.key])
                        music_dirty |= info["moved"]
                        if terminated or truncated:
                            if terminated and soundtrack is not None:
                                soundtrack.play_goal()
                            pygame.time.wait(350)
                            observation, _ = env.reset(seed=args.seed)
                            music_dirty = True
            if running and soundtrack is not None:
                if music_dirty:
                    soundtrack.set_state(forecaster(observation))
                soundtrack.update()
            env.render()
    finally:
        try:
            if soundtrack is not None:
                soundtrack.close()
        finally:
            env.close()


if __name__ == "__main__":
    main()

