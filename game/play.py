"""Keyboard-controlled entry point for the v0.1 environment."""

from __future__ import annotations

import argparse

from game.env import NeuralWorldEnv
from game.objects import Action


def main() -> None:
    parser = argparse.ArgumentParser(description="Play NeuralWorld v0.1")
    parser.add_argument("--seed", type=int, default=0, help="deterministic reset seed")
    args = parser.parse_args()

    try:
        import pygame
    except ImportError as exc:
        raise SystemExit('Pygame is required: pip install "pygame>=2.5,<3"') from exc

    env = NeuralWorldEnv(render_mode="human")
    env.reset(seed=args.seed)
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

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        env.reset(seed=args.seed)
                    elif event.key in key_actions:
                        _, _, terminated, truncated, _ = env.step(key_actions[event.key])
                        if terminated or truncated:
                            pygame.time.wait(350)
                            env.reset(seed=args.seed)
            env.render()
    finally:
        env.close()


if __name__ == "__main__":
    main()

