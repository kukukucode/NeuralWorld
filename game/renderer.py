"""NumPy headless renderer plus an optional lazy-loaded Pygame window."""

from __future__ import annotations

from typing import Any

import numpy as np

from game.objects import GameState

BACKGROUND = np.asarray((18, 24, 38), dtype=np.uint8)
GRID = np.asarray((38, 49, 69), dtype=np.uint8)
WALL = np.asarray((83, 101, 130), dtype=np.uint8)
GOAL = np.asarray((67, 201, 123), dtype=np.uint8)
BOX = np.asarray((244, 162, 76), dtype=np.uint8)
PLAYER = np.asarray((87, 166, 255), dtype=np.uint8)


def render_rgb(
    state: GameState, *, width: int, height: int, tile_size: int
) -> np.ndarray:
    """Render to RGB without initializing any display or importing Pygame."""

    frame = np.empty((height * tile_size, width * tile_size, 3), dtype=np.uint8)
    frame[:] = BACKGROUND

    def fill_tile(position: tuple[int, int], color: np.ndarray, margin: int = 0) -> None:
        x, y = position
        x0, y0 = x * tile_size + margin, y * tile_size + margin
        x1, y1 = (x + 1) * tile_size - margin, (y + 1) * tile_size - margin
        frame[y0:y1, x0:x1] = color

    for x in range(width):
        frame[:, x * tile_size : x * tile_size + 1] = GRID
    for y in range(height):
        frame[y * tile_size : y * tile_size + 1, :] = GRID
    for wall in state.walls:
        fill_tile(wall, WALL, 1)
    fill_tile(state.goal, GOAL, max(3, tile_size // 5))
    fill_tile(state.box, BOX, max(2, tile_size // 8))
    fill_tile(state.player, PLAYER, max(3, tile_size // 6))
    return frame


class HumanRenderer:
    """A tiny Pygame adapter kept separate from headless simulation."""

    def __init__(self, *, width: int, height: int, tile_size: int, fps: int = 30):
        try:
            import pygame
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                'Human rendering requires Pygame: pip install "pygame>=2.5,<3"'
            ) from exc

        self.pygame: Any = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width * tile_size, height * tile_size))
        pygame.display.set_caption("NeuralWorld v0.1")
        self.clock = pygame.time.Clock()
        self.fps = fps

    def draw(self, frame: np.ndarray) -> None:
        surface = self.pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        self.screen.blit(surface, (0, 0))
        self.pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self) -> None:
        self.pygame.display.quit()
        self.pygame.quit()

