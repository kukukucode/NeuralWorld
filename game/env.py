"""Gymnasium environment for the deterministic NeuralWorld v0.1 game."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from game.layouts import LAYOUTS, parse_layout
from game.objects import Action, GameState
from game.physics import apply_action
from game.renderer import HumanRenderer, render_rgb

RenderMode = Literal["human", "rgb_array"] | None


class NeuralWorldEnv(gym.Env[dict[str, object], int]):
    """A deterministic tile world with walls, one box, and one goal."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        *,
        render_mode: RenderMode = None,
        layout: Sequence[str] | None = None,
        max_steps: int = 200,
        tile_size: int = 32,
    ) -> None:
        if render_mode not in self.metadata["render_modes"] and render_mode is not None:
            raise ValueError(f"Unsupported render mode: {render_mode!r}")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if tile_size < 8:
            raise ValueError("tile_size must be at least 8")

        initial, width, height = parse_layout(layout or LAYOUTS[0])
        self._custom_layout = tuple(layout) if layout is not None else None
        self._state = initial
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.tile_size = tile_size
        self.render_mode = render_mode
        self._human_renderer: HumanRenderer | None = None

        coordinate_low = np.asarray((0, 0), dtype=np.int32)
        coordinate_high = np.asarray((width - 1, height - 1), dtype=np.int32)
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Dict(
            {
                "player": spaces.Box(coordinate_low, coordinate_high, dtype=np.int32),
                "box": spaces.Box(coordinate_low, coordinate_high, dtype=np.int32),
                "goal": spaces.Box(coordinate_low, coordinate_high, dtype=np.int32),
                "walls": spaces.MultiBinary((height, width)),
                "step_count": spaces.Discrete(max_steps + 1),
            }
        )

    @property
    def state(self) -> GameState:
        """Return the immutable ground-truth state."""

        return self._state

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, object], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        if self._custom_layout is not None:
            rows = self._custom_layout
            layout_index: int | None = None
        else:
            requested = options.get("layout_index")
            if requested is None:
                layout_index = int(self.np_random.integers(len(LAYOUTS)))
            else:
                layout_index = int(requested)
                if not 0 <= layout_index < len(LAYOUTS):
                    raise ValueError(f"layout_index must be in [0, {len(LAYOUTS) - 1}]")
            rows = LAYOUTS[layout_index]

        self._state, width, height = parse_layout(rows)
        if (width, height) != (self.width, self.height):
            raise ValueError("All layouts used by an environment must have equal dimensions")

        observation = self._observation()
        info = {"layout_index": layout_index, "seed": seed, "success": False}
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(
        self, action: int | Action
    ) -> tuple[dict[str, object], float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action!r}")

        result = apply_action(self._state, Action(int(action)))
        self._state = result.state

        terminated = self._state.player == self._state.goal
        truncated = self._state.step_count >= self.max_steps and not terminated
        reward = -0.01
        if result.collided:
            reward -= 0.04
        if result.pushed:
            reward += 0.02
        if terminated:
            reward += 1.0

        info = {
            "collided": result.collided,
            "moved": result.moved,
            "pushed": result.pushed,
            "success": terminated,
        }
        if self.render_mode == "human":
            self.render()
        return self._observation(), reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        frame = render_rgb(
            self._state,
            width=self.width,
            height=self.height,
            tile_size=self.tile_size,
        )
        if self.render_mode == "rgb_array":
            return frame
        if self.render_mode == "human":
            if self._human_renderer is None:
                self._human_renderer = HumanRenderer(
                    width=self.width,
                    height=self.height,
                    tile_size=self.tile_size,
                    fps=self.metadata["render_fps"],
                )
            self._human_renderer.draw(frame)
        return None

    def close(self) -> None:
        if self._human_renderer is not None:
            self._human_renderer.close()
            self._human_renderer = None

    def _observation(self) -> dict[str, object]:
        return self._state.observation(width=self.width, height=self.height)

