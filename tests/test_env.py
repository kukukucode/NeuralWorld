from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from game import Action, NeuralWorldEnv


def assert_observations_equal(left: dict[str, object], right: dict[str, object]) -> None:
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


def test_environment_passes_gymnasium_checker() -> None:
    check_env(NeuralWorldEnv())


def test_same_seed_and_actions_reproduce_identical_trajectory() -> None:
    actions = [Action.RIGHT, Action.DOWN, Action.DOWN, Action.LEFT, Action.NOOP] * 3
    first = NeuralWorldEnv()
    second = NeuralWorldEnv()

    first_observation, first_info = first.reset(seed=1729)
    second_observation, second_info = second.reset(seed=1729)
    assert_observations_equal(first_observation, second_observation)
    assert first_info == second_info

    for action in actions:
        first_step = first.step(action)
        second_step = second.step(action)
        assert_observations_equal(first_step[0], second_step[0])
        assert first_step[1:] == second_step[1:]


def test_wall_collision_preserves_player_position() -> None:
    env = NeuralWorldEnv()
    env.reset(seed=0, options={"layout_index": 0})
    before = env.state.player
    _, reward, _, _, info = env.step(Action.UP)
    assert env.state.player == before
    assert reward == pytest.approx(-0.05)
    assert info["collided"] is True


def test_box_is_pushed_into_free_cell() -> None:
    layout = (
        "#####",
        "#PBG#",
        "#...#",
        "#...#",
        "#####",
    )
    env = NeuralWorldEnv(layout=layout)
    env.reset(seed=1)
    _, reward, _, _, info = env.step(Action.RIGHT)
    assert env.state.player == (2, 1)
    assert env.state.box == (3, 1)
    assert reward == pytest.approx(0.01)
    assert info["pushed"] is True


def test_box_cannot_be_pushed_through_wall() -> None:
    layout = (
        "#####",
        "#PB##",
        "#..G#",
        "#...#",
        "#####",
    )
    env = NeuralWorldEnv(layout=layout)
    env.reset(seed=1)
    env.step(Action.RIGHT)
    assert env.state.player == (1, 1)
    assert env.state.box == (2, 1)


def test_goal_terminates_episode_with_reward() -> None:
    layout = (
        "#####",
        "#PG.#",
        "#B..#",
        "#...#",
        "#####",
    )
    env = NeuralWorldEnv(layout=layout)
    env.reset(seed=1)
    _, reward, terminated, truncated, info = env.step(Action.RIGHT)
    assert terminated is True
    assert truncated is False
    assert reward == pytest.approx(0.99)
    assert info["success"] is True


def test_time_limit_truncates_episode() -> None:
    env = NeuralWorldEnv(max_steps=2)
    env.reset(seed=0)
    env.step(Action.NOOP)
    _, _, terminated, truncated, _ = env.step(Action.NOOP)
    assert terminated is False
    assert truncated is True


def test_rgb_array_render_is_headless_and_deterministic() -> None:
    env = NeuralWorldEnv(render_mode="rgb_array", tile_size=16)
    env.reset(seed=99)
    first = env.render()
    second = env.render()
    assert isinstance(first, np.ndarray)
    assert first.shape == (144, 144, 3)
    assert first.dtype == np.uint8
    np.testing.assert_array_equal(first, second)

