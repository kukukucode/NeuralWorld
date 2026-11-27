from __future__ import annotations

import pytest

from game import Action, NeuralWorldEnv
from game.layouts import CHALLENGE_LAYOUTS, parse_layout


def test_layout_requires_closed_boundary() -> None:
    with pytest.raises(ValueError, match="boundary"):
        parse_layout(("#P###", "#B.G#", "#...#", "#...#", "##.##"))


def test_layout_requires_one_of_each_object() -> None:
    with pytest.raises(ValueError, match="exactly one goal"):
        parse_layout(("#####", "#P..#", "#B..#", "#...#", "#####"))


@pytest.mark.parametrize(
    ("name", "route", "pushes"),
    [
        ("maze", "RDDDDDDRRUUUUUURRR", 0),
        ("box-gate", "DDRRRRUUURR", 3),
    ],
)
def test_challenge_layout_can_be_solved(name: str, route: str, pushes: int) -> None:
    actions = {"R": Action.RIGHT, "D": Action.DOWN, "U": Action.UP}
    env = NeuralWorldEnv(layout=CHALLENGE_LAYOUTS[name])
    try:
        observation, _ = env.reset(seed=42)
        assert observation["walls"].shape == (9, 9)
        pushed = 0
        for index, move in enumerate(route):
            _, _, terminated, truncated, info = env.step(actions[move])
            assert not info["collided"]
            assert not truncated
            assert terminated == (index == len(route) - 1)
            pushed += info["pushed"]
        assert pushed == pushes
    finally:
        env.close()

