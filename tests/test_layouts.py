from __future__ import annotations

from collections import deque

import pytest

from game import Action, NeuralWorldEnv
from game.layouts import CHALLENGE_LAYOUTS, generate_maze, parse_layout
from game.objects import ACTION_DELTAS


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


def test_generated_maze_seed_is_reproducible_and_varies() -> None:
    first = generate_maze(42)
    assert first == generate_maze(42)
    assert first != generate_maze(43)


def test_generated_mazes_have_a_path_without_moving_the_box() -> None:
    for seed in range(100):
        state, width, height = parse_layout(generate_maze(seed))
        assert (width, height) == (9, 9)
        blocked = state.walls | {state.box}
        reachable = {state.player}
        pending = deque([state.player])
        while pending:
            x, y = pending.popleft()
            for dx, dy in ACTION_DELTAS.values():
                neighbor = (x + dx, y + dy)
                if neighbor not in blocked and neighbor not in reachable:
                    reachable.add(neighbor)
                    pending.append(neighbor)
        assert state.goal in reachable, f"seed {seed}"
        assert len(reachable) == width * height - len(blocked), f"seed {seed}"

