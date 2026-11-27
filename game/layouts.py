"""Layouts for experiments and player challenges."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from random import Random

from game.objects import Coordinate, GameState

LAYOUTS: tuple[tuple[str, ...], ...] = (
    (
        "#########",
        "#P......#",
        "#..##...#",
        "#..B....#",
        "#.......#",
        "#...#...#",
        "#......G#",
        "#.......#",
        "#########",
    ),
    (
        "#########",
        "#......G#",
        "#.#.....#",
        "#.#.B...#",
        "#.#.....#",
        "#...###.#",
        "#P......#",
        "#.......#",
        "#########",
    ),
    (
        "#########",
        "#P......#",
        "#...#...#",
        "#...#...#",
        "#...B...#",
        "#.......#",
        "#.###...#",
        "#G......#",
        "#########",
    ),
)

# Explicit choices for play. A separate catalog keeps seeded experiments reproducible.
CHALLENGE_LAYOUTS: dict[str, tuple[str, ...]] = {
    "maze": (
        "#########",
        "#P.#...G#",
        "#..#.#..#",
        "##.#.#..#",
        "#..#.#.##",
        "#..#.#..#",
        "#..#.##.#",
        "#....#B.#",
        "#########",
    ),
    "box-gate": (
        "#########",
        "#..##..G#",
        "#P.##.#.#",
        "#.###.#.#",
        "#..B...##",
        "#..##.#.#",
        "#..##...#",
        "#..##...#",
        "#########",
    ),
}


def _neighbors(cell: Coordinate, step: int = 1) -> tuple[Coordinate, ...]:
    x, y = cell
    return ((x + step, y), (x - step, y), (x, y + step), (x, y - step))


def _farthest_cell(grid: list[list[str]], start: Coordinate) -> Coordinate:
    pending = deque([start])
    seen = {start}
    farthest = start
    while pending:
        farthest = pending.popleft()
        for x, y in _neighbors(farthest):
            cell = (x, y)
            if grid[y][x] == "." and cell not in seen:
                seen.add(cell)
                pending.append(cell)
    return farthest


def generate_maze(seed: int) -> tuple[str, ...]:
    """Carve a seeded 9×9 maze with a route that never requires moving the box."""
    rng = Random(seed)
    size = 9
    grid = [["#"] * size for _ in range(size)]
    rooms = [(x, y) for y in range(1, size, 2) for x in range(1, size, 2)]
    box = rng.choice(((1, 1), (1, size - 2), (size - 2, 1), (size - 2, size - 2)))
    rooms.remove(box)
    start = rng.choice(rooms)
    remaining = set(rooms) - {start}
    grid[start[1]][start[0]] = "."
    trail = [start]
    # Each newly visited room gets exactly one passage to the existing maze.
    while trail:
        x, y = trail[-1]
        choices = [room for room in _neighbors((x, y), 2) if room in remaining]
        if not choices:
            trail.pop()
            continue
        nx, ny = rng.choice(choices)
        grid[(y + ny) // 2][(x + nx) // 2] = "."
        grid[ny][nx] = "."
        remaining.remove((nx, ny))
        trail.append((nx, ny))

    # The two farthest-point searches find the ends of a longest path in the tree.
    player = _farthest_cell(grid, start)
    goal = _farthest_cell(grid, player)
    grid[player[1]][player[0]] = "P"
    grid[goal[1]][goal[0]] = "G"

    # Attach the reserved corner as a dead end so the box cannot block the solution.
    bx, by = box
    nx, ny = rng.choice([room for room in _neighbors(box, 2) if room in rooms])
    grid[(by + ny) // 2][(bx + nx) // 2] = "."
    grid[by][bx] = "B"
    return tuple("".join(row) for row in grid)


def parse_layout(rows: Sequence[str]) -> tuple[GameState, int, int]:
    """Validate and parse an ASCII layout into an initial game state."""

    if not rows:
        raise ValueError("A layout must contain at least one row")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("All layout rows must have the same non-zero width")

    found: dict[str, list[Coordinate]] = {"P": [], "B": [], "G": []}
    walls: set[Coordinate] = set()
    valid = {"#", ".", "P", "B", "G"}
    for y, row in enumerate(rows):
        for x, tile in enumerate(row):
            if tile not in valid:
                raise ValueError(f"Unknown layout tile {tile!r} at ({x}, {y})")
            if tile == "#":
                walls.add((x, y))
            elif tile in found:
                found[tile].append((x, y))

    for tile, name in (("P", "player"), ("B", "box"), ("G", "goal")):
        if len(found[tile]) != 1:
            raise ValueError(f"Layout must contain exactly one {name} ({tile})")

    height = len(rows)
    boundary = (
        {(x, 0) for x in range(width)}
        | {(x, height - 1) for x in range(width)}
        | {(0, y) for y in range(height)}
        | {(width - 1, y) for y in range(height)}
    )
    if not boundary.issubset(walls):
        raise ValueError("The layout boundary must be entirely enclosed by walls")

    state = GameState(
        player=found["P"][0],
        box=found["B"][0],
        goal=found["G"][0],
        walls=frozenset(walls),
    )
    return state, width, height

