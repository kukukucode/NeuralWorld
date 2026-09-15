"""Small curated layouts selected deterministically from the reset seed."""

from __future__ import annotations

from collections.abc import Sequence

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

