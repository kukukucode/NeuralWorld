from __future__ import annotations

import pytest

from game.layouts import parse_layout


def test_layout_requires_closed_boundary() -> None:
    with pytest.raises(ValueError, match="boundary"):
        parse_layout(("#P###", "#B.G#", "#...#", "#...#", "##.##"))


def test_layout_requires_one_of_each_object() -> None:
    with pytest.raises(ValueError, match="exactly one goal"):
        parse_layout(("#####", "#P..#", "#B..#", "#...#", "#####"))

