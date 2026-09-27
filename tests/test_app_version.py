from __future__ import annotations

from importlib.metadata import version

import pytest

from game._version import APP_NAME, __version__
from game.play import main
from game.renderer import HumanRenderer


def test_package_and_app_versions_match() -> None:
    assert version("neuralworld") == __version__


def test_play_help_uses_current_app_name(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr("sys.argv", ["neuralworld-play", "--help"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 0
    assert APP_NAME in capsys.readouterr().out
    assert __version__ in APP_NAME


def test_window_caption_uses_current_app_name(monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercise the real renderer without opening a visible window in pytest/CI.
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame

    renderer = HumanRenderer(width=9, height=9, tile_size=16)
    try:
        assert pygame.display.get_caption()[0] == APP_NAME
    finally:
        renderer.close()
