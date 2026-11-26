from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pygame
import pytest
import torch
from mido import MidiFile, bpm2tempo

from data.dataset import StateCodec
from game import NeuralWorldEnv
from game.play import main as play
from models.state_mlp import StateMLP, TrainingConfig, save_checkpoint
from music.midi import save_midi
from music.playback import SoundtrackPlayer
from music.score import BAR_TICKS, MusicRules, choose_style, compose_bar
from music.state import GOAL_STATE, MusicForecaster, MusicState

CALM = MusicState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
TENSE = replace(CALM, danger=1.0)


def test_future_collision_changes_music_from_the_same_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = NeuralWorldEnv()
    observation, _ = env.reset(seed=0, options={"layout_index": 0})
    codec = StateCodec(env.width, env.height)
    model = StateMLP(codec.state_size, hidden_sizes=(8,))
    forecast = MusicForecaster(model, codec)

    def predict(
        _model: StateMLP, _codec: StateCodec, states: torch.Tensor, actions: torch.Tensor,
    ) -> torch.Tensor:
        assert actions.shape == (3, 25)
        trajectory = states.expand(4, -1, -1).clone()
        trajectory[1:, :, 0] = future_x[:, None] / (codec.width - 1)
        return trajectory

    monkeypatch.setattr("music.state.rollout", predict)
    future_x = torch.tensor([2, 3, 4])
    safe = forecast(observation)
    future_x = torch.tensor([0, 0, 0])
    risky = forecast(observation)
    env.close()

    assert safe.predicted_collision_probability == 0
    assert risky.predicted_collision_probability == 1
    assert risky.danger > safe.danger
    assert choose_style(safe, MusicRules()).mood == "calm"
    assert choose_style(risky, MusicRules()).mood == "tension"


def test_midi_changes_tempo_harmony_and_motif_with_danger(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.json"
    rules_path.write_text('{"tension_bpm": 108}', encoding="utf-8")
    rules = MusicRules.from_json(rules_path)
    calm = compose_bar(choose_style(CALM, rules), 3)
    tense = compose_bar(choose_style(TENSE, rules), 3)
    goal = compose_bar(choose_style(GOAL_STATE, rules), 1)
    track = MidiFile(save_midi([calm, tense, goal], tmp_path / "test.mid")).tracks[0]
    assert [message.tempo for message in track if message.type == "set_tempo"] == [
        bpm2tempo(84), bpm2tempo(108), bpm2tempo(96),
    ]
    assert sum(message.time for message in track) == 3 * BAR_TICKS

    notes: list[set[tuple[int, int]]] = [set(), set(), set()]
    tick = 0
    for message in track:
        tick += message.time
        if message.type == "note_on":
            notes[tick // BAR_TICKS].add((message.channel, message.note))
    assert (0, 65) not in notes[0] and (0, 65) in notes[1]  # G7 adds the seventh.
    assert (1, 71) in notes[1]  # Tension motif.
    assert (1, 72) in notes[2] and (0, 48) in notes[2]  # Final tonic.


def test_play_command_starts_audio_and_updates_on_movement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    codec = StateCodec(9, 9)
    model = StateMLP(codec.state_size, hidden_sizes=(8,))
    checkpoint = save_checkpoint(
        tmp_path / "model.pt", model, codec=codec, training_config=TrainingConfig(epochs=1),
    )
    monkeypatch.setattr("sys.argv", ["neuralworld-play", "--music", "--music-checkpoint", str(checkpoint)])
    events = iter([
        [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT)],
        [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)],
        [pygame.event.Event(pygame.QUIT)],
    ])
    monkeypatch.setattr(pygame.event, "get", lambda: next(events))
    audio_started: list[bool] = []
    state_updates: list[MusicState] = []
    original_update = SoundtrackPlayer.update
    original_set_state = SoundtrackPlayer.set_state

    def update(self: SoundtrackPlayer) -> None:
        original_update(self)
        current = pygame.mixer.Channel(0).get_sound()
        audio_started.append(current is not None and any(current.get_raw()))

    def set_state(self: SoundtrackPlayer, state: MusicState) -> None:
        state_updates.append(state)
        original_set_state(self, state)

    monkeypatch.setattr(SoundtrackPlayer, "update", update)
    monkeypatch.setattr(SoundtrackPlayer, "set_state", set_state)
    play()
    assert audio_started and all(audio_started)
    assert len(state_updates) == 1
    assert pygame.mixer.get_init() is None


def test_audio_queue_adapts_at_the_next_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    rules = MusicRules()
    player = SoundtrackPlayer(rules, CALM)
    try:
        channel = pygame.mixer.Channel(0)
        player.update()
        player.update()
        before = channel.get_queue()
        assert before is not None
        player.set_state(TENSE)
        player.update()
        after = channel.get_queue()
        assert after is not None
        assert before.get_length() == pytest.approx(120 / rules.calm_bpm, abs=0.001)
        assert after.get_length() == pytest.approx(120 / rules.tension_bpm, abs=0.001)
    finally:
        player.close()
        pygame.mixer.quit()
