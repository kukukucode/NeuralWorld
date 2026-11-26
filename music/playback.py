"""Play the generated note events through Pygame's audio mixer."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pygame

from music.score import BAR_TICKS, TICKS_PER_BEAT, Bar, MusicRules, MusicStyle, choose_style, compose_bar
from music.state import GOAL_STATE, MusicState


def render_bar(bar: Bar, sample_rate: int, channels: int) -> np.ndarray:
    """Render one bar to signed 16-bit PCM using a lightweight synth."""
    samples_per_tick = sample_rate * 60 / (bar.style.bpm * TICKS_PER_BEAT)
    samples = np.zeros(round(BAR_TICKS * samples_per_tick), dtype=np.float32)
    for note in bar.notes:
        start = round(note.start * samples_per_tick)
        end = round((note.start + note.duration) * samples_per_tick)
        time = np.arange(end - start, dtype=np.float32) / sample_rate
        frequency = 440 * 2 ** ((note.pitch - 69) / 12)
        phase = 2 * np.pi * frequency * time
        fade_in = time / 0.012
        fade_out = (time[-1] - time) / 0.055
        envelope = np.minimum(1.0, np.minimum(fade_in, fade_out))
        if note.channel == 1:
            wave = (np.sin(phase) + 0.20 * np.sin(phase * 2)) * np.exp(-2.4 * time)
            gain = 0.16
        else:
            wave = np.sin(phase) + 0.08 * np.sin(phase * 2)
            gain = 0.08
        samples[start:end] += wave * envelope * (gain * note.velocity / 127)
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    return pcm if channels == 1 else np.repeat(pcm[:, None], channels, axis=1)


class SoundtrackPlayer:
    """Queue short bars and apply a new style at the next bar boundary."""

    def __init__(self, rules: MusicRules, state: MusicState) -> None:
        if pygame.mixer.get_init() is None:
            pygame.mixer.init()
        sample_rate, sample_format, channels = pygame.mixer.get_init()
        if sample_format != -16:
            raise pygame.error("adaptive music requires a signed 16-bit audio mixer")
        self._channel = pygame.mixer.Channel(0)
        self._rules = rules
        self._style = choose_style(state, rules)
        self._queued_style: MusicStyle | None = None
        self._next_bar = 0

        @lru_cache(maxsize=24)
        def sound(style: MusicStyle, phase: int) -> pygame.mixer.Sound:
            pcm = render_bar(compose_bar(style, phase), sample_rate, channels)
            return pygame.sndarray.make_sound(pcm)

        self._sound = sound

    def set_state(self, state: MusicState) -> None:
        self._style = choose_style(state, self._rules)

    def update(self) -> None:
        if self._channel.get_queue() is None:
            self._channel.queue(self._sound(self._style, self._next_bar % 4))
            self._next_bar += 1
            self._queued_style = self._style
        elif self._queued_style != self._style:
            self._channel.queue(self._sound(self._style, (self._next_bar - 1) % 4))
            self._queued_style = self._style

    def play_goal(self) -> None:
        resolution = choose_style(GOAL_STATE, self._rules)
        self._channel.play(self._sound(resolution, 1), fade_ms=12)
        self._next_bar = 0
        self._queued_style = None

    def close(self) -> None:
        self._channel.stop()
        self._sound.cache_clear()
