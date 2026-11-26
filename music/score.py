"""Compose short passages from functional harmony, rhythm, and motifs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from music.state import GOAL_STATE, MusicState

TICKS_PER_BEAT = 480
BAR_TICKS = 2 * TICKS_PER_BEAT
Mood = Literal["calm", "tension", "resolution"]

# C-major voicings use common tones and inversions to keep adjacent voices close.
_CHORDS = {
    "I": (48, (60, 64, 67)),
    "IV": (53, (60, 65, 69)),
    "V": (55, (59, 62, 67)),
    "vi": (45, (60, 64, 69)),
    "ii": (50, (62, 65, 69)),
    "V7": (55, (59, 62, 65, 67)),
}
_PROGRESSIONS = {
    "calm": ("I", "IV", "V", "I"),
    "tension": ("I", "vi", "ii", "V7"),
    "resolution": ("V7", "I"),
}
_MOTIFS = {
    "calm": (72, 76, 79),
    "tension": (74, 77, 71),
    "resolution": (71, 74, 72),
}


@dataclass(frozen=True, slots=True)
class MusicRules:
    danger_threshold: float = 0.52
    progress_threshold: float = 0.82
    uncertainty_threshold: float = 0.48
    movement_threshold: float = 0.48
    calm_bpm: int = 84
    tension_bpm: int = 112
    resolution_bpm: int = 96

    def __post_init__(self) -> None:
        for threshold in (
            self.danger_threshold, self.progress_threshold,
            self.uncertainty_threshold, self.movement_threshold,
        ):
            if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
                raise ValueError("music thresholds must be numbers between 0 and 1")
        for bpm in (self.calm_bpm, self.tension_bpm, self.resolution_bpm):
            if type(bpm) is not int or not 40 <= bpm <= 240:
                raise ValueError("music tempos must be integers between 40 and 240 BPM")

    @classmethod
    def from_json(cls, path: Path | None) -> MusicRules:
        if path is None:
            return cls()
        return cls(**json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class MusicStyle:
    mood: Mood
    bpm: int
    syncopated: bool
    dense: bool
    sparse_accompaniment: bool


@dataclass(frozen=True, slots=True)
class Note:
    start: int
    duration: int
    pitch: int
    velocity: int
    channel: int


@dataclass(frozen=True, slots=True)
class Bar:
    style: MusicStyle
    notes: tuple[Note, ...]


def choose_style(state: MusicState, rules: MusicRules) -> MusicStyle:
    if state == GOAL_STATE:
        mood, bpm = "resolution", rules.resolution_bpm
    elif state.danger >= rules.danger_threshold:
        mood, bpm = "tension", rules.tension_bpm
    elif state.progress >= rules.progress_threshold:
        mood, bpm = "resolution", rules.resolution_bpm
    else:
        mood, bpm = "calm", rules.calm_bpm
    return MusicStyle(
        mood, bpm, state.uncertainty >= rules.uncertainty_threshold,
        state.movement_intensity >= rules.movement_threshold, state.isolation >= 0.75,
    )


def compose_bar(style: MusicStyle, index: int) -> Bar:
    progression = _PROGRESSIONS[style.mood]
    root, chord = _CHORDS[progression[index % len(progression)]]
    notes = [Note(0, BAR_TICKS, pitch, 46, 0) for pitch in chord]
    if not style.sparse_accompaniment:
        notes.append(Note(0, BAR_TICKS, root, 57, 0))

    motif = _MOTIFS[style.mood]
    starts = (0, 300, 660) if style.syncopated else (0, 240, 480)
    if style.mood == "resolution" and index % 2 == 1:
        notes.append(Note(0, BAR_TICKS, motif[-1], 72, 1))
    else:
        count = 3 if style.dense or style.mood == "tension" else 2
        for start, pitch in zip(starts[:count], motif):
            notes.append(Note(start, 210, pitch, 74 if start == 0 else 62, 1))
    return Bar(style, tuple(notes))
