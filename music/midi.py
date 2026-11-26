"""Save the soundtrack's note events as a standard MIDI file."""

from collections.abc import Iterable
from pathlib import Path

from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo

from music.score import BAR_TICKS, TICKS_PER_BEAT, Bar


def save_midi(bars: Iterable[Bar], path: str | Path) -> Path:
    midi = MidiFile(type=0, ticks_per_beat=TICKS_PER_BEAT)
    track = MidiTrack()
    midi.tracks.append(track)
    track.append(MetaMessage("track_name", name="NeuralWorld adaptive soundtrack"))
    track.append(MetaMessage("time_signature", numerator=2, denominator=4))
    track.append(MetaMessage("key_signature", key="C"))
    track.append(Message("program_change", program=0, channel=0))
    track.append(Message("program_change", program=8, channel=1))

    offset = last_tick = 0
    last_bpm = None
    for bar in bars:
        events: list[tuple[int, int, Message | MetaMessage]] = []
        if bar.style.bpm != last_bpm:
            events.append((offset, 1, MetaMessage("set_tempo", tempo=bpm2tempo(bar.style.bpm))))
            last_bpm = bar.style.bpm
        for note in bar.notes:
            on = Message("note_on", channel=note.channel, note=note.pitch, velocity=note.velocity)
            off = Message("note_off", channel=note.channel, note=note.pitch, velocity=0)
            events.append((offset + note.start, 2, on))
            events.append((offset + note.start + note.duration, 0, off))
        for tick, _, message in sorted(events, key=lambda item: (item[0], item[1])):
            track.append(message.copy(time=tick - last_tick))
            last_tick = tick
        offset += BAR_TICKS
    track.append(MetaMessage("end_of_track", time=offset - last_tick))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(path)
    return path
