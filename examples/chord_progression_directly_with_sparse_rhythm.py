import os

from midi_skirt import (
    ChordBuilder,
    ChordProgression,
    ChordProgressionRhythm,
    PatternConstants,
    Rhythm,
    TrackBuilder,
)

# Track constants
bpm = 90
time_signature = '4/4'
pattern_constants = PatternConstants(resolution=440, time_signature=time_signature)
track_instance = TrackBuilder(pattern_constants=pattern_constants, bpm=bpm, time_signature=time_signature)

# Rhythm constants
rhythm_length = pattern_constants.bar * 64
rhythm_quantization = pattern_constants.sixteenth_note
rhythm_note_density = 0.3
rhythm_note_len_choices = [pattern_constants.quarter_note, pattern_constants.eighth_note, pattern_constants.half_note]

# ChordProgression constants
chord_progression_changes = [pattern_constants.bar] * 4
chord_progression_octave = 4
chord_progression_repeat = 16

# ChordProgressionRhythm constants
tick_method = "direct"  # "direct" or "random" or "random_once" or "random_asc" or "random_desc"
vel_method = "random"  # "direct" or "random"
tick_noise = [-55, 55]  # only used if tick_method is done randomly

# Build the rhythm, chord progression, chord progression rhythm, and write to a midi file
rhythm_instance = Rhythm(rhythm_len=rhythm_length, start_tick=0, quantization=rhythm_quantization)
rhythm_instance.build_rhythm_randomly(note_density=rhythm_note_density, note_len_choices=rhythm_note_len_choices)


am7 = ChordBuilder().build_from_intervals(root="A", octave=chord_progression_octave, intervals=["1", "b3", "5", "b7"])
g6 = ChordBuilder().build_from_intervals(root="G", octave=chord_progression_octave, intervals=["1", "3", "5", "6"])
f69 = ChordBuilder().build_from_intervals(root="F", octave=chord_progression_octave,
                                          intervals=["1", "3", "5", "6", "9"])
bb13 = ChordBuilder().build_from_intervals(root="Bb", octave=chord_progression_octave,
                                           intervals=["1", "3", "5", "b7", "9", "13"])
chord_progression_instance = ChordProgression(
    chords=[am7, g6, f69, bb13],
    changes=chord_progression_changes
)
chord_progression_instance.repeat_progression(chord_progression_repeat)

cpr = ChordProgressionRhythm(
    rhythm=rhythm_instance,
    chord_progression=chord_progression_instance,
    tick_method=tick_method,
    vel_method=vel_method,
    tick_noise=tick_noise)

filename = "chord_progression"
track_instance.write_chord_progression_to_midi(chord_progression_rhythm=cpr, filename=filename)

os.system("timidity -A {amplification} {filename}".format(
    amplification=200,
    filename=track_instance.track_filename))
