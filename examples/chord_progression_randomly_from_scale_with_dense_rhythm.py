import os

from midi_skirt import (
    ChordProgression,
    ChordProgressionRhythm,
    PatternConstants,
    Rhythm,
    TrackBuilder,
)

# Track constants
bpm = 160
time_signature = '3/4'
pattern_constants = PatternConstants(resolution=440, time_signature=time_signature)
track_instance = TrackBuilder(pattern_constants=pattern_constants, bpm=bpm, time_signature=time_signature)

# Rhythm constants
rhythm_length = pattern_constants.bar * 64
rhythm_quantization = pattern_constants.beat
rhythm_note_density = 10.0
rhythm_note_len_choices = [pattern_constants.beat]

# ChordProgression constants
chord_progression_changes = [pattern_constants.bar] * 64
chord_progression_root = "G"
chord_progression_octave = 4
chord_progression_scale_name = "locrian"
number_of_chords_and_changes = 64

# ChordProgressionRhythm constants
tick_method = "direct"  # "direct" or "random" or "random_once" or "random_asc" or "random_desc"
vel_method = "random"  # "direct" or "random"
tick_noise = [-55, 55]  # only used if tick_method is done randomly

# Build the rhythm, chord progression, chord progression rhythm, and write to a midi file
rhythm_instance = Rhythm(rhythm_len=rhythm_length, start_tick=0, quantization=rhythm_quantization)
rhythm_instance.build_rhythm_randomly(note_density=rhythm_note_density, note_len_choices=rhythm_note_len_choices)

chord_progression_instance = ChordProgression(changes=chord_progression_changes)
chord_progression_instance.build_progression_randomly_from_scale(
    root=chord_progression_root, octave=chord_progression_octave, scale_name=chord_progression_scale_name,
    num_chords=number_of_chords_and_changes)

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
