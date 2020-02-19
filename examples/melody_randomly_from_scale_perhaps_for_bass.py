import os

from midi_skirt import (
    Melody,
    TrackBuilder,
)

# Track constants
bpm = 120
time_signature = '3/4'
pattern_constants = PatternConstants(resolution=440, time_signature=time_signature)
track_instance = TrackBuilder(pattern_constants=pattern_constants, bpm=bpm, time_signature=time_signature)

# Melody constants
melody_root_note = 'G'
melody_octave = 4
melody_scale = "locrian"
melody_len = pattern_constants.bar * 64
melody_quantization = pattern_constants.eighth_note
melody_note_density = .4
melody_note_len_choices = [pattern_constants.quarter_note, pattern_constants.eighth_note, pattern_constants.half_note]

melody = Melody(
    root_note=melody_root_note,
    octave=melody_octave,
    scale_name=melody_scale,
    melody_len=melody_len,
    quantization=melody_quantization,
    note_density=melody_note_density,
    note_len_choices=melody_note_len_choices)
my_melody = melody.create_melody()

filename = "melody"
track_instance.write_melody_to_midi(melody=my_melody, filename=filename)

os.system("timidity -A {amplification} {filename}".format(
    amplification=200,
    filename=track_instance.track_filename))
