import os

from midi_skirt import (
    Melody,
    TrackBuilder,
)

# Track constants
bpm = 120
time_signature = '3/4'
track_instance = TrackBuilder(bpm=bpm, time_signature=time_signature)

# Melody constants
melody_root_note = 'G'
melody_octave = 4
melody_scale = "locrian"
melody_len = track_instance.pc.bar * 64
melody_quantization = track_instance.pc.eighth_note
melody_note_density = .4
melody_note_len_choices = [track_instance.pc.quarter_note, track_instance.pc.eighth_note, track_instance.pc.half_note]

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
