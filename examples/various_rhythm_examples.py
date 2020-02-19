"""
This file has examples for how to use midi_skirt.py. I show a few examples of how to make a chord progression, more
examples of how to make a rhythm, how to combine the chord progressions with the rhythm, how to create a melody,
and finally how to write the data to a midi file and play the audio.
"""
from midi_skirt import Rhythm


# Example 3: Triplets Durations (subtle)
def get_rhythm_example_3(track):
    rhythm_len = track.pc.bar * 64
    start_tick = 0
    quantization = track.pc.half_note
    note_density = 10.0
    note_len_choices = track.pc.get_notes(3)

    rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
    rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)
    return rhythm


# Example 4: Triplet Quantization
def get_rhythm_example_4(track):
    rhythm_len = track.pc.bar * 64
    start_tick = 0
    quantization = track.pc.half_triplet
    note_density = 10.0
    note_len_choices = track.pc.get_notes(3)

    rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
    rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)
    return rhythm


# Example 5: Fifth Quantization
def get_rhythm_example_5(track):
    rhythm_len = track.pc.bar * 64
    start_tick = 0
    quantization = track.pc.half_fifth
    note_density = 10.0
    note_len_choices = track.pc.get_notes(5)

    rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
    rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)
    return rhythm


# Example 6: Seventh Quantization
def get_rhythm_example_6(track):
    rhythm_len = track.pc.bar * 64
    start_tick = 0
    quantization = track.pc.half_seventh
    note_density = 10.0
    note_len_choices = track.pc.get_notes(7)

    rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
    rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)
    return rhythm


# Example 7: Ninth Quantization
def get_rhythm_example_7(track):
    rhythm_len = track.pc.bar * 64
    start_tick = 0
    quantization = track.pc.half_ninth
    note_density = 10.0
    note_len_choices = track.pc.get_notes(9)

    rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
    rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)
    return rhythm


# Example 8: Adding emphasis randomly
def get_rhythm_example_8(track):
    rhythm_len = track.pc.bar * 64
    start_tick = 0
    quantization = track.pc.beat
    note_density = 10.0
    note_len_choices = [track.pc.beat]

    rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
    rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)
    rhythm.build_emphasis_ticks_randomly(emphasis_quantization=track.pc.beat, container=track.pc.bar)
    return rhythm


# Example 9: Adding emphasis directly
def get_rhythm_example_9(track):
    rhythm_len = track.pc.bar * 64
    start_tick = 0
    quantization = track.pc.beat
    note_density = 10.0
    note_len_choices = [track.pc.beat]

    rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
    rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)
    rhythm.build_emphasis_ticks_directly(emphasis_quantization=track.pc.beat, container=track.pc.bar,
                                         emphasis_divisions=[2, 5])  # sounds nice in 11/4 (3+8)
    return rhythm


# Example 10: Adding emphasis randomly 2
def get_rhythm_example_10(track):
    rhythm_len = track.pc.bar * 64
    start_tick = 0
    quantization = track.pc.eighth_note
    note_density = 1
    note_len_choices = [track.pc.eighth_note, track.pc.quarter_note]

    rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
    rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)
    rhythm.build_emphasis_ticks_randomly(emphasis_quantization=track.pc.eighth_note, container=track.pc.bar)
    return rhythm
