"""
This file has examples for how to use midi_skirt.py. I show a few examples of how to make a chord progression, more
examples of how to make a rhythm, how to combine the chord progressions with the rhythm, how to create a melody,
and finally how to write the data to a midi file and play the audio.
"""
from midi_skirt import (
    ChordBuilder,
    ChordProgression,
    ChordProgressionRhythm,
    Melody,
    Rhythm,
    TrackBuilder,
)


# Chord Progression Examples
# Method 1: Building from intervals
def get_chord_progression_example_1(track):
    am7 = ChordBuilder().build_from_intervals(root="A", octave=5, intervals=["1", "b3", "5", "b7"])
    g6 = ChordBuilder().build_from_intervals(root="G", octave=5, intervals=["1", "3", "5", "6"])
    f69 = ChordBuilder().build_from_intervals(root="F", octave=5, intervals=["1", "3", "5", "6", "9"])
    bb13 = ChordBuilder().build_from_intervals(root="Bb", octave=5, intervals=["1", "3", "5", "b7", "9", "13"])

    chord_progression = ChordProgression(
        chords=[am7, g6, f69, bb13],
        changes=[track.pc.bar, track.pc.bar, track.pc.bar, track.pc.bar]
    )
    chord_progression.repeat_progression(18)
    return chord_progression


# Method 2: Building randomly from scale
def get_chord_progression_example_2(track):
    num_chords_and_changes = 64
    chord_progression = ChordProgression(
        changes=[track.pc.bar] * num_chords_and_changes
    )

    # To see which scales are available: `scale.NAMED_SCALES.keys()`
    chord_progression.build_progression_randomly_from_scale(root="G", octave=4, scale="locrian",
                                                            num_chords=num_chords_and_changes)
    return chord_progression


# Rhythm Examples
# Example 1: Medium note density, granular quantization, varying note durations
def get_rhythm_example_1(track):
    rhythm_len = track.pc.bar * 64
    start_tick = 0
    quantization = track.pc.sixteenth_note
    note_density = .5
    note_len_choices = track.pc.get_notes_2()

    rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
    rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)
    return rhythm


# Example 2: High note density, coarse quantization, bar-length durations
def get_rhythm_example_2(track):
    rhythm_len = track.pc.bar * 64
    start_tick = 0
    quantization = track.pc.beat
    note_density = 10.0
    note_len_choices = [track.pc.beat]

    rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
    rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)
    return rhythm


# Example 3: Triplets Durations (subtle)
def get_rhythm_example_3(track):
    rhythm_len = track.pc.bar * 64
    start_tick = 0
    quantization = track.pc.half_note
    note_density = 10.0
    note_len_choices = track.pc.get_notes_3()

    rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
    rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)
    return rhythm


# Example 4: Triplet Quantization
def get_rhythm_example_4(track):
    rhythm_len = track.pc.bar * 64
    start_tick = 0
    quantization = track.pc.half_triplet
    note_density = 10.0
    note_len_choices = track.pc.get_notes_3()

    rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
    rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)
    return rhythm


# Example 5: Fifth Quantization
def get_rhythm_example_5(track):
    rhythm_len = track.pc.bar * 64
    start_tick = 0
    quantization = track.pc.half_fifth
    note_density = 10.0
    note_len_choices = track.pc.get_notes_5()

    rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
    rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)
    return rhythm


# Example 6: Seventh Quantization
def get_rhythm_example_6(track):
    rhythm_len = track.pc.bar * 64
    start_tick = 0
    quantization = track.pc.half_seventh
    note_density = 10.0
    note_len_choices = track.pc.get_notes_7()

    rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
    rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)
    return rhythm


# Example 7: Ninth Quantization
def get_rhythm_example_7(track):
    rhythm_len = track.pc.bar * 64
    start_tick = 0
    quantization = track.pc.half_ninth
    note_density = 10.0
    note_len_choices = track.pc.get_notes_9()

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
                                         emphasis_divisions=[2,5])  # sounds nice in 11/4 (3+8)
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


track = TrackBuilder(bpm=180, time_signature_numerator=7, time_signature_denominator=8)

# Sync the chord progression with the rhythm
rhythm = get_rhythm_example_2(track)
chord_progression = get_chord_progression_example_2(track)
cpr = ChordProgressionRhythm(
    rhythm=rhythm,
    chord_progression=chord_progression,
    tick_method="direct",  # "direct" or "random" or "random_once" or "random_asc" or "random_desc"
    vel_method="random",  # "direct" or "random"
    tick_noise=[-55, 55])  # only used if tick_method is done randomly

melody = Melody(
    root_note='G',
    octave=4,
    scale_name="locrian",
    melody_len=track.pc.bar*64,
    quantization=track.pc.eighth_note,
    note_density=.4,
    note_len_choices=[track.pc.quarter_note, track.pc.eighth_note, track.pc.half_note])

my_melody = melody.create_melody()


# to play the midi file, import into GarageBand or Logic or run with the command line:
# `timidity --adjust-tempo=120 melody2.mid`
track.write_chord_progression_to_midi(chord_progression_rhythm=cpr, filename="chord_progression.mid")
track.write_melody_to_midi(melody=my_melody, filename="melody2.mid")
