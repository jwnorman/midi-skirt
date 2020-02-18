"""
This file has examples for how to use midi_skirt.py. I show a few examples of how to make a chord progression, more
examples of how to make a rhythm, how to combine the chord progressions with the rhythm, how to create a melody,
and finally how to write the data to a midi file and play the audio.
"""
import midi
import os

from musical.theory import Chord, Note, scale, Scale
from midi_skirt import (
    PatternConstants,
    Melody,
    MidiEventStager,
    MidiChord,
    ChordBuilder,
    ChordProgression,
    Rhythm,
    ChordProgressionRhythm,
    add_tuples_to_track,
    convert_staging_events_to_dataframe,
    get_max_tick,
    make_ticks_rel
)


# Set the track foundation for the chord progression, including time signature, BPM, and get an empty midi Track object
bpm = 180
time_signature_numerator = 7
time_signature_denominator = 8
pc = PatternConstants(resolution=440, beats_per_bar=time_signature_numerator)

# Chord progression foundation
chord_progression_pattern = midi.Pattern(resolution=pc.resolution)
chord_progression_track = midi.Track()
chord_progression_pattern.append(chord_progression_track)
chord_progression_track.append(midi.SetTempoEvent(bpm=bpm))
chord_progression_track.append(midi.TimeSignatureEvent(numerator=time_signature_numerator,
                                                       denominator=time_signature_denominator))

# Chord progression foundation
melody_pattern = midi.Pattern(resolution=pc.resolution)
melody_track = midi.Track()
melody_pattern.append(melody_track)
melody_track.append(midi.SetTempoEvent(bpm=bpm))
melody_track.append(midi.TimeSignatureEvent(numerator=time_signature_numerator,
                                            denominator=time_signature_denominator))


# Chord Progression Examples
# Method 1: Building from intervals
def get_chord_progression_example_1():
    am7 = ChordBuilder().build_from_intervals(root="A", octave=5, intervals=["1", "b3", "5", "b7"])
    g6 = ChordBuilder().build_from_intervals(root="G", octave=5, intervals=["1", "3", "5", "6"])
    f69 = ChordBuilder().build_from_intervals(root="F", octave=5, intervals=["1", "3", "5", "6", "9"])
    bb13 = ChordBuilder().build_from_intervals(root="Bb", octave=5, intervals=["1", "3", "5", "b7", "9", "13"])

    chord_progression = ChordProgression(
        chords=[am7, g6, f69, bb13],
        changes=[pc.bar, pc.bar, pc.bar, pc.bar]
    )
    chord_progression.repeat_progression(18)
    return chord_progression


# Method 2: Building randomly from scale
def get_chord_progression_example_2():
    num_chords_and_changes = 64
    chord_progression = ChordProgression(
        changes=[pc.bar] * num_chords_and_changes
    )

    # To see which scales are available: `scale.NAMED_SCALES.keys()`
    chord_progression.build_progression_randomly_from_scale(root="G", octave=4, scale="locrian",
                                                            num_chords=num_chords_and_changes)
    return chord_progression


# Rhythm Examples
# Example 1: Medium note density, granular quantization, varying note durations
def get_rhythm_example_1():
    rhythm_len = pc.bar * 64
    start_tick = 0
    quantization = pc.sixteenth_note
    note_density = .5
    note_len_choices = pc.get_notes_2()

    rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
    rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)
    return rhythm


# Example 2: High note density, coarse quantization, bar-length durations
def get_rhythm_example_2():
    rhythm_len = pc.bar * 64
    start_tick = 0
    quantization = pc.beat
    note_density = 10.0
    note_len_choices = [pc.beat]

    rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
    rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)
    return rhythm


# Example 3: Triplets Durations (subtle)
def get_rhythm_example_3():
    rhythm_len = pc.bar * 64
    start_tick = 0
    quantization = pc.half_note
    note_density = 10.0
    note_len_choices = pc.get_notes_3()

    rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
    rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)
    return rhythm


# Example 4: Triplet Quantization
def get_rhythm_example_4():
    rhythm_len = pc.bar * 64
    start_tick = 0
    quantization = pc.half_triplet
    note_density = 10.0
    note_len_choices = pc.get_notes_3()

    rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
    rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)
    return rhythm


# Example 5: Fifth Quantization
def get_rhythm_example_5():
    rhythm_len = pc.bar * 64
    start_tick = 0
    quantization = pc.half_fifth
    note_density = 10.0
    note_len_choices = pc.get_notes_5()

    rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
    rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)
    return rhythm


# Example 6: Seventh Quantization
def get_rhythm_example_6():
    rhythm_len = pc.bar * 64
    start_tick = 0
    quantization = pc.half_seventh
    note_density = 10.0
    note_len_choices = pc.get_notes_7()

    rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
    rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)
    return rhythm


# Example 7: Ninth Quantization
def get_rhythm_example_7():
    rhythm_len = pc.bar * 64
    start_tick = 0
    quantization = pc.half_ninth
    note_density = 10.0
    note_len_choices = pc.get_notes_9()

    rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
    rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)
    return rhythm


# Example 8: Adding emphasis randomly
def get_rhythm_example_8():
    rhythm_len = pc.bar * 64
    start_tick = 0
    quantization = pc.beat
    note_density = 10.0
    note_len_choices = [pc.beat]

    rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
    rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)
    rhythm.build_emphasis_ticks_randomly(emphasis_quantization=pc.beat, container=pc.bar)
    return rhythm


# Example 9: Adding emphasis directly
def get_rhythm_example_9():
    rhythm_len = pc.bar * 64
    start_tick = 0
    quantization = pc.beat
    note_density = 10.0
    note_len_choices = [pc.beat]

    rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
    rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)
    rhythm.build_emphasis_ticks_directly(emphasis_quantization=pc.beat, container=pc.bar,
                                         emphasis_divisions=[2,5])  # sounds nice in 11/4 (3+8)
    return rhythm


# Example 10: Adding emphasis randomly 2
def get_rhythm_example_10():
    rhythm_len = pc.bar * 64
    start_tick = 0
    quantization = pc.eighth_note
    note_density = 1
    note_len_choices = [pc.eighth_note, pc.quarter_note]

    rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
    rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)
    rhythm.build_emphasis_ticks_randomly(emphasis_quantization=pc.eighth_note, container=pc.bar)
    return rhythm


def write_chord_progression_to_midi(track, pattern, chord_progression, filename):
    # Order the chord progression rhythm by tick and duration
    all_staged_events = []
    for chord in chord_progression.chords:
        for staged_event in chord.staged_events:
            all_staged_events.append(staged_event)
    # Using Pandas df here because it's how I pictured the data in my head (tabularly); probably not the most efficient
    staged_events_df = convert_staging_events_to_dataframe(all_staged_events)
    staged_events_df.sort_values(by=["tick", "duration"], inplace=True)

    track = add_tuples_to_track(track, staged_events_df)

    # Add the end of track event, append it to the track
    eot = midi.EndOfTrackEvent(tick=get_max_tick(track) + 2 * pc.whole_note)
    track.append(eot)
    track = make_ticks_rel(track)
    midi.write_midifile(filename, pattern)


def write_melody_to_midi(track, pattern, melody, filename):
    staged_events_df = convert_staging_events_to_dataframe(melody)
    staged_events_df.sort_values(by=["tick", "duration"], inplace=True)

    track = add_tuples_to_track(track, staged_events_df)
    eot = midi.EndOfTrackEvent(tick=get_max_tick(track) + 2 * pc.whole_note)
    track.append(eot)
    track = make_ticks_rel(track)
    midi.write_midifile(filename, pattern)


# Sync the chord progression with the rhythm
rhythm = get_rhythm_example_2()
chord_progression = get_chord_progression_example_2()
cpr = ChordProgressionRhythm(
    rhythm=rhythm,
    chord_progression=chord_progression,
    tick_method="direct",  # "direct" or "random" or "random_once" or "random_asc" or "random_desc"
    vel_method="random",  # "direct" or "random"
    tick_noise=[-55, 55])  # only used if tick_method is done randomly


root = Note(('G', 4))
named_scale = scale.NAMED_SCALES['locrian']
my_scale = Scale(root, named_scale)
# chords_in_scale = Chord.progression(my_scale, 5)
# my_chords = [chords_in_scale[0]] + [chords_in_scale[5]] + [chords_in_scale[3]] + [chords_in_scale[4]]
# chord_notes = [ChordBuilder().get_list_of_chord_notes_from_chord(my_chord) for my_chord in my_chords]

melody = Melody(
    melody_len=pc.bar*64,
    quantization=pc.sixteenth_note,
    note_density=.4,
    note_len_choices=[pc.quarter_note, pc.eighth_note, pc.sixteenth_note, pc.sixty_forth_note,
                      pc.half_note, pc.thirty_second_note],
    available_notes=[my_scale.get(x) for x in range(28, 37)])

my_melody = melody.create_melody()


write_chord_progression_to_midi(track=chord_progression_track, pattern=chord_progression_pattern, chord_progression=cpr,
                                filename="chord_progression.mid")
write_melody_to_midi(track=melody_track, pattern=melody_pattern, melody=my_melody, filename="melody.mid")
os.system("timidity --adjust-tempo={} example.mid".format(bpm))
