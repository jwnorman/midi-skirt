import os

from midi_skirt import *

pattern = midi.Pattern(resolution=pc.resolution)
track = midi.Track()
pattern.append(track)

bpm = 120
numerator = 7
denominator = 8
track.append(midi.SetTempoEvent(bpm=bpm))
track.append(midi.TimeSignatureEvent(numerator=numerator, denominator=denominator))

pc = PatternConstants(resolution=440, beats_per_bar=numerator*2)

scale.NAMED_SCALES.keys()

# Chord Progression Examples

## Method 1: Building from intervals

c7sharp9 = ChordBuilder().build_from_intervals("C", 5, ["1", "3", "5", "b7", "#9"])
am7 = ChordBuilder().build_from_intervals("A", 5, ["1", "b3", "5", "b7"])
g6 = ChordBuilder().build_from_intervals("G", 5, ["1", "3", "5", "6"])
f69 = ChordBuilder().build_from_intervals("F", 5, ["1", "3", "5", "6", "9"])
bb13 = ChordBuilder().build_from_intervals("Bb", 5, ["1", "3", "5", "b7", "9", "13"])

chord_progression = ChordProgression(
    chords=[am7, g6, f69, bb13],
    changes=[pc.bar, pc.bar, pc.bar, pc.bar]
)

chord_progression.repeat_progression(18)

## Method 2: Building randomly from scale

num_chords_and_changes = 64
chord_progression = ChordProgression(
    changes=[pc.bar] * num_chords_and_changes
)

chord_progression.build_progression_randomly_from_scale("G", 4, "locrian", num_chords_and_changes)

# Rhythm Examples

## Example 1: Medium note density, granular quantization, varying note durations

# rhythm specs
rhythm_len = pc.bar * 64
start_tick = 0
quantization = pc.sixteenth_note
note_density = .5
note_len_choices = pc.get_notes_2()

rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)

## Example 2: High note density, coarse quantization, bar-length durations

# rhythm specs
rhythm_len = pc.bar * 64
start_tick = 0
quantization = pc.beat
note_density = 10.0
note_len_choices = [pc.beat]

rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)

## Example 3: Triplets Durations (subtle)

# # rhythm specs
# rhythm_len = pc.bar * 64
# start_tick = 0
# quantization = pc.half_note
# note_density = 10.0
# note_len_choices = pc.get_notes_3()

rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)

## Example 4: Triplet Quantization

# # rhythm specs
# rhythm_len = pc.bar * 64
# start_tick = 0
# quantization = pc.half_triplet
# note_density = 10.0
# note_len_choices = pc.get_notes_3()

rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)

## Example 5: Fifth Quantization

# # rhythm specs
# rhythm_len = pc.bar * 64
# start_tick = 0
# quantization = pc.half_fifth
# note_density = 10.0
# note_len_choices = pc.get_notes_5()

rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)

## Example 6: Seventh Quantization

# # rhythm specs
# rhythm_len = pc.bar * 64
# start_tick = 0
# quantization = pc.half_seventh
# note_density = 10.0
# note_len_choices = pc.get_notes_7()

rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)

## Example 7: Ninth Quantization

# # rhythm specs
# rhythm_len = pc.bar * 64
# start_tick = 0
# quantization = pc.half_ninth
# note_density = 10.0
# note_len_choices = pc.get_notes_9()

rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)

## Example 8: Adding emphasis randomly

def example_8():
    # rhythm specs
    rhythm_len = pc.bar * 64
    start_tick = 0
    quantization = pc.beat
    note_density = 10.0
    note_len_choices = [pc.beat]

    rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
    rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)
    rhythm.build_emphasis_ticks_randomly(emphasis_quantization=pc.beat, container=pc.bar)

    return rhythm

## Example 9: Adding emphasis directly

def example_9():
    rhythm_len = pc.bar * 64
    start_tick = 0
    quantization = pc.beat
    note_density = 10.0
    note_len_choices = [pc.beat]

    rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
    rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)
    rhythm.build_emphasis_ticks_directly(emphasis_quantization=pc.beat, container=pc.bar, emphasis_divisions=[2,5])  # sounds nice in 11/4 (3+8)
    return rhythm

## Example 10: Adding emphasis randomly 2

def example_10():
    # rhythm specs
    rhythm_len = pc.bar * 64
    start_tick = 0
    quantization = pc.eighth_note
    note_density = 1
    note_len_choices = [pc.eighth_note, pc.quarter_note]

    rhythm = Rhythm(rhythm_len=rhythm_len, start_tick=start_tick, quantization=quantization)
    rhythm.build_rhythm_randomly(note_density=note_density, note_len_choices=note_len_choices)
    rhythm.build_emphasis_ticks_randomly(emphasis_quantization=pc.eighth_note, container=pc.bar)

    return rhythm



rhythm = example_10()

# Sync the chord progression with the rhythm

cpr = ChordProgressionRhythm(
    rhythm=rhythm,
    chord_progression=chord_progression,
    tick_method="direct",  # "direct" or "random" or "random_once" or "random_asc" or "random_desc"
    vel_method="random",  # "direct" or "random"
    tick_noise=[-55, 55])  # only used if tick_method is done randomly

all_staged_events = []
for chord in cpr.chords:
    for staged_event in chord.staged_events:
        all_staged_events.append(staged_event)
df = convert_staging_events_to_dataframe(all_staged_events)
df.sort_values(by=["tick", "duration"], inplace=True)



# Finish and write to midi file

# add to track
track = add_tuples_to_track(track, df)

# Add the end of track event, append it to the track
eot = midi.EndOfTrackEvent(tick=get_max_tick(track) + 2 * pc.whole_note)
track.append(eot)

track = make_ticks_rel(track)

midi.write_midifile("example.mid", pattern)

os.system("timidity --adjust-tempo={} example.mid".format(tempo))