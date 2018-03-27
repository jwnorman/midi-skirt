import os

from midi_skirt import *


pc = PatternConstants()

pattern = midi.Pattern(resolution=pc.resolution)
track = midi.Track()
pattern.append(track)


# Chord progression examples
# 1st method
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

# 2nd method
num_chords_and_changes = 64
chord_progression = ChordProgression(
    changes=[pc.bar] * num_chords_and_changes
)

chord_progression.build_progression_randomly_from_scale("C", 5, "major", num_chords_and_changes)


# Rhythm examples
# 1st example
rhythm = Rhythm(rhythm_len=pc.bar * 64, start_tick=2, quantization=pc.sixteenth_note)
rhythm.build_rhythm_randomly(
    note_density=.5,
    note_len_choices=[pc.sixteenth_note, pc.thirty_second_note, pc.eighth_note, pc.quarter_note, pc.whole_note])

# 2nd example
rhythm = Rhythm(rhythm_len=pc.bar * 64, start_tick=2, quantization=pc.bar)
rhythm.build_rhythm_randomly(
    note_density=10.0,
    note_len_choices=[pc.whole_note])

# Once you have the progression and rhythm, sync them up and end the track:
# put chord progression and rhythm together
cpr = ChordProgressionRhythm(rhythm, chord_progression)

all_staged_events = []
for chord in cpr.chords:
    for staged_event in chord.staged_events:
        all_staged_events.append(staged_event)
df = convert_staging_events_to_dataframe(all_staged_events)
df.sort_values(by=["tick", "duration"], inplace=True)

# add to track
track = add_tuples_to_track(track, df)

# Add the end of track event, append it to the track
eot = midi.EndOfTrackEvent(tick=get_max_tick(track) + 2 * pc.whole_note)
track.append(eot)

track = make_ticks_rel(track)

midi.write_midifile("example.mid", pattern)

os.system("timidity /Users/jacknorman1/Documents/Programming/midi-skirt/example.mid")
