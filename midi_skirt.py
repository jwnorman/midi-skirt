"""Tools to help build random rhythms, chord progressions, and melodies."""

import copy
import midi
import numpy as np
import pandas as pd
import random

from musical.theory import Note, scale, Scale


class PatternConstants:
    def __init__(self, resolution=440, beats_per_bar=4):
        self.resolution = resolution
        self.beat = resolution
        self.quarter_note = resolution
        self.beats_per_bar = beats_per_bar
        self.bar = self.beat * beats_per_bar
        self.half_note = self.quarter_note * 2
        self.eighth_note = int(self.quarter_note / 2.0)
        self.sixteenth_note = int(self.quarter_note / 4.0)
        self.thirty_second_note = int(self.quarter_note / 8.0)
        self.sixty_forth_note = int(self.quarter_note / 16.0)
        self.whole_note = self.bar

        (self.half_triplet, self.quarter_triplet, self.eighth_triplet, self.sixteenth_triplet,
         self.thirty_second_triplet, self.sixty_forth_triplet) = self.get_notes(3)
        (self.half_fifth, self.quarter_fifth, self.eighth_fifth, self.sixteenth_fifth, self.thirty_second_fifth,
         self.sixty_forth_fifth) = self.get_notes(5)
        (self.half_seventh, self.quarter_seventh, self.eighth_seventh, self.sixteenth_seventh,
         self.thirty_second_seventh, sixty_forth_seventh) = self.get_notes(7)
        (self.half_ninth, self.quarter_ninth, self.eighth_ninth, self.sixteenth_ninth, self.thirty_second_ninth,
         self.sixty_forth_ninth) = self.get_notes(9)

    def get_notes(self, divisions=3):
        half = int(self.half_note / float(divisions))
        quarter = int(self.quarter_note / float(divisions))
        eighth = int(self.eighth_note / float(divisions))
        sixteenth = int(self.sixteenth_note / float(divisions))
        thirty_second = int(self.thirty_second_note / float(divisions))
        sixty_forth = int(self.sixty_forth_note / float(divisions))
        return[half, quarter, eighth, sixteenth, thirty_second, sixty_forth]

    def get_notes_2(self):
        return [self.sixty_forth_note, self.thirty_second_note, self.sixteenth_note, self.eighth_note,
                self.quarter_note, self.half_note, self.whole_note]

    def get_notes_3(self):
        return [self.sixty_forth_triplet, self.thirty_second_triplet, self.sixteenth_triplet,
                self.eighth_triplet, self.quarter_triplet, self.half_triplet, self.whole_note]

    def get_notes_5(self):
        return [self.sixteenth_fifth, self.eighth_fifth, self.quarter_fifth, self.half_fifth, self.whole_note]

    def get_notes_7(self):
        return [self.sixteenth_seventh, self.eighth_seventh, self.quarter_seventh, self.half_seventh, self.whole_note]

    def get_notes_9(self):
        return [self.sixteenth_ninth, self.eighth_ninth, self.quarter_ninth, self.half_ninth, self.whole_note]


def make_ticks_rel(track):
    number_before_negative = 0
    running_tick = 0
    for event in track:
        event.tick -= running_tick
        if event.tick >= 0:
            number_before_negative += 1
        else:
            print number_before_negative
        running_tick += event.tick
    return track


def get_max_tick(track):
    return max([event.tick for event in track])


def get_min_tick(track):
    return min([event.tick for event in track])


class MidiEventStager:
    # class to store events nearly ready to be converted to events
    def __init__(self, midi_event_fun, start_tick, duration, pitch, velocity):
        self.midi_event_fun = midi_event_fun
        self.start_tick = start_tick
        self.duration = duration
        self.pitch = pitch
        self.velocity = velocity

#     def apply_midi_event_fun():
#         self.midi_event_fun(tick=self.start_tick, velocity=self.velocity, pitch=self.pitch)

    def __repr__(self):
        return "{} {}".format(self.pitch, self.midi_event_fun.name)


def convert_staging_events_to_dataframe(staging_events):
    event_tuples = [(se.midi_event_fun, se.start_tick, se.duration, se.pitch, se.velocity) for se in staging_events]
    event_df = pd.DataFrame(event_tuples)
    event_df.columns = ["event_type_fun", "tick", "duration", "pitch", "velocity"]
    return event_df


class MidiChord:
    """Build a list of staging events from input notes.

    Note: a MidiChord has no conception of start_tick but set_start_tick helps set the tick.
    """

    def __init__(self, chord_notes=None):
        """
        chord_notes: a list of strings that represent the notes and octaves of the chord.

        usage:
            chord = MidiChord(['c1', 'e1', 'g1'])
        """
        self.chord_notes = chord_notes
        self.staged_events = []
        # self.chord_name = ""

    def build_chord(self):
        """Populate the staged_events list with on and off of events, one for each note in the chord."""
        for note in self.chord_notes:
            # velocity = random.randint(50, 90)
            notes_created = self._create_note_tuple(note)
            self.staged_events.extend(notes_created)

    def set_start_tick(self, start_tick):
        for event in self.staged_events:
            event.start_tick += start_tick

    def set_start_tick_uniformly_noisily(self, start_tick, noise_range, direction=None):
        noises = [random.randint(noise_range[0], noise_range[1]) for _ in range(len(self.staged_events))]
        if direction is None:
            for event, noise in zip(self.staged_events, noises):
                event.start_tick += start_tick + noise
        elif direction == "ascending":
            for event, noise in zip(self.staged_events, sorted(noises, reverse=False)):
                event.start_tick += start_tick + noise
        elif direction == "descending":
            for event, noise in zip(self.staged_events, sorted(noises, reverse=True)):
                event.start_tick += start_tick + noise
        else:
            print "wat"
        for event in self.staged_events:
            event.start_tick += start_tick + random.randint(noise_range[0], noise_range[1])

    def set_start_tick_uniformly_noisily_ascending(self, start_tick, noise_range):
        pass

    def set_start_tick_uniformly_noisily_once(self, start_tick, noise_range):
        noise = random.randint(noise_range[0], noise_range[1])
        for event in self.staged_events:
            event.start_tick += start_tick + noise

    def set_duration(self, duration):
        for event in self.staged_events:
            if event.midi_event_fun.name == "Note Off":
                event.start_tick += duration
            event.duration += duration

    def set_velocity(self, velocity):
        for event in self.staged_events:
            if event.midi_event_fun.name == "Note On":
                event.velocity = velocity

    def set_velocity_randomly_uniform(self, min_vel, max_vel):
        for event in self.staged_events:
            if event.midi_event_fun.name == "Note On":
                event.velocity = random.randint(min_vel, max_vel)

    def _create_note_tuple(self, note):
        return [
            MidiEventStager(midi.NoteOnEvent, 0, 0, note, 0),
            MidiEventStager(midi.NoteOffEvent, 0, 0, note, 0)
        ]


class ChordBuilder:
    """A class with functions to help create MidiChord instances."""

    def build_from_intervals(self, root, octave, intervals, scale_name="major"):
        """Given chord specs return a MidiChord object of said chord.

        root: string of the note.
        octave: an integer between 0 and 8 (or 9 or something)
        intervals: a list of note intervals relative to the root. Use 'b' for flat and '#' for sharp.

        usage:
            chord = ChordBuilder().build_from_intervals('c', 6, ["1", "3", "5", "b7", "#9"])

        returns:
            a MidiChord object
        """
        named_scale = scale.NAMED_SCALES[scale_name]
        my_scale = Scale(Note((root.upper(), octave)), named_scale)
        num_notes_in_scale = len(my_scale)
        scale_start_num = octave * num_notes_in_scale
        intervals = [self._deal_with_pitch_accidentals(interval) for interval in intervals]
        notes = [my_scale.get(scale_start_num + interval[0] - 1).transpose(interval[1]) for interval in intervals]
        chord = MidiChord([note.note + str(note.octave) for note in notes])
        chord.build_chord()
        return chord

    def build_directly(self, notes):
        chord = MidiChord([note.note + str(note.octave) for note in notes])
        chord.build_chord()
        return chord

    def build_randomly_from_scale(self, root, octave, scale_name="major", num_note_choices=[3, 4, 5]):
        named_scale = scale.NAMED_SCALES[scale_name]
        my_scale = Scale(Note((root.upper(), octave)), named_scale)
        num_notes_in_scale = len(my_scale)
        scale_start_num = octave * num_notes_in_scale
        num_notes_in_chord = np.random.choice(num_note_choices)
        possible_notes = [my_scale.get(temp_note)
                          for temp_note in range(scale_start_num, scale_start_num + num_notes_in_scale * 2)]
        notes = np.random.choice(possible_notes, size=num_notes_in_chord, replace=False)
        chord = MidiChord([note.note + str(note.octave) for note in notes])
        chord.build_chord()
        return chord

    def _get_list_of_chord_notes_from_chord(self, notes):
        return [Note.index_from_string(chord_note.note + str(chord_note.octave)) for chord_note in notes]

    def _deal_with_pitch_accidentals(self, interval):
        # input "b9" output (9, -1)
        # input "#9" output (9, 1)
        # input "9" output (9, 0)
        if "b" in interval:
            transposition = -1
        elif "#" in interval:
            transposition = 1
        else:
            transposition = 0
        note = int(interval.replace("b", "").replace("#", ""))
        return((note, transposition))


class ChordProgression:
    """A ChordProgression consists of chords and changes (start ticks).

    The changes are just the underlying chord progression. A rhythm is created separately and applied
    later by using the chord progression object to know which chord to play.
    """

    def __init__(self, chords=[], changes=[]):
        self.chords = chords
        self.changes = changes

    def build_progression_randomly_from_scale(self, root, octave, scale, num_chords):
        # for now the number of notes in each chord is random choice of 3, 4, or 5
        self.chords = [ChordBuilder().build_randomly_from_scale(root, octave, scale) for _ in range(num_chords)]

    def build_changes_randomly(self, duration_choices):
        pass

    def build_progression_from_major(self, root, octave, roman_numerals=["I", "IV", "V7", "I"]):
        pass

    def repeat_progression(self, num_repeats):
        temp_chords = []
        temp_changes = []
        for _ in range(num_repeats):
            for chord in self.chords:
                temp_chords.append(copy.deepcopy(chord))
            for change in self.changes:
                temp_changes.append(copy.deepcopy(change))
        self.chords = temp_chords
        self.changes = temp_changes

    @property
    def length(self):
        return sum(self.changes)

    def get_chord_from_tick(self, tick):
        temp = [0] + self.changes + [np.inf]
        pos = np.sum(np.cumsum(temp) < tick)
        if pos > len(self.chords):
            pos -= 2
        if pos == len(self.chords):
            pos -= 1
        return self.chords[pos]


class Rhythm:
    def __init__(self, rhythm_len=None, start_tick=None, quantization=None):
        self.rhythm_len = rhythm_len
        self.start_tick = start_tick
        self.quantization = quantization
        self.start_ticks = []
        self.note_lengths = []

    def build_rhythm_randomly(self, note_density, note_len_choices):
        number_of_notes = self._compute_num_notes(note_density)
        self.start_ticks = self._get_random_start_ticks(number_of_notes)
        self.note_lengths = [np.random.choice(note_len_choices) for _ in self.start_ticks]

    def build_rhythm_directly(self):
        pass

    def _compute_num_notes(self, note_density):
        return int(self.rhythm_len * note_density / float(self.quantization))

    def _get_random_start_ticks(self, number_of_notes):
        return np.unique(sorted([
            self._find_nearest_note(
                value=random.randint(0, self.rhythm_len) + self.start_tick,
                note_type=self.quantization)
            for _ in range(number_of_notes)]))

    def _find_nearest_note(self, value, note_type):
        mod = value % note_type
        if (mod > (note_type / 2.0)):  # round up
            return value + note_type - mod
        else:
            return value - mod


class ChordProgressionRhythm:
    def __init__(self, rhythm, chord_progression, tick_method="direct", vel_method="random", tick_noise=[-55, 55],
                 match_lengths=True):
        self.rhythm = rhythm
        self.chord_progression = chord_progression
        self.tick_method = tick_method
        self.vel_method = vel_method
        self.tick_noise = tick_noise
        self.chords = self.build_staging_events()
        if match_lengths:
            self.make_chord_and_rhythm_same_length()

    def build_staging_events(self):
        chords = []
        for tick, duration in zip(self.rhythm.start_ticks, self.rhythm.note_lengths):
            chord = copy.deepcopy(self.chord_progression.get_chord_from_tick(tick))
            if self.tick_method == "direct":
                chord.set_start_tick(tick)
            elif self.tick_method == "random":
                chord.set_start_tick_uniformly_noisily(tick, self.tick_noise)
            elif self.tick_method == "random_once":
                chord.set_start_tick_uniformly_noisily_once(tick, self.tick_noise)
            elif self.tick_method == "random_asc":
                chord.set_start_tick_uniformly_noisily(tick, self.tick_noise, "ascending")
            elif self.tick_method == "random_desc":
                chord.set_start_tick_uniformly_noisily(tick, self.tick_noise, "descending")
            else:
                chord.set_start_tick(tick)
            if self.vel_method == "random":
                chord.set_velocity_randomly_uniform(50, 90)
            elif self.vel_method == "direct":
                chord.set_velocity(random.randint(50, 90))
            else:
                chord.set_velocity_randomly_uniform(50, 90)
            chord.set_duration(duration)
            chords.append(chord)
        return chords

    def make_chord_and_rhythm_same_length(self):
        pass


def add_tuples_to_track(track, df):
    for row in df.iterrows():
        data = row[1]
        track.append(data["event_type_fun"](tick=data["tick"],
                                            velocity=data["velocity"],
                                            pitch=Note.index_from_string(data["pitch"])))
    return track

# class Melody:
#     def __init__(self, melody_len=None, scale=None, quantization=None, note_density=None, note_len_choices=None,
#                  available_notes=None):
#         self.melody_len = melody_len
#         self.quantization = quantization  # this maybe should be at note level only
#         self.note_density = note_density  # proportion of available ticks (determined by quantization) occupied by notes
#         self.note_len_choices = note_len_choices
#         self.available_notes = available_notes
#         self.number_of_notes = self._compute_num_notes()
#         self.start_ticks = self._get_start_ticks()

#     def _compute_num_notes(self):
#         return self.melody_len / self.quantization

#     def _get_start_ticks(self):
#         return np.unique(sorted([find_nearest_note(random.randint(0, self.melody_len), self.quantization)
#                                  for _ in range(self.number_of_notes)]))

#     def create_melody(self):
#         melody_notes = []
#         for tick in self.start_ticks:
#             melody_tuples = self._create_melody_note_tuple(tick)
#             melody_notes.extend(melody_tuples)
#         return melody_notes

#     def _create_melody_note_tuple(self, start_tick):
#         note_id = str(uuid.uuid1())
#         velocity = random.randint(50, 90)
#         cur_note = random.choice(self.available_notes)
#         cur_note = Note.index_from_string(cur_note.note + str(cur_note.octave))
#         note_length = random.choice(self.note_len_choices)
#         return [
#             (note_id, midi.NoteOnEvent, start_tick, cur_note, note_length, velocity),
#             (note_id, midi.NoteOffEvent, start_tick+note_length, cur_note, note_length, 0)
#         ]
