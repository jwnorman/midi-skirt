"""Tools to help build random rhythms, chord progressions, and melodies."""

import copy
import midi
import numpy as np
import pandas as pd
import random
import uuid

from musical.theory import (
    Note,
    scale,
    Scale,
)


class PatternConstants:
    """A helper class to pre-compute all note durations given a time signature and a resolution."""

    def __init__(self, resolution=440, time_signature='4/4'):
        """Pre-compute all note durations given resolution and time signature.

        :param resolution: from the python-midi README: A tick represents the lowest level resolution of a MIDI track.
            Tempo is always analogous with Beats per Minute (BPM) which is the same thing as Quarter notes per Minute
            (QPM). resolution is also known as the Pulses per Quarter note (PPQ). It analogous to Ticks per Beat (TPM).
        :param time_signature: the time signature (<beats per measure> / <value of a beat>) of the pattern

        :return:
        """
        self.beats_per_bar, _ = [int(i) for i in time_signature.split('/')]
        self.resolution = resolution
        self.beat = resolution
        self.quarter_note = resolution
        self.bar = self.beat * self.beats_per_bar
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
        """Return notes based on desired number of divisions within a bar."""
        half = int(self.half_note / float(divisions))
        quarter = int(self.quarter_note / float(divisions))
        eighth = int(self.eighth_note / float(divisions))
        sixteenth = int(self.sixteenth_note / float(divisions))
        thirty_second = int(self.thirty_second_note / float(divisions))
        sixty_forth = int(self.sixty_forth_note / float(divisions))
        return [half, quarter, eighth, sixteenth, thirty_second, sixty_forth]


class MidiEventStager:
    """Class to store events nearly ready to be converted to events."""
    def __init__(self, midi_event_fun, start_tick, duration, pitch, velocity):
        self.midi_event_fun = midi_event_fun
        self.start_tick = start_tick
        self.duration = duration
        self.pitch = pitch
        self.velocity = velocity

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
        chord_notes: a list of strings, each of which represent a note (a pitch and octave concatenated)

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
        """Set the start tick of the chord directly using `start_tick`.

        :param start_tick: the value to add or subtract from the absolute tick position (which will be influenced by
            the chord progression).
        :return:
        """
        for event in self.staged_events:
            event.start_tick += start_tick

    def set_start_tick_uniformly_noisily(self, start_tick, noise_range, direction=None):
        """Set the start tick of the chord with some noise distributed to each note of the chord.

        :param start_tick: the value to add or subtract from the absolute tick position (which will be influenced by
            the chord progression).
        :param noise_range: The range to be used when selecting note displacements at random.
        :param direction: after note displacements have been computed, the order (if any) for the displacements to be
            applied in relationship to the note pitch.

        :return:
        """
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
            print("wat")
        for event in self.staged_events:
            event.start_tick += start_tick + random.randint(noise_range[0], noise_range[1])

    def set_start_tick_uniformly_noisily_once(self, start_tick, noise_range):
        """Set the start tick of the chord randomly given a start_tick and a noise range.

        This is similar to `set_start_tick_uniformly_noisily` but the displacement is applied to the entire chord
        instead of individual notes.

        :param start_tick: the value to add or subtract from the absolute tick position (which will be influenced by
            the chord progression).
        :param noise_range: the range to be used when selecting the chord displacement at random.

        :return:
        """
        noise = random.randint(noise_range[0], noise_range[1])
        for event in self.staged_events:
            event.start_tick += start_tick + noise

    def set_duration(self, duration):
        """Set the duration of all the notes in the chord directly.

        :param duration: the duration of each note

        :return:
        """
        for event in self.staged_events:
            if event.midi_event_fun.name == "Note Off":
                event.start_tick += duration
            event.duration += duration

    def set_velocity(self, velocity):
        """Set the velocity of the chord directly.

        Each note within the chord will be played with the same velocity.

        :param velocity: the velocity value (representing volume/loudness of the notes)

        :return:
        """
        for event in self.staged_events:
            if event.midi_event_fun.name == "Note On":
                event.velocity = velocity

    def set_velocity_randomly_uniform(self, min_vel, max_vel):
        """Set the velocity of the notes of the chord randomly given a velocity range.

        :param min_vel: the minimum value of the range used to compute random velocities
        :param max_vel: the maximum value of the range used to compute random velocities

        :return:
        """
        for event in self.staged_events:
            if event.midi_event_fun.name == "Note On":
                event.velocity = random.randint(min_vel, max_vel)

    @staticmethod
    def _create_note_tuple(note):
        return [
            MidiEventStager(midi.NoteOnEvent, 0, 0, note, 0),
            MidiEventStager(midi.NoteOffEvent, 0, 0, note, 0)
        ]

    def __repr__(self):
        return "<MidiChord: {}>".format(self.chord_notes)


class ChordBuilder:
    """A class with functions to help create MidiChord instances."""
    def __init__(self):
        pass

    def build_from_intervals(self, root, octave, intervals, scale_name="major"):
        """Given chord specs return a MidiChord object of said chord.

        usage:
            chord = ChordBuilder().build_from_intervals('c', 6, ["1", "3", "5", "b7", "#9"])

        :param root: string of the note.
        :param octave: an integer between 0 and 8 (or 9 or something)
        :param intervals: a list of note intervals relative to the root. Use 'b' for flat and '#' for sharp.
        :param scale_name: the scale from which to select notes

        :return: a Chord object
        """
        named_scale = scale.NAMED_SCALES[scale_name]
        my_scale = Scale(Note((root.upper(), octave)), named_scale)
        num_notes_in_scale = len(my_scale)
        # TODO: is this the correct way to calculate the scale_start_num?
        scale_start_num = octave * num_notes_in_scale
        intervals = [self._deal_with_pitch_accidentals(interval) for interval in intervals]
        notes = [my_scale.get(scale_start_num + interval[0] - 1).transpose(interval[1]) for interval in intervals]
        chord = MidiChord([note.note + str(note.octave) for note in notes])
        chord.build_chord()
        return chord

    @staticmethod
    def build_directly(notes):
        chord = MidiChord([note.note + str(note.octave) for note in notes])
        chord.build_chord()
        return chord

    @staticmethod
    def build_randomly_from_scale(root, octave, scale_name="major", num_note_choices=None):
        """Create a MidiChord object based on a given scale.

        :param root: the root note of the scale
        :param octave: the octave to be used when creating the chord
        :param scale_name: the scale name to be used
        :param num_note_choices: a list of how integers representing the number of notes allowed to be chosen at random
            when constructing the chord randomly. None will default to [3, 4, 5].

        :return: a MidiChord
        """
        if num_note_choices is None:
            num_note_choices = [3, 4, 5]
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

    @staticmethod
    def get_list_of_chord_notes_from_chord(chord_notes):
        """Extract the string representations of a list of Note objects.

        :param chord_notes: a list of Note objects

        :return: a list of string representations of a list of Note objects.
        """
        return [Note.index_from_string(chord_note.note + str(chord_note.octave)) for chord_note in chord_notes]

    @staticmethod
    def _deal_with_pitch_accidentals(interval):
        """Return a tuple of the note and how much to transpose the note.

        This is a helper function to convert interval strings (like "b13") and output the note itself with a second
        integer that represents how much to transpose the note (-1 for "b", 1 for "#", and 0 for nothing).

        Examples
            "b9" -> (9, -1)
            "#9" -> (9, 1)
            "9" -> (9, 0)

        :param interval: a string representing a musical interval, e.g. "b9", "#9", or "9"

        :return: a tuple of the note and how much to transpose the note
        """
        if "b" in interval:
            transposition = -1
        elif "#" in interval:
            transposition = 1
        else:
            transposition = 0
        note = int(interval.replace("b", "").replace("#", ""))
        return note, transposition


class ChordProgression:
    """A ChordProgression consists of chords and changes (start ticks)."""

    def __init__(self, chords, changes):
        """
        :param chords: a list of MidiChord instances
        :param changes: a list of start ticks which denote where the underlying chords change. Note that these changes
            are merely the *underlying* chord progression and don't necessarily denote when a chord will be played:
            that will be left to the a Rhythm instance, which is created separately and applied later by using the
            ChordProgression object to know which chord to play.

        :return:
        """
        self.chords = chords
        self.changes = changes

    def build_progression_randomly_from_scale(self, root, octave, scale_name, num_chords):
        # for now the number of notes in each chord is random choice of 3, 4, or 5
        self.chords = [ChordBuilder().build_randomly_from_scale(root, octave, scale_name) for _ in range(num_chords)]

    def build_changes_randomly(self, duration_choices):
        pass

    def build_progression_from_major(self, root, octave, roman_numerals):
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
        """Return the chord of self.chords given a all chord start ticks and a target tick.

        :param tick: the tick of interest, which corresponds to the desired chord

        :return: a MidiChord
        """
        pos = np.sum(np.cumsum(np.array(self.changes)) - self.changes[0] <= tick)
        return self.chords[pos - 1]


class Rhythm:
    def __init__(self, rhythm_len=None, start_tick=None, quantization=None, emphasis_velocities=(45, 95)):
        self.rhythm_len = rhythm_len
        self.start_tick = start_tick
        self.quantization = quantization
        self.emphasis_velocities = emphasis_velocities
        self.start_ticks = []
        self.note_lengths = []
        self.emphases = []

    def build_rhythm_randomly(self, note_density, note_len_choices):
        number_of_notes = self._compute_num_notes(note_density)
        self.start_ticks = self._get_random_start_ticks(number_of_notes)
        self.note_lengths = [np.random.choice(note_len_choices) for _ in self.start_ticks]
        self.emphases = [self.emphasis_velocities[0]] * number_of_notes

    def build_rhythm_directly(self):
        pass

    def build_emphasis_ticks_randomly(self, emphasis_quantization, container):
        # note: don't forget start_tick
        emphasis_ticks = []
        num_container_divisions = container / emphasis_quantization
        num_emphasis_notes = random.choice(range(num_container_divisions))
        random_emphasis_combo = np.random.choice(range(num_container_divisions), num_emphasis_notes, replace=False)
        # print("Emphasis on: " + str(np.array(random_emphasis_combo) + 1))
        emphasis_ticks_temp = np.arange(0, container, emphasis_quantization).take(random_emphasis_combo)
        for tick in emphasis_ticks_temp:
            emphasis_ticks.extend(np.arange(tick, self.rhythm_len, container))
        emphases = []
        for st in self.start_ticks:
            if st in emphasis_ticks:
                emphases.append(self.emphasis_velocities[1])
            else:
                emphases.append(self.emphasis_velocities[0])
        self.emphases = emphases

    def build_emphasis_ticks_directly(self, emphasis_quantization, container, emphasis_divisions):
        # note: don't forget start_tick
        emphasis_ticks = []
        emphasis_divisions = np.array(emphasis_divisions) - 1
        epmhasis_ticks_temp = np.arange(0, container, emphasis_quantization).take(emphasis_divisions)
        for tick in epmhasis_ticks_temp:
            emphasis_ticks.extend(np.arange(tick, self.rhythm_len, container))
        emphases = []
        for st in self.start_ticks:
            if st in emphasis_ticks:
                emphases.append(self.emphasis_velocities[1])
            else:
                emphases.append(self.emphasis_velocities[0])
        self.emphases = emphases

    def _compute_num_notes(self, note_density):
        return int(self.rhythm_len * note_density / float(self.quantization))

    def _get_random_start_ticks(self, number_of_notes):
        return np.unique(sorted([
            self.find_nearest_note(
                value=random.randint(0, self.rhythm_len) + self.start_tick,
                note_type=self.quantization)
            for _ in range(number_of_notes)]))

    @staticmethod
    def find_nearest_note(value, note_type):
        mod = value % note_type
        if mod > (note_type / 2.0):  # round up
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
        for tick, duration, emphasis in zip(self.rhythm.start_ticks, self.rhythm.note_lengths, self.rhythm.emphases):
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
                chord.set_velocity_randomly_uniform(emphasis - 10, emphasis + 10)
            elif self.vel_method == "direct":
                chord.set_velocity(random.randint(emphasis - 10, emphasis + 10))
            else:
                chord.set_velocity_randomly_uniform(emphasis - 10, emphasis + 10)
            chord.set_duration(duration)
            chords.append(chord)
        return chords

    def make_chord_and_rhythm_same_length(self):
        pass


def add_tuples_to_track(track, df):
    for row in df.iterrows():
        data = row[1]
        pitch = data["pitch"] if isinstance(data["pitch"], int) else Note.index_from_string(data["pitch"])
        track.append(data["event_type_fun"](tick=data["tick"],
                                            velocity=data["velocity"],
                                            pitch=pitch))
    return track


class Melody:
    def __init__(self, root_note=None, octave=None, scale_name=None, melody_len=None, quantization=None,
                 note_density=None, note_len_choices=None):
        self.root_note = root_note
        self.octave = octave
        self.scale_name = scale_name
        self.melody_len = melody_len
        self.quantization = quantization  # this maybe should be at note level only
        self.note_density = note_density  # proportion of available ticks (determined by quantization) occupied by notes
        self.note_len_choices = note_len_choices

        self.root = Note((self.root_note, self.octave))
        self.named_scale = scale.NAMED_SCALES[self.scale_name]
        self.scale = Scale(self.root, self.named_scale)

        self.available_notes = [self.scale.get(x) for x in range(21, 30)]
        self.number_of_notes = self._compute_num_notes()
        self.start_ticks = self._get_start_ticks()

    def _compute_num_notes(self):
        return int(self.melody_len * self.note_density / float(self.quantization))

    def _get_start_ticks(self):
        return np.unique(sorted([Rhythm().find_nearest_note(random.randint(0, self.melody_len), self.quantization)
                                 for _ in range(self.number_of_notes)]))

    def _create_melody_note_tuple(self, start_tick):
        velocity = random.randint(50, 90)
        cur_note = random.choice(self.available_notes)
        cur_note = Note.index_from_string(cur_note.note + str(cur_note.octave))
        note_length = random.choice(self.note_len_choices)
        # ["event_type_fun", "tick", "duration", "pitch", "velocity"]
        return [
            MidiEventStager(midi.NoteOnEvent, start_tick, note_length, cur_note, velocity),
            MidiEventStager(midi.NoteOffEvent, start_tick+note_length, note_length, cur_note, 0)
        ]

    def create_melody(self):
        melody_notes = []
        for tick in self.start_ticks:
            melody_tuples = self._create_melody_note_tuple(tick)
            melody_notes.extend(melody_tuples)
        return melody_notes


class TrackBuilder:
    def __init__(self, pattern_constants, bpm, time_signature):
        # Set the track foundation for the chord progression, including time signature, BPM, and get an empty midi
        # Track object
        self.pc = pattern_constants
        self.bpm = bpm
        self.time_signature_numerator, self.time_signature_denominator = [int(i) for i in time_signature.split('/')]
        self.resolution = 440

        # The following will be set in initialize_or_reset_state()
        self.pattern = None
        self.track = None
        self.track_uuid = None
        self.track_filename = None
        self._initialize_or_reset_state()

    def _initialize_or_reset_state(self):
        self.track_uuid = str(uuid.uuid4())[0:7]
        self.pattern = midi.Pattern(resolution=self.pc.resolution)
        self.track = midi.Track()
        self.pattern.append(self.track)
        self.track.append(midi.SetTempoEvent(bpm=self.bpm))
        self.track.append(midi.TimeSignatureEvent(numerator=self.time_signature_numerator,
                                                  denominator=self.time_signature_denominator))

    def make_ticks_rel(self):
        number_before_negative = 0
        running_tick = 0
        for event in self.track:
            event.tick -= running_tick
            if event.tick >= 0:
                number_before_negative += 1
            else:
                print(number_before_negative)
            running_tick += event.tick

    def get_max_tick(self):
        return max([event.tick for event in self.track])

    def get_min_tick(self):
        return min([event.tick for event in self.track])

    def write_chord_progression_to_midi(self, chord_progression_rhythm, filename):
        self._initialize_or_reset_state()

        # Order the chord progression rhythm by tick and duration
        all_staged_events = []
        for chord in chord_progression_rhythm.chords:
            for staged_event in chord.staged_events:
                all_staged_events.append(staged_event)

        staged_events_df = convert_staging_events_to_dataframe(all_staged_events)
        staged_events_df.sort_values(by=["tick", "duration"], inplace=True)

        self.track = add_tuples_to_track(self.track, staged_events_df)

        # Add the end of track event, append it to the track
        eot = midi.EndOfTrackEvent(tick=self.get_max_tick() + 2 * self.pc.whole_note)
        self.track.append(eot)
        self.make_ticks_rel()
        self.track_filename = "example_outputs/" + filename + "_" + self.track_uuid + ".mid"
        midi.write_midifile(self.track_filename, self.pattern)

    def write_melody_to_midi(self, melody, filename):
        self._initialize_or_reset_state()

        staged_events_df = convert_staging_events_to_dataframe(melody)
        staged_events_df.sort_values(by=["tick", "duration"], inplace=True)

        self.track = add_tuples_to_track(self.track, staged_events_df)
        eot = midi.EndOfTrackEvent(tick=self.get_max_tick() + 2 * self.pc.whole_note)
        self.track.append(eot)
        self.make_ticks_rel()
        self.track_filename = "example_outputs/" + filename + "_" + self.track_uuid + ".mid"
        midi.write_midifile(self.track_filename, self.pattern)
