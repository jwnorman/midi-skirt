# import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import peakutils
import sys
# import seaborn as sns

from detect_peaks import detect_peaks
from scipy import signal
from scipy.io import wavfile
from scipy.stats import rankdata
from sklearn.cluster import KMeans

from midi_skirt import *


class DrumMapping:
    def kick_and_snare(self, cluster):
        return {
            1: "C3",
            2: "D3"
        }[cluster]

    def kick_snare_and_high_hat(self, cluster):
        return {
            1: "C3",
            2: "D3",
            3: "F#3"
        }[cluster]

    def snare_and_high_hat(self):
        pass

    def kick_snare_high_hat_open_high_hat(self):
        pass


def get_peak_info(data, min_dist):
    print "Finding peaks"
    thres = np.percentile(data, 98)
    ids = detect_peaks(data, mph=thres, mpd=min_dist)
    peaks = data[ids]
    filtered_ids_and_peaks = [(id_temp, peak) for (id_temp, peak) in zip(ids, peaks) if peak > thres]
    ids = [temp[0] for temp in filtered_ids_and_peaks]
    peaks = [temp[1] for temp in filtered_ids_and_peaks]
    if len(peaks) == 0:
        sys.exit("No peaks are found. Did you record really quietly?")
    return (ids, peaks)


def get_fft(segment, rate):
    segment = np.append(segment, np.zeros(int(2**np.ceil(np.log2(len(segment))) - len(segment))))
    fd = abs(np.fft.fft(segment))
    freq = abs(np.fft.fftfreq(len(fd), 1 / float(rate)))
    return fd, freq


def get_lit(bins):
    """
    input: [30,40,80,120,180,300]
    output: [(30,40), (40,80), (80,120), (120,180), (180,300)]
    """
    bins = zip(bins, np.roll(bins, -1))
    bins.pop()
    return bins


def load_wav_file(filename):
    print "Loading wav file"
    fs, data = wavfile.read(filename)
    try:
        channel0 = data[:,0]
    except:
        channel0 = data
    return channel0


class WavToMidi:
    def __init__(self, filename, rate, drum_mapping, k):
        self.filename = filename
        self.rate = rate
        self.pc = PatternConstants(resolution=440)
        self.pattern = midi.Pattern(resolution=self.pc.resolution)
        self.track = midi.Track()
        self.pattern.append(self.track)
        self.all_staged_events = []
        self.drum_mapping = drum_mapping
        self.k = k

    def _get_bin_markers(self, peak_ids, max_time_id):
        print "Finding bin markers"
        bin_markers = []
        bin_markers.append(0)
        for index in range(len(peak_ids) - 1):
            temp_mean = (peak_ids[index] + peak_ids[index + 1]) / 2
            bin_markers.append(temp_mean)
        bin_markers.append(max_time_id)
        return bin_markers

    def _get_prominent_freq(self, segment):
        fd, freq = get_fft(segment, self.rate)
        smoothed = signal.savgol_filter(fd, 169, 3)
        max_freq = np.argmax(smoothed)
        freq_with_max_amplitude = freq[max_freq]
        return freq_with_max_amplitude

    def _create_signatures(self, segments):
        print "Generating sound signatures"
        promiment_freqs = [self._get_prominent_freq(segment) for segment in segments]
        return pd.DataFrame({
            "prominent_freq": promiment_freqs
        })

    def _cluster_signatures(self, signatures, k=2):
        print "Clustering signatures"
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=100).fit(signatures)
        rank_mapping = dict(zip(range(k), rankdata(kmeans.cluster_centers_, "dense")))
        return pd.DataFrame({
            "clusters": kmeans.labels_,
            "rank": [rank_mapping[label] for label in kmeans.labels_]
        })

    def _get_midi_info(self, time_ids, peaks, signatures, clusters):
        info = pd.DataFrame({
            "time_ids": time_ids,
            "peaks": peaks
        })
        return pd.concat([info, clusters, signatures], axis=1)

    def _get_midi_info_from_wav(self):
        data = load_wav_file(self.filename)
        time_ids, peaks = get_peak_info(data, self.rate/8.0)
        bin_markers = self._get_bin_markers(time_ids, len(data))
        buckets = get_lit(bin_markers)
        segments = [data[np.arange(int(bucket[0]), int(bucket[1]))] for bucket in buckets]
        signatures = self._create_signatures(segments)
        clusters = self._cluster_signatures(signatures, self.k)
        midi_info = self._get_midi_info(time_ids, peaks, signatures, clusters)
        return midi_info

    def convert_amplitude_to_velocity(self, amp, min_amp, max_amp, min_vel, max_vel):
        try:
            return int((max_vel - min_vel) / float(max_amp - min_amp) * amp + min_vel)
        except:
            return 0

    def convert_sample_to_tick(self, pc, time_id, num_samples=44100):
        spb = 60 / 120.0
        tpb = pc.resolution
        return int((time_id / float(num_samples)) * (1.0 / spb * tpb))

    def convert_wav_to_midi(self):
        mi = self._get_midi_info_from_wav()
        min_amp = mi.peaks.min()
        max_amp = mi.peaks.max()
        for row in mi.itertuples():
            drum = self.drum_mapping(row.rank)
            vel = self.convert_amplitude_to_velocity(row.peaks, min_amp, max_amp, 50, 100)
            tick = self.convert_sample_to_tick(self.pc, row.time_ids, self.rate)
            self.all_staged_events.append(MidiEventStager(midi.NoteOnEvent, tick, self.pc.eighth_note, drum, vel))
            self.all_staged_events.append(MidiEventStager(midi.NoteOffEvent, tick + self.pc.eighth_note, self.pc.eighth_note, drum, vel))
        df = convert_staging_events_to_dataframe(self.all_staged_events)
        df.sort_values(by=["tick", "duration"], inplace=True)
        self.track = add_tuples_to_track(self.track, df)
        eot = midi.EndOfTrackEvent(tick=get_max_tick(self.track) + 2 * self.pc.whole_note)
        self.track.append(eot)
        self.track = make_ticks_rel(self.track)
        return self.pattern


filename = "/Users/admin/Desktop/BeatBoxExamples/sample3_5.wav"
filename = "~/Desktop/beatbox_2.wav"
rate = 44100
drum_mapping = DrumMapping().kick_and_snare
k = 2
os.system("sox {} --rate {} temp_file.wav".format(filename, rate))
wav_to_midi = WavToMidi("temp_file.wav", rate, drum_mapping, k)
pattern = wav_to_midi.convert_wav_to_midi()
midi.write_midifile("example.mid", pattern)
os.system("rm temp_file.wav")


filename = "/Users/admin/Desktop/BeatBoxExamples/sample3_7.wav"
rate = 44100
drum_mapping = DrumMapping().kick_snare_and_high_hat
k = 3
os.system("sox {} --rate {} temp_file.wav".format(filename, rate))
wav_to_midi = WavToMidi("temp_file.wav", rate, drum_mapping, k)
pattern = wav_to_midi.convert_wav_to_midi()
midi.write_midifile("example.mid", pattern)
os.system("rm temp_file.wav")

