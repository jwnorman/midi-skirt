# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import peakutils
# import seaborn as sns

from scipy import signal
from scipy.io import wavfile
from scipy.stats import rankdata
from sklearn.cluster import KMeans

from midi_skirt import *


def load_wav_file(filename):
    fs, data = wavfile.read(filename)
    try:
        channel0 = data[:,0]
    except:
        channel0 = data
    return channel0


def get_peak_info(data, thres=7000, min_dist=44100 / 4.0):
    ids = peakutils.indexes(data, thres=thres, min_dist=min_dist)
    peaks = data[ids]
    filtered_ids_and_peaks = [(id_temp, peak) for (id_temp, peak) in zip(ids, peaks) if peak > thres]
    ids = [temp[0] for temp in filtered_ids_and_peaks]
    peaks = [temp[1] for temp in filtered_ids_and_peaks]
    return (ids, peaks)


def get_bin_markers(peak_ids, max_time_id):
    bin_markers = []
    bin_markers.append(0)
    for index in range(len(peak_ids) - 1):
        temp_mean = (peak_ids[index] + peak_ids[index + 1]) / 2
        bin_markers.append(temp_mean)
    bin_markers.append(max_time_id)
    return bin_markers


def get_lit(bins):
    """
    input: [30,40,80,120,180,300]
    output: [(30,40), (40,80), (80,120), (120,180), (180,300)]
    """
    bins = zip(bins, np.roll(bins, -1))
    bins.pop()
    return bins


def get_fft(segment):
    fd = abs(np.fft.fft(segment))
    freq = abs(np.fft.fftfreq(len(fd), 1 / float(44100)))
    return fd, freq


def get_prominent_freq(segment):
    fd, freq = get_fft(segment)
    smoothed = signal.savgol_filter(fd, 169, 3)
    max_freq = np.argmax(smoothed)
    freq_with_max_amplitude = freq[max_freq]
    return freq_with_max_amplitude


def create_signatures(segments):
    promiment_freqs = [get_prominent_freq(segment) for segment in segments]
    return pd.DataFrame({
        "prominent_freq": promiment_freqs
    })


def cluster_signatures(signatures, k=2):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=100).fit(signatures)
#    rank_mapping = dict(zip(kmeans.labels_, rankdata(kmeans.cluster_centers_, "dense")))
    rank_mapping = dict(zip(range(k), rankdata(kmeans.cluster_centers_, "dense")))
    print kmeans.labels_
    print kmeans.cluster_centers_
    import pdb; pdb.set_trace()
    return pd.DataFrame({
        "clusters": kmeans.labels_,
        "rank": [rank_mapping[label] for label in kmeans.labels_]
    })


# signatures = pd.DataFrame([340, 5000, 250, 4000, 290, 4600, 468, 2668], columns=["prominent_freq"])
signatures = pd.DataFrame([259, 4400, 280, 5400, 5300, 5499, 1702], columns=["prominent_freq"])

def get_midi_info(time_ids, peaks, signatures, clusters):
    info = pd.DataFrame({
        "time_ids": time_ids,
        "peaks": peaks
    })
    return pd.concat([info, clusters, signatures], axis=1)


def get_midi_info_from_wav(filename):
    data = load_wav_file(filename)
    time_ids, peaks = get_peak_info(data, 6000, 44100 / 8.0)
    bin_markers = get_bin_markers(time_ids, len(data))
    buckets = get_lit(bin_markers)
    segments = [data[np.arange(int(bucket[0]), int(bucket[1]))] for bucket in buckets]
    signatures = create_signatures(segments)
    clusters = cluster_signatures(signatures, 2)
    midi_info = get_midi_info(time_ids, peaks, signatures, clusters)
    return midi_info




# map from cluster to drum
cluster_to_drum_mapping = {
    1: "C3",
    2: "D3"
}


class DrumMapping:
    def kick_and_snare(self, cluster):
        return {
            1: "C3",
            2: "D3"
        }[cluster]

    def kick_snare_and_high_hat(self):
        pass

    def snare_and_high_hat(self):
        pass

    def kick_snare_high_hat_open_high_hat(self):
        pass


def convert_amplitude_to_velocity(amp, min_amp, max_amp, min_vel, max_vel):
    try:
        return int((max_vel - min_vel) / float(max_amp - min_amp) * amp + min_vel)
    except:
        return 0


def convert_sample_to_tick(pc, time_id, num_samples=44000):
    spb = 60 / 120.0
    tpb = pc.resolution
    return int((time_id / float(num_samples)) * (1.0 / spb * tpb))


def convert_wav_to_midi(filename):
    mi = get_midi_info_from_wav(filename)

    pc = PatternConstants(resolution=440)
    pattern = midi.Pattern(resolution=pc.resolution)
    track = midi.Track()
    pattern.append(track)
    all_staged_events = []
    min_amp = mi.peaks.min()
    max_amp = mi.peaks.max()
    import pdb; pdb.set_trace()
    drum_mapping = DrumMapping().kick_and_snare
    for row in mi.itertuples():
        # drum = cluster_to_drum_mapping[row.clusters]
        drum = drum_mapping(row.rank)
        vel = convert_amplitude_to_velocity(row.peaks, min_amp, max_amp, 50, 100)
        tick = convert_sample_to_tick(pc, row.time_ids)
        all_staged_events.append(MidiEventStager(midi.NoteOnEvent, tick, pc.eighth_note, drum, vel))
        all_staged_events.append(MidiEventStager(midi.NoteOffEvent, tick + pc.eighth_note, pc.eighth_note, drum, vel))
    df = convert_staging_events_to_dataframe(all_staged_events)
    df.sort_values(by=["tick", "duration"], inplace=True)
    track = add_tuples_to_track(track, df)
    eot = midi.EndOfTrackEvent(tick=get_max_tick(track) + 2 * pc.whole_note)
    track.append(eot)
    track = make_ticks_rel(track)
    return pattern

# os.system("sox /Users/jacknorman1/Desktop/beatbox_2.wav /Users/jacknorman1/Desktop/beatbox_2.wav")

filename = "/Users/jacknorman1/Desktop/beatbox_copy.wav"
pattern = convert_wav_to_midi(filename)
midi.write_midifile("example.mid", pattern)


filename = "/Users/jacknorman1/Desktop/beatbox_2_copy.wav"
pattern = convert_wav_to_midi(filename)
midi.write_midifile("example.mid", pattern)




# entire_fd, entire_freq = get_fft(data)
# plt.plot(entire_freq, entire_fd)
# plt.show()


# # plot smoothed segment fft with identified max
# smoothed = signal.savgol_filter(fd, 169, 3)
# max_freq = np.argmax(smoothed)
# plt.plot(freq, smoothed)
# plt.scatter(max_freq, max(smoothed), color="red", linewidth=20)
# plt.show()



# plot segment with its fft
# def plot_segment_with_fft(fig1, plot_num, r, c, p):
#     ax1 = fig1.add_subplot(r, c, p) # 821
#     bucket = buckets[plot_num]
#     segment = data[np.arange(int(bucket[0]), int(bucket[1]))]
#     ax1.plot(data)
#     ax1.axvline(x=bucket[0])
#     ax1.axvline(x=bucket[1])
#     ax1.scatter(peak_info[0][plot_num], peak_info[1][plot_num])

#     ax2 = fig1.add_subplot(r, c, p + 1)
#     fd = abs(np.fft.fft(segment))
#     freq = abs(np.fft.fftfreq(len(fd), 1 / float(44100)))
#     freq = abs(np.fft.fftfreq(len(fd), 1 / float(44100)))
#     ax2.scatter(freq, fd)

# fig1 = plt.figure()
# plot_segment_with_fft(fig1, 0, 8, 2, 1)
# plot_segment_with_fft(fig1, 1, 8, 2, 3)
# plot_segment_with_fft(fig1, 2, 8, 2, 5)
# plot_segment_with_fft(fig1, 3, 8, 2, 7)
# plot_segment_with_fft(fig1, 4, 8, 2, 9)
# plot_segment_with_fft(fig1, 5, 8, 2, 11)
# plot_segment_with_fft(fig1, 6, 8, 2, 13)
# plot_segment_with_fft(fig1, 7, 8, 2, 15)
# plt.show()
