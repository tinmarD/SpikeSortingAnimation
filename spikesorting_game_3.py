import pygame
import numpy as np
import pandas as pd
import mne
import os
import re
import seaborn as sns
from scipy import signal
import h5py
import time
from scipy.io import wavfile
import matplotlib.pyplot as plt
from spikeAnalysis_utils_local import *

win_w, win_h = 1280, 720

edf_file_path = r'C:\Users\deudon\Desktop\SpikeSorting\_Data\VL14\DAY2_Elodie\data\edf_5kHz'
edf_filename = '20170113-104634_p1_0_600s_5kHz.edf'
edf_filename_30kHz = '20170113-104634_p1_0_600s_30kHz.edf'
spykingcircus_dirpath = r'C:\Users\deudon\Desktop\SpikeSorting\_Data\VL14\DAY2_Elodie\spykingcircus_results'
spykingcircus_results_filename = '20170113-104634-001_0.result.hdf5'
spykingcircus_clusters_filename = '20170113-104634-001_0.clusters.hdf5'
time_sel, chansel_pos = [152.7, 180], 0
artefact_filepath = r'C:\Users\deudon\Desktop\SpikeSorting\_Data\VL14\DAY2_Elodie\artefact_day2_epifar.csv'
# Sound
sound_samples_dir = r'C:\Users\deudon\Desktop\SpikeSorting\_Scripts\_Python\Animation\samples_shortsound_2'
# sound_samples_dir = r'C:\Users\deudon\Desktop\SpikeSorting\_Scripts\_Python\Animation\99Sounds99DrumSamples\SampleOGG_sel'
unit_sound_path = r'C:\Users\deudon\Desktop\SpikeSorting\_Scripts\_Python\Animation\pluck_3ms.ogg'

# Read artefact periods file
df = pd.read_csv(artefact_filepath, sep=';', decimal=',')
filename, keep_start, keep_end = np.array(df['Filename']), np.array(df['t_start']).astype(float), np.array(
    df['t_end']).astype(float)

# Read Spiking Circus results file
f_results = h5py.File(os.path.join(spykingcircus_dirpath, spykingcircus_results_filename), 'r')
f_clusters = h5py.File(os.path.join(spykingcircus_dirpath, spykingcircus_clusters_filename), 'r')

spktimes_grp = f_results['spiketimes']
# Sort data in increasing number of the unit
unit_names_temp = list(spktimes_grp.keys())
spktrains_temp = list(spktimes_grp.values())
dataset_num = np.zeros(len(unit_names_temp))
for idx, name in enumerate(unit_names_temp):
    dataset_num[idx] = re.findall('\d+', name)[0]
sort_vect = dataset_num.argsort()
# Order the spike trains and unit names
unit_names = np.array([unit_names_temp[i] for i in sort_vect])
spktrainsdataset = [spktrains_temp[i] for i in sort_vect]
# Get units appearing on the electrode selected
pref_el_temp = np.array(f_clusters['electrodes']).ravel().astype(int)
pref_el = pref_el_temp[sort_vect]
# Get all neurons of the appearing on the tetrode
tetrode_num = np.ceil((1 + chansel_pos) / 4.0)
tetrode_micro_wire_pos = np.arange(4 * (tetrode_num - 1), 4 * tetrode_num, 1).astype(int)
unit_sel_pos = np.where(np.isin(pref_el, tetrode_micro_wire_pos))[0]
unit_pref_wire = pref_el[unit_sel_pos] - min(tetrode_micro_wire_pos)
# unit_sel_pos = np.where(np.isin(pref_el, chansel_pos))[0]
n_unit_sel = len(unit_sel_pos)
if n_unit_sel == 0:
    raise ValueError('No spike on the selected channel {}. Select another one.'.format(chansel_pos))
else:
    print('{} units found on channel {} - tetrode {}'.format(n_unit_sel, chansel_pos, tetrode_num))
unit_names_sel = unit_names[unit_sel_pos]
spikestimes = [np.array(f_results['spiketimes']['temp_{}'.format(i)]).flatten() / 30000 - time_sel[0] for i in unit_sel_pos]
amplitudes = [np.array(f_results['amplitudes']['temp_{}'.format(i)]).flatten() / 30000 - time_sel[0] for i in unit_sel_pos]

# Get the spike times in the original file
spikestimes_sync = []
spikestimes = [np.array(f_results['spiketimes']['temp_{}'.format(i)]).flatten() / 30000 - time_sel[0] for i in unit_sel_pos]
for i in range(len(spikestimes)):
    spiketimes_i = spikestimes[i]
    # Select spikes of first file part ! The edf file must be the first part !
    spiketimes_i = spiketimes_i[spiketimes_i < 600]
    spiketimes_i_sync = np.array([find_original_time(spiketimes_i[j], filename, keep_start, keep_end)[0] for j in range(spiketimes_i.size)])
    filename_i = np.array([find_original_time(spiketimes_i[j], filename, keep_start, keep_end)[1] for j in range(spiketimes_i.size)])
    spiketimes_i_sync = spiketimes_i_sync[filename_i == edf_filename_30kHz]
    if edf_filename_30kHz not in filename_i:
        print('Could not find filename {} in the list of filename'.format(edf_filename))
    spikestimes_sync.append(np.array(spiketimes_i_sync))

spikestimes = spikestimes_sync
if sum([not spikes.any() for spikes in spikestimes]) == n_unit_sel:
    raise ValueError('Could not find spikes in the selected channel and time period. Change them and check that filenames correponds in the artefact-free periods file')

# Read edf file
raw = mne.io.read_raw_edf(os.path.join(edf_file_path, edf_filename))
sample_sel = raw.time_as_index(time_sel)
raw_sig = raw.get_data(tetrode_micro_wire_pos, sample_sel[0], sample_sel[1]).squeeze()
channame = raw.info['ch_names'][chansel_pos]
fs = int(raw.info['sfreq'])
raw_sig_min = np.tile(raw_sig.min(axis=1), (raw_sig.shape[1], 1)).T
raw_sig_max = np.tile(raw_sig.max(axis=1), (raw_sig.shape[1], 1)).T
raw_sig_scaled_unit = (raw_sig - raw_sig_min) / (raw_sig_max - raw_sig_min) - 0.5
unit_half_duration = 0.001
unit_half_duration_samples = int(unit_half_duration*fs)

spikestimessel = [spikestimes_i[spikestimes_i<(fs*time_sel[1])] for spikestimes_i in spikestimes]
for i in range(0, n_unit_sel):
    print('Unit {} : {} spikes'.format(i, len(spikestimessel[i])))
# Scale signals
n_sig_total = 10
offset = win_h / n_sig_total
sig_offset = np.tile(np.arange(n_sig_total)*offset + int(offset/2), (raw_sig.shape[1], 1)).T
sig_spread_raw = win_h/2
raw_sig_scaled_win = raw_sig_scaled_unit * sig_spread_raw + sig_offset[:4]
vspace_unit = win_h / 20
unit_offset = np.array([-1.5, -0.5, 0.5, 1.5])*vspace_unit + 0.75*win_h


def find_ap_in_time_window(spiketimes, t_start, t_end):
    times, units = [], []
    for i, spiketime_i in enumerate(spiketimes):
        sel_i = (spiketime_i > t_start) & (spiketime_i < t_end)
        times.append(spiketime_i[sel_i])
        units.append(i*np.ones(sel_i.sum()))
    return times, units

# PyGame
n_channels = 4
pygame.init()
pygame.mixer.pre_init(44100, size=-16, channels=4, buffer=256)
pygame.mixer.init()
pygame.mixer.set_num_channels(n_channels)
pygame.font.init()
myfont = pygame.font.SysFont('Comic Sans MS', 15)
clock = pygame.time.Clock()
# Sounds
# Load a sound for each neuron
sound_list = np.array(os.listdir(sound_samples_dir))
unit_sound_name = sound_list[np.random.randint(0, len(sound_list), n_unit_sel)]
# unit_sounds = [pygame.mixer.Sound(os.path.join(sound_samples_dir, unit_sound_name_i)) for unit_sound_name_i in unit_sound_name]
unit_sounds = [pygame.mixer.Sound(os.path.join(sound_samples_dir, sound_list[i % len(sound_list)])) for i in range(n_unit_sel)]
for i in range(0, n_unit_sel):
    unit_sounds[i].set_volume(0.5)
color_pal = sns.color_palette(n_colors=2)
# Create channels
channels = [pygame.mixer.Channel(j) for j in range(0, n_channels)]
unit_color_pal = [[1, 0, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0.5], [1, 0.5, 0],
                  [0, 1, 0.5], [0, 1, 0]]
color_units = [(255*np.array(unit_color_pal[i % len(unit_color_pal)])).astype(int) for i in range(n_unit_sel)]
color_rawsig = 255*np.array(sns.color_palette("Paired")[0])
win = pygame.display.set_mode((win_w, win_h))
running, i = 1, 0
fps, i_step, i_step_old = 130, 3, 3
t_start, t_end = i / fs, (i + win_w) / fs
plot_raw, plot_units, play_ap = True, True, True
draw_rect = True
ap_to_play = []  # List of Action Potential to be played ( AP corresponding unit sound is played when the AP is on
                 # the middle of the screen
ap_played = []   # List of Action Potential already played
last_ap_time = -10
clock_start = time.clock()
i_chan = 0
# Main pygame loop
while running:
    lines_raw = [np.vstack([np.arange(win_w), raw_sig_scaled_win[i_sig][i:i+win_w]]).T for i_sig in range(4)]
    win.fill((0, 0, 0))
    for i_sig in range(4):
        pygame.draw.aalines(win, color_rawsig,  False, lines_raw[i_sig], 5)
    # Plot each unit of the current time segment with a different color
    if plot_units:
        # Find if there are neurons in the time window
        times, units = find_ap_in_time_window(spikestimes, t_start, t_end)
        for i_unit in range(n_unit_sel):
            unit_offset_i = unit_offset[unit_pref_wire[i_unit]]
            # pygame.draw.aaline(win, [10, 10, 10], [0, unit_offset_i], [win_w, unit_offset_i], False)
            for t_ap in times[i_unit]:
                # Plot a vertical bar a the time of each AP
                line_ap_x = int((t_ap-t_start)*fs)
                pygame.draw.line(win, color_units[i_unit], [line_ap_x, unit_offset_i - win_h/80],
                                   [line_ap_x, unit_offset_i + win_h/80], 2)
                # Color the raw signal
                tind_range = np.arange(int(t_ap * fs) - unit_half_duration_samples,
                                       int(t_ap * fs) + unit_half_duration_samples)
                for j in range(4):
                    lines_ap = np.vstack([tind_range - int(t_start * fs),
                                          raw_sig_scaled_win[j][tind_range]]).T
                    pygame.draw.aalines(win, color_units[i_unit], False, lines_ap, 8)
                if (t_ap, i_unit) not in ap_played:
                    ap_to_play.append((t_ap, i_unit))
                    ap_played.append((t_ap, i_unit))
        if play_ap:
            # Play sounds if the AP is on the center of the screen
            t_center_offset = t_start + 0.55 * (t_end - t_start)
            ap_deleted_pos = []
            if ap_to_play:
                for i_ap, (t_ap_sound, i_unit) in enumerate(ap_to_play):
                    if t_ap_sound < t_center_offset:
                        channels[i_chan].play(unit_sounds[i_unit])
                        i_chan = 0 if i_chan >= (n_channels-1) else i_chan + 1
                        ap_deleted_pos.append(i_ap)
                        last_ap_time = max(t_ap_sound, last_ap_time)
                        last_ap_color = color_units[i_unit]
                ap_to_play = [item for j, item in enumerate(ap_to_play) if j not in ap_deleted_pos]
            # Draw box
            rect_w, rect_h = 15, 50
            last_ap_delay = (t_center_offset - last_ap_time)
            if last_ap_delay > 0.03 or last_ap_delay <= 0:
                rect_color = (200, 200, 200)
            else:
                rect_color = (int(255 * (last_ap_delay / 0.03)), int(255 * (last_ap_delay / 0.03)), 0)
            pygame.draw.rect(win, rect_color, (win_w / 2 - rect_w / 2, 0.92 * win_h, rect_w, rect_h))
    # Draw time
    time_surf = myfont.render('t = {:.1f}s'.format(t_start+0.5*(t_end-t_start)), False, (200, 200, 200))
    win.blit(time_surf, (5, 5))
    # Refresh
    pygame.display.flip()
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = 0
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_UP:
                fps = min(fps+10, 240)
            if ev.key == pygame.K_DOWN:
                fps = max(15, fps-10)
            if ev.key == pygame.K_SPACE:
                if not i_step:
                    i_step = i_step_old
                elif i_step:
                    i_step_old = i_step
                    i_step = 0
            if ev.key == pygame.K_3:
                plot_units = not plot_units
            if ev.key == pygame.K_4:
                play_ap = not play_ap
    clock.tick(fps)
    i += i_step
    t_start, t_end = i/fs, (i+win_w)/fs

pygame.quit()
pygame.mixer.quit()


