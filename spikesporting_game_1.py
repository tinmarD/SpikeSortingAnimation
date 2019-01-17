import pygame
import numpy as np
import mne
import os
import re
import seaborn as sns
from scipy import signal
import h5py
import time
from scipy.io import wavfile


## Change these parameters
win_w, win_h = 1280, 720
edf_file_path = r'C:\Users\deudon\Desktop\SpikeSorting\_Data\002RM_day4_pointes\signal\monopolaire_5kHz_d4_post_crise'
edf_filename = 'p30d4p1_0_600s_5kHz_micro_m.edf'
time_sel, chansel_pos = [2, 14], 40

tetrode_num = np.ceil((1 + chansel_pos) / 4.0)
tetrode_micro_wire_pos = np.arange(4 * (tetrode_num - 1), 4 * tetrode_num, 1).astype(int)

# Read edf file
raw = mne.io.read_raw_edf(os.path.join(edf_file_path, edf_filename))
sample_sel = raw.time_as_index(time_sel)
raw_sig = raw.get_data(tetrode_micro_wire_pos, sample_sel[0], sample_sel[1]).squeeze()
channame = raw.info['ch_names'][chansel_pos]
fs = int(raw.info['sfreq'])
raw_sig_min = np.tile(raw_sig.min(axis=1), (raw_sig.shape[1], 1)).T
raw_sig_max = np.tile(raw_sig.max(axis=1), (raw_sig.shape[1], 1)).T
raw_sig_scaled_unit = (raw_sig - raw_sig_min) / (raw_sig_max - raw_sig_min) - 0.5

# Scale signals
n_sig_total = 4
offset = win_h / n_sig_total
sig_offset = np.tile(np.arange(n_sig_total)*offset + int(offset/2), (raw_sig.shape[1], 1)).T
sig_spread = win_h / 2
raw_sig_scaled_win = raw_sig_scaled_unit * sig_spread + sig_offset

# PyGame
pygame.init()
pygame.mixer.pre_init(44100, size=-16, channels=2, buffer=128)
pygame.mixer.init()
pygame.font.init()
myfont = pygame.font.SysFont('Comic Sans MS', 15)
clock = pygame.time.Clock()
color_pal = sns.color_palette(n_colors=2)
color_rawsig, color_filtsig = (255*np.array(color_pal[0])).astype(int), (255*np.array(color_pal[1])).astype(int)
win = pygame.display.set_mode((win_w, win_h))
running, i = 1, 0
fps, i_step, i_step_old = 130, 3, 3
t_start, t_end = i / fs, (i + win_w) / fs
clock_start = time.clock()
while running:
    lines_raw = [np.vstack([np.arange(win_w), raw_sig_scaled_win[i_sig][i:i+win_w]]).T for i_sig in range(0, n_sig_total)]
    win.fill((0, 0, 0))
    for i_sig in range(0, n_sig_total):
        pygame.draw.aalines(win, color_rawsig,  False, lines_raw[i_sig], 5)
    # Draw time
    time_surf = myfont.render('t = {:.2f}s'.format(t_start+0.5*(t_end-t_start)), False, (200, 200, 200))
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
    clock.tick(fps)
    i += i_step
    t_start, t_end = i/fs, (i+win_w)/fs


pygame.quit()
pygame.mixer.quit()

t_final = t_end
elapsed_time = time.clock() - clock_start
# Write first signal of the tetrode as a wavfile
wavfile.write('raw_sig.wav', fs, raw_sig_scaled_unit[0, 0:int(t_final*fs)])
